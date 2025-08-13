# src/generate.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from cvae import CVAE


# ------------------------
# Latent refinement helper
# ------------------------
def _latent_refine(model, y_star_t, steps=200, lr=0.2, lam=1e-3):
    """
    Optimize z so that prop_head(z) ≈ y* (y in scaled space).
    L2 regularization on z helps keep samples reasonable.
    """
    model.eval()

    # Start from conditional prior mean given target y*
    with torch.no_grad():
        p_mu, _ = model.prior(y_star_t)
    z = p_mu.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        y_pred = model.prop_head(z)                   # scaled prediction
        loss = torch.mean((y_pred - y_star_t) ** 2) + lam * torch.mean(z ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return z.detach()                                 # keep graph out of sampling


# ------------------------
# Suggestion generator
# ------------------------
def suggest(model, x_scaler, y_scaler, feature_cols, y_target_scalar=1200.0, N=25, refine=True):
    # Scale target y
    y_star = np.array(y_target_scalar, dtype=np.float32).reshape(1, -1)
    y_star_s = y_scaler.transform(y_star)
    y_star_t = torch.tensor(y_star_s, dtype=torch.float32)

    # Get a base latent vector
    if refine:
        z_base = _latent_refine(model, y_star_t, steps=200, lr=0.2, lam=1e-3)
    else:
        with torch.no_grad():
            z_base, _ = model.prior(y_star_t)

    X_suggestions_s = []
    y_pred_s_list = []

    # Sample around z_base (no grads needed)
    with torch.no_grad():
        for _ in range(N):
            z = z_base + 0.05 * torch.randn_like(z_base)
            x_hat_s = model.dec(z, y_star_t).cpu().numpy()           # scaled X
            y_pred_s = model.prop_head(z).cpu().numpy()              # scaled y

            X_suggestions_s.append(x_hat_s[0])
            y_pred_s_list.append(y_pred_s[0, 0])

    # Unscale back to original units
    X_suggestions = x_scaler.inverse_transform(np.array(X_suggestions_s))
    y_pred = y_scaler.inverse_transform(np.array(y_pred_s_list).reshape(-1, 1)).ravel()

    # Build DataFrame
    df = pd.DataFrame(X_suggestions, columns=feature_cols)

    # Optional: clamp and renormalize any *_frac columns to [0,1] and sum≈1
    frac_cols = [c for c in feature_cols if c.endswith("_frac")]
    if frac_cols:
        # Clamp to [0,1]
        df[frac_cols] = df[frac_cols].clip(lower=0.0, upper=1.0)
        # Renormalize rows with positive sum to exactly 1
        row_sums = df[frac_cols].sum(axis=1).replace(0, np.nan)
        df.loc[row_sums.notna(), frac_cols] = (
            df.loc[row_sums.notna(), frac_cols].div(row_sums[row_sums.notna()], axis=0)
        )

    # Round categorical encodings to nearest int
    cat_cols = [
        "Equilibrium_Conditions_encoded",
        "Single_Multiphase_encoded",
        "Type_Present_Phases_encoded",
        "Tension_Compression_encoded",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].round().astype(int)

    # Add model’s predicted yield strength (in MPa)
    df.insert(0, "Predicted_Yield_Strength", y_pred)
    # (No *_pct columns added — fractions are sufficient.)

    return df


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True, help="Target yield strength (MPa)")
    parser.add_argument("--N", type=int, default=25, help="Number of candidates to generate")
    parser.add_argument("--refine", action="store_true", help="Use latent refinement of z")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load scalers and feature names
    scalers = joblib.load(
        "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
        "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib"
    )
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]
    feature_cols = scalers["feature_cols"]

    # Load model
    x_dim = len(feature_cols)
    y_dim = 1
    model = CVAE(x_dim=x_dim, y_dim=y_dim, z_dim=4, hidden=128).to(device)
    model.load_state_dict(torch.load(
        "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
        "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt",
        map_location=device
    ))
    model.eval()

    # Generate suggestions
    df_suggestions = suggest(
        model, x_scaler, y_scaler, feature_cols,
        y_target_scalar=args.target, N=args.N, refine=args.refine
    )

    # Save
    out_path = Path(
        f"/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
        f"Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/outputs/suggestions_y{int(args.target)}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_suggestions.to_csv(out_path, index=False)
    print(f"Saved {args.N} suggestions for target {args.target} MPa → {out_path}")


if __name__ == "__main__":
    main()