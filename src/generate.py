# src/generate.py
import argparse
import torch
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from cvae import CVAE

# ------------------------
# Latent refinement helper
# ------------------------
def _latent_refine(model, y_star_t, steps=30, lr=0.05, lam=1e-3):
    """Optimize z so that prop_head(z) ≈ y*; small L2 regularization on z."""
    model.eval()
    with torch.no_grad():
        p_mu, _ = model.prior(y_star_t)
    z = p_mu.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([z], lr=0.2)
    for _ in range(200):
        y_pred = model.prop_head(z)
        loss = torch.mean((y_pred - y_star_t) ** 2) + lam * torch.mean(z ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return z.detach()

# ------------------------
# Suggestion generator
# ------------------------
def suggest(model, x_scaler, y_scaler, feature_cols, y_target_scalar=1200.0, N=25, refine=True):
    # Scale target y
    y_target_scaled = y_scaler.transform(np.array(y_target_scalar).reshape(1, -1))
    y_star_t = torch.tensor(y_target_scaled, dtype=torch.float32)

    # Refinement step (requires gradients)
    if refine:
        z_refined = _latent_refine(model, y_star_t, steps=40, lr=0.05, lam=1e-3)
    else:
        with torch.no_grad():
            p_mu, _ = model.prior(y_star_t)
            z_refined = p_mu

    # Sampling loop — no grads needed
    with torch.no_grad():
        X_suggestions = []
        for _ in range(N):
            z = z_refined + 0.05 * torch.randn_like(z_refined)
            x_hat = model.dec(z, y_star_t).numpy()
            X_suggestions.append(x_hat[0])

    # Unscale back to original units
    X_suggestions_unscaled = x_scaler.inverse_transform(np.array(X_suggestions))

    # Convert to DataFrame
    df_suggestions = pd.DataFrame(X_suggestions_unscaled, columns=feature_cols)

    # Round categorical encodings
    cat_cols = [
        "Equilibrium_Conditions_encoded",
        "Single_Multiphase_encoded",
        "Type_Present_Phases_encoded",
        "Tension_Compression_encoded",
    ]
    for col in cat_cols:
        if col in df_suggestions.columns:
            df_suggestions[col] = df_suggestions[col].round().astype(int)

    return df_suggestions

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True, help="Target yield strength")
    parser.add_argument("--N", type=int, default=25, help="Number of candidates to generate")
    parser.add_argument("--refine", action="store_true", help="Use latent refinement")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load scalers and feature names from dict
    scalers = joblib.load("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib")
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]
    feature_cols = scalers["feature_cols"]

    # Load model
    x_dim = len(feature_cols)
    y_dim = 1
    model = CVAE(x_dim=x_dim, y_dim=y_dim, z_dim=4, hidden=128).to(device)
    model.load_state_dict(torch.load("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt", map_location=device))
    model.eval()

    # Generate suggestions
    df_suggestions = suggest(
        model, x_scaler, y_scaler, feature_cols,
        args.target, args.N, refine=args.refine
    )

    # Save to CSV
    out_path = Path(f"/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/outputs/suggestions_y{int(args.target)}.csv")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df_suggestions.to_csv(out_path, index=False)
    print(f"Saved {args.N} suggestions for target {args.target} MPa → {out_path}")

if __name__ == "__main__":
    main()