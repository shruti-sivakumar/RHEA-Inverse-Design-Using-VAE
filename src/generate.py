# src/generate.py
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from cvae import CVAE

# ------------------------
# Latent refinement helper
# ------------------------
def _latent_refine(model, y_cond_t, y_prop_t, steps=200, lr=0.2, lam=1e-3):
    """
    Optimize z so that prop_head(z) ≈ y_target.
    L2 regularization on z helps keep samples reasonable.
    """
    model.eval()
    with torch.no_grad():
        p_mu, _ = model.prior(y_cond_t)
    z = p_mu.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        y_pred = model.prop_head(z)
        loss = torch.mean((y_pred - y_prop_t) ** 2) + lam * torch.mean(z ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach()

# ------------------------
# Suggestion generator
# ------------------------
def suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
            y_target_scalar=1200.0, temp=None, N=50, refine=True, max_elements=None):

    # Scale property target (yield strength)
    y_prop_s = y_prop_scaler.transform(np.array([[y_target_scalar]], dtype=np.float32))
    y_prop_t = torch.tensor(y_prop_s, dtype=torch.float32)

    # Scale conditioning (temperature)
    if temp is not None:
        y_cond_s = y_cond_scaler.transform(np.array([[temp]], dtype=np.float32))
    else:
        y_cond_s = np.zeros((1, len(y_cond_scaler.mean_)))  # neutral conditioning
    y_cond_t = torch.tensor(y_cond_s, dtype=torch.float32)

    # Get base latent
    if refine:
        z_base = _latent_refine(model, y_cond_t, y_prop_t)
    else:
        with torch.no_grad():
            z_base, _ = model.prior(y_cond_t)

    X_suggestions_s, y_pred_s_list = [], []
    with torch.no_grad():
        for _ in range(N):
            z = z_base + 0.05 * torch.randn_like(z_base)
            x_hat_s = model.dec(z, y_cond_t).cpu().numpy()
            y_pred_s = model.prop_head(z).cpu().numpy()
            X_suggestions_s.append(x_hat_s[0])
            y_pred_s_list.append(y_pred_s[0, 0])

    # Inverse transform
    X_suggestions = x_scaler.inverse_transform(np.array(X_suggestions_s))
    y_pred = y_prop_scaler.inverse_transform(
        np.array(y_pred_s_list).reshape(-1, 1)
    ).ravel()

    # Build DataFrame
    df = pd.DataFrame(X_suggestions, columns=feature_cols)

    # Handle composition fractions
    frac_cols = [c for c in feature_cols if c.endswith("_frac")]
    if frac_cols:
        df[frac_cols] = df[frac_cols].clip(0, 1)
        row_sums = df[frac_cols].sum(axis=1).replace(0, np.nan)
        df.loc[row_sums.notna(), frac_cols] = (
            df.loc[row_sums.notna(), frac_cols].div(row_sums[row_sums.notna()], axis=0)
        )

        # Optional: flexible masking (limit number of elements)
        if max_elements is not None:
            for i in range(len(df)):
                fracs = df.loc[i, frac_cols].values
                # Keep top-k, zero out rest
                top_idx = np.argsort(fracs)[::-1][:max_elements]
                mask = np.zeros_like(fracs)
                mask[top_idx] = 1
                fracs = fracs * mask
                fracs /= fracs.sum() if fracs.sum() > 0 else 1
                df.loc[i, frac_cols] = fracs

    # Round categorical encodings
    for c in ["Equilibrium_Conditions_encoded",
              "Single_Multiphase_encoded",
              "Type_Present_Phases_encoded",
              "Tension_Compression_encoded"]:
        if c in df.columns:
            df[c] = df[c].round().astype(int)

    # Add outputs
    df.insert(0, "Predicted_Yield_Strength", y_pred)
    if temp is not None:
        df.insert(1, "Conditioned_Temp", temp)

    return df

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True,
                        help="Target yield strength (MPa)")
    parser.add_argument("--N", type=int, default=25,
                        help="Number of candidates to generate")
    parser.add_argument("--refine", action="store_true",
                        help="Use latent refinement of z")
    parser.add_argument("--temp", type=float, default=None,
                        help="Optional conditioning temperature (°C)")
    parser.add_argument("--max_elements", type=int, default=None,
                        help="Limit number of elements in composition (e.g. 4 or 5)")
    args = parser.parse_args()

    device = torch.device("cpu")
    scalers = joblib.load(
        "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
        "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib"
    )
    x_scaler = scalers["x_scaler"]
    y_prop_scaler = scalers["y_prop_scaler"]
    y_cond_scaler = scalers["y_cond_scaler"]
    feature_cols = scalers["feature_cols"]

    model = CVAE(x_dim=len(feature_cols),
                 y_cond_dim=len(scalers["cond_cols"]),
                 y_prop_dim=1,
                 z_dim=4, hidden=128).to(device)
    model.load_state_dict(torch.load(
        "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
        "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt",
        map_location=device
    ))
    model.eval()

    # Generate suggestions
    df_suggestions = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                             y_target_scalar=args.target, temp=args.temp,
                             N=args.N, refine=args.refine,
                             max_elements=args.max_elements)

    # Save
    out_dir = Path("../outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    fname_parts = [f"suggestions_y{int(args.target)}"]
    if args.temp is not None:
        fname_parts.append(f"T{int(args.temp)}")
    if args.max_elements is not None:
        fname_parts.append(f"max{args.max_elements}el")
    out_path = out_dir / ("_".join(fname_parts) + ".csv")

    df_suggestions.to_csv(out_path, index=False)
    print(f"Saved {args.N} suggestions → {out_path}")

if __name__ == "__main__":
    main()