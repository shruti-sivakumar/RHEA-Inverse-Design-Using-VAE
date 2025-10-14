import torch, numpy as np, pandas as pd
from cvae import CVAE

# -----------------------
# Helper: latent refinement (optional)
# -----------------------
def _latent_refine(model, y_cond_t, y_target_t, steps=200, lr=0.1, lam=1e-3):
    """
    Gradient-based refinement of latent z to better match desired property.
    - y_cond_t: conditioning (temperature, scaled tensor)
    - y_target_t: desired target strength (scaled tensor)
    """
    z = model.sample_prior(y_cond_t, n=1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        y_pred = model.prop_head(z, y_cond_t)
        loss = torch.mean((y_pred - y_target_t) ** 2) + lam * torch.mean(z ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach()

# -----------------------
# Element limiting logic
# -----------------------
def enforce_element_limits(df, frac_cols, max_elements=None, threshold=0.01):
    """
    Enforce realistic element limits:
    - Drop elements below threshold (default 1%).
    - If more than max_elements remain, keep only top-k.
    - Renormalize to sum to 1.0.
    Adds column 'Active_Elements' for convenience.
    """
    for i in range(len(df)):
        fracs = df.loc[i, frac_cols].values

        # Clip tiny negatives to 0 first
        fracs = np.clip(fracs, 0.0, None)

        # Drop small fractions
        mask = fracs >= threshold
        fracs = fracs * mask

        # If too many nonzero elements, keep only top-k
        if max_elements is not None:
            order = np.argsort(fracs)[::-1]  # descending
            keep_idx = order[:max_elements]
            mask = np.zeros_like(fracs, dtype=np.float32)
            mask[keep_idx] = 1.0
            fracs = fracs * mask

        # Renormalize
        s = fracs.sum()
        if s > 0:
            fracs = fracs / s

        df.loc[i, frac_cols] = fracs.astype(np.float32)

    df["Active_Elements"] = (df[frac_cols] > 0).sum(axis=1).astype(int)
    return df

# -----------------------
# Suggest alloys
# -----------------------
def suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
            y_target_scalar=1200.0, temp=25.0, N=50, refine=True,
            max_elements=None, reuse_z=None):
    """
    Generate candidate alloys given a target property and temperature.
    - model: trained CVAE
    - x_scaler, y_prop_scaler, y_cond_scaler: fitted scalers
    - feature_cols: list of feature column names
    - y_target_scalar: desired Yield Strength (in MPa, unscaled)
    - temp: conditioning temperature (in °C)
    - N: number of candidates to generate
    - refine: whether to apply latent refinement
    - max_elements: if set, keep <= k nonzero elements (with domain-aware thresholding)
    - reuse_z: optional torch.Tensor latent vector to reuse for multi-condition tests
    """
    device = next(model.parameters()).device

    # Conditioning and target
    y_cond = np.array([[temp]])
    y_target = np.array([[y_target_scalar]])

    y_cond_s = y_cond_scaler.transform(y_cond)
    y_target_s = y_prop_scaler.transform(y_target)

    y_cond_t = torch.tensor(y_cond_s, dtype=torch.float32, device=device)
    y_target_t = torch.tensor(y_target_s, dtype=torch.float32, device=device)

    # ---------------------
    # Latent sampling / reuse
    # ---------------------
    if reuse_z is not None:
        Z = reuse_z.to(device)
        if Z.ndim == 1:
            Z = Z.unsqueeze(0)
    else:
        if refine:
            z0 = _latent_refine(model, y_cond_t, y_target_t)
        else:
            z0 = model.sample_prior(y_cond_t, n=1)
        Z = z0.repeat(N, 1) + 0.05 * torch.randn(N, z0.size(-1), device=device)

    # Decode + predict
    with torch.no_grad():
        X_hat_s = model.dec(Z, y_cond_t.repeat(Z.size(0), 1)).cpu().numpy()
        y_pred_s = model.prop_head(Z, y_cond_t.repeat(Z.size(0), 1)).cpu().numpy()

    # Inverse scaling
    X_hat = x_scaler.inverse_transform(X_hat_s)
    y_pred = y_prop_scaler.inverse_transform(y_pred_s).ravel()

    # Keep only fraction columns
    frac_cols = [c for c in feature_cols if c.endswith("_frac")]
    df_all = pd.DataFrame(X_hat, columns=feature_cols)
    df = df_all[frac_cols].copy()

    # Clean and renormalize
    df[frac_cols] = np.clip(df[frac_cols].values, 0.0, None)
    row_sums = df[frac_cols].sum(axis=1).replace(0, 1.0)
    df[frac_cols] = (df[frac_cols].T / row_sums).T
    df = enforce_element_limits(df, frac_cols, max_elements=max_elements, threshold=0.01)

    df.insert(0, "Predicted_Yield_Strength", y_pred)
    df.insert(1, "Conditioned_Temp", temp)

    # Attach latent vector for reuse (optional)
    df.attrs["latent_z"] = Z[0].detach().cpu()  # store first latent (same for all)
    return df