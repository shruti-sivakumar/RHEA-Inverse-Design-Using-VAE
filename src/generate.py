# src/generate.py
import torch, numpy as np, pandas as pd
from cvae import CVAE

# -----------------------
# Helper: latent refinement (optional)
# -----------------------
def _latent_refine(model, y_cond_t, y_target_t, steps=200, lr=0.1, lam=1e-3):
    """
    Gradient-based refinement of latent z to better match desired property.
    - y_cond_t: conditioning (temperature, etc.)
    - y_target_t: desired target strength (scaled)
    """
    z = model.sample_prior(y_cond_t, n=1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        y_pred = model.prop_head(z)
        loss = torch.mean((y_pred - y_target_t) ** 2) + lam * torch.mean(z ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach()

# -----------------------
# Suggest alloys
# -----------------------
def suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
            y_target_scalar=1200.0, temp=None, N=50, refine=True, max_elements=None):
    """
    Generate candidate alloys given a target property and optional temp.
    - max_elements: if set, enforce flexible masking (keep <= k elements nonzero).
    """

    device = next(model.parameters()).device

    # Build conditioning + target vector
    y_cond = np.array([[temp]]) if temp is not None else np.zeros((1,1))
    y_target = np.array([[y_target_scalar]])

    y_cond_s = y_cond_scaler.transform(y_cond)
    y_target_s = y_prop_scaler.transform(y_target)

    y_cond_t = torch.tensor(y_cond_s, dtype=torch.float32, device=device)
    y_target_t = torch.tensor(y_target_s, dtype=torch.float32, device=device)

    # Sample z
    if refine:
        z0 = _latent_refine(model, y_cond_t, y_target_t)
    else:
        z0 = model.sample_prior(y_cond_t, n=1)

    # Expand to N candidates
    Z = z0.repeat(N, 1) + 0.05 * torch.randn(N, z0.size(-1), device=device)

    # Decode
    with torch.no_grad():
        X_hat_s = model.dec(Z, y_cond_t.repeat(N,1)).cpu().numpy()
        y_pred_s = model.prop_head(Z).cpu().numpy()

    # Inverse scale
    X_hat = x_scaler.inverse_transform(X_hat_s)
    y_pred = y_prop_scaler.inverse_transform(y_pred_s).ravel()

    df = pd.DataFrame(X_hat, columns=feature_cols)
    df.insert(0, "Predicted_Yield_Strength", y_pred)
    if temp is not None:
        df.insert(1, "Conditioned_Temp", temp)

    # -----------------------
    # Flexible masking: enforce max_elements if requested
    # -----------------------
    if max_elements is not None:
        frac_cols = [c for c in feature_cols if c.endswith("_frac")]
        for i in range(len(df)):
            fracs = df.loc[i, frac_cols].values
            topk = np.argsort(fracs)[::-1][:max_elements]  # keep only top k
            mask = np.zeros_like(fracs); mask[topk] = 1
            fracs = fracs * mask
            fracs = np.clip(fracs, 0, None)
            if fracs.sum() > 0:
                fracs /= fracs.sum()
            df.loc[i, frac_cols] = fracs.astype(np.float32)

    return df