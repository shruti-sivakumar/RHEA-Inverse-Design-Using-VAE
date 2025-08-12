# generate.py
import numpy as np, torch, joblib
from cvae import CVAE

def load_artifacts():
    scalers = joblib.load("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib")
    data = np.load("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/data_splits.npz")
    x_dim = data["X_train"].shape[1]
    y_dim = data["y_train"].shape[1]
    model = CVAE(x_dim=x_dim, y_dim=y_dim, z_dim=6, hidden=128)
    model.load_state_dict(torch.load("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt", map_location="cpu"))
    model.eval()
    return model, scalers

@torch.no_grad()
def suggest(y_target_scalar, N=20):
    model, scalers = load_artifacts()
    x_scaler = scalers["x_scaler"]; y_scaler = scalers["y_scaler"]; feature_cols = scalers["feature_cols"]

    # Scale y* to [0,1]
    y_star = np.array([[y_target_scalar]], dtype=np.float32)
    y_star_s = y_scaler.transform(y_star)
    y_star_t = torch.tensor(y_star_s, dtype=torch.float32)

    # Conditional prior p(z|y*)
    p_mu, p_logvar = model.prior(y_star_t)
    std = torch.exp(0.5 * p_logvar)

    # Sample K latents, decode
    out = []
    for _ in range(N):
        eps = torch.randn_like(std)
        z = p_mu + eps * std
        x_hat = model.dec(z, y_star_t).numpy()
        # Unscale to original feature space
        x_hat_unscaled = x_scaler.inverse_transform(x_hat)
        out.append(x_hat_unscaled[0])

    # Return as DataFrame with column names
    import pandas as pd
    df_out = pd.DataFrame(out, columns=feature_cols)
    return df_out

if __name__ == "__main__":
    df_suggestions = suggest(y_target_scalar=1200.0, N=25)
    df_suggestions.to_csv("outputs/suggestions_y1200.csv", index=False)
    print("Saved suggestions to suggestions_y1200.csv")