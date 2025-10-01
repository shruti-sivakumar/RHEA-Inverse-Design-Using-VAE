import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import joblib

from cvae import CVAE

# --------------------------
# Config
# --------------------------
DATA_FILE = "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/encoded_data.csv"
SCALER_FILE = "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib"
MODEL_FILE = "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt"
OUT_DIR = Path("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

LATENT_DIM = 8   # must match training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Load dataset + scalers
# --------------------------
df = pd.read_csv(DATA_FILE)
scalers = joblib.load(SCALER_FILE)
feature_cols = scalers["feature_cols"]
cond_cols = scalers["cond_cols"]

X = df[feature_cols].fillna(0).values.astype(np.float32)
y_cond = df[cond_cols].fillna(25).values.astype(np.float32)

# Filter out rows without target property
if "Yield_Strength" not in df.columns:
    raise KeyError("Dataset must contain 'Yield_Strength' column.")

df_prop = df.dropna(subset=["Yield_Strength"]).reset_index(drop=True)
X_prop = df_prop[feature_cols].fillna(0).values.astype(np.float32)
y_prop = df_prop[["Yield_Strength"]].values.astype(np.float32)
y_cond_prop = df_prop[cond_cols].fillna(25).values.astype(np.float32)

# --------------------------
# Load trained CVAE
# --------------------------
model = CVAE(
    x_dim=X.shape[1],
    y_cond_dim=y_cond.shape[1],
    y_prop_dim=1,
    z_dim=LATENT_DIM,
    hidden=256
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

# --------------------------
# Encode data into latent space
# --------------------------
def get_latents(X, cond):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    cond_t = torch.tensor(cond, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        q_mu, q_logvar = model.enc(X_t, cond_t)
        z = q_mu.cpu().numpy()
    return z

latents = get_latents(X, y_cond)

# --------------------------
# t-SNE
# --------------------------
tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
emb = tsne.fit_transform(latents)

# --------------------------
# Plot 1: General Latent Distribution
# --------------------------
plt.figure(figsize=(8,6))
plt.scatter(emb[:,0], emb[:,1], s=10, alpha=0.7)
plt.title("t-SNE of CVAE Latent Space")
plt.tight_layout()
plt.savefig(OUT_DIR / "latent_tsne_all.png", dpi=300)
plt.show()

# --------------------------
# Plot 2: Latents colored by Yield Strength
# --------------------------
df_prop_latents = get_latents(X_prop, y_cond_prop)
emb_prop = tsne.fit_transform(df_prop_latents)

plt.figure(figsize=(8,6))
plt.scatter(emb_prop[:,0], emb_prop[:,1],
            c=y_prop.flatten(), cmap="viridis", s=12)
plt.colorbar(label="Yield Strength (MPa)")
plt.title("t-SNE of Training Latents (Colored by Yield Strength)")
plt.tight_layout()
plt.savefig(OUT_DIR / "latent_tsne_yield_strength.png", dpi=300)
plt.show()

# --------------------------
# Plot 3: Latents colored by Temperature
# --------------------------
plt.figure(figsize=(8,6))
plt.scatter(emb_prop[:,0], emb_prop[:,1],
            c=y_cond_prop.flatten(), cmap="coolwarm", s=12)
plt.colorbar(label="Temperature (°C)")
plt.title("t-SNE of Training Latents (Colored by Temperature)")
plt.tight_layout()
plt.savefig(OUT_DIR / "latent_tsne_temperature.png", dpi=300)
plt.show()
