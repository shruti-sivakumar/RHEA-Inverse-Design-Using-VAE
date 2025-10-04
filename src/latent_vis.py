# src/latent_vis.py
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
BASE = Path(__file__).resolve().parents[1]
DATA_FILE = BASE / "data" / "encoded_data.csv"
SCALER_FILE = BASE / "data" / "processed" / "scalers.joblib"
MODEL_FILE = BASE / "models" / "cvae_best.pt"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

LATENT_DIM = 8
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

# Filter out rows with missing Yield_Strength
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
    return q_mu.cpu().numpy()

latents_all = get_latents(X, y_cond)
latents_prop = get_latents(X_prop, y_cond_prop)

# Optional: z-score standardization for t-SNE stability
latents_all_std = (latents_all - latents_all.mean(axis=0)) / (latents_all.std(axis=0) + 1e-8)
latents_prop_std = (latents_prop - latents_prop.mean(axis=0)) / (latents_prop.std(axis=0) + 1e-8)

# --------------------------
# t-SNE 1: overall distribution
# --------------------------
tsne_all = TSNE(n_components=2, perplexity=30, max_iter=1500,
                random_state=42, init="pca", learning_rate="auto")
emb_all = tsne_all.fit_transform(latents_all_std)

plt.figure(figsize=(8,6))
plt.scatter(emb_all[:,0], emb_all[:,1], s=10, alpha=0.7, color="#4682B4")
plt.title("t-SNE of CVAE Latent Space (All Samples)")
plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
plt.tight_layout()
plt.savefig(OUT_DIR / "latent_tsne_all.png", dpi=300)
plt.show()

# --------------------------
# t-SNE 2: color by Yield Strength
# --------------------------
tsne_y = TSNE(n_components=2, perplexity=30, max_iter=1500,
              random_state=42, init="pca", learning_rate="auto")
emb_y = tsne_y.fit_transform(latents_prop_std)

plt.figure(figsize=(8,6))
sc = plt.scatter(emb_y[:,0], emb_y[:,1],
                 c=y_prop.flatten(), cmap="viridis", s=18, alpha=0.8)
plt.colorbar(sc, label="Yield Strength (MPa)")
plt.title("t-SNE of Latent Space Colored by Yield Strength")
plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
plt.tight_layout()
plt.savefig(OUT_DIR / "latent_tsne_yield_strength.png", dpi=300)
plt.show()

# --------------------------
# t-SNE 3: color by Temperature
# --------------------------
tsne_T = TSNE(n_components=2, perplexity=30, max_iter=1500,
              random_state=42, init="pca", learning_rate="auto")
emb_T = tsne_T.fit_transform(latents_prop_std)

plt.figure(figsize=(8,6))
sc = plt.scatter(emb_T[:,0], emb_T[:,1],
                 c=y_cond_prop.flatten(), cmap="coolwarm", s=18, alpha=0.8)
plt.colorbar(sc, label="Testing Temperature (°C)")
plt.title("t-SNE of Latent Space Colored by Temperature")
plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
plt.tight_layout()
plt.savefig(OUT_DIR / "latent_tsne_temperature.png", dpi=300)
plt.show()

# --------------------------
# Save embedding data (optional, for paper)
# --------------------------
emb_df = pd.DataFrame({
    "tSNE1": emb_y[:,0],
    "tSNE2": emb_y[:,1],
    "Yield_Strength": y_prop.flatten(),
    "Temperature": y_cond_prop.flatten()
})
emb_df.to_csv(OUT_DIR / "latent_tsne_embeddings.csv", index=False)
print(f"[INFO] Saved t-SNE embeddings → {OUT_DIR/'latent_tsne_embeddings.csv'}")