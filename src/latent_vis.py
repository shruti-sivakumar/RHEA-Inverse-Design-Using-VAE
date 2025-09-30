# src/latent_vis.py
import joblib, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from cvae import CVAE
from generate import suggest

# -------------------
# Load model + scalers
# -------------------
scalers = joblib.load(
    "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
    "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib"
)
x_scaler = scalers["x_scaler"]
y_prop_scaler = scalers["y_prop_scaler"]
y_cond_scaler = scalers["y_cond_scaler"]
feature_cols = scalers["feature_cols"]
cond_cols = scalers["cond_cols"]

device = torch.device("cpu")
model = CVAE(x_dim=len(feature_cols),
             y_cond_dim=len(cond_cols),
             y_prop_dim=1,
             z_dim=4, hidden=128).to(device)
model.load_state_dict(torch.load(
    "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
    "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt",
    map_location=device
))
model.eval()

# -------------------
# Helper: encode to z
# -------------------
def get_latent(X, y_cond):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_cond, dtype=torch.float32).to(device)
    with torch.no_grad():
        q_mu, q_logvar = model.enc(X_t, y_t)
        z = q_mu.cpu().numpy()
    return z

# -------------------
# Load data
# -------------------
data = np.load(
    "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
    "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/data_splits.npz"
)
X_train, y_cond_train = data["X_train"], data["y_cond_train"]

# Encode training latents
z_train = get_latent(X_train, y_cond_train)

# -------------------
# Generate alloys for eval
# -------------------
df_gen = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler,
                 feature_cols, y_target_scalar=1200.0,
                 temp=800, N=200, refine=True)

X_gen = df_gen[feature_cols].values
y_gen = np.full((len(df_gen), 1), 800.0)  # same temp
y_gen_scaled = y_cond_scaler.transform(y_gen)

z_gen = get_latent(X_gen, y_gen_scaled)

# -------------------
# Clean invalid latents
# -------------------
def clean_latents(z, label):
    mask = np.isfinite(z).all(axis=1)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} invalid {label} latents")
    return z[mask]

z_train = clean_latents(z_train, "train")
z_gen = clean_latents(z_gen, "generated")

# -------------------
# Run t-SNE
# -------------------
all_latent = np.vstack([z_train, z_gen])
labels = np.array([0]*len(z_train) + [1]*len(z_gen))  # 0=train, 1=generated

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
emb = tsne.fit_transform(all_latent)

# -------------------
# Plot
# -------------------
plt.figure(figsize=(8,6))
plt.scatter(emb[labels==0,0], emb[labels==0,1], c="blue", alpha=0.5, label="Training Latents")
plt.scatter(emb[labels==1,0], emb[labels==1,1], c="red", alpha=0.7, label="Generated Alloys")
plt.title("t-SNE of CVAE Latent Space")
plt.legend()
plt.tight_layout()

OUT_DIR = Path("../outputs"); OUT_DIR.mkdir(exist_ok=True, parents=True)
out_path = OUT_DIR / "latent_tsne.png"
plt.savefig(out_path, dpi=300)
plt.show()
print(f"[INFO] Saved t-SNE plot → {out_path}")
