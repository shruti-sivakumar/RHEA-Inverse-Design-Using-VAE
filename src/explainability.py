# src/explainability.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance

from cvae import CVAE
from generate import suggest

# --------------------------
# Config
# --------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "encoded_data.csv"
SCALER_FILE = ROOT / "data" / "processed" / "scalers.joblib"
MODEL_FILE = ROOT / "models" / "cvae_best.pt"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Load dataset + scalers
# --------------------------
df = pd.read_csv(DATA_FILE)
scalers = joblib.load(SCALER_FILE)
feature_cols = scalers["feature_cols"]
cond_cols = scalers["cond_cols"]
x_scaler = scalers["x_scaler"]
y_prop_scaler = scalers["y_prop_scaler"]
y_cond_scaler = scalers["y_cond_scaler"]

X = df[feature_cols].fillna(0).values.astype(np.float32)
y_cond = df[cond_cols].fillna(25).values.astype(np.float32)
y_prop = df[["Yield_Strength"]].fillna(0).values.astype(np.float32)

# Scale
X_s = x_scaler.transform(X)
y_cond_s = y_cond_scaler.transform(y_cond)
y_prop_s = y_prop_scaler.transform(y_prop)

# --------------------------
# Load trained CVAE
# --------------------------
model = CVAE(
    x_dim=X_s.shape[1],
    y_cond_dim=y_cond_s.shape[1],
    y_prop_dim=1,
    z_dim=LATENT_DIM,
    hidden=256
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

# Wrapper for property prediction
def predict_strength(X_block):
    """Predict yield strength given scaled X and fixed y_cond (mean temp)."""
    X_t = torch.tensor(X_block, dtype=torch.float32, device=DEVICE)
    y_mean = torch.tensor(y_cond_s.mean(axis=0, keepdims=True),
                          dtype=torch.float32, device=DEVICE)
    y_rep = y_mean.repeat(X_t.shape[0], 1)
    with torch.no_grad():
        _, y_pred, _ = model(X_t, y_rep)
    return y_prop_scaler.inverse_transform(y_pred.cpu().numpy())

# --------------------------
# Correlation check (Train vs Generated)
# --------------------------
corr_train = pd.DataFrame(np.c_[X, y_prop], columns=feature_cols + ["Yield_Strength"]).corr()["Yield_Strength"]

# Generate alloys for comparison
gen_df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler,
                 feature_cols, y_target_scalar=1500.0, temp=800, N=200)

corr_gen = gen_df.corr()["Predicted_Yield_Strength"]

# Heatmap
corr_df = pd.concat([corr_train, corr_gen], axis=1, keys=["Train", "Generated"])
plt.figure(figsize=(8,10))
sns.heatmap(corr_df, annot=False, cmap="coolwarm", center=0, cbar=True)
plt.title("Feature–Strength Correlation: Train vs Generated")
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_train_vs_generated.png", dpi=300)
plt.close()
print(f"[OK] Correlation heatmap saved → {OUT_DIR/'correlation_train_vs_generated.png'}")

# --------------------------
# Permutation Feature Importance (PFI)
# --------------------------
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

class SklearnWrapper:
    def fit(self, X, y): pass  # not needed
    def predict(self, X):
        return predict_strength(X).ravel()
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

wrapper = SklearnWrapper()

result = permutation_importance(
    wrapper, X_s, y_prop.ravel(),
    n_repeats=10, random_state=42, n_jobs=-1, scoring="r2"
)

imp_means = result.importances_mean
imp_std = result.importances_std

# Plot PFI
sorted_idx = np.argsort(imp_means)[::-1]
plt.figure(figsize=(8,6))
plt.barh(np.array(feature_cols)[sorted_idx], imp_means[sorted_idx], xerr=imp_std[sorted_idx])
plt.xlabel("Permutation Importance (Δ R²)")
plt.title("Permutation Feature Importance (Yield Strength)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "pfi_importance.png", dpi=300)
plt.close()
print(f"[OK] Permutation Feature Importance saved → {OUT_DIR/'pfi_importance.png'}")