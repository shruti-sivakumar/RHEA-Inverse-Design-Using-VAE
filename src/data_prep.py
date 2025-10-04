# src/data_prep.py

import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path

# -----------------------------
# Set reproducibility
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)

# -----------------------------
# Base directory = repo root
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH    = BASE_DIR / "data" / "encoded_data.csv"
OUT_SCALERS = BASE_DIR / "data" / "processed" / "scalers.joblib"
OUT_SPLITS  = BASE_DIR / "data" / "processed" / "data_splits.npz"

# Make sure processed folder exists
OUT_SCALERS.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Define conditioning and property targets
# -----------------------------
cond_cols = ["Testing_Temp"]        # conditioning variable(s)
prop_col  = "Yield_Strength"        # supervised target

# Feature columns: everything except targets + percentages
feature_cols = [
    c for c in df.columns
    if c not in cond_cols + [prop_col] and not c.endswith("_pct")
]

# -----------------------------
# Build X, y_cond, y_prop
# -----------------------------
X = df[feature_cols].copy()
y_cond = df[cond_cols].copy()
y_prop = df[[prop_col]].copy()

# Handle missing values (median imputation)
X = X.fillna(X.median(numeric_only=True))
y_cond = y_cond.fillna(y_cond.median(numeric_only=True))
y_prop = y_prop.fillna(y_prop.median(numeric_only=True))

# -----------------------------
# Train/val split
# -----------------------------
X_train, X_val, y_cond_train, y_cond_val, y_prop_train, y_prop_val = train_test_split(
    X.values, y_cond.values, y_prop.values,
    test_size=0.2, random_state=seed, shuffle=True
)

# -----------------------------
# Scale features
# -----------------------------
x_scaler = StandardScaler()
y_cond_scaler = StandardScaler()   # keep conditioning in scaled units
y_prop_scaler = MinMaxScaler()     # keep target in [0,1]

X_train_s = x_scaler.fit_transform(X_train)
X_val_s   = x_scaler.transform(X_val)

y_cond_train_s = y_cond_scaler.fit_transform(y_cond_train)
y_cond_val_s   = y_cond_scaler.transform(y_cond_val)

y_prop_train_s = y_prop_scaler.fit_transform(y_prop_train)
y_prop_val_s   = y_prop_scaler.transform(y_prop_val)

# -----------------------------
# Save scalers and splits
# -----------------------------
joblib.dump({
    "x_scaler": x_scaler,
    "y_cond_scaler": y_cond_scaler,
    "y_prop_scaler": y_prop_scaler,
    "feature_cols": feature_cols,
    "cond_cols": cond_cols,
    "prop_col": prop_col,
}, OUT_SCALERS)

np.savez(OUT_SPLITS,
         X_train=X_train_s, X_val=X_val_s,
         y_cond_train=y_cond_train_s, y_cond_val=y_cond_val_s,
         y_prop_train=y_prop_train_s, y_prop_val=y_prop_val_s)

print(f"[OK] Saved scalers → {OUT_SCALERS}")
print(f"[OK] Saved splits  → {OUT_SPLITS}")
print(f"[INFO] Feature dims: X={X_train_s.shape[1]}, y_cond={y_cond_train_s.shape[1]}, y_prop={y_prop_train_s.shape[1]}")