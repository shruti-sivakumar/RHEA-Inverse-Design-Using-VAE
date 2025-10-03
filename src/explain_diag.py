# src/explain_diag.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from cvae import CVAE
from generate import suggest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_FILE   = "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt"
SCALERS_FILE = "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib"
OUT_DIR = Path("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/outputs/explain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Elements and feature names (12 fraction features)
ELEMENTS  = ["Al","Co","Cr","Hf","Mo","Nb","Si","Ta","Ti","V","W","Zr"]
FRAC_COLS = [f"{el}_frac" for el in ELEMENTS]

def load_model(x_dim, y_cond_dim, y_prop_dim=1, z_dim=8, hidden=256):
    model = CVAE(x_dim, y_cond_dim, y_prop_dim, z_dim=z_dim, hidden=hidden).to(DEVICE)
    state = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def build_cond_scaler_func(y_cond_scaler, cond_cols, temp_value: float):
    """
    Returns a function f(batch_size:int)->Tensor that produces the scaled
    conditioning tensor for the given temperature, with the requested batch size.
    """
    if y_cond_scaler is None or len(cond_cols) == 0:
        # No conditioning
        def make_cond(bs: int):
            return torch.zeros((bs, 0), dtype=torch.float32, device=DEVICE)
        return make_cond

    # Precompute a single row then repeat; or recompute per batch — both fine.
    def make_cond(bs: int):
        raw = np.full((bs, len(cond_cols)), float(temp_value), dtype=np.float32)
        scaled = y_cond_scaler.transform(raw)
        return torch.tensor(scaled, dtype=torch.float32, device=DEVICE)
    return make_cond

def run_ig_on_generated(df, model, make_cond, feature_cols, n_steps=50):
    """
    Runs Integrated Gradients on the model's property head w.r.t. the input features.
    df: DataFrame containing the generated alloys (from suggest()).
    model: trained CVAE.
    make_cond: function that returns a (B, cond_dim) tensor for any requested batch size B.
    feature_cols: list of input feature names (order must match training).
    """
    # Use the same features the model was trained on
    X_np = df[feature_cols].astype(np.float32).values
    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)

    # Forward for Captum; builds cond to match the current x_in batch size
    def fwd(x_in):
        Ct = make_cond(x_in.shape[0])
        _, y_pred, _ = model(x_in, Ct)
        return y_pred.squeeze(1)

    ig = IntegratedGradients(fwd)

    # Baseline: zeros (you could also use mean composition if you prefer)
    baseline = torch.zeros_like(X)

    # IG
    attributions = ig.attribute(X, baselines=baseline, n_steps=n_steps)
    return attributions.detach().cpu().numpy()

def aggregate_element_importance(attributions, feature_cols):
    """
    Maps the attribution vector (for all model input features) down to the 12 element fractions.
    If your model only uses the 12 fraction features, this is just a direct mapping.
    """
    # Build an index map for element fraction features inside feature_cols
    idx_map = {col: i for i, col in enumerate(feature_cols)}
    elem_idx = [idx_map[c] for c in FRAC_COLS if c in idx_map]

    # If any FRAC_COLS missing in feature_cols, guard:
    if len(elem_idx) != len(FRAC_COLS):
        # Create zeros for missing and fill those present
        agg = np.zeros((attributions.shape[0], len(FRAC_COLS)), dtype=np.float32)
        present = [c in idx_map for c in FRAC_COLS]
        for j, present_flag in enumerate(present):
            if present_flag:
                agg[:, j] = attributions[:, idx_map[FRAC_COLS[j]]]
        return agg

    # Slice the 12 fraction features
    return attributions[:, elem_idx]

def plot_importance(mean_scores, title, out_path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(ELEMENTS))
    plt.bar(ELEMENTS, mean_scores)
    plt.ylabel("Mean IG Attribution")
    plt.title(title)
    plt.xticks(x, ELEMENTS, rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

def main(temp: float, N: int, n_steps: int = 50, compare_dataset: bool = False):
    # Load scalers / metadata
    scalers = joblib.load(SCALERS_FILE)
    x_scaler = scalers["x_scaler"]
    y_prop_scaler = scalers["y_prop_scaler"]
    y_cond_scaler = scalers["y_cond_scaler"]
    feature_cols = scalers["feature_cols"]
    cond_cols    = scalers["cond_cols"]

    # Load model (dims from scalers)
    model = load_model(x_dim=len(feature_cols), y_cond_dim=len(cond_cols), y_prop_dim=1)

    # Generate alloys at the requested temp using the existing pipeline
    df = suggest(
        model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
        y_target_scalar=2000.0, temp=temp, N=N, refine=True
    )

    # Make conditioning function (batch-safe)
    make_cond = build_cond_scaler_func(y_cond_scaler, cond_cols, temp_value=temp)

    # Run IG on the generated set
    attributions = run_ig_on_generated(df, model, make_cond, feature_cols, n_steps=n_steps)

    # Reduce to element-level scores (12 fraction features)
    elem_attr = aggregate_element_importance(attributions, feature_cols)
    elem_mean = elem_attr.mean(axis=0)

    # Save CSV of mean IG per element
    out_csv = OUT_DIR / f"ig_elements_T{int(temp)}.csv"
    pd.DataFrame([dict(zip(ELEMENTS, elem_mean))]).to_csv(out_csv, index=False)
    print(f"[INFO] Saved mean IG per element → {out_csv}")

    # Plot
    title = f"Integrated Gradients: Element Importance @ {temp:.0f} °C (N={N}, steps={n_steps})"
    out_png = OUT_DIR / f"ig_elements_T{int(temp)}.png"
    plot_importance(elem_mean, title, out_png)

    # (Optional) Simple dataset comparison — average YS per element presence at this temp
    if compare_dataset:
        data_file = Path("../data/encoded_data.csv")
        if data_file.exists():
            raw = pd.read_csv(data_file)
            if "Testing_Temp" in raw.columns and "Yield_Strength" in raw.columns:
                # Keep rows near the chosen temp (±50°C band to be forgiving)
                rows = raw[raw["Testing_Temp"].between(temp - 50, temp + 50)]
                comp = []
                for el in ELEMENTS:
                    col = f"{el}_frac"
                    if col in rows.columns:
                        present = rows[rows[col] > 0]
                        avg = present["Yield_Strength"].mean() if not present.empty else np.nan
                        comp.append((el, avg))
                    else:
                        comp.append((el, np.nan))
                comp_df = pd.DataFrame(comp, columns=["Element", "AvgYS_at_temp"])
                comp_df.to_csv(OUT_DIR / f"dataset_avgYS_T{int(temp)}.csv", index=False)
                print(f"[INFO] Saved dataset AvgYS near {temp:.0f} °C → {OUT_DIR / f'dataset_avgYS_T{int(temp)}.csv'}")
            else:
                print("[WARN] dataset comparison skipped: Testing_Temp or Yield_Strength missing.")
        else:
            print("[WARN] dataset comparison skipped: encoded_data.csv not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, default=800.0)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--compare-dataset", action="store_true")
    args = parser.parse_args()

    main(temp=args.temp, N=args.N, n_steps=args.steps, compare_dataset=args.compare_dataset)
