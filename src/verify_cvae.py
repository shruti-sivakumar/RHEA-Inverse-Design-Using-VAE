import numpy as np, pandas as pd, torch, joblib, matplotlib.pyplot as plt
from pathlib import Path
from cvae import CVAE
from generate import suggest

BASE = Path(__file__).resolve().parents[1]
DATA_CSV     = BASE / "data" / "encoded_data.csv"
SCALERS_PATH = BASE / "data" / "processed" / "scalers.joblib"
MODEL_PATH   = BASE / "models" / "cvae_best.pt"
OUT_DIR      = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ELEMENTS = ["Al","Co","Cr","Hf","Mo","Nb","Si","Ta","Ti","V","W","Zr"]
FRAC_COLS = [f"{el}_frac" for el in ELEMENTS]

# ---------- helpers ----------
def make_alloy_formula_from_fracs(fracs, eps=1e-6):
    fracs = np.array(fracs, dtype=float).clip(0)
    s = fracs.sum()
    if s <= eps: return ""
    fracs = np.round(fracs / s, 2)
    parts = []
    for el, f in zip(ELEMENTS, fracs):
        if f > 0:
            parts.append(el if abs(f-1.0) < 1e-2 else f"{el}{f:.2f}".rstrip("0").rstrip("."))
    return "".join(parts)

def row_formula(row):  # dataset row -> formula
    return make_alloy_formula_from_fracs([row[c] for c in FRAC_COLS])

def l1_frac_distance(a, b):  # sum |ai - bi|
    return float(np.sum(np.abs(np.array(a) - np.array(b))))

def closest_by_fraction(df_gen, target_fracs):
    dists = df_gen[FRAC_COLS].apply(lambda r: l1_frac_distance(r.values, target_fracs), axis=1)
    idx = int(dists.idxmin())
    return idx, float(dists.min())

# ---------- load ----------
df = pd.read_csv(DATA_CSV)
scalers = joblib.load(SCALERS_PATH)
x_scaler = scalers["x_scaler"]; y_prop_scaler = scalers["y_prop_scaler"]; y_cond_scaler = scalers["y_cond_scaler"]
feature_cols = scalers["feature_cols"]; cond_cols = scalers["cond_cols"]

device = torch.device("cpu")
model = CVAE(len(feature_cols), len(cond_cols), 1, z_dim=8, hidden=256).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------- verification ----------
def verify_query1(temp, top_k_dataset=3, gen_N=400, max_elements=None, save_name=None):
    df_t = df[df["Testing_Temp"].round(0) == round(temp)].dropna(subset=["Yield_Strength"])
    if df_t.empty:
        print(f"[Q1] No rows at T={temp}°C in dataset."); return None
    df_t = df_t.sort_values("Yield_Strength", ascending=False).head(top_k_dataset).reset_index(drop=True)
    df_gen = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                     y_target_scalar=float(df_t["Yield_Strength"].max()),
                     temp=float(temp), N=gen_N, refine=True, max_elements=max_elements)

    rows = []
    for i, r in df_t.iterrows():
        tgt_fracs = [r[c] for c in FRAC_COLS]
        tgt_form  = row_formula(r)
        j, d = closest_by_fraction(df_gen, tgt_fracs)
        g = df_gen.iloc[j]
        rows.append({
            "Case": f"Q1_T{int(temp)}_{i+1}",
            "Temp": temp,
            "Dataset_Yield": float(r["Yield_Strength"]),
            "Dataset_Formula": tgt_form,
            "Gen_Yield": float(g["Predicted_Yield_Strength"]),
            "Gen_Formula": make_alloy_formula_from_fracs([g[c] for c in FRAC_COLS]),
            "L1_frac_distance": d,
            "Yield_Error": float(abs(r["Yield_Strength"] - g["Predicted_Yield_Strength"])),
            "Yield_Error_%": float(100 * abs(r["Yield_Strength"] - g["Predicted_Yield_Strength"]) / r["Yield_Strength"]),
        })
    out = pd.DataFrame(rows)
    if save_name:
        path = OUT_DIR / f"{save_name}.csv"
        out.to_csv(path, index=False); print(f"[Q1] Saved → {path}")
    return out

def verify_query2(n_cases=10, gen_N=300, max_elements=None, seed=42, save_name=None):
    rnd = df.dropna(subset=["Yield_Strength","Testing_Temp"]).sample(n_cases, random_state=seed).reset_index(drop=True)
    rows = []
    for i, r in rnd.iterrows():
        temp = float(r["Testing_Temp"]); target = float(r["Yield_Strength"])
        df_gen = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                         y_target_scalar=target, temp=temp, N=gen_N, refine=True, max_elements=max_elements)
        tgt_fracs = [r[c] for c in FRAC_COLS]; tgt_form = row_formula(r)
        j, d = closest_by_fraction(df_gen, tgt_fracs)
        g = df_gen.iloc[j]
        rows.append({
            "Case": f"Q2_row{i+1}",
            "Temp": temp,
            "Target_Yield": target,
            "Dataset_Formula": tgt_form,
            "Gen_Yield": float(g["Predicted_Yield_Strength"]),
            "Gen_Formula": make_alloy_formula_from_fracs([g[c] for c in FRAC_COLS]),
            "L1_frac_distance": d,
            "Yield_Error": float(abs(target - g["Predicted_Yield_Strength"])),
            "Yield_Error_%": float(100 * abs(target - g["Predicted_Yield_Strength"]) / target),
        })
    out = pd.DataFrame(rows)
    if save_name:
        path = OUT_DIR / f"{save_name}.csv"
        out.to_csv(path, index=False); print(f"[Q2] Saved → {path}")
    return out

# ---------- summary + plots ----------
def summarize(dfv, name="Qx", l1_thr=0.4, pct_thr=5.0):
    if dfv is None or dfv.empty: return
    mae = dfv["Yield_Error"].mean()
    pct_hits = (dfv["Yield_Error_%"] <= pct_thr).mean() * 100
    close_comp = (dfv["L1_frac_distance"] <= l1_thr).mean() * 100
    print(f"[{name}] MAE = {mae:.2f} MPa | Strength ±{pct_thr}% match: {pct_hits:.1f}% | L1 ≤ {l1_thr}: {close_comp:.1f}%")
    # scatter
    plt.figure(figsize=(6,5))
    plt.scatter(dfv["L1_frac_distance"], dfv["Yield_Error_%"], c="dodgerblue", s=50, alpha=0.7, edgecolors='k')
    plt.axhline(pct_thr, color='r', ls='--', lw=1)
    plt.axvline(l1_thr, color='g', ls='--', lw=1)
    plt.xlabel("Composition L1 Distance")
    plt.ylabel("Yield Strength Error (%)")
    plt.title(f"{name}: Composition vs Strength Error")
    plt.tight_layout()
    figpath = OUT_DIR / f"{name}_scatter.png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"[{name}] Scatter saved → {figpath}")

# ---------- correlation plot ----------
def plot_correlation(dfv, name="Qx"):
    """Plot Predicted vs Dataset (or Target) yield strength."""
    if dfv is None or dfv.empty: 
        return
    y_true = dfv["Dataset_Yield"] if "Dataset_Yield" in dfv.columns else dfv["Target_Yield"]
    y_pred = dfv["Gen_Yield"]

    plt.figure(figsize=(5.5,5.5))
    plt.scatter(y_true, y_pred, color="royalblue", s=60, edgecolors='k', alpha=0.8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=1.2, label="Ideal y = x")
    plt.xlabel("Dataset / Target Yield Strength (MPa)")
    plt.ylabel("Generated Predicted Yield (MPa)")
    plt.title(f"{name}: Predicted vs Dataset Yield Strength")
    plt.legend()
    plt.tight_layout()
    figpath = OUT_DIR / f"{name}_correlation.png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"[{name}] Correlation plot saved → {figpath}")

# ---------- run ----------
if __name__ == "__main__":
    q1 = verify_query1(temp=800, top_k_dataset=3, gen_N=400, max_elements=6, save_name="verify_q1_T800")
    q2 = verify_query2(n_cases=10, gen_N=300, max_elements=6, seed=7, save_name="verify_q2_random10")
    summarize(q1, name="Q1_T800")
    summarize(q2, name="Q2_random10")

    # New correlation plots
    plot_correlation(q1, name="Q1_T800")
    plot_correlation(q2, name="Q2_random10")
