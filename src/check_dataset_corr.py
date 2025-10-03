# src/check_dataset_corr.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# -------------------
# Config
# -------------------
DATA_FILE = "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/encoded_data.csv"
TEMP = 800
TEMP_TOL = 25  # filter around 800 ± 25 °C

# IG results you got earlier (dictionary form)
ig_scores = {
    "Al": -5.2204017e-05,
    "Co": 0.00010577723,
    "Cr": 0.0002033065,
    "Hf": 1.9630204e-06,
    "Mo": 3.4195054e-05,
    "Nb": -0.00016078891,
    "Si": -1.571862e-05,
    "Ta": 5.68867e-05,
    "Ti": -0.0005007902,
    "V": -3.2723071e-06,
    "W": 3.657056e-05,
    "Zr": -6.4118576e-05
}

# -------------------
# Load and filter
# -------------------
df = pd.read_csv(DATA_FILE)

# Filter rows close to target temp
df_temp = df[(df["Testing_Temp"] >= TEMP - TEMP_TOL) &
             (df["Testing_Temp"] <= TEMP + TEMP_TOL)].dropna(subset=["Yield_Strength"])

print(f"[INFO] Using {len(df_temp)} rows near {TEMP} °C for correlation check.")

# -------------------
# Correlation analysis
# -------------------
elements = ["Al","Co","Cr","Hf","Mo","Nb","Si","Ta","Ti","V","W","Zr"]
results = []

for el in elements:
    col = f"{el}_frac"
    if col not in df_temp.columns:
        continue
    x = df_temp[col].values
    y = df_temp["Yield_Strength"].values

    if len(np.unique(x)) < 2:
        pear, spear = np.nan, np.nan
    else:
        pear, _ = pearsonr(x, y)
        spear, _ = spearmanr(x, y)

    results.append({
        "Element": el,
        "Pearson": pear,
        "Spearman": spear,
        "IG Attribution": ig_scores.get(el, np.nan)
    })

# -------------------
# Save/print results
# -------------------
df_out = pd.DataFrame(results)
df_out = df_out.sort_values("Pearson", ascending=False)

print(df_out.to_string(index=False, float_format="%.4f"))

# Save as CSV for inspection
df_out.to_csv("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/outputs/explain/dataset_vs_ig_corr.csv", index=False)
print("[INFO] Results saved to /outputs/explain/dataset_vs_ig_corr.csv")