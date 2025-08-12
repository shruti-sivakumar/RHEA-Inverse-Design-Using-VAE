import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


IN_FILE  = "corrected_materials_dataset.csv"
OUT_FILE = "materials_rhea_encoded.csv"

NUM_COLS = [
    "Density",
    "Young_Modulus_ROM",
    "Young_Modulus_Exp",
    "Testing_Temp",
    "Yield_Strength",
    "Specific_Strength",
]

CAT_COLS = [
    "Equilibrium_Conditions",
    "Single_Multiphase",
    "Type_Present_Phases",
    "Tension_Compression",
]


def clean_numerical_value(v):
    """Convert messy numeric strings to float, else NaN."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().strip("\"'")
    if s in {"", "-"}:
        return np.nan
    if s.upper() == "RT":
        return 25.0
    m = re.search(r"\(([\d.,]+)\)", s)
    if m:
        s = m.group(1)
    s = re.sub(r"\[.*?\]", "", s)
    if "/" in s:
        s = s.split("/", 1)[0]
    s = re.sub(r"[^\d.,+\-eE]", " ", s).strip()
    s = re.sub(r"\s+", "", s)
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def pick_best_max_yield_then_specific(g: pd.DataFrame) -> pd.Series:
    """Pick row with max Yield_Strength, tie-break by max Specific_Strength."""
    max_y = g["Yield_Strength"].max()
    top = g[g["Yield_Strength"] == max_y]
    if len(top) == 1:
        return top.iloc[0]
    # tie-breaker: highest Specific_Strength
    return top.loc[top["Specific_Strength"].fillna(-np.inf).idxmax()]


def main():
    df = pd.read_csv(IN_FILE)

    # Clean numeric columns
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(clean_numerical_value)

    # Deduplicate per Composition
    df_best = (
        df.groupby("Composition", as_index=False, group_keys=False)
          .apply(pick_best_max_yield_then_specific)
          .reset_index(drop=True)
    )

    # Label encode categorical columns
    for col in CAT_COLS:
        if col not in df_best.columns:
            df_best[col] = "Unknown"
        df_best[col] = df_best[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df_best[f"{col}_encoded"] = le.fit_transform(df_best[col])

    # Save final CSV
    df_best.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved: {OUT_FILE}  shape={df_best.shape}")

if __name__ == "__main__":
    main()
