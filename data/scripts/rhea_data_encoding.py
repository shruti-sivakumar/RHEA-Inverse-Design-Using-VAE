import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

IN_FILE  = "data/data.csv"
OUT_FILE_MODEL = "data/encoded_data.csv"          # numeric/model-ready
OUT_FILE_HUMAN = "data/encoded_data_human.csv"    # readable for reports

ELEMENTS = ['Al', 'Co', 'Cr', 'Hf', 'Mo', 'Nb', 'Si', 'Ta', 'Ti', 'V', 'W', 'Zr']

NUM_COLS = [
    "Density",
    "Young_Modulus_ROM",
    "Young_Modulus_Exp",     # will be dropped for model CSV
    "Testing_Temp",
    "Yield_Strength",
    "Specific_Strength",     # will be dropped for model CSV
]

CAT_COLS = [
    "Equilibrium_Conditions",
    "Single_Multiphase",
    "Type_Present_Phases",
    "Tension_Compression",
]

LEAK_COLS = ["Specific_Strength", "Young_Modulus_Exp"]

EL_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d+)?")

def clean_numerical_value(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().strip("\"'")
    if s in {"", "-"}: return np.nan
    if s.upper() == "RT": return 25.0
    m = re.search(r"\(([\d.,]+)\)", s)
    if m: s = m.group(1)
    s = re.sub(r"\[.*?\]", "", s)
    if "/" in s: s = s.split("/", 1)[0]
    s = re.sub(r"[^\d.,+\-eE]", " ", s).strip()
    s = re.sub(r"\s+", "", s)
    if "," in s and "." in s: s = s.replace(",", "")
    elif "," in s: s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_composition_to_percents(comp_str: str, elements=ELEMENTS, eps=1e-12):
    counts = {el: 0.0 for el in elements}
    if not isinstance(comp_str, str) or not comp_str.strip():
        return counts
    total = 0.0
    for sym, num in EL_RE.findall(comp_str):
        amt = float(num) if num not in (None, "",) else 1.0
        if sym in counts:
            counts[sym] += amt
            total += amt
    if total < eps:  # all zeros
        return counts
    # to percentages
    for el in counts:
        counts[el] = 100.0 * counts[el] / total
    # renorm to exact 100
    s = sum(counts.values())
    if abs(s - 100.0) > 1e-8:
        for el in counts:
            counts[el] *= (100.0 / s)
    return counts

def pick_best_max_yield(g: pd.DataFrame) -> pd.Series:
    """Keep the row with the maximum Yield_Strength for this (Composition, Temp)."""
    return g.loc[g["Yield_Strength"].idxmax()]

def main():
    df = pd.read_csv(IN_FILE)

    # Clean numeric columns
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(clean_numerical_value)

    # Deduplicate per (Composition, Testing_Temp) (keep highest Yield_Strength)
    df_best = (
        df.groupby(["Composition", "Testing_Temp"], as_index=False, group_keys=False)
        .apply(pick_best_max_yield, include_groups=False)
        .reset_index(drop=True)
    )

    # Composition -> percentages (+ fractions for modeling)
    comp_pct_rows = [parse_composition_to_percents(c) for c in df_best["Composition"].fillna("")]
    comp_pct_df = pd.DataFrame(comp_pct_rows, columns=ELEMENTS)

    # check sums ~ 100
    assert np.allclose(comp_pct_df.sum(axis=1).values, 100.0, atol=1e-6)

    # rename columns
    comp_pct_df = comp_pct_df.add_suffix("_pct")
    comp_frac_df = comp_pct_df / 100.0
    comp_frac_df.columns = [c.replace("_pct", "_frac") for c in comp_frac_df.columns]

    # Encode categoricals (keep both original for human CSV and encoded for model CSV)
    for col in CAT_COLS:
        if col not in df_best.columns:
            df_best[col] = "Unknown"
        df_best[col] = df_best[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df_best[f"{col}_encoded"] = le.fit_transform(df_best[col])

    # Drop Ref if present
    if "Ref" in df_best.columns:
        df_best = df_best.drop(columns=["Ref"])

    # ---- HUMAN CSV ----
    df_human = df_best.copy()
    for col in comp_pct_df.columns[::-1]:
        df_human.insert(1, col, comp_pct_df[col].values)

    human_cols = (
        ["Composition"] +
        list(comp_pct_df.columns) +
        CAT_COLS +
        NUM_COLS +
        [f"{c}_encoded" for c in CAT_COLS]
    )
    human_cols = [c for c in human_cols if c in df_human.columns]
    df_human = df_human[human_cols]

    # ---- MODEL CSV ----
    model_cols = (
        list(comp_frac_df.columns) +
        [c for c in NUM_COLS if c not in LEAK_COLS] +
        [f"{c}_encoded" for c in CAT_COLS]
    )
    df_model = pd.concat([comp_frac_df, df_best], axis=1)
    model_cols = [c for c in model_cols if c in df_model.columns]
    df_model = df_model[model_cols]

    # Save
    df_model.to_csv(OUT_FILE_MODEL, index=False)
    df_human.to_csv(OUT_FILE_HUMAN, index=False)
    print(f"Saved: {OUT_FILE_MODEL}  shape={df_model.shape}")
    print(f"Saved: {OUT_FILE_HUMAN}  shape={df_human.shape}")

if __name__ == "__main__":
    main()