import sys, joblib, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from cvae import CVAE
from generate import suggest

# --------------------------
# Paths
# --------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALERS_PATH = BASE_DIR / "data" / "processed" / "scalers.joblib"
MODEL_PATH   = BASE_DIR / "models" / "cvae_best.pt"

# --------------------------
# Alloy formula builder
# --------------------------
def make_alloy_formula(row):
    elements = ["Al","Co","Cr","Hf","Mo","Nb","Si","Ta","Ti","V","W","Zr"]
    frac_cols = [f"{el}_frac" for el in elements]
    fracs = row[frac_cols].astype(float).values
    total = fracs.sum()
    if total <= 0:
        return ""
    fracs = np.round(fracs / total, 2)
    parts = []
    for el, frac in zip(elements, fracs):
        if frac > 0:
            if abs(frac - 1.0) < 1e-2:
                parts.append(el)
            else:
                parts.append(f"{el}{frac:.2f}".rstrip("0").rstrip("."))
    return "".join(parts)

# --------------------------
# Clean export columns
# --------------------------
def trim_for_export(df):
    frac_cols = [c for c in df.columns if c.endswith("_frac")]
    keep = ["Predicted_Yield_Strength", "Conditioned_Temp", "Active_Elements"] + frac_cols + ["Alloy_Formula"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

# --------------------------
# Deduplicate & top_n selector
# --------------------------
def get_top_unique(df, top_n, query_type="2", target=None):
    df = df.copy()
    df["Alloy_Formula"] = df.apply(make_alloy_formula, axis=1)
    df = df.drop_duplicates(subset=["Alloy_Formula"])
    if query_type == "1":
        df_sorted = df.sort_values("Predicted_Yield_Strength", ascending=False).reset_index(drop=True)
    else:
        order = (df["Predicted_Yield_Strength"] - target).abs().argsort()
        df_sorted = df.iloc[order].reset_index(drop=True)
    return df_sorted.head(top_n), df_sorted

def save_results(df, query_name, top_n=None, target=None, temp=None):
    parts = [query_name]
    if target is not None: parts.append(f"y{int(target)}")
    if temp is not None: parts.append(f"T{int(temp)}")
    if top_n is not None: parts.append(f"top{top_n}")
    out_path = OUT_DIR / ("_".join(parts) + ".csv")
    df_export = trim_for_export(df)
    df_export.to_csv(out_path, index=False)
    print(f"[INFO] Results saved → {out_path}")

# --------------------------
# Main
# --------------------------
def main():
    # Load scalers + model
    scalers = joblib.load(SCALERS_PATH)
    x_scaler = scalers["x_scaler"]; y_prop_scaler = scalers["y_prop_scaler"]
    y_cond_scaler = scalers["y_cond_scaler"]; feature_cols = scalers["feature_cols"]
    cond_cols = scalers["cond_cols"]

    device = torch.device("cpu")
    model = CVAE(len(feature_cols), len(cond_cols), 1, z_dim=8, hidden=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    while True:
        print("\n--- Alloy Design Query Menu ---")
        print("1. Highest yield strength at a given temperature")
        print("2. Alloy close to a target yield strength (optionally at a given temp)")
        print("3. Two-constraint (dual-temperature) design query")
        print("4. Exit")
        choice = input("Select query type (1-4): ").strip()
        if choice == "4":
            break

        top_n = int(input("How many top results? ").strip())
        max_elements = int(input("Max number of elements (0=none): ").strip() or 0) or None

        if choice == "1":
            temp = float(input("Enter temperature (°C): "))
            df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                         y_target_scalar=2000.0, temp=temp, N=300, refine=True,
                         max_elements=max_elements)
            df_top, df_all_sorted = get_top_unique(df, top_n, query_type="1")

            # --- Plot with rank on x-axis ---
            plt.figure(figsize=(8,6))
            plt.scatter(df_all_sorted.index, df_all_sorted["Predicted_Yield_Strength"],
                        label="All candidates")
            plt.scatter(df_top.index, df_top["Predicted_Yield_Strength"],
                        label=f"Top {top_n}")
            plt.title(f"Top {top_n} Highest Yield Strength Alloys @ {temp} °C")
            plt.xlabel("Rank (0 = best)"); plt.ylabel("Yield Strength (MPa)")
            plt.legend(); plt.tight_layout()
            plot_path = OUT_DIR / f"query1_T{int(temp)}_top{top_n}.png"
            plt.savefig(plot_path, dpi=300); plt.show()
            print(f"[INFO] Plot saved → {plot_path}")

            print(trim_for_export(df_top)[["Predicted_Yield_Strength","Alloy_Formula","Active_Elements"]])
            save_results(df_top,"query1",top_n,temp=temp)

        elif choice == "2":
            target = float(input("Target yield strength (MPa): "))
            t_in = input("Condition on temperature? (y/n): ").strip().lower()
            temp = float(input("Enter temperature (°C): ")) if t_in=="y" else 25.0
            df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                         y_target_scalar=target, temp=temp, N=300, refine=True,
                         max_elements=max_elements)
            df_top, df_all_sorted = get_top_unique(df, top_n, query_type="2", target=target)

            # Plot ordered by closeness to target
            plt.figure(figsize=(8,6))
            plt.scatter(df_all_sorted.index, df_all_sorted["Predicted_Yield_Strength"], label="All candidates")
            plt.scatter(df_top.index, df_top["Predicted_Yield_Strength"], label=f"Top {top_n}")
            plt.axhline(y=target, linestyle="--", label="Target")
            plt.title(f"Generated Candidates for {target} MPa {('@'+str(temp)+'°C') if temp else ''}")
            plt.xlabel("Rank by closeness"); plt.ylabel("Yield Strength (MPa)")
            plt.legend(); plt.tight_layout()
            plt.savefig(OUT_DIR / f"query2_y{int(target)}{'_T'+str(int(temp)) if temp else ''}_top{top_n}.png", dpi=300)
            plt.show()

            print(trim_for_export(df_top)[["Predicted_Yield_Strength","Alloy_Formula","Active_Elements"]])
            save_results(df_top,"query2",top_n,target=target,temp=temp)

        elif choice == "3":
            print("Two-constraint design query:")
            T1 = float(input("Enter first temperature (°C): "))
            M1 = float(input("Enter minimum strength at T1 (MPa): "))
            T2 = float(input("Enter second temperature (°C): "))
            M2 = float(input("Enter minimum strength at T2 (MPa): "))

            target_avg = np.mean([M1, M2])
            df = suggest(
                model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                y_target_scalar=target_avg, temp=T1,
                N=400, refine=True, max_elements=max_elements
            )

            valid, near = [], []
            for _, row in df.iterrows():
                # Reuse same latent z for the same alloy across temps
                z_same = df.attrs["latent_z"]

                ok, near_ok = True, True
                for (T, M) in [(T1, M1), (T2, M2)]:
                    dfc = suggest(
                        model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                        y_target_scalar=row["Predicted_Yield_Strength"], temp=T,
                        N=1, refine=False, max_elements=max_elements,
                        reuse_z=z_same
                    )
                    val = dfc["Predicted_Yield_Strength"].iloc[0]
                    if val < M:
                        ok = False
                        if val < 0.95 * M:
                            near_ok = False
                            break
                if ok:
                    valid.append(row)
                elif near_ok:
                    near.append(row)

            if valid:
                df_valid, _ = get_top_unique(
                    pd.DataFrame(valid), top_n, query_type="2", target=target_avg
                )
                print("[INFO] Alloys satisfying both constraints:")
                print(trim_for_export(df_valid)[["Predicted_Yield_Strength","Alloy_Formula","Active_Elements"]])
                save_results(df_valid, "query3_two_constraints", top_n)
            elif near:
                df_near, _ = get_top_unique(
                    pd.DataFrame(near), top_n, query_type="2", target=target_avg
                )
                print("[INFO] No alloys strictly satisfy constraints. Showing near-satisfying alloys (within 5%):")
                print(trim_for_export(df_near)[["Predicted_Yield_Strength","Alloy_Formula","Active_Elements"]])
                save_results(df_near, "query3_two_constraints_near", top_n)
            else:
                print("[INFO] No alloys satisfy or come close to the two constraints.")

if __name__=="__main__":
    main()