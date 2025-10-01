# src/evaluate_cvae.py
import sys, joblib, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from cvae import CVAE
from generate import suggest

OUT_DIR = Path("../outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Normalize and round to 2 decimals
    fracs = np.round(fracs / total, 2)

    parts = []
    for el, frac in zip(elements, fracs):
        if frac > 0:
            if abs(frac - 1.0) < 1e-2:
                parts.append(el)
            else:
                # Clean trailing zeros, so Al0.20 → Al0.2
                parts.append(f"{el}{frac:.2f}".rstrip("0").rstrip("."))
    return "".join(parts)

# --------------------------
# Deduplicate & top_n selector
# --------------------------
def get_top_unique(df, top_n, query_type="2", target=None):
    df = df.copy()
    df["Alloy_Formula"] = df.apply(make_alloy_formula, axis=1)
    df = df.drop_duplicates(subset=["Alloy_Formula"])
    if query_type == "1":
        df_sorted = df.sort_values("Predicted_Yield_Strength", ascending=False)
    else:
        df_sorted = df.iloc[(df["Predicted_Yield_Strength"] - target).abs().argsort()]
    return df_sorted.head(top_n), df_sorted

def save_results(df, query_name, top_n=None, target=None, temp=None):
    parts = [query_name]
    if target is not None: parts.append(f"y{int(target)}")
    if temp is not None: parts.append(f"T{int(temp)}")
    if top_n is not None: parts.append(f"top{top_n}")
    out_path = OUT_DIR / ("_".join(parts) + ".csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Results saved → {out_path}")

# --------------------------
# Main
# --------------------------
def main():
    scalers = joblib.load("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/processed/scalers.joblib")
    x_scaler = scalers["x_scaler"]; y_prop_scaler = scalers["y_prop_scaler"]
    y_cond_scaler = scalers["y_cond_scaler"]; feature_cols = scalers["feature_cols"]
    cond_cols = scalers["cond_cols"]

    device = torch.device("cpu")
    model = CVAE(len(feature_cols), len(cond_cols), 1, z_dim=8, hidden=256).to(device)
    model.load_state_dict(torch.load("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt", map_location=device))
    model.eval()

    while True:
        print("\n--- Alloy Design Query Menu ---")
        print("1. Highest yield strength at a given temperature")
        print("2. Alloy close to a target yield strength (optionally at a given temp)")
        print("3. Multi-objective design (constraints at multiple temps)")
        print("4. Exit")
        choice = input("Select query type (1-4): ").strip()
        if choice == "4": break

        top_n = int(input("How many top results? ").strip())
        max_elements = int(input("Max number of elements (0=none): ").strip() or 0) or None

        if choice == "1":
            temp = float(input("Enter temperature (°C): "))
            df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                        y_target_scalar=2000.0, temp=temp, N=300, refine=True,
                        max_elements=max_elements)
            df_top, df_all = get_top_unique(df, top_n, query_type="1")
            
            # --- Plot ---
            plt.figure(figsize=(8,6))
            plt.scatter(range(len(df_all)), df_all["Predicted_Yield_Strength"],
                        color="blue", label="All candidates")
            plt.scatter(df_top.index, df_top["Predicted_Yield_Strength"],
                        color="red", label=f"Top {top_n}")
            plt.title(f"Top {top_n} Highest Yield Strength Alloys @ {temp} °C")
            plt.xlabel("Candidate Index (sorted by strength)")
            plt.ylabel("Yield Strength (MPa)")
            plt.legend()
            plt.tight_layout()
            plot_path = OUT_DIR / f"query1_T{int(temp)}_top{top_n}.png"
            plt.savefig(plot_path, dpi=300)
            plt.show()
            print(f"[INFO] Plot saved → {plot_path}")
            
            # Print + save results
            print(df_top[["Predicted_Yield_Strength","Alloy_Formula"]])
            save_results(df_top,"query1",top_n,temp=temp)

        elif choice == "2":
            target = float(input("Target yield strength (MPa): "))
            t_in = input("Condition on temperature? (y/n): ").strip().lower()
            temp = float(input("Enter temperature (°C): ")) if t_in=="y" else None
            df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                         y_target_scalar=target, temp=temp, N=300, refine=True,
                         max_elements=max_elements)
            df_top, df_all = get_top_unique(df, top_n, query_type="2", target=target)

            # Plot
            plt.figure(figsize=(8,6))
            plt.scatter(range(len(df_all)), df_all["Predicted_Yield_Strength"], color="blue", label="All candidates")
            plt.scatter(df_top.index, df_top["Predicted_Yield_Strength"], color="red", label=f"Top {top_n}")
            plt.axhline(y=target, color="r", linestyle="--", label="Target")
            plt.title(f"Generated Candidates for {target} MPa {('@'+str(temp)+'°C') if temp else ''}")
            plt.xlabel("Candidate index (sorted by closeness)"); plt.ylabel("Yield Strength (MPa)")
            plt.legend(); plt.tight_layout()
            plt.savefig(OUT_DIR / f"query2_y{int(target)}{'_T'+str(int(temp)) if temp else ''}_top{top_n}.png", dpi=300)
            plt.show()

            print(df_top[["Predicted_Yield_Strength","Alloy_Formula"]])
            save_results(df_top,"query2",top_n,target=target,temp=temp)

        elif choice == "3":
            n_constraints = int(input("Enter number of constraints: "))
            constraints = [
                (float(input(f"T{i+1} (°C): ")),
                 float(input(f"Min strength{i+1} (MPa): ")))
                for i in range(n_constraints)
            ]

            df = suggest(
                model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                y_target_scalar=np.mean([m for _, m in constraints]),
                N=400, refine=True, max_elements=max_elements
            )

            valid, near = [], []
            for _, row in df.iterrows():
                ok, near_ok = True, True
                for (t, m) in constraints:
                    dfc = suggest(
                        model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                        y_target_scalar=row["Predicted_Yield_Strength"], temp=t,
                        N=1, refine=True, max_elements=max_elements
                    )
                    val = dfc["Predicted_Yield_Strength"].iloc[0]
                    if val < m:
                        ok = False
                        if val < 0.95 * m:  # more than 5% short
                            near_ok = False
                            break
                if ok:
                    valid.append(row)
                elif near_ok:
                    near.append(row)

            if valid:
                df_valid, _ = get_top_unique(
                    pd.DataFrame(valid), top_n, query_type="2",
                    target=np.mean([m for _, m in constraints])
                )
                print("[INFO] Alloys satisfying all constraints:")
                print(df_valid[["Predicted_Yield_Strength", "Alloy_Formula"]])
                save_results(df_valid, "query3_multi", top_n)
            elif near:
                df_near, _ = get_top_unique(
                    pd.DataFrame(near), top_n, query_type="2",
                    target=np.mean([m for _, m in constraints])
                )
                print("[INFO] No alloys strictly satisfy constraints. Showing near-satisfying alloys (within 5%):")
                print(df_near[["Predicted_Yield_Strength", "Alloy_Formula"]])
                save_results(df_near, "query3_multi_near", top_n)
            else:
                print("[INFO] No alloys satisfy or come close to constraints.")

if __name__=="__main__":
    main()