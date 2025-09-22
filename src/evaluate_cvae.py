# src/evaluate_cvae.py
import sys, joblib, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cvae import CVAE
from generate import suggest

OUT_DIR = Path("/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Alloy formula builder
# --------------------------
def make_alloy_formula(row):
    elements = ["Al","Co","Cr","Hf","Mo","Nb","Si","Ta","Ti","V","W","Zr"]
    frac_cols = [f"{el}_frac" for el in elements]
    fracs = row[frac_cols].values
    total = fracs.sum()
    if total <= 0:
        return ""
    fracs = np.round(fracs/total, 2)
    return "".join([f"{el}{frac}".rstrip("0").rstrip(".")
                    for el, frac in zip(elements, fracs) if frac > 0])

# --------------------------
# Deduplicate & top_n selector
# --------------------------
def get_top_unique(df, top_n, query_type="2", target=None):
    df = df.copy()
    df["Alloy_Formula"] = df.apply(make_alloy_formula, axis=1)
    df = df.drop_duplicates(subset=["Alloy_Formula"])  # remove duplicates

    if query_type == "1":  # highest strength
        df_sorted = df.sort_values("Predicted_Yield_Strength", ascending=False)
    else:  # closeness to target
        if target is None:
            raise ValueError("Target must be provided for query type != 1")
        df_sorted = df.iloc[(df["Predicted_Yield_Strength"] - target).abs().argsort()]

    return df_sorted.head(top_n).reset_index(drop=True), df_sorted.reset_index(drop=True)

def save_results(df, query_name, top_n=None, target=None, temp=None):
    parts = [query_name]
    if target is not None:
        parts.append(f"y{int(target)}")
    if temp is not None:
        parts.append(f"T{int(temp)}")
    if top_n is not None:
        parts.append(f"top{top_n}")
    out_path = OUT_DIR / ("_".join(parts) + ".csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Results saved → {out_path}")

# --------------------------
# Main
# --------------------------
def main():
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
    model = CVAE(x_dim=len(feature_cols), y_cond_dim=len(cond_cols), y_prop_dim=1,
                 z_dim=4, hidden=128).to(device)
    model.load_state_dict(torch.load(
        "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/"
        "Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/models/cvae_best.pt",
        map_location=device
    ))
    model.eval()

    while True:
        print("\n--- Alloy Design Query Menu ---")
        print("1. Highest yield strength at a given temperature")
        print("2. Alloy close to a target yield strength (optionally at a given temp)")
        print("3. Multi-objective design (constraints at multiple temps)")
        print("4. Exit")

        choice = input("Select query type (1-4): ").strip()
        if choice == "4":
            print("Exiting.")
            sys.exit(0)

        top_n = int(input("How many top results to display/save? ").strip())
        limit_choice = input("Do you want to limit number of elements? (y/n): ").strip().lower()
        max_elements = int(input("Enter max number of elements: ").strip()) if limit_choice == "y" else None

        if choice == "1":
            temp = float(input("Enter temperature (°C): "))

            # load training data and get max strength at this temp 
            data_csv = "/Users/shrutisivakumar/Library/CloudStorage/OneDrive-Personal/College Stuff/Sem 5/Projects/DDMM/RHEA-Inverse-Design-Using-VAE/data/encoded_data.csv"
            df_train = pd.read_csv(data_csv)

            subset = df_train[df_train["Testing_Temp"] == temp]
            if subset.empty:
                print(f"[WARN] No training data at {temp} °C. Using global max instead.")
                target_strength = df_train["Yield_Strength"].max()
            else:
                target_strength = subset["Yield_Strength"].max()

            print(f"[INFO] Using target strength = {target_strength:.1f} MPa for {temp} °C")

            # --- Generate candidates with target based on training max ---
            df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                         y_target_scalar=target_strength, temp=temp, N=200, refine=True,
                         max_elements=max_elements)

            df_top, df_all = get_top_unique(df, top_n=top_n, query_type="1")

            # Plot
            plt.figure(figsize=(8,6))
            plt.scatter(range(len(df_all)), df_all["Predicted_Yield_Strength"], color="blue", label="All unique candidates")
            plt.scatter(df_top.index, df_top["Predicted_Yield_Strength"], color="red", label=f"Top {top_n}")
            plt.title(f"Top {top_n} Highest Strength Alloys @ {temp} °C")
            plt.xlabel("Candidate Index (unique, sorted by strength)")
            plt.ylabel("Yield Strength (MPa)")
            plt.legend()
            plt.tight_layout()
            plot_path = OUT_DIR / f"query1_T{int(temp)}_top{top_n}.png"
            plt.savefig(plot_path, dpi=300)
            plt.show()
            print(f"[INFO] Plot saved → {plot_path}")

            print(df_top[["Predicted_Yield_Strength","Alloy_Formula"]])
            save_results(df_top,"query1",top_n=top_n,temp=temp)

        elif choice == "2":
            target = float(input("Enter target yield strength (MPa): "))
            t_in = input("Condition on temperature? (y/n): ").strip().lower()
            temp = float(input("Enter temperature (°C): ")) if t_in == "y" else None

            df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                        y_target_scalar=target, temp=temp, N=200,
                        refine=True, max_elements=max_elements)

            df_top, df_all = get_top_unique(df, top_n=top_n, query_type="2", target=target)

            # Plot
            plt.figure(figsize=(8,6))
            plt.scatter(range(len(df_all)), df_all["Predicted_Yield_Strength"], color="blue", label="All unique candidates")
            plt.scatter(df_top.index, df_top["Predicted_Yield_Strength"], color="red", label=f"Top {top_n}")
            plt.axhline(y=target, color="r", linestyle="--", label="Target")
            title = f"Generated Candidates for {target} MPa"
            if temp is not None:
                title += f" @ {temp} °C"
            plt.title(title)
            plt.xlabel("Candidate Index (unique, sorted by closeness)")
            plt.ylabel("Yield Strength (MPa)")
            plt.legend()
            plt.tight_layout()
            plot_path = OUT_DIR / f"query2_y{int(target)}{'_T'+str(int(temp)) if temp else ''}_top{top_n}.png"
            plt.savefig(plot_path, dpi=300)
            plt.show()
            print(f"[INFO] Plot saved → {plot_path}")

            print(df_top[["Predicted_Yield_Strength","Alloy_Formula"]])
            save_results(df_top, query_name="query2", target=target, temp=temp, top_n=top_n)

        elif choice == "3":
            n_constraints = int(input("Enter number of constraints: "))
            constraints = [(float(input(f"T{i+1} (°C): ")),
                            float(input(f"Min strength{i+1} (MPa): "))) for i in range(n_constraints)]
            df = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                         y_target_scalar=np.mean([m for _,m in constraints]),
                         N=400, refine=True, max_elements=max_elements)

            valid, near = [], []
            for _, row in df.iterrows():
                ok, near_ok = True, True
                for (t,m) in constraints:
                    dfc = suggest(model, x_scaler, y_prop_scaler, y_cond_scaler, feature_cols,
                                  y_target_scalar=row["Predicted_Yield_Strength"], temp=t,
                                  N=1, refine=True, max_elements=max_elements)
                    val = dfc["Predicted_Yield_Strength"].iloc[0]
                    if val < m:
                        ok = False
                        if val < 0.95 * m:  # >5% short
                            near_ok = False
                            break
                if ok: valid.append(row)
                elif near_ok: near.append(row)

            if valid:
                df_valid, df_all = get_top_unique(pd.DataFrame(valid), top_n=top_n, query_type="2",
                                                  target=np.mean([m for _,m in constraints]))
                print("[INFO] Alloys satisfying all constraints:")
                print(df_valid[["Predicted_Yield_Strength","Alloy_Formula"]])
                save_results(df_valid,"query3_multi",top_n=top_n)
            elif near:
                df_near, df_all = get_top_unique(pd.DataFrame(near), top_n=top_n, query_type="2",
                                                 target=np.mean([m for _,m in constraints]))
                print("[INFO] No alloys strictly satisfy constraints. Showing near-satisfying alloys (within 5%):")
                print(df_near[["Predicted_Yield_Strength","Alloy_Formula"]])
                save_results(df_near,"query3_multi_near",top_n=top_n)
            else:
                print("[INFO] No alloys satisfy or come close to constraints.")

if __name__=="__main__":
    main()