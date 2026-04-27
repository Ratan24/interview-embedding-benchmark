import pandas as pd
from pathlib import Path

def main():
    results_dir = Path(__file__).resolve().parent / "results"
    
    # 1. Merge
    csvs = list(results_dir.glob("full_results_binary_*.csv"))
    if not csvs:
        print("No binary CSV chunks found!")
        return
        
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    out_path = results_dir / "full_results_binary.csv"
    df.to_csv(out_path, index=False)
    print(f"Merged {len(csvs)} files into {out_path.name} ({len(df)} rows)")

    # 2. Binary Primary Comparison (model x conditions)
    best = df.loc[df.groupby(["model", "condition", "skill"])["macro_f1_mean"].idxmax()]
    pivot = best.groupby(["model", "condition"])["macro_f1_mean"].mean().unstack("condition")
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)
    pivot.round(3).to_csv(results_dir / "primary_comparison_binary.csv")

    print("\n=== 2/3. BINARY PRIMARY COMPARISON (10-Model Ranking) ===")
    print(pivot.round(3).to_string())

    # 4. Compare vs 4-class results
    df_4_path = results_dir / "primary_comparison.csv"
    if df_4_path.exists():
        df_4 = pd.read_csv(df_4_path, index_col=0)
        ranking = pivot[["mean"]].copy()
        ranking.columns = ["Binary F1"]
        ranking["4-class F1"] = df_4["mean"]
        ranking["Diff / Lift"] = ranking["Binary F1"] - ranking["4-class F1"]
        
        print("\n=== 4. BINARY VS 4-CLASS COMPARISON ===")
        print(ranking.round(3).to_string())

    # 5. Winning condition for binary?
    print("\n=== 5. WINNING CONDITION FOR BINARY ===")
    cond_means = pivot.drop(columns=["mean"]).mean(axis=0).sort_values(ascending=False)
    print(cond_means.round(4).to_string())


if __name__ == "__main__":
    main()
