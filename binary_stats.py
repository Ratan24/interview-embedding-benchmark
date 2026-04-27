import pandas as pd
import numpy as np
from scipy import stats
import ast

def main():
    # Q1: Significance test
    df = pd.read_csv("results/full_results_binary.csv")

    df_c2a = df[df["condition"] == "C2a"]

    def get_folds(model_name):
        sub = df_c2a[df_c2a["model"] == model_name]
        folds = []
        for f_str in sub["_macro_f1_folds"]:
            lst = ast.literal_eval(f_str)
            folds.extend(lst)
        return folds

    openai_folds = get_folds("openai-large")
    voyage_folds = get_folds("voyage")
    kalm_folds = get_folds("kalm-12b")

    t_stat_ov, p_val_ov = stats.ttest_rel(openai_folds, voyage_folds)
    t_stat_ok, p_val_ok = stats.ttest_rel(openai_folds, kalm_folds)
    
    print(f"=== Q1: Significance Testing (C2a) ===")
    print(f"openai-large vs voyage:")
    print(f"  Means: openai={np.mean(openai_folds):.4f}, voyage={np.mean(voyage_folds):.4f}")
    print(f"  t={t_stat_ov:.4f}, p={p_val_ov:.4f} -> Significant: {p_val_ov < 0.05}")
    print()
    print(f"openai-large vs kalm-12b:")
    print(f"  Means: openai={np.mean(openai_folds):.4f}, kalm={np.mean(kalm_folds):.4f}")
    print(f"  t={t_stat_ok:.4f}, p={p_val_ok:.4f} -> Significant: {p_val_ok < 0.05}")

    # Q2: Class Distribution
    labels = pd.read_csv("parsed/labels.csv")
    skills = ["react", "javascript", "html_css"]
    total = 0
    total_pass = 0
    total_fail = 0

    print("\n=== Q2: Class Distribution (Fail < 2 vs Pass >= 2) ===")
    for skill in skills:
        grades = labels[f"{skill}_grade"].dropna().values
        fails = np.sum(grades < 2)
        passes = np.sum(grades >= 2)
        total += len(grades)
        total_fail += fails
        total_pass += passes
        print(f"  {skill.capitalize()}: {fails} Fail, {passes} Pass (Pass % = {passes/len(grades)*100:.1f}%)")

    pass_pct = total_pass / total * 100
    print(f"\nTotal Dataset: {total_fail} Fail, {total_pass} Pass")
    print(f"Base Pass Rate: {pass_pct:.1f}%")

if __name__ == "__main__":
    main()
