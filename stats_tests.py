import ast
import pandas as pd
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
results_dir = BASE_DIR / "results"

# 1. Load the 10 partial CSVs
models = [
    "voyage", "gemini", "openai-large", "openai-small", "cohere",
    "kalm-12b", "qwen3-8b", "qwen3-4b", "qwen3-0.6b", "jina-v5-small"
]

dfs = []
for m in models:
    path = results_dir / f"full_results_{m}.csv"
    dfs.append(pd.read_csv(path))
df = pd.concat(dfs, ignore_index=True)

# 2. Filter for C2a Ordinal
focus_df = df[(df['condition'] == 'C2a') & (df['classifier'] == 'ordinal')]

def get_fold_scores(model_name):
    """Extracts all 15 fold scores (5 folds * 3 skills) for a given model."""
    model_data = focus_df[focus_df['model'] == model_name]
    scores = []
    for _, row in model_data.iterrows():
        # Clean the string representation of the list if necessary
        fold_str = str(row['_macro_f1_folds']).replace(' ', ',').replace(',,', ',')
        try:
            folds = ast.literal_eval(fold_str)
        except Exception:
            # Fallback if it's already a list or differently formatted
            folds = eval(row['_macro_f1_folds'])
        scores.extend(folds)
    return scores

# 3. Define Pairs to Test
pairs_to_test = [
    ("voyage", "gemini"),
    ("voyage", "qwen3-8b"),
    ("voyage", "kalm-12b"),
    ("voyage", "openai-large"),
    ("qwen3-8b", "openai-small"),
    ("qwen3-8b", "kalm-12b"),
    ("cohere", "jina-v5-small")
]

print("="*80)
print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test on 5x3 CV folds)")
print("Configuration: C2a (per-skill Q&A), Ordinal Classifier")
print("="*80)
print(f"{'Comparison':<30} | {'Mean Δ':<8} | {'t-stat':<8} | {'p-value':<8} | {'Significant (p<0.05)?'}")
print("-" * 80)

for m1, m2 in pairs_to_test:
    scores1 = get_fold_scores(m1)
    scores2 = get_fold_scores(m2)
    
    mean_diff = (sum(scores1) / len(scores1)) - (sum(scores2) / len(scores2))
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(scores1, scores2)
    
    sig = "YES" if p_val < 0.05 else "NO"
    
    print(f"{m1} vs {m2:<15} | {mean_diff:>8.4f} | {t_stat:>8.3f} | {p_val:>8.4f} | {sig}")

print("="*80)
