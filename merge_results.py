"""Merge partial benchmark results into a unified full_results.csv.

Combines the Phase 4 paid-model results, kalm-12b results, and the
4 re-run open-source model results into one 480-row file, then
regenerates all derived output files (pivot tables, confusion matrices).

Usage:
    python merge_results.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Import output functions from benchmark.py
from benchmark import (
    ALL_CONDITIONS,
    MODELS,
    RESULTS_DIR,
    save_primary_comparison,
    save_classifier_pivots,
    save_per_skill_breakdown,
    print_summary,
)


def main():
    """Merge partial results and regenerate outputs."""
    base = Path(__file__).resolve().parent
    results_dir = base / "results"
    backup_dir = base / "results_phase4_backup"

    parts = []

    # Phase 4: 5 paid models (240 rows)
    p4 = backup_dir / "full_results.csv"
    if not p4.exists():
        print(f"ERROR: {p4} not found")
        sys.exit(1)
    df4 = pd.read_csv(p4)
    print(f"Phase 4 backup: {len(df4)} rows, models: {sorted(df4['model'].unique())}")
    parts.append(df4)

    # kalm-12b (48 rows)
    pk = results_dir / "full_results_kalm12b_only.csv"
    if not pk.exists():
        # Try the model-specific naming
        pk = results_dir / "full_results_kalm-12b.csv"
    if not pk.exists():
        print(f"ERROR: kalm-12b results not found")
        sys.exit(1)
    dfk = pd.read_csv(pk)
    print(f"kalm-12b: {len(dfk)} rows")
    parts.append(dfk)

    # 4 open-source models (48 rows each)
    for model in ["qwen3-0.6b", "qwen3-4b", "qwen3-8b", "jina-v5-small"]:
        path = results_dir / f"full_results_{model}.csv"
        if not path.exists():
            print(f"ERROR: {path} not found — run benchmark.py --model {model} first")
            sys.exit(1)
        df_m = pd.read_csv(path)
        print(f"{model}: {len(df_m)} rows")
        parts.append(df_m)

    # Merge
    combined = pd.concat(parts, ignore_index=True)
    n_models = combined["model"].nunique()
    print(f"\nCombined: {len(combined)} rows, {n_models} models")
    print(f"Models: {sorted(combined['model'].unique())}")

    # Validate
    if len(combined) != 480:
        print(f"WARNING: Expected 480 rows, got {len(combined)}")
    if n_models != 10:
        print(f"WARNING: Expected 10 models, got {n_models}")

    # Check for duplicates
    dupes = combined.duplicated(subset=["model", "condition", "skill", "classifier", "intro_strategy"])
    if dupes.any():
        print(f"WARNING: {dupes.sum()} duplicate rows found — removing")
        combined = combined.drop_duplicates(
            subset=["model", "condition", "skill", "classifier", "intro_strategy"],
            keep="last",
        )
        print(f"After dedup: {len(combined)} rows")

    # Save unified full_results.csv
    combined.to_csv(results_dir / "full_results.csv", index=False)
    print(f"\nSaved results/full_results.csv ({len(combined)} rows)")

    # Regenerate derived outputs
    print("\nRegenerating derived outputs...")
    save_primary_comparison(combined)
    save_classifier_pivots(combined)
    save_per_skill_breakdown(combined)

    # Confusion matrices need _confusion_matrix and _macro_f1_folds fields
    # which are not in the CSV. Skip — they were saved during the original runs.
    print("Note: confusion_matrices.json not regenerated (requires fold-level data)")

    print_summary(combined)
    print("\nDone. All outputs regenerated from merged data.")


if __name__ == "__main__":
    main()
