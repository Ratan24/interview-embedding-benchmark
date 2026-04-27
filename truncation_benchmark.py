"""Benchmark truncated C3 embeddings to produce a transcript-length curve.

Loads each voyage_C3_trunc{N}.npy file plus the full voyage_C3.npy baseline,
runs 5-fold stratified CV with both 4-class ordinal and binary (pass/fail)
classifiers, and saves a unified results table.

Usage:
    python truncation_benchmark.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PARSED_DIR = BASE_DIR / "parsed"
VECTORS_DIR = BASE_DIR / "vectors"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_SEED = 42
N_FOLDS = 5

TOKEN_LIMITS = [256, 512, 1024, 2048, 4096, 8192]
SKILLS = ["react", "javascript", "html_css"]
SKILL_GRADE_COLS = {
    "react": "react_grade",
    "javascript": "javascript_grade",
    "html_css": "html_css_grade",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_labels():
    """Load labels from parsed/labels.csv."""
    return pd.read_csv(PARSED_DIR / "labels.csv")


def load_ids(suffix):
    """Load the IDs JSON for a given embedding file suffix."""
    if suffix == "full":
        ids_path = VECTORS_DIR / "voyage_C3_ids.json"
    else:
        ids_path = VECTORS_DIR / f"voyage_C3_trunc{suffix}_ids.json"
    with open(ids_path) as f:
        return json.load(f)


def load_embeddings(suffix):
    """Load embeddings for a given token limit.

    Args:
        suffix: Token limit int or 'full' for the baseline.

    Returns:
        Tuple of (embeddings_array, ids_list).
    """
    if suffix == "full":
        npy_path = VECTORS_DIR / "voyage_C3.npy"
    else:
        npy_path = VECTORS_DIR / f"voyage_C3_trunc{suffix}.npy"

    embeddings = np.load(npy_path)
    ids = load_ids(suffix)
    return embeddings, ids


def align_labels(labels_df, embed_ids):
    """Reorder labels to match embedding ID order."""
    id_to_idx = {aid: i for i, aid in enumerate(labels_df["job_application_id"])}
    indices = [id_to_idx[aid] for aid in embed_ids]
    return labels_df.iloc[indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(X, y, binary=False):
    """Run 5-fold stratified CV and return Macro F1.

    Args:
        X: Feature matrix (n, d).
        y: Label array.
        binary: If True, transform y to pass/fail first.

    Returns:
        Dict with macro_f1_mean, macro_f1_std, per-fold scores,
        and per-class F1 scores.
    """
    if binary:
        y = (y >= 2).astype(int)

    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    fold_f1s = []
    labels_list = [0, 1] if binary else [0, 1, 2, 3]
    per_class_folds = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_f1s.append(macro_f1)

        per_class = f1_score(
            y_test, y_pred, average=None, labels=labels_list, zero_division=0
        )
        per_class_folds.append(per_class)

    per_class_arr = np.array(per_class_folds)

    result = {
        "macro_f1_mean": np.mean(fold_f1s),
        "macro_f1_std": np.std(fold_f1s),
        "macro_f1_folds": fold_f1s,
    }

    if binary:
        result["f1_fail_mean"] = per_class_arr[:, 0].mean()
        result["f1_pass_mean"] = per_class_arr[:, 1].mean()
    else:
        for i, name in enumerate(["not_exp", "junior", "midlevel", "senior"]):
            result[f"f1_{name}_mean"] = per_class_arr[:, i].mean()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    labels_df = load_labels()
    all_rows = []

    # Token limits to evaluate: truncated + full baseline
    limits_to_run = TOKEN_LIMITS + ["full"]

    for limit in limits_to_run:
        label = str(limit) if limit != "full" else "full"
        log.info("=== Token limit: %s ===", label)

        embeddings, embed_ids = load_embeddings(limit)
        aligned = align_labels(labels_df, embed_ids)

        for skill in SKILLS:
            y = aligned[SKILL_GRADE_COLS[skill]].values
            X = embeddings  # C3 = full transcript, same embedding for all skills

            # 4-class evaluation
            res_4class = evaluate(X, y, binary=False)

            # Binary evaluation
            res_binary = evaluate(X, y, binary=True)

            row = {
                "token_limit": label,
                "skill": skill,
                "macro_f1_4class": res_4class["macro_f1_mean"],
                "macro_f1_4class_std": res_4class["macro_f1_std"],
                "macro_f1_binary": res_binary["macro_f1_mean"],
                "macro_f1_binary_std": res_binary["macro_f1_std"],
                "f1_fail": res_binary["f1_fail_mean"],
                "f1_pass": res_binary["f1_pass_mean"],
                "f1_not_exp": res_4class["f1_not_exp_mean"],
                "f1_junior": res_4class["f1_junior_mean"],
                "f1_midlevel": res_4class["f1_midlevel_mean"],
                "f1_senior": res_4class["f1_senior_mean"],
            }
            all_rows.append(row)

            log.info(
                "  %s: 4-class=%.3f (±%.3f)  binary=%.3f (±%.3f)",
                skill,
                res_4class["macro_f1_mean"], res_4class["macro_f1_std"],
                res_binary["macro_f1_mean"], res_binary["macro_f1_std"],
            )

    # Save results
    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "truncation_curve.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved %s (%d rows)", out_path.name, len(df))

    # Print summary table (averaged across skills)
    print("\n" + "=" * 70)
    print("TRUNCATION CURVE SUMMARY (averaged across 3 skills)")
    print("=" * 70)

    summary = df.groupby("token_limit").agg({
        "macro_f1_4class": "mean",
        "macro_f1_binary": "mean",
    })

    # Reorder rows
    order = [str(l) for l in TOKEN_LIMITS] + ["full"]
    summary = summary.reindex(order)
    summary.columns = ["4-Class F1", "Binary F1"]
    print(summary.round(3).to_string())

    # Show the knee
    best_4class = summary["4-Class F1"].max()
    for idx, row in summary.iterrows():
        if row["4-Class F1"] >= best_4class * 0.95:
            print(f"\n95% of best 4-class F1 reached at token limit: {idx}")
            break

    print("=" * 70)


if __name__ == "__main__":
    main()
