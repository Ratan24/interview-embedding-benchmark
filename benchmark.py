"""Benchmark embedding models via probing classifiers on skill-grade prediction.

Evaluates 5 embedding models × 6 parsing conditions × 3 skills using both
nominal (LogisticRegression) and ordinal (LogisticAT) classifiers with
5-fold stratified cross-validation.

Usage:
    python benchmark.py                              # full sweep
    python benchmark.py --model voyage --condition C3
    python benchmark.py --dry-run                    # print combos and exit
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
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

ALL_CONDITIONS = ["C1a", "C1b", "C2a", "C2b", "C3", "C4"]
SKILLS = ["react", "javascript", "html_css"]
SKILL_GRADE_COLS = {
    "react": "react_grade",
    "javascript": "javascript_grade",
    "html_css": "html_css_grade",
}

# Segment indices for per-skill conditions
SEGMENT_IDX_NO_INTRO = {"react": 0, "javascript": 1, "html_css": 2}
SEGMENT_IDX_WITH_INTRO = {"intro": 0, "react": 1, "javascript": 2, "html_css": 3}

# Models that have .npy files in vectors/
MODELS = ["gemini", "openai-large", "openai-small", "voyage", "cohere",
          "kalm-12b", "qwen3-8b", "qwen3-4b", "jina-v5-small", "qwen3-0.6b"]

# Conditions that have per-skill segments with intro
CONDITIONS_WITH_INTRO = ["C1b", "C2b"]
# Conditions that are per-skill (have segment dimension)
CONDITIONS_PER_SKILL = ["C1a", "C1b", "C2a", "C2b"]
# Conditions that are full-transcript
CONDITIONS_FULL = ["C3", "C4"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
    """Load labels from parsed/labels.csv.

    Returns:
        DataFrame with job_application_id and grade columns.
    """
    path = PARSED_DIR / "labels.csv"
    df = pd.read_csv(path)
    return df


def load_embeddings(model_key, condition):
    """Load embedding array and IDs for a model-condition pair.

    Args:
        model_key: Model name (e.g. 'voyage').
        condition: Condition ID (e.g. 'C3').

    Returns:
        Tuple of (embeddings_array, ids_list).
    """
    npy_path = VECTORS_DIR / f"{model_key}_{condition}.npy"
    ids_path = VECTORS_DIR / f"{model_key}_{condition}_ids.json"

    embeddings = np.load(npy_path)
    with open(ids_path) as f:
        ids = json.load(f)

    return embeddings, ids


def align_labels_to_embeddings(labels_df, embed_ids):
    """Reorder labels to match embedding ID order.

    Args:
        labels_df: Labels DataFrame.
        embed_ids: List of job_application_ids from the embedding file.

    Returns:
        DataFrame aligned to embed_ids order.
    """
    id_to_idx = {aid: i for i, aid in enumerate(labels_df["job_application_id"])}
    indices = [id_to_idx[aid] for aid in embed_ids]
    return labels_df.iloc[indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(embeddings, condition, skill, intro_strategy=None):
    """Extract the feature vector for a given skill from the embedding array.

    For per-skill conditions: selects the skill's segment embedding.
    For full-transcript conditions: uses the full embedding directly.
    For conditions with intro (C1b/C2b): applies the intro_strategy.

    Args:
        embeddings: Numpy array from .npy file.
        condition: Condition ID string.
        skill: Skill key ('react', 'javascript', 'html_css').
        intro_strategy: For C1b/C2b only. One of 'concat' or 'average'.

    Returns:
        2D numpy array of shape (n_candidates, feature_dim).
    """
    if condition in CONDITIONS_FULL:
        # Full transcript — same embedding for all skills
        return embeddings

    if condition in CONDITIONS_WITH_INTRO:
        # Per-skill with intro
        skill_idx = SEGMENT_IDX_WITH_INTRO[skill]
        intro_idx = SEGMENT_IDX_WITH_INTRO["intro"]

        skill_emb = embeddings[:, skill_idx, :]
        intro_emb = embeddings[:, intro_idx, :]

        if intro_strategy == "concat":
            return np.concatenate([intro_emb, skill_emb], axis=1)
        elif intro_strategy == "average":
            return (intro_emb + skill_emb) / 2.0
        else:
            raise ValueError(f"Unknown intro_strategy: {intro_strategy}")

    # Per-skill without intro (C1a, C2a)
    skill_idx = SEGMENT_IDX_NO_INTRO[skill]
    return embeddings[:, skill_idx, :]


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def make_classifier(clf_type):
    """Create a classifier instance.

    Args:
        clf_type: One of 'nominal' or 'ordinal'.

    Returns:
        Classifier object.
    """
    if clf_type == "nominal":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
    elif clf_type == "ordinal":
        return LogisticAT(alpha=1.0, max_iter=1000)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def evaluate_single(X, y, clf_type, binary=False):
    """Run stratified K-fold CV and compute metrics.

    Args:
        X: Feature matrix (n, d).
        y: Label array (n,) with values 0-3 (or 0-1 if binary).
        clf_type: 'nominal' or 'ordinal'.
        binary: True if doing Pass/Fail binary classification.

    Returns:
        Dict with per-fold and mean metrics, plus per-class scores.
    """
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    fold_metrics = {
        "macro_f1": [],
        "weighted_f1": [],
        "mae": [],
        "kappa": [],
    }
    per_class_f1_folds = []
    per_class_prec_folds = []
    per_class_rec_folds = []
    all_y_true = []
    all_y_pred = []
    labels_list = [0, 1] if binary else [0, 1, 2, 3]

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Replace NaN/inf from scaling (e.g. near-zero-variance features)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        clf = make_classifier(clf_type)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Clip ordinal predictions to valid range
        if not binary:
            y_pred = np.clip(y_pred, 0, 3).astype(int)

        fold_metrics["macro_f1"].append(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )
        fold_metrics["weighted_f1"].append(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        fold_metrics["mae"].append(mean_absolute_error(y_test, y_pred))
        fold_metrics["kappa"].append(cohen_kappa_score(y_test, y_pred))

        # Per-class metrics
        per_class_f1_folds.append(
            f1_score(y_test, y_pred, average=None, labels=labels_list, zero_division=0)
        )
        if binary:
            per_class_prec_folds.append(
                precision_score(y_test, y_pred, average=None, labels=labels_list, zero_division=0)
            )
            per_class_rec_folds.append(
                recall_score(y_test, y_pred, average=None, labels=labels_list, zero_division=0)
            )

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Aggregate
    result = {}
    for metric, values in fold_metrics.items():
        result[f"{metric}_mean"] = np.mean(values)
        result[f"{metric}_std"] = np.std(values)
        result[f"{metric}_folds"] = values

    # Per-class metrics averaged across folds
    per_class_arr = np.array(per_class_f1_folds)
    if binary:
        p_arr = np.array(per_class_prec_folds)
        r_arr = np.array(per_class_rec_folds)
        for grade in range(2):
            g_name = {0: "fail", 1: "pass"}[grade]
            result[f"f1_{g_name}_mean"] = per_class_arr[:, grade].mean()
            result[f"f1_{g_name}_std"] = per_class_arr[:, grade].std()
            result[f"prec_{g_name}_mean"] = p_arr[:, grade].mean()
            result[f"rec_{g_name}_mean"] = r_arr[:, grade].mean()
    else:
        for grade in range(4):
            g_name = {0: "not_exp", 1: "junior", 2: "midlevel", 3: "senior"}[grade]
            result[f"f1_{g_name}_mean"] = per_class_arr[:, grade].mean()
            result[f"f1_{g_name}_std"] = per_class_arr[:, grade].std()

    # Full confusion matrix (accumulated across folds)
    result["confusion_matrix"] = confusion_matrix(
        all_y_true, all_y_pred, labels=labels_list
    ).tolist()

    return result


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(model_keys, conditions, binary=False):
    """Run the full benchmark for specified models and conditions.

    Args:
        model_keys: List of model names.
        conditions: List of condition IDs.
        binary: True if running Pass/Fail classification.

    Returns:
        List of result dicts (one per evaluation).
    """
    labels_df = load_labels()
    all_results = []
    total_combos = 0

    # Count combos
    for model_key in model_keys:
        for condition in conditions:
            for skill in SKILLS:
                clf_types = ["nominal"] if binary else ["nominal", "ordinal"]
                if condition in CONDITIONS_WITH_INTRO:
                    total_combos += len(clf_types) * 2  # concat + average
                else:
                    total_combos += len(clf_types)

    # Pre-flight validation: check all .npy files for data quality issues
    log.info("Validating embedding files...")
    skip_set = set()
    for model_key in model_keys:
        for condition in conditions:
            npy_path = VECTORS_DIR / f"{model_key}_{condition}.npy"
            if not npy_path.exists():
                continue
            arr = np.load(npy_path)
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            if nan_count > 0 or inf_count > 0:
                pct = nan_count / arr.size * 100
                log.warning(
                    "  %s/%s: %d NaN (%.1f%%), %d inf — SKIPPING",
                    model_key, condition, nan_count, pct, inf_count,
                )
                skip_set.add((model_key, condition))
            else:
                log.info("  %s/%s: OK  shape=%s", model_key, condition, arr.shape)
    if skip_set:
        log.warning(
            "Skipping %d model-condition pairs due to data quality issues",
            len(skip_set),
        )
    log.info("Validation complete.")

    combo_num = 0
    for model_key in model_keys:
        for condition in conditions:
            npy_path = VECTORS_DIR / f"{model_key}_{condition}.npy"
            if not npy_path.exists():
                log.warning("Skipping %s/%s — .npy not found", model_key, condition)
                continue
            if (model_key, condition) in skip_set:
                continue

            embeddings, embed_ids = load_embeddings(model_key, condition)
            aligned_labels = align_labels_to_embeddings(labels_df, embed_ids)

            for skill in SKILLS:
                y = aligned_labels[SKILL_GRADE_COLS[skill]].values
                if binary:
                    y = (y >= 2).astype(int)

                # Determine intro strategies
                if condition in CONDITIONS_WITH_INTRO:
                    intro_strategies = ["concat", "average"]
                else:
                    intro_strategies = [None]

                for intro_strategy in intro_strategies:
                    X = extract_features(
                        embeddings, condition, skill, intro_strategy
                    )

                    clf_types = ["nominal"] if binary else ["nominal", "ordinal"]
                    for clf_type in clf_types:
                        combo_num += 1
                        strategy_label = intro_strategy or "none"
                        log.info(
                            "[%d/%d] %s / %s / %s / %s / intro=%s",
                            combo_num, total_combos, model_key, condition,
                            skill, clf_type, strategy_label,
                        )

                        t0 = time.time()
                        metrics = evaluate_single(X, y, clf_type, binary=binary)
                        elapsed = time.time() - t0

                        row = {
                            "model": model_key,
                            "condition": condition,
                            "skill": skill,
                            "classifier": clf_type,
                            "intro_strategy": strategy_label,
                            "macro_f1_mean": metrics["macro_f1_mean"],
                            "macro_f1_std": metrics["macro_f1_std"],
                            "weighted_f1_mean": metrics["weighted_f1_mean"],
                            "weighted_f1_std": metrics["weighted_f1_std"],
                            "mae_mean": metrics["mae_mean"],
                            "mae_std": metrics["mae_std"],
                            "kappa_mean": metrics["kappa_mean"],
                            "kappa_std": metrics["kappa_std"],
                            "n_samples": len(y),
                            "feature_dim": X.shape[1],
                            "elapsed_s": round(elapsed, 1),
                        }

                        if binary:
                            row.update({
                                "f1_fail_mean": metrics["f1_fail_mean"],
                                "f1_pass_mean": metrics["f1_pass_mean"],
                                "prec_fail_mean": metrics["prec_fail_mean"],
                                "prec_pass_mean": metrics["prec_pass_mean"],
                                "rec_fail_mean": metrics["rec_fail_mean"],
                                "rec_pass_mean": metrics["rec_pass_mean"],
                                "f1_fail_std": metrics["f1_fail_std"],
                                "f1_pass_std": metrics["f1_pass_std"],
                            })
                        else:
                            row.update({
                                "f1_not_exp_mean": metrics["f1_not_exp_mean"],
                                "f1_junior_mean": metrics["f1_junior_mean"],
                                "f1_midlevel_mean": metrics["f1_midlevel_mean"],
                                "f1_senior_mean": metrics["f1_senior_mean"],
                                "f1_not_exp_std": metrics["f1_not_exp_std"],
                                "f1_junior_std": metrics["f1_junior_std"],
                                "f1_midlevel_std": metrics["f1_midlevel_std"],
                                "f1_senior_std": metrics["f1_senior_std"],
                            })

                        # Store fold-level results and confusion matrix
                        row["_macro_f1_folds"] = metrics["macro_f1_folds"]
                        row["_confusion_matrix"] = metrics["confusion_matrix"]

                        all_results.append(row)

                        log.info(
                            "  macro_f1=%.3f (±%.3f)  kappa=%.3f  "
                            "mae=%.3f  [%.1fs]",
                            metrics["macro_f1_mean"],
                            metrics["macro_f1_std"],
                            metrics["kappa_mean"],
                            metrics["mae_mean"],
                            elapsed,
                        )

    return all_results


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def save_full_results(results, model_key=None, binary=False):
    """Save the complete results table.

    Args:
        results: List of result dicts.
        model_key: If set, save to model-specific file instead of unified.
        binary: True if running Pass/Fail classification.

    Returns:
        DataFrame of results.
    """
    # Drop internal fields for CSV
    export_cols = [
        c for c in results[0].keys() if (not c.startswith("_") or "fold" in c)
    ]
    df = pd.DataFrame(results)[export_cols]
    
    prefix = "full_results_binary" if binary else "full_results"
    
    if model_key:
        path = RESULTS_DIR / f"{prefix}_{model_key}.csv"
    else:
        path = RESULTS_DIR / f"{prefix}.csv"
    df.to_csv(path, index=False)
    log.info("Saved %s (%d rows)", path.name, len(df))
    return df


def save_primary_comparison(df):
    """Save the professor's comparison table: models × conditions, Macro F1.

    Uses the best classifier/intro_strategy per model-condition-skill combo,
    then averages across 3 skills.

    Args:
        df: Full results DataFrame.
    """
    # For each model-condition-skill, pick the best macro_f1_mean
    best = df.loc[
        df.groupby(["model", "condition", "skill"])["macro_f1_mean"].idxmax()
    ]

    # Average across skills
    pivot = best.groupby(["model", "condition"])["macro_f1_mean"].mean()
    pivot = pivot.unstack("condition")

    # Reorder columns and rows
    cond_order = [c for c in ALL_CONDITIONS if c in pivot.columns]
    model_order = [m for m in MODELS if m in pivot.index]
    pivot = pivot.reindex(index=model_order, columns=cond_order)

    # Add row mean
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)

    path = RESULTS_DIR / "primary_comparison.csv"
    pivot.round(3).to_csv(path)
    log.info("Saved %s", path.name)

    # Print for terminal
    log.info("")
    log.info("=" * 70)
    log.info("PRIMARY COMPARISON: Macro F1 (avg across 3 skills, best classifier)")
    log.info("=" * 70)
    log.info("\n%s", pivot.round(3).to_string())
    log.info("")


def save_classifier_pivots(df):
    """Save separate pivot tables for nominal and ordinal classifiers.

    Args:
        df: Full results DataFrame.
    """
    for clf_type in ["nominal", "ordinal"]:
        sub = df[df["classifier"] == clf_type]
        if sub.empty:
            continue

        # For intro conditions, pick the best intro_strategy
        best = sub.loc[
            sub.groupby(["model", "condition", "skill"])["macro_f1_mean"].idxmax()
        ]
        pivot = best.groupby(["model", "condition"])["macro_f1_mean"].mean()
        pivot = pivot.unstack("condition")

        cond_order = [c for c in ALL_CONDITIONS if c in pivot.columns]
        model_order = [m for m in MODELS if m in pivot.index]
        pivot = pivot.reindex(index=model_order, columns=cond_order)
        pivot["mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("mean", ascending=False)

        path = RESULTS_DIR / f"pivot_{clf_type}.csv"
        pivot.round(3).to_csv(path)
        log.info("Saved %s", path.name)


def save_per_skill_breakdown(df, binary=False):
    """Save per-skill results breakdown.

    Args:
        df: Full results DataFrame.
        binary: True if processing binary results.
    """
    # Best classifier/strategy per model-condition-skill
    best = df.loc[
        df.groupby(["model", "condition", "skill"])["macro_f1_mean"].idxmax()
    ]
    
    cls_cols = (
        ["f1_fail_mean", "f1_pass_mean", "prec_pass_mean", "rec_pass_mean"]
        if binary
        else ["f1_not_exp_mean", "f1_junior_mean", "f1_midlevel_mean", "f1_senior_mean"]
    )
    
    cols = [
        "model", "condition", "skill", "classifier", "intro_strategy",
        "macro_f1_mean", "macro_f1_std", "weighted_f1_mean", "mae_mean",
        "kappa_mean"
    ] + cls_cols
    
    out = best[cols].sort_values(
        ["skill", "macro_f1_mean"], ascending=[True, False]
    )
    
    filename = "per_skill_breakdown_binary.csv" if binary else "per_skill_breakdown.csv"
    path = RESULTS_DIR / filename
    out.to_csv(path, index=False)
    log.info("Saved %s (%d rows)", path.name, len(out))


def save_confusion_matrices(results):
    """Save confusion matrices for top model-condition combos.

    Args:
        results: List of result dicts (with _confusion_matrix field).
    """
    # Find top 3 by macro_f1_mean (best classifier per model-condition)
    df = pd.DataFrame(results)
    best_per_combo = df.loc[
        df.groupby(["model", "condition", "skill"])["macro_f1_mean"].idxmax()
    ]
    top = best_per_combo.nlargest(9, "macro_f1_mean")  # top 3 combos × 3 skills

    cm_data = {}
    for _, row in top.iterrows():
        key = f"{row['model']}_{row['condition']}_{row['skill']}_{row['classifier']}"
        # Find the matching result to get confusion matrix
        for r in results:
            if (
                r["model"] == row["model"]
                and r["condition"] == row["condition"]
                and r["skill"] == row["skill"]
                and r["classifier"] == row["classifier"]
                and r["intro_strategy"] == row["intro_strategy"]
            ):
                cm_data[key] = {
                    "confusion_matrix": r["_confusion_matrix"],
                    "macro_f1": round(r["macro_f1_mean"], 3),
                    "labels": ["Not Exp", "Junior", "Mid-level", "Senior"],
                }
                break

    path = RESULTS_DIR / "confusion_matrices.json"
    
    if path.exists():
        with open(path, "r") as f:
            try:
                existing_data = json.load(f)
                existing_data.update(cm_data)
                cm_data = existing_data
            except json.JSONDecodeError:
                pass

    with open(path, "w") as f:
        json.dump(cm_data, f, indent=2)
    log.info("Saved %s (%d matrices total)", path.name, len(cm_data))


def print_summary(df):
    """Print the final summary to the terminal.

    Args:
        df: Full results DataFrame.
    """
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    # Overall best model-condition
    best_per_combo = df.loc[
        df.groupby(["model", "condition", "skill"])["macro_f1_mean"].idxmax()
    ]
    avg_by_model_cond = best_per_combo.groupby(
        ["model", "condition"]
    )["macro_f1_mean"].mean()
    best_idx = avg_by_model_cond.idxmax()
    best_val = avg_by_model_cond.max()
    log.info(
        "  Best model-condition: %s / %s  (Macro F1 = %.3f)",
        best_idx[0], best_idx[1], best_val,
    )

    # Ordinal vs nominal
    for clf in ["nominal", "ordinal"]:
        sub = df[df["classifier"] == clf]
        best_sub = sub.loc[
            sub.groupby(["model", "condition", "skill"])["macro_f1_mean"].idxmax()
        ]
        mean_f1 = best_sub["macro_f1_mean"].mean()
        log.info("  %s classifier avg Macro F1: %.3f", clf, mean_f1)

    # Concat vs average (for intro conditions only)
    intro_df = df[df["condition"].isin(CONDITIONS_WITH_INTRO)]
    if not intro_df.empty:
        for strategy in ["concat", "average"]:
            sub = intro_df[intro_df["intro_strategy"] == strategy]
            if not sub.empty:
                mean_f1 = sub["macro_f1_mean"].mean()
                log.info("  Intro strategy '%s' avg Macro F1: %.3f", strategy, mean_f1)

    # Per-skill summary
    log.info("")
    log.info("  Per-skill best (across all models/conditions):")
    for skill in SKILLS:
        skill_df = best_per_combo[best_per_combo["skill"] == skill]
        if skill_df.empty:
            continue
        best_row = skill_df.loc[skill_df["macro_f1_mean"].idxmax()]
        log.info(
            "    %s: %.3f  (%s / %s / %s)",
            skill,
            best_row["macro_f1_mean"],
            best_row["model"],
            best_row["condition"],
            best_row["classifier"],
        )

    # Worst-performing class
    log.info("")
    log.info("  Per-class F1 (averaged across all combos):")
    if "f1_pass_mean" in df.columns:
        cls_list = [
            ("f1_fail_mean", "Fail"),
            ("f1_pass_mean", "Pass"),
        ]
    else:
        cls_list = [
            ("f1_not_exp_mean", "Not Experienced"),
            ("f1_junior_mean", "Junior"),
            ("f1_midlevel_mean", "Mid-level"),
            ("f1_senior_mean", "Senior"),
        ]
    for cls_col, cls_name in cls_list:
        mean_val = df[cls_col].mean()
        log.info("    %s: %.3f", cls_name, mean_val)

    log.info("")
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with model, condition, and dry_run fields.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models with probing classifiers."
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=None,
        help="Run a single model (default: all).",
    )
    parser.add_argument(
        "--condition",
        choices=ALL_CONDITIONS,
        default=None,
        help="Run a single condition (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned combinations and exit without running.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Run as binary classification (Pass/Fail) rather than 4-class ordinal.",
    )
    return parser.parse_args()


def main():
    """Entry point for the benchmark pipeline."""
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_keys = [args.model] if args.model else MODELS
    conditions = [args.condition] if args.condition else ALL_CONDITIONS

    # Warn if running partial with unified file present
    if args.model and (RESULTS_DIR / "full_results.csv").exists():
        log.warning(
            "Running single model '%s'. Results saved to "
            "full_results_%s.csv (unified file NOT updated)",
            args.model, args.model,
        )

    # Dry run: just print what would be evaluated
    if args.dry_run:
        combo_count = 0
        for model_key in model_keys:
            for condition in conditions:
                npy_path = VECTORS_DIR / f"{model_key}_{condition}.npy"
                exists = npy_path.exists()
                for skill in SKILLS:
                    if condition in CONDITIONS_WITH_INTRO:
                        strategies = ["concat", "average"]
                    else:
                        strategies = ["none"]
                    for strategy in strategies:
                        for clf_type in ["nominal", "ordinal"]:
                            combo_count += 1
                            status = "OK" if exists else "MISSING"
                            print(
                                f"  [{status}] {model_key:15s} {condition:4s} "
                                f"{skill:12s} {clf_type:8s} "
                                f"intro={strategy}"
                            )
        print(f"\nTotal combinations: {combo_count}")
        return

    # Run benchmark
    log.info("Starting benchmark: %d models × %d conditions (binary=%s)",
             len(model_keys), len(conditions), args.binary)
    t0 = time.time()
    results = run_benchmark(model_keys, conditions, binary=args.binary)
    elapsed = time.time() - t0

    if not results:
        log.error("No results produced. Check that .npy files exist.")
        return

    log.info("Benchmark complete: %d evaluations in %.0fs", len(results), elapsed)

    # Save outputs
    df = save_full_results(results, model_key=args.model, binary=args.binary)
    save_confusion_matrices(results)
    if not args.model:
        save_primary_comparison(df)
        save_classifier_pivots(df)
        save_per_skill_breakdown(df, binary=args.binary)
    print_summary(df)

    log.info("All results saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
