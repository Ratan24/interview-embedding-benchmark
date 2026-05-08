"""Sequential stopping-agent classifier for binary pass/fail prediction.

Adapts the methodology of Manzoor, Ascarza, Netzer (2025), "Learning When
to Quit in Sales Conversations", to AI-paced technical interviews.  The
upstream notebook fine-tunes Llama 3.2 3B on raw tokens; here we instead
run a probing logistic regression on top of pre-computed Voyage `C3`
embeddings, evaluated at six transcript-prefix checkpoints (256, 512,
1024, 2048, 4096 candidate-only `cl100k_base` tokens, and the full
transcript).

Phase 1 (per-checkpoint outcome classifier) and Phase 2 (asymmetric
confidence-thresholded stopping rule) are produced in a single run.
A pooled-model variant with checkpoint as a one-hot covariate is fit
as a sanity check.

Inputs:
    data/transcripts_6400_records.csv
    data/Ai-Vetted-ranked.csv
    <vectors>/voyage_C3_trunc{256,512,1024,2048,4096}.npy + ids
    <vectors>/voyage_C3.npy + ids

The vectors directory defaults to repo-local `vectors/` and is overridable
via the `HAZARD_VECTORS_DIR` environment variable, mirroring the existing
`plot_hazard_embeddings.py` convention.

Outputs:
    results/stopping_agent_per_checkpoint.csv
    results/stopping_agent_oof_predictions.csv
    results/stopping_agent_threshold_sweep.csv
    figures/stopping_agent_auc_curve.png
    figures/stopping_agent_savings.png
    figures/stopping_agent_threshold_heatmap.png
"""

import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from parse_transcripts import format_answer_only, parse_single_transcript
from plot_hazard_embeddings import align, load_length_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_CSV = BASE_DIR / "data" / "transcripts_6400_records.csv"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

VECTORS_DIR = Path(
    os.environ.get("HAZARD_VECTORS_DIR", BASE_DIR / "vectors")
)

CHECKPOINTS = [256, 512, 1024, 2048, 4096, "full"]
PREFIX_CHECKPOINTS = [c for c in CHECKPOINTS if c != "full"]
N_FOLDS = 5
RANDOM_SEED = 42
LOGREG_C = 1.0

POSITION_COLUMNS = [
    "tokens_consumed",
    "candidate_words_so_far",
    "n_messages_so_far",
    "candidate_words_per_turn",
    "fraction_of_overall_median_tokens",
]

TAU_GRID = np.array(
    [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49]
)

ACCENT_COLOR = "#1f3a5f"
ACCENT_ALT = "#a23b3b"
REFERENCE_COLOR = "#888888"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_aligned_inputs():
    """Build candidate-level frame and matching parsed-message dictionary.

    Returns
    -------
    frame : pd.DataFrame
        Duration-filtered candidates with `job_application_id`, `is_passed`,
        `fail`, and `c3_full_tokens` (cl100k_base token count of the
        candidate-only transcript).  Indexed sequentially.
    transcripts : dict[str, list[dict]]
        Map from `job_application_id` to the parsed message list.
    median_full_tokens : float
        Median full-transcript candidate-token count over `frame`.
    """
    log.info("Loading duration-filtered candidate frame …")
    frame = load_length_frame()
    frame = frame[["job_application_id", "is_passed", "fail",
                   "duration_min", "duration_sec"]].copy()

    log.info("Re-parsing transcripts for kept candidates …")
    raw = pd.read_csv(
        TRANSCRIPTS_CSV, encoding="latin1", on_bad_lines="skip"
    )
    keep_ids = set(frame["job_application_id"])
    raw = raw[raw["job_application_id"].isin(keep_ids)].copy()

    encoder = tiktoken.get_encoding("cl100k_base")
    transcripts = {}
    c3_tokens = {}
    for _, row in raw.iterrows():
        msgs = parse_single_transcript(row["interview_transcript"])
        if msgs is None or not isinstance(msgs, list) or not msgs:
            continue
        c3_text = format_answer_only(msgs)
        if not c3_text.strip():
            continue
        transcripts[row["job_application_id"]] = msgs
        c3_tokens[row["job_application_id"]] = len(encoder.encode(c3_text))

    frame = frame[frame["job_application_id"].isin(transcripts)].copy()
    frame["c3_full_tokens"] = frame["job_application_id"].map(c3_tokens)

    # Intersect with embedding cohort up front so every checkpoint aligns
    # losslessly downstream.  All voyage_C3 ID files share the same 6338-id
    # set, so any one of them suffices.
    sample_ids_path = VECTORS_DIR / "voyage_C3_ids.json"
    if not sample_ids_path.exists():
        raise FileNotFoundError(
            f"Could not find {sample_ids_path}.  Set HAZARD_VECTORS_DIR to "
            f"the directory containing the benchmark's pre-computed "
            f"*.npy / *_ids.json files."
        )
    embedding_ids = set(json.load(open(sample_ids_path)))
    before = len(frame)
    frame = frame[frame["job_application_id"].isin(embedding_ids)].copy()
    frame = frame.reset_index(drop=True)
    log.info(
        "  After intersecting with embedding cohort: %d (dropped %d)",
        len(frame), before - len(frame),
    )

    median_full = float(frame["c3_full_tokens"].median())
    log.info(
        "  N=%d  pass_rate=%.1f%%  median_c3_tokens=%.0f",
        len(frame), 100 * (1 - frame["fail"].mean()), median_full,
    )
    return frame, transcripts, median_full


# ---------------------------------------------------------------------------
# Position features
# ---------------------------------------------------------------------------


def position_features(messages, budget, candidate_full_tokens,
                      median_full_tokens, encoder):
    """Compute the five position scalars for one (candidate, checkpoint).

    The budget unit is candidate-only `cl100k_base` tokens, matching the
    truncation used to produce the on-disk `voyage_C3_trunc{N}.npy`
    embeddings.  Interviewer turns count toward `n_messages_so_far` but
    not toward the budget.
    """
    if budget == "full":
        tokens_consumed = candidate_full_tokens
        n_messages = sum(
            1 for m in messages if (m.get("content") or "").strip()
        )
        candidate_words = sum(
            len((m.get("content") or "").strip().split())
            for m in messages if m.get("role") == "user"
        )
    else:
        cumulative = 0
        n_messages = 0
        candidate_words = 0
        for m in messages:
            content = (m.get("content") or "").strip()
            if not content:
                continue
            n_messages += 1
            if m.get("role") == "user":
                msg_tokens = len(encoder.encode(content))
                if cumulative + msg_tokens > budget:
                    fraction = (budget - cumulative) / max(msg_tokens, 1)
                    fraction = max(0.0, min(1.0, fraction))
                    candidate_words += int(round(
                        len(content.split()) * fraction
                    ))
                    cumulative = budget
                    break
                cumulative += msg_tokens
                candidate_words += len(content.split())
                if cumulative >= budget:
                    break
        tokens_consumed = min(int(budget), int(candidate_full_tokens))

    return {
        "tokens_consumed": float(tokens_consumed),
        "candidate_words_so_far": float(candidate_words),
        "n_messages_so_far": float(n_messages),
        "candidate_words_per_turn": (
            float(candidate_words) / float(max(n_messages, 1))
        ),
        "fraction_of_overall_median_tokens": (
            float(tokens_consumed) / float(max(median_full_tokens, 1))
        ),
    }


def build_position_matrix(frame, transcripts, checkpoint, median_full,
                          encoder):
    """Return position-feature matrix `[N, 5]` aligned to `frame` row order."""
    rows = []
    for _, row in frame.iterrows():
        msgs = transcripts[row["job_application_id"]]
        feats = position_features(
            msgs, checkpoint, row["c3_full_tokens"], median_full, encoder
        )
        rows.append([feats[c] for c in POSITION_COLUMNS])
    return np.asarray(rows, dtype=np.float32)


def load_voyage_checkpoint(checkpoint, frame):
    """Load the Voyage C3 embedding for one checkpoint, aligned to frame."""
    fname = (
        "voyage_C3.npy" if checkpoint == "full"
        else f"voyage_C3_trunc{checkpoint}.npy"
    )
    ids_fname = fname.replace(".npy", "_ids.json")
    npy_path = VECTORS_DIR / fname
    ids_path = VECTORS_DIR / ids_fname
    if not npy_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Missing {npy_path.name} or {ids_path.name} under {VECTORS_DIR}.  "
            f"Set HAZARD_VECTORS_DIR to the directory containing the "
            f"benchmark's pre-computed *.npy / *_ids.json files."
        )
    X = np.load(npy_path).astype(np.float32, copy=False)
    ids = json.load(open(ids_path))
    aligned_frame, X_aligned = align(frame.copy(), X, ids)
    if len(aligned_frame) != len(frame):
        raise RuntimeError(
            f"Alignment dropped {len(frame) - len(aligned_frame)} candidates "
            f"at checkpoint {checkpoint}; expected zero loss given prior filter."
        )
    if not (aligned_frame["job_application_id"].values
            == frame["job_application_id"].values).all():
        raise RuntimeError(
            f"Row order divergence at checkpoint {checkpoint}."
        )
    return X_aligned


# ---------------------------------------------------------------------------
# Out-of-fold logistic regression
# ---------------------------------------------------------------------------


def predict_oof(X, y, groups=None):
    """5-fold OOF logistic regression with class-balanced weights.

    Uses StratifiedKFold when `groups is None` and StratifiedGroupKFold
    otherwise.  Returns (p_fail, metrics) where metrics is a dict with
    ROC-AUC, PR-AUC, Macro-F1, recall@P=0.95, n_pos and n_neg.
    """
    n_splits = N_FOLDS
    if groups is None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
        )
        split_iter = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
        )
        split_iter = splitter.split(X, y, groups)

    p_fail = np.full(len(y), np.nan, dtype=np.float64)
    fold_aucs = []
    for fold, (tr, te) in enumerate(split_iter, start=1):
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(
            C=LOGREG_C,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        clf.fit(X_tr, y[tr])
        p_fail[te] = clf.predict_proba(X_te)[:, 1]
        fold_aucs.append(float(roc_auc_score(y[te], p_fail[te])))

    auc = float(roc_auc_score(y, p_fail))
    # PR-AUC is reported for the *rare* class (pass) since y is the
    # fail indicator and 95.5% of candidates fail.  Computing AP on the
    # majority class would be trivially close to 1.
    y_pass = 1 - y
    p_pass = 1.0 - p_fail
    pr_auc_pass = float(average_precision_score(y_pass, p_pass))
    y_pred = (p_fail >= 0.5).astype(int)
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    recall_p95_pass = recall_at_precision(
        y_pass, p_pass, target_precision=0.95
    )
    recall_p50_pass = recall_at_precision(
        y_pass, p_pass, target_precision=0.50
    )

    metrics = {
        "auc": auc,
        "pr_auc_pass": pr_auc_pass,
        "macro_f1": macro_f1,
        "recall_at_p95_pass": recall_p95_pass,
        "recall_at_p50_pass": recall_p50_pass,
        "fold_aucs": fold_aucs,
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
    }
    return p_fail, metrics


def recall_at_precision(y_true, scores, target_precision):
    """Highest recall achievable while keeping precision >= target.

    Returns 0.0 if no operating point reaches the target.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    feasible = precision[:-1] >= target_precision
    if not feasible.any():
        return 0.0
    return float(recall[:-1][feasible].max())


# ---------------------------------------------------------------------------
# Per-checkpoint and pooled fitting
# ---------------------------------------------------------------------------


def fit_per_checkpoint_models(frame, transcripts, median_full, encoder):
    """Fit one model per checkpoint and return (per_ckpt_metrics, P_oof)."""
    y = frame["fail"].to_numpy(dtype=int)
    P_oof = np.full((len(frame), len(CHECKPOINTS)), np.nan)
    rows = []

    for j, checkpoint in enumerate(CHECKPOINTS):
        log.info("Fitting per-checkpoint model: %s", checkpoint)
        X_pos = build_position_matrix(
            frame, transcripts, checkpoint, median_full, encoder
        )
        X_emb = load_voyage_checkpoint(checkpoint, frame)
        X = np.hstack([X_emb, X_pos]).astype(np.float32, copy=False)

        p_fail, metrics = predict_oof(X, y)
        P_oof[:, j] = p_fail

        log.info(
            "  AUC=%.3f  AP_pass=%.3f  Macro-F1=%.3f  "
            "Recall@P50_pass=%.3f  Recall@P95_pass=%.3f",
            metrics["auc"], metrics["pr_auc_pass"], metrics["macro_f1"],
            metrics["recall_at_p50_pass"], metrics["recall_at_p95_pass"],
        )
        log.info("  per-fold AUC: %s",
                 ", ".join(f"{a:.3f}" for a in metrics["fold_aucs"]))

        # Ablation: same checkpoint, embedding only (no position features).
        _, abl_metrics = predict_oof(
            X_emb.astype(np.float32, copy=False), y
        )
        rows.append({
            "checkpoint": str(checkpoint),
            "n_pos": metrics["n_pos"],
            "n_neg": metrics["n_neg"],
            "auc": metrics["auc"],
            "pr_auc_pass": metrics["pr_auc_pass"],
            "macro_f1": metrics["macro_f1"],
            "recall_at_p50_pass": metrics["recall_at_p50_pass"],
            "recall_at_p95_pass": metrics["recall_at_p95_pass"],
            "auc_no_position_features": abl_metrics["auc"],
            "auc_delta_position": metrics["auc"] - abl_metrics["auc"],
            "fold_aucs": ";".join(f"{a:.4f}" for a in metrics["fold_aucs"]),
        })

    per_ckpt_metrics = pd.DataFrame(rows)
    return per_ckpt_metrics, P_oof


def fit_pooled_model(frame, transcripts, median_full, encoder):
    """Stack six per-checkpoint matrices and fit a single grouped model."""
    log.info("Fitting pooled model with checkpoint as covariate …")
    y_blocks = []
    X_blocks = []
    g_blocks = []
    for j, checkpoint in enumerate(CHECKPOINTS):
        X_pos = build_position_matrix(
            frame, transcripts, checkpoint, median_full, encoder
        )
        X_emb = load_voyage_checkpoint(checkpoint, frame)
        block = np.hstack([X_emb, X_pos])
        # Append one-hot checkpoint indicator (drop first to avoid collinearity).
        dummies = np.zeros((len(frame), len(CHECKPOINTS) - 1),
                           dtype=np.float32)
        if j > 0:
            dummies[:, j - 1] = 1.0
        block = np.hstack([block, dummies]).astype(np.float32, copy=False)
        X_blocks.append(block)
        y_blocks.append(frame["fail"].to_numpy(dtype=int))
        g_blocks.append(frame["job_application_id"].to_numpy())

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    groups = np.concatenate(g_blocks)

    _, metrics = predict_oof(X, y, groups=groups)
    log.info(
        "  pooled  AUC=%.3f  AP_pass=%.3f  Macro-F1=%.3f  "
        "Recall@P50_pass=%.3f  Recall@P95_pass=%.3f",
        metrics["auc"], metrics["pr_auc_pass"], metrics["macro_f1"],
        metrics["recall_at_p50_pass"], metrics["recall_at_p95_pass"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Phase 2 — asymmetric stopping rule
# ---------------------------------------------------------------------------


def simulate_stopping(P_oof, y, c3_full_tokens, tau_fail, tau_pass):
    """Apply the asymmetric stopping rule across checkpoints, vectorised.

    Returns a dict of summary metrics for one (tau_fail, tau_pass) pair.
    """
    n = len(y)
    decided = np.zeros(n, dtype=bool)
    pred = np.full(n, -1, dtype=int)
    tokens_used = np.zeros(n, dtype=np.int64)

    for j, checkpoint in enumerate(PREFIX_CHECKPOINTS):
        if decided.all():
            break
        idx = np.where(~decided)[0]
        p = P_oof[idx, j]
        commit_fail = p >= 0.5 + tau_fail
        commit_pass = p <= 0.5 - tau_pass
        fail_idx = idx[commit_fail]
        pass_idx = idx[commit_pass & ~commit_fail]

        pred[fail_idx] = 1
        pred[pass_idx] = 0
        tokens_used[fail_idx] = checkpoint
        tokens_used[pass_idx] = checkpoint
        decided[fail_idx] = True
        decided[pass_idx] = True

    # Fallback: full-transcript prediction for anyone still undecided.
    fallback_idx = np.where(~decided)[0]
    if len(fallback_idx):
        p_full = P_oof[fallback_idx, len(CHECKPOINTS) - 1]
        pred[fallback_idx] = (p_full >= 0.5).astype(int)
        tokens_used[fallback_idx] = c3_full_tokens[fallback_idx]

    n_committed_fail = int((decided & (pred == 1)).sum())
    n_committed_pass = int((decided & (pred == 0)).sum())

    correct = (pred == y)
    accuracy = float(correct.mean())
    macro_f1 = float(f1_score(y, pred, average="macro"))

    pred_fail_mask = (pred == 1)
    pred_pass_mask = (pred == 0)
    precision_fail = (
        float(((pred == 1) & (y == 1)).sum() / max(pred_fail_mask.sum(), 1))
    )
    precision_pass = (
        float(((pred == 0) & (y == 0)).sum() / max(pred_pass_mask.sum(), 1))
    )

    return {
        "tau_fail": float(tau_fail),
        "tau_pass": float(tau_pass),
        "coverage_early": float(decided.mean()),
        "n_committed_fail": n_committed_fail,
        "n_committed_pass": n_committed_pass,
        "n_fallback": int(len(fallback_idx)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision_fail": precision_fail,
        "precision_pass": precision_pass,
        "mean_tokens": float(tokens_used.mean()),
        "median_tokens": float(np.median(tokens_used)),
    }


def threshold_sweep(P_oof, y, c3_full_tokens):
    """Sweep the (tau_fail, tau_pass) grid and return one row per pair."""
    rows = []
    for tau_fail in TAU_GRID:
        for tau_pass in TAU_GRID:
            rows.append(simulate_stopping(
                P_oof, y, c3_full_tokens, tau_fail, tau_pass
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(direction="out", length=4, color="#333333")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_auc_curve(per_ckpt, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=200)
    x = np.arange(len(per_ckpt))
    labels = per_ckpt["checkpoint"].tolist()

    ax.plot(
        x, per_ckpt["auc"], "-o", color=ACCENT_COLOR, markersize=6,
        linewidth=2, markerfacecolor=ACCENT_COLOR,
        markeredgecolor="white", markeredgewidth=1.0, label="ROC-AUC",
    )
    ax.plot(
        x, per_ckpt["pr_auc_pass"], "--s", color=ACCENT_ALT, markersize=5,
        linewidth=1.6, markerfacecolor=ACCENT_ALT,
        markeredgecolor="white", markeredgewidth=0.8,
        label="Average Precision (pass class)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(per_ckpt) - 0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Transcript prefix (candidate-only cl100k_base tokens)")
    ax.set_ylabel("Score")
    ax.set_title(
        "Stopping-Agent Probe: AUC vs Transcript Prefix",
        loc="left", pad=12, fontsize=12, fontweight="bold",
    )
    ax.text(
        0.0, 1.01,
        f"Voyage C3 + position features  |  N = {int(per_ckpt['n_pos'].iloc[0] + per_ckpt['n_neg'].iloc[0]):,}  "
        f"|  5-fold CV out-of-fold predictions",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
        ha="left", va="bottom",
    )
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


def plot_savings_curve(sweep_df, full_baseline, out_path):
    """Trade-off plot anchored at the always-full baseline.

    `full_baseline` is a dict {mean_tokens, accuracy, precision_fail}
    representing "never stop early" (everyone falls back to the full
    transcript), used as a reference line.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=200)

    # Asymmetric "fail-only stopping": never commit pass (tau_pass = max),
    # vary tau_fail.  This is the practically interesting curve given the
    # 4.5% pass base rate.
    tau_pass_max = sweep_df["tau_pass"].max()
    fail_only = sweep_df[sweep_df["tau_pass"] == tau_pass_max]\
        .sort_values("tau_fail")

    diag = sweep_df[sweep_df["tau_fail"] == sweep_df["tau_pass"]]\
        .sort_values("tau_fail")

    ax.plot(
        fail_only["mean_tokens"], fail_only["accuracy"], "-o",
        color=ACCENT_COLOR, markersize=6, linewidth=2,
        markerfacecolor=ACCENT_COLOR,
        markeredgecolor="white", markeredgewidth=1.0,
        label="Fail-only stopping (vary τ_fail; never commit pass)",
    )
    ax.plot(
        diag["mean_tokens"], diag["accuracy"], "--s",
        color=ACCENT_ALT, markersize=5, linewidth=1.6,
        markerfacecolor=ACCENT_ALT, markeredgecolor="white",
        markeredgewidth=0.8,
        label="Symmetric stopping (τ_fail = τ_pass)",
    )

    # Always-full reference
    ax.axhline(
        full_baseline["accuracy"], linestyle=":",
        color=REFERENCE_COLOR, linewidth=1.2,
        label=(
            f"Always-full baseline "
            f"(acc={full_baseline['accuracy']:.3f}, "
            f"~{full_baseline['mean_tokens']:.0f} tokens)"
        ),
    )
    ax.axvline(
        full_baseline["mean_tokens"], linestyle=":",
        color=REFERENCE_COLOR, linewidth=1.2,
    )

    # Annotate τ_fail values at the endpoints of the asymmetric curve.
    if len(fail_only):
        first = fail_only.iloc[0]
        last = fail_only.iloc[-1]
        ax.annotate(
            f"τ_fail={first['tau_fail']:.2f}\n"
            f"acc={first['accuracy']:.3f}, {first['mean_tokens']:.0f} tok",
            xy=(first["mean_tokens"], first["accuracy"]),
            xytext=(10, -28), textcoords="offset points",
            fontsize=7.5, color="#222",
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.6),
        )
        ax.annotate(
            f"τ_fail={last['tau_fail']:.2f}\n"
            f"acc={last['accuracy']:.3f}, {last['mean_tokens']:.0f} tok",
            xy=(last["mean_tokens"], last["accuracy"]),
            xytext=(10, 12), textcoords="offset points",
            fontsize=7.5, color="#222",
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.6),
        )

    ax.set_xlabel("Mean tokens consumed per candidate")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.83, 0.95)
    ax.set_xlim(200, full_baseline["mean_tokens"] * 1.05)
    ax.set_title(
        "Stopping Trade-off: Savings vs. Accuracy",
        loc="left", pad=12, fontsize=12, fontweight="bold",
    )
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


def plot_threshold_heatmap(sweep_df, out_path):
    taus = sorted(sweep_df["tau_fail"].unique())
    n = len(taus)
    grid_tokens = np.full((n, n), np.nan)
    grid_acc = np.full((n, n), np.nan)
    for i, tf in enumerate(taus):
        for j, tp in enumerate(taus):
            row = sweep_df[(sweep_df["tau_fail"] == tf)
                           & (sweep_df["tau_pass"] == tp)]
            if len(row):
                grid_tokens[i, j] = row["mean_tokens"].iloc[0]
                grid_acc[i, j] = row["accuracy"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=200)
    for ax, grid, title, cmap, fmt in [
        (axes[0], grid_tokens, "Mean tokens consumed",
         "Blues", "{:.0f}"),
        (axes[1], grid_acc, "Accuracy",
         "Greens", "{:.2f}"),
    ]:
        im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{t:.2f}" for t in taus], fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{t:.2f}" for t in taus], fontsize=8)
        ax.set_xlabel(r"$\tau_{\mathrm{pass}}$ (commit-pass margin)")
        ax.set_ylabel(r"$\tau_{\mathrm{fail}}$ (commit-fail margin)")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold",
                     pad=8)
        for i in range(n):
            for j in range(n):
                if np.isnan(grid[i, j]):
                    continue
                ax.text(
                    j, i, fmt.format(grid[i, j]),
                    ha="center", va="center", fontsize=6.5,
                    color="white" if grid[i, j] > np.nanmedian(grid) else "#222",
                )
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle(
        "Stopping-Agent Threshold Sweep",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def run_verification(per_ckpt, sweep_df):
    """Log the four sanity checks from the plan."""
    log.info("=" * 70)
    log.info("VERIFICATION")
    log.info("=" * 70)

    voyage_csv = RESULTS_DIR / "hazard_decile_summary__voyage.csv"
    if voyage_csv.exists():
        prior = pd.read_csv(voyage_csv)
        if "oof_auc" in prior.columns:
            prior_auc = float(prior["oof_auc"].iloc[0])
            full_row = per_ckpt[per_ckpt["checkpoint"] == "full"].iloc[0]
            delta = full_row["auc"] - prior_auc
            log.info(
                "  [1] full AUC=%.3f  (Issue#1 voyage AUC=%.3f, delta=%+.3f, "
                "tolerance=±0.020)",
                full_row["auc"], prior_auc, delta,
            )
            if abs(delta) > 0.02:
                log.warning("      Δ exceeds ±0.020 — investigate.")
        else:
            log.info("  [1] Issue#1 voyage CSV present but lacks oof_auc column.")
    else:
        log.info("  [1] Issue#1 voyage CSV not found; skipping cross-check.")

    fold_matrix = np.array([
        [float(x) for x in s.split(";")]
        for s in per_ckpt["fold_aucs"]
    ])
    n_folds = fold_matrix.shape[1]
    # Allow a small tolerance per pair (-0.01) — AUC saturates between
    # 4096 and full, where fold-level noise can produce sub-0.01 dips.
    tolerance = 0.01
    monotone_count = 0
    for k in range(n_folds):
        diffs = np.diff(fold_matrix[:, k])
        if (diffs >= -tolerance).all():
            monotone_count += 1
    log.info(
        "  [2] AUC monotonic-within-%.2f in budget for %d/%d folds "
        "(target ≥4).",
        tolerance, monotone_count, n_folds,
    )

    short_row = per_ckpt[per_ckpt["checkpoint"] == "256"]
    if len(short_row):
        delta_pos = float(short_row["auc_delta_position"].iloc[0])
        log.info(
            "  [3] Position-feature ablation at 256 tokens: ΔAUC=%+.3f "
            "(target (0, 0.03]).",
            delta_pos,
        )

    zero_row = sweep_df[(sweep_df["tau_fail"] == 0.0)
                        & (sweep_df["tau_pass"] == 0.0)]
    if len(zero_row):
        z = zero_row.iloc[0]
        log.info(
            "  [4] τ=0 sweep row: mean_tokens=%.0f (target 256), "
            "coverage=%.3f (target 1.000)",
            z["mean_tokens"], z["coverage_early"],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    encoder = tiktoken.get_encoding("cl100k_base")
    frame, transcripts, median_full = load_aligned_inputs()

    per_ckpt, P_oof = fit_per_checkpoint_models(
        frame, transcripts, median_full, encoder
    )

    pooled_metrics = fit_pooled_model(
        frame, transcripts, median_full, encoder
    )
    pooled_row = pd.DataFrame([{
        "checkpoint": "pooled",
        "n_pos": pooled_metrics["n_pos"],
        "n_neg": pooled_metrics["n_neg"],
        "auc": pooled_metrics["auc"],
        "pr_auc_pass": pooled_metrics["pr_auc_pass"],
        "macro_f1": pooled_metrics["macro_f1"],
        "recall_at_p50_pass": pooled_metrics["recall_at_p50_pass"],
        "recall_at_p95_pass": pooled_metrics["recall_at_p95_pass"],
        "auc_no_position_features": np.nan,
        "auc_delta_position": np.nan,
        "fold_aucs": ";".join(
            f"{a:.4f}" for a in pooled_metrics["fold_aucs"]
        ),
    }])
    per_ckpt_full = pd.concat([per_ckpt, pooled_row], ignore_index=True)

    per_ckpt_path = RESULTS_DIR / "stopping_agent_per_checkpoint.csv"
    per_ckpt_full.to_csv(per_ckpt_path, index=False)
    log.info("Wrote %s", per_ckpt_path)

    oof_long = []
    for j, checkpoint in enumerate(CHECKPOINTS):
        for i in range(len(frame)):
            oof_long.append({
                "job_application_id": frame["job_application_id"].iloc[i],
                "checkpoint": str(checkpoint),
                "p_fail": float(P_oof[i, j]),
                "is_passed": bool(frame["is_passed"].iloc[i]),
                "fail": int(frame["fail"].iloc[i]),
                "c3_full_tokens": int(frame["c3_full_tokens"].iloc[i]),
            })
    oof_path = RESULTS_DIR / "stopping_agent_oof_predictions.csv"
    pd.DataFrame(oof_long).to_csv(oof_path, index=False)
    log.info("Wrote %s", oof_path)

    log.info("Sweeping (tau_fail, tau_pass) grid (%d × %d) …",
             len(TAU_GRID), len(TAU_GRID))
    c3_full_tokens = frame["c3_full_tokens"].to_numpy()
    y = frame["fail"].to_numpy(dtype=int)
    sweep_df = threshold_sweep(P_oof, y, c3_full_tokens)
    sweep_path = RESULTS_DIR / "stopping_agent_threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    log.info("Wrote %s", sweep_path)

    # Always-full baseline: predict from the full-checkpoint OOF probability,
    # consume each candidate's full transcript.
    p_full = P_oof[:, len(CHECKPOINTS) - 1]
    pred_full = (p_full >= 0.5).astype(int)
    full_baseline = {
        "mean_tokens": float(c3_full_tokens.mean()),
        "accuracy": float((pred_full == y).mean()),
        "precision_fail": float(
            ((pred_full == 1) & (y == 1)).sum()
            / max((pred_full == 1).sum(), 1)
        ),
    }
    log.info(
        "  always-full baseline: mean_tokens=%.0f  accuracy=%.3f  "
        "precision_fail=%.3f",
        full_baseline["mean_tokens"], full_baseline["accuracy"],
        full_baseline["precision_fail"],
    )

    plot_auc_curve(per_ckpt, FIGURES_DIR / "stopping_agent_auc_curve.png")
    plot_savings_curve(
        sweep_df, full_baseline,
        FIGURES_DIR / "stopping_agent_savings.png",
    )
    plot_threshold_heatmap(
        sweep_df, FIGURES_DIR / "stopping_agent_threshold_heatmap.png"
    )

    run_verification(per_ckpt, sweep_df)


if __name__ == "__main__":
    main()
