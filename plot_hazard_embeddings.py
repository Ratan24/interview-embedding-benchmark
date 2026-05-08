"""Plot interview length vs. predicted failure risk decile, using real
embedding vectors (rather than the TF-IDF baseline in plot_hazard.py).

Loads pre-computed embeddings produced by the benchmark pipeline,
trains a 5-fold cross-validated logistic regression to predict
``is_passed``, and bins candidates by out-of-fold predicted failure
probability into deciles.  Mean interview duration per decile is
plotted alongside companion length metrics.

Defaults to the binary-pass/fail champion (``openai-large`` + ``C2a``)
and also runs the 4-class champion (``voyage`` + ``C2a``) for
robustness.

Inputs:
    data/transcripts_6400_records.csv             # for duration/length
    data/Ai-Vetted-ranked.csv                      # for is_passed
    <VECTORS_DIR>/<model>_<condition>.npy
    <VECTORS_DIR>/<model>_<condition>_ids.json

Outputs (per model):
    figures/hazard_of_failing__<model>.png
    figures/hazard_of_failing_panels__<model>.png
    figures/hazard_of_failing_validation__<model>.png
    results/hazard_decile_summary__<model>.csv

Plus a comparison plot stitched against the TF-IDF baseline:
    figures/hazard_of_failing_comparison.png
"""

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_CSV = BASE_DIR / "data" / "transcripts_6400_records.csv"
VETTING_CSV = BASE_DIR / "data" / "Ai-Vetted-ranked.csv"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

VECTORS_DIR = Path(
    "/Users/ratanpyla/Desktop/Micro1 - Emil Palikot/AI_recruiter_internal"
    "/experiments/embeddings/benchmark_2026/vectors"
)

# Models to evaluate.  Each entry is (model_key, condition).
# openai-large + C2a is the binary-pass/fail champion (README: F1=0.830).
# voyage + C2a is the 4-class champion, included for robustness.
MODELS = [
    ("openai-large", "C2a"),
    ("voyage", "C2a"),
]

RANDOM_SEED = 42
N_DECILES = 10
N_FOLDS = 5

MIN_DURATION_MIN = 5
MAX_DURATION_MIN = 120

LOGREG_C = 1.0

ACCENT_COLOR = "#1f3a5f"
ACCENT_ALT = "#a23b3b"
ACCENT_TFIDF = "#888888"
BAND_COLOR = "#1f3a5f"
REFERENCE_COLOR = "#888888"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transcript helpers (length features only — text not needed here)
# ---------------------------------------------------------------------------


def clean_json(raw):
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw)


def parse_transcript(raw_json):
    if not isinstance(raw_json, str) or not raw_json.strip():
        return None
    try:
        return json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        try:
            return json.loads(clean_json(raw_json))
        except (json.JSONDecodeError, TypeError):
            return None


def candidate_word_count(messages):
    return sum(
        len((m.get("content") or "").strip().split())
        for m in messages
        if m.get("role") == "user"
    )


def message_count(messages):
    return sum(1 for m in messages if (m.get("content") or "").strip())


def transcript_token_count(messages, encoder):
    parts = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = m.get("role", "")
        prefix = (
            "Interviewer: " if role == "interviewer"
            else "Candidate: " if role == "user" else ""
        )
        parts.append(prefix + content)
    text = "\n\n".join(parts)
    return len(encoder.encode(text)) if text else 0


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_length_frame():
    """Build a candidate-level frame with duration and length features."""
    log.info("Loading vetting CSV …")
    vdf = pd.read_csv(VETTING_CSV)
    vdf = vdf[
        vdf["job_application_id"].notna()
        & vdf["is_passed"].notna()
        & vdf["vetting_creation_date"].notna()
        & vdf["vetting_completed_date"].notna()
    ].copy()
    vdf["start"] = pd.to_datetime(vdf["vetting_creation_date"], errors="coerce")
    vdf["end"] = pd.to_datetime(vdf["vetting_completed_date"], errors="coerce")
    vdf["duration_min"] = (vdf["end"] - vdf["start"]).dt.total_seconds() / 60.0
    vdf = vdf.dropna(subset=["duration_min"])
    vdf = vdf[["job_application_id", "is_passed", "duration_min"]]

    log.info("Loading transcripts CSV …")
    tdf = pd.read_csv(
        TRANSCRIPTS_CSV, encoding="latin1", on_bad_lines="skip"
    )
    tdf = tdf[["job_application_id", "interview_transcript"]]

    log.info("Joining on job_application_id …")
    df = tdf.merge(vdf, on="job_application_id", how="inner")
    log.info("  Joined rows: %d", len(df))

    encoder = tiktoken.get_encoding("cl100k_base")
    n_tokens, n_words, n_messages, keep = [], [], [], []
    for raw in df["interview_transcript"]:
        msgs = parse_transcript(raw)
        if msgs is None or not isinstance(msgs, list) or not msgs:
            keep.append(False)
            n_tokens.append(0); n_words.append(0); n_messages.append(0)
            continue
        keep.append(True)
        n_tokens.append(transcript_token_count(msgs, encoder))
        n_words.append(candidate_word_count(msgs))
        n_messages.append(message_count(msgs))
    df["n_tokens"] = n_tokens
    df["candidate_words"] = n_words
    df["n_messages"] = n_messages
    df = df[pd.Series(keep, index=df.index)].drop(columns=["interview_transcript"])
    log.info("  After dropping unparseable transcripts: %d", len(df))

    before = len(df)
    df = df[
        (df["duration_min"] >= MIN_DURATION_MIN)
        & (df["duration_min"] <= MAX_DURATION_MIN)
    ].reset_index(drop=True)
    log.info(
        "  After clipping duration to [%d, %d] min: %d (dropped %d)",
        MIN_DURATION_MIN, MAX_DURATION_MIN, len(df), before - len(df),
    )

    df["fail"] = (~df["is_passed"].astype(bool)).astype(int)
    df["duration_sec"] = df["duration_min"] * 60.0

    log.info(
        "  Pass rate: %.1f%% (n_pass=%d, n_fail=%d)",
        100 * (1 - df["fail"].mean()),
        int((df["fail"] == 0).sum()),
        int((df["fail"] == 1).sum()),
    )
    return df


def load_embedding_features(model, condition):
    """Return (X, ids) for the given (model, condition) embedding."""
    npy_path = VECTORS_DIR / f"{model}_{condition}.npy"
    ids_path = VECTORS_DIR / f"{model}_{condition}_ids.json"
    log.info("Loading %s …", npy_path.name)
    X = np.load(npy_path)
    ids = json.load(open(ids_path))
    log.info("  shape=%s  dtype=%s  n_ids=%d", X.shape, X.dtype, len(ids))

    # Per-skill conditions are 3D: (N, 3 skills, dim).  Flatten to (N, 3*dim).
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
        log.info("  flattened per-skill -> %s", X.shape)
    elif X.ndim != 2:
        raise ValueError(f"Unexpected embedding rank: {X.shape}")
    return X.astype(np.float32, copy=False), ids


def align(df, X, ids):
    """Restrict df and X to candidates present in both, in shared order."""
    id_to_idx = {jid: i for i, jid in enumerate(ids)}
    mask = df["job_application_id"].isin(id_to_idx)
    df = df[mask].reset_index(drop=True).copy()
    rows = df["job_application_id"].map(id_to_idx).to_numpy()
    X = X[rows]
    log.info("  Aligned candidates: %d", len(df))
    return df, X


# ---------------------------------------------------------------------------
# Out-of-fold predicted failure probability
# ---------------------------------------------------------------------------


def predict_failure_oof(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    p_fail = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
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
        log.info("  fold %d done", fold)
    auc = roc_auc_score(y, p_fail)
    log.info("  OOF ROC-AUC: %.3f", auc)
    return p_fail, auc


def assign_deciles(p_fail):
    ranks = pd.Series(p_fail).rank(method="first")
    return (pd.qcut(ranks, N_DECILES, labels=False) + 1).to_numpy()


def mean_with_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean, mean
    sem = float(stats.sem(values))
    t = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return mean, mean - t * sem, mean + t * sem


def summarise_by_decile(df):
    metric_cols = ["duration_sec", "duration_min", "n_tokens",
                   "candidate_words", "n_messages"]
    rows = []
    for decile, group in df.groupby("decile"):
        rec = {"decile": int(decile), "n": len(group),
               "pass_rate": float((group["fail"] == 0).mean())}
        for col in metric_cols:
            m, lo, hi = mean_with_ci(group[col].values)
            rec[f"{col}_mean"] = m
            rec[f"{col}_ci_lo"] = lo
            rec[f"{col}_ci_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)


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


def plot_primary(summary, df, auc, model_label, out_path):
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=200)
    x = summary["decile"].values
    y = summary["duration_sec_mean"].values
    lo = summary["duration_sec_ci_lo"].values
    hi = summary["duration_sec_ci_hi"].values

    ax.fill_between(x, lo, hi, color=BAND_COLOR, alpha=0.18,
                    linewidth=0, label="95% CI")
    ax.plot(x, y, "-o", color=ACCENT_COLOR, markersize=6,
            linewidth=2, markerfacecolor=ACCENT_COLOR,
            markeredgecolor="white", markeredgewidth=1.0,
            label="Mean per decile")

    median_sec = float(df["duration_sec"].median())
    ax.axhline(median_sec, linestyle="--", color=REFERENCE_COLOR,
               linewidth=1.0, label=f"Overall median ({median_sec/60:.0f} min)")

    ymax = float(np.nanmax(hi)) * 1.10
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(1, N_DECILES + 1))
    ax.set_xlim(0.5, N_DECILES + 0.5)
    ax.set_xlabel("Predicted Failure Risk Decile")
    ax.set_ylabel("Average Call Duration (s)")
    ax.set_title(
        "Interview Length vs. Predicted Failure Risk",
        loc="left", pad=12, fontsize=12, fontweight="bold",
    )
    ax.text(
        0.0, 1.01,
        f"{model_label}   |   N = {len(df):,}   |   "
        f"out-of-fold AUC = {auc:.3f}",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
        ha="left", va="bottom",
    )
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


def plot_panels(summary, df, model_label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=200)
    panels = [
        ("duration_sec", "Average call duration (s)"),
        ("candidate_words", "Average candidate words"),
        ("n_tokens", "Average transcript tokens"),
        ("n_messages", "Average messages"),
    ]
    x = summary["decile"].values
    for ax, (col, ylabel) in zip(axes.ravel(), panels):
        y = summary[f"{col}_mean"].values
        lo = summary[f"{col}_ci_lo"].values
        hi = summary[f"{col}_ci_hi"].values
        ax.fill_between(x, lo, hi, color=BAND_COLOR, alpha=0.18, linewidth=0)
        ax.plot(x, y, "-o", color=ACCENT_COLOR, markersize=5,
                linewidth=1.8, markerfacecolor=ACCENT_COLOR,
                markeredgecolor="white", markeredgewidth=0.8)
        median = float(df[col].median())
        ax.axhline(median, linestyle="--", color=REFERENCE_COLOR, linewidth=0.9,
                   label=f"overall median = {median:.0f}")
        ax.set_xticks(range(1, N_DECILES + 1))
        ax.set_xlim(0.5, N_DECILES + 0.5)
        ax.set_xlabel("Predicted Failure Risk Decile")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend(frameon=False, loc="best", fontsize=8)

    fig.suptitle(
        f"Interview Length Metrics by Predicted Failure Risk Decile  ·  {model_label}",
        fontsize=12.5, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


def plot_validation(summary, model_label, out_path):
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=200)
    x = summary["decile"].values
    y = summary["pass_rate"].values * 100
    ax.bar(x, y, color=ACCENT_COLOR, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(1, N_DECILES + 1))
    ax.set_xlabel("Predicted Failure Risk Decile")
    ax.set_ylabel("Actual pass rate (%)")
    ax.set_title(
        f"Classifier Ranking Validation · {model_label}",
        loc="left", pad=10, fontsize=11.5, fontweight="bold",
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


def plot_comparison(summaries, df, out_path):
    """Overlay duration-vs-decile curves from multiple risk models."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    palette = [ACCENT_COLOR, ACCENT_ALT, ACCENT_TFIDF]
    markers = ["o", "s", "^"]
    for (label, summary, auc), color, mk in zip(summaries, palette, markers):
        x = summary["decile"].values
        y = summary["duration_sec_mean"].values
        lo = summary["duration_sec_ci_lo"].values
        hi = summary["duration_sec_ci_hi"].values
        ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.plot(x, y, "-", color=color, linewidth=1.8, marker=mk,
                markersize=5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"{label}  (AUC={auc:.3f})")

    median_sec = float(df["duration_sec"].median())
    ax.axhline(median_sec, linestyle="--", color=REFERENCE_COLOR, linewidth=1.0,
               label=f"Overall median ({median_sec/60:.0f} min)")

    ymax = max(
        float(np.nanmax(s["duration_sec_ci_hi"])) for _, s, _ in summaries
    ) * 1.10
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(1, N_DECILES + 1))
    ax.set_xlim(0.5, N_DECILES + 0.5)
    ax.set_xlabel("Predicted Failure Risk Decile")
    ax.set_ylabel("Average Call Duration (s)")
    ax.set_title(
        "Interview Length vs. Predicted Failure Risk — model comparison",
        loc="left", pad=12, fontsize=12, fontweight="bold",
    )
    ax.text(
        0.0, 1.01,
        f"AI-paced technical interviews   |   N = {len(df):,}   |   "
        f"5-fold CV out-of-fold predictions",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
        ha="left", va="bottom",
    )
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_for_model(df_full, model, condition):
    """Train, summarise, and plot for one (model, condition).

    Returns (label, summary_df, oof_auc).
    """
    label = f"{model} + {condition}"
    log.info("=" * 70)
    log.info("Model: %s", label)
    log.info("=" * 70)

    X, ids = load_embedding_features(model, condition)
    df, X = align(df_full.copy(), X, ids)

    log.info(
        "Fitting LogReg on standardised %d-d embeddings  (n=%d) …",
        X.shape[1], X.shape[0],
    )
    p_fail, auc = predict_failure_oof(X, df["fail"].to_numpy())
    df["p_fail"] = p_fail
    df["decile"] = assign_deciles(p_fail)

    summary = summarise_by_decile(df)

    summary_path = RESULTS_DIR / f"hazard_decile_summary__{model}.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Wrote %s", summary_path)

    log.info("Decile summary (mean call duration in seconds):")
    for _, row in summary.iterrows():
        log.info(
            "  decile %2d  n=%4d  pass=%5.1f%%  "
            "duration=%6.1f s [%6.1f, %6.1f]",
            row["decile"], row["n"], row["pass_rate"] * 100,
            row["duration_sec_mean"],
            row["duration_sec_ci_lo"], row["duration_sec_ci_hi"],
        )

    plot_primary(
        summary, df, auc, label,
        FIGURES_DIR / f"hazard_of_failing__{model}.png",
    )
    plot_panels(
        summary, df, label,
        FIGURES_DIR / f"hazard_of_failing_panels__{model}.png",
    )
    plot_validation(
        summary, label,
        FIGURES_DIR / f"hazard_of_failing_validation__{model}.png",
    )

    return label, summary, auc


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df_full = load_length_frame()

    summaries = []
    for model, condition in MODELS:
        label, summary, auc = run_for_model(df_full, model, condition)
        summaries.append((label, summary, auc))

    # Comparison vs TF-IDF baseline (if present)
    tfidf_path = RESULTS_DIR / "hazard_decile_summary.csv"
    if tfidf_path.exists():
        log.info("Loading TF-IDF baseline from %s", tfidf_path)
        tfidf_summary = pd.read_csv(tfidf_path)
        # AUC is logged but not stored in CSV — recompute or hard-set.
        # We hard-set here; the TF-IDF run logs its OOF AUC.
        tfidf_auc = 0.877
        summaries.append(("tfidf + C4 text  (baseline)", tfidf_summary, tfidf_auc))

    plot_comparison(
        summaries, df_full,
        FIGURES_DIR / "hazard_of_failing_comparison.png",
    )


if __name__ == "__main__":
    main()
