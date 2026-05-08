"""Plot interview length vs. predicted failure risk decile.

Trains a TF-IDF + logistic regression classifier on full Q&A
transcripts to predict the binary pass/fail outcome, then bins
candidates by out-of-fold predicted failure probability into
deciles and reports the mean interview length per decile.

Mirrors the call-duration-by-risk-decile diagnostic used in the
stopping-agents literature.  In a human-paced interview the
curve slopes downward (interviewers cut struggling candidates
short); in our AI-paced protocol the curve is expected to be
flat.

Inputs:
    data/transcripts_6400_records.csv
    data/Ai-Vetted-ranked.csv

Outputs:
    figures/hazard_of_failing.png             primary plot (matches reference)
    figures/hazard_of_failing_panels.png      4-panel: duration / words / tokens / messages
    figures/hazard_of_failing_validation.png  pass rate by decile (sanity check)
    results/hazard_decile_summary.csv         per-decile means and CIs
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_CSV = BASE_DIR / "data" / "transcripts_6400_records.csv"
VETTING_CSV = BASE_DIR / "data" / "Ai-Vetted-ranked.csv"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_SEED = 42
N_DECILES = 10
N_FOLDS = 5

# Duration sanity range (minutes) — outside this we treat the row as
# a paused/resumed session rather than a real interview length.
MIN_DURATION_MIN = 5
MAX_DURATION_MIN = 120

# TF-IDF / classifier
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
LOGREG_C = 1.0

# Plot styling
ACCENT_COLOR = "#1f3a5f"
BAND_COLOR = "#1f3a5f"
REFERENCE_COLOR = "#888888"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------


def clean_json(raw):
    """Strip control characters that break json.loads."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw)


def parse_transcript(raw_json):
    """Parse one transcript JSON string into a list of message dicts."""
    if not isinstance(raw_json, str) or not raw_json.strip():
        return None
    try:
        return json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        try:
            return json.loads(clean_json(raw_json))
        except (json.JSONDecodeError, TypeError):
            return None


def format_full_qa(messages):
    """Concatenate all messages as a single Q&A string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "interviewer":
            parts.append(f"Interviewer: {content}")
        elif role == "user":
            parts.append(f"Candidate: {content}")
    return "\n\n".join(parts)


def candidate_word_count(messages):
    """Count whitespace-separated words across candidate turns only."""
    words = 0
    for msg in messages:
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                words += len(content.split())
    return words


def message_count(messages):
    """Count non-empty messages."""
    return sum(
        1 for m in messages
        if (m.get("content") or "").strip()
    )


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_dataset():
    """Build a candidate-level frame with text, length, and label."""
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

    # Parse transcripts and compute length features
    encoder = tiktoken.get_encoding("cl100k_base")
    parsed_text = []
    n_tokens = []
    n_words = []
    n_messages = []
    keep = []

    for raw in df["interview_transcript"]:
        msgs = parse_transcript(raw)
        if msgs is None or not isinstance(msgs, list) or not msgs:
            keep.append(False)
            parsed_text.append("")
            n_tokens.append(0)
            n_words.append(0)
            n_messages.append(0)
            continue
        text = format_full_qa(msgs)
        keep.append(bool(text.strip()))
        parsed_text.append(text)
        n_tokens.append(len(encoder.encode(text)) if text else 0)
        n_words.append(candidate_word_count(msgs))
        n_messages.append(message_count(msgs))

    df["text"] = parsed_text
    df["n_tokens"] = n_tokens
    df["candidate_words"] = n_words
    df["n_messages"] = n_messages
    df = df[pd.Series(keep, index=df.index)].copy()

    log.info("  After dropping empty/unparseable transcripts: %d", len(df))

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


# ---------------------------------------------------------------------------
# Out-of-fold predicted failure probability
# ---------------------------------------------------------------------------


def predict_failure_oof(df):
    """Compute out-of-fold P(fail) via 5-fold stratified CV TF-IDF + LogReg."""
    log.info(
        "Fitting TF-IDF (max_features=%d, ngram=%s) + LogReg on %d candidates …",
        TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, len(df),
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    p_fail = np.full(len(df), np.nan)

    for fold, (tr, te) in enumerate(skf.split(df["text"], df["fail"]), start=1):
        vec = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
        )
        X_tr = vec.fit_transform(df["text"].iloc[tr])
        X_te = vec.transform(df["text"].iloc[te])

        clf = LogisticRegression(
            C=LOGREG_C,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        )
        clf.fit(X_tr, df["fail"].iloc[tr])
        p_fail[te] = clf.predict_proba(X_te)[:, 1]
        log.info("  fold %d done", fold)

    auc = roc_auc_score(df["fail"], p_fail)
    log.info("  Out-of-fold ROC-AUC: %.3f", auc)
    return p_fail, auc


# ---------------------------------------------------------------------------
# Decile aggregation
# ---------------------------------------------------------------------------


def assign_deciles(p_fail):
    """Rank-based decile assignment (1 = lowest risk, 10 = highest)."""
    ranks = pd.Series(p_fail).rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
    return deciles.to_numpy()


def mean_with_ci(values, alpha=0.05):
    """Sample mean and t-based 95% CI."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean, mean
    sem = float(stats.sem(values))
    t = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return mean, mean - t * sem, mean + t * sem


def summarise_by_decile(df):
    """Aggregate mean ± 95% CI for each length metric per decile."""
    metrics = {
        "duration_sec": "Average call duration (s)",
        "duration_min": "Average call duration (min)",
        "n_tokens": "Average tokens (cl100k_base)",
        "candidate_words": "Average candidate words",
        "n_messages": "Average messages",
    }
    rows = []
    for decile, group in df.groupby("decile"):
        rec = {"decile": int(decile), "n": len(group),
               "pass_rate": float((group["fail"] == 0).mean())}
        for col in metrics:
            m, lo, hi = mean_with_ci(group[col].values)
            rec[f"{col}_mean"] = m
            rec[f"{col}_ci_lo"] = lo
            rec[f"{col}_ci_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True), metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def style_axes(ax):
    """Clean publication-style axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(direction="out", length=4, color="#333333")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_primary(summary, df, auc, out_path):
    """Reference-style plot: duration_sec vs. risk decile."""
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

    # Reference line at overall median duration
    median_sec = float(df["duration_sec"].median())
    ax.axhline(median_sec, linestyle="--", color=REFERENCE_COLOR,
               linewidth=1.0, label=f"Overall median ({median_sec/60:.0f} min)")

    # Zero-anchored y-axis so flatness is visually honest.
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
        f"AI-paced technical interviews   |   N = {len(df):,}   |   "
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


def plot_panels(summary, df, out_path):
    """Four-panel grid showing all length metrics by decile."""
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
        "Interview Length Metrics by Predicted Failure Risk Decile",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


def plot_validation(summary, out_path):
    """Sanity check: pass rate per decile (confirms classifier ordering)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=200)
    x = summary["decile"].values
    y = summary["pass_rate"].values * 100

    ax.bar(x, y, color=ACCENT_COLOR, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(1, N_DECILES + 1))
    ax.set_xlabel("Predicted Failure Risk Decile")
    ax.set_ylabel("Actual pass rate (%)")
    ax.set_title(
        "Classifier Ranking Validation: Actual Pass Rate by Decile",
        loc="left", pad=10, fontsize=11.5, fontweight="bold",
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    p_fail, auc = predict_failure_oof(df)
    df["p_fail"] = p_fail
    df["decile"] = assign_deciles(p_fail)

    summary, _ = summarise_by_decile(df)

    summary_path = RESULTS_DIR / "hazard_decile_summary.csv"
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

    plot_primary(summary, df, auc, FIGURES_DIR / "hazard_of_failing.png")
    plot_panels(summary, df, FIGURES_DIR / "hazard_of_failing_panels.png")
    plot_validation(summary, FIGURES_DIR / "hazard_of_failing_validation.png")


if __name__ == "__main__":
    main()
