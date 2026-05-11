"""Pull long, high-risk, failed interview transcripts for qualitative review.

Follow-up to Issue #1: Emil's question on the merged PR was "those long
interviews are just interviews that never lead to the candidate passing —
could you pull a few from decile 9/10 and see what is happening?".

Filtering pipeline (in order):
1. Take per-candidate OOF P(fail) at the `full` checkpoint from
   `results/stopping_agent_oof_predictions.csv` (the Voyage C3 classifier
   from Issue #2, which has per-candidate predictions on disk).
2. Compute rank-based deciles over P(fail); keep deciles 9 and 10.
3. Keep only candidates with `is_passed = False` (belt-and-braces; these
   deciles are already 0-0.6% pass rate, but explicit filter avoids any
   rare-pass leakage).
4. Within that subset, keep the top quartile by `duration_min` — the
   "long" interviews relative to the high-risk-failed subset.
5. Sample 10 with random_state=42 for reproducibility.

Outputs (all under `report/`, all gitignored):
- `inspection_long_high_risk.csv` — one row per selected interview, columns
  for metadata, red-flag metrics, and the full Q&A transcript text.
- `inspection_long_high_risk_transcripts.md` — same interviews rendered as
  readable Q&A blocks for top-to-bottom reading.

Also writes a one-line-per-interview summary table to stdout.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd

from parse_transcripts import (
    format_qa,
    parse_single_transcript,
)
from plot_hazard_embeddings import load_length_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_CSV = BASE_DIR / "data" / "transcripts_6400_records.csv"
OOF_PREDICTIONS = BASE_DIR / "results" / "stopping_agent_oof_predictions.csv"
REPORT_DIR = BASE_DIR / "report"

OUT_CSV = REPORT_DIR / "inspection_long_high_risk.csv"
OUT_MD = REPORT_DIR / "inspection_long_high_risk_transcripts.md"

DECILES_TO_INSPECT = {9, 10}
N_SAMPLES = 10
LONG_QUANTILE = 0.75
RANDOM_SEED = 42

DONTKNOW_RE = re.compile(
    r"\b(i\s*don'?t\s*know|don'?t\s*know|not\s*sure|no\s*idea|"
    r"i\s*forgot|i\s*forget|skip\s*it|i\s*can'?t\s*remember|i\s*have\s*no\s*idea)\b",
    re.IGNORECASE,
)
FILLER_RE = re.compile(
    r"\b(uh+|um+|hmm+|ehh+|errr+|like\s+you\s+know)\b",
    re.IGNORECASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate-level frame
# ---------------------------------------------------------------------------


def short_hash(jid):
    return hashlib.sha1(str(jid).encode()).hexdigest()[:8]


def load_cohort():
    """Build the candidate-level frame with predicted P(fail) and decile."""
    log.info("Loading OOF predictions …")
    oof = pd.read_csv(OOF_PREDICTIONS)
    full = oof[oof["checkpoint"] == "full"].copy()
    full = full[["job_application_id", "p_fail",
                 "c3_full_tokens", "fail", "is_passed"]]

    log.info("Loading duration-filtered candidate frame …")
    length_frame = load_length_frame()
    length_frame = length_frame[[
        "job_application_id", "duration_min", "duration_sec",
        "n_tokens", "candidate_words", "n_messages",
    ]]

    df = full.merge(length_frame, on="job_application_id", how="inner")

    # Rank-based deciles over P(fail).
    ranks = df["p_fail"].rank(method="first")
    df["decile"] = (pd.qcut(ranks, 10, labels=False) + 1).astype(int)

    log.info(
        "  N=%d after join; pass_rate=%.1f%% overall",
        len(df), 100 * df["is_passed"].mean(),
    )
    return df


def select_candidates(df):
    """Apply the three filters and sample N_SAMPLES rows."""
    cohort = df[
        df["decile"].isin(DECILES_TO_INSPECT)
        & (~df["is_passed"].astype(bool))
    ].copy()
    log.info(
        "  After deciles ∈ %s and is_passed=False: %d candidates "
        "(decile 9: %d, decile 10: %d)",
        sorted(DECILES_TO_INSPECT), len(cohort),
        int((cohort["decile"] == 9).sum()),
        int((cohort["decile"] == 10).sum()),
    )

    threshold = cohort["duration_min"].quantile(LONG_QUANTILE)
    cohort = cohort[cohort["duration_min"] >= threshold].copy()
    log.info(
        "  After top-quartile duration (>= %.1f min within subset): %d",
        threshold, len(cohort),
    )

    chosen = cohort.sample(
        n=min(N_SAMPLES, len(cohort)),
        random_state=RANDOM_SEED,
    ).sort_values("duration_min", ascending=False).reset_index(drop=True)
    log.info("  Sampled %d candidates", len(chosen))
    return chosen


# ---------------------------------------------------------------------------
# Transcript parsing and red-flag metrics
# ---------------------------------------------------------------------------


def load_transcripts_for(ids):
    """Return dict[job_application_id -> list[message dicts]]."""
    log.info("Loading raw transcripts CSV …")
    raw = pd.read_csv(
        TRANSCRIPTS_CSV, encoding="latin1", on_bad_lines="skip"
    )
    raw = raw[raw["job_application_id"].isin(ids)]
    log.info("  Rows for selected ids: %d", len(raw))

    out = {}
    for _, row in raw.iterrows():
        msgs = parse_single_transcript(row["interview_transcript"])
        if msgs is None or not isinstance(msgs, list) or not msgs:
            continue
        out[row["job_application_id"]] = msgs
    return out


def red_flags(messages):
    """Heuristic metrics that help triage qualitative reading."""
    candidate_turns = [
        (m.get("content") or "").strip()
        for m in messages if m.get("role") == "user"
    ]
    interviewer_turns = [
        (m.get("content") or "").strip()
        for m in messages if m.get("role") == "interviewer"
    ]
    candidate_turns = [t for t in candidate_turns if t]
    interviewer_turns = [t for t in interviewer_turns if t]

    if candidate_turns:
        words_per_turn = [len(t.split()) for t in candidate_turns]
        mean_words = sum(words_per_turn) / len(words_per_turn)
        shortest = min(words_per_turn)
        pct_short = sum(1 for w in words_per_turn if w <= 5) / len(words_per_turn)
    else:
        mean_words = 0.0
        shortest = 0
        pct_short = 0.0

    # Longest run of consecutive interviewer messages (re-prompts).
    longest_interviewer_run = 0
    current_run = 0
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if m.get("role") == "interviewer":
            current_run += 1
            longest_interviewer_run = max(longest_interviewer_run, current_run)
        else:
            current_run = 0

    candidate_text = "\n".join(candidate_turns)
    dontknow_hits = len(DONTKNOW_RE.findall(candidate_text))
    filler_hits = len(FILLER_RE.findall(candidate_text))

    first50 = " ".join(candidate_text.split()[:50])
    last50 = " ".join(candidate_text.split()[-50:])

    return {
        "n_candidate_turns": len(candidate_turns),
        "n_interviewer_turns": len(interviewer_turns),
        "mean_candidate_words_per_turn": round(mean_words, 2),
        "shortest_candidate_turn_words": int(shortest),
        "pct_candidate_turns_under_5_words": round(pct_short, 3),
        "longest_interviewer_run": int(longest_interviewer_run),
        "dontknow_hits": int(dontknow_hits),
        "filler_hits": int(filler_hits),
        "first_50_candidate_words": first50,
        "last_50_candidate_words": last50,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_one_markdown(row, messages, flags):
    """Markdown block for one candidate."""
    h = short_hash(row["job_application_id"])
    grade_passed = "PASS" if row["is_passed"] else "FAIL"

    header = [
        f"## Candidate `{h}`  ({grade_passed}, decile {int(row['decile'])})",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| `job_application_id` (hash) | `{h}` |",
        f"| predicted P(fail) | {row['p_fail']:.4f} |",
        f"| decile (over P(fail)) | {int(row['decile'])} |",
        f"| actual outcome | **{grade_passed}** |",
        f"| duration (min) | {row['duration_min']:.1f} |",
        f"| full-transcript candidate tokens | {int(row['c3_full_tokens'])} |",
        f"| candidate words (full transcript) | {int(row['candidate_words'])} |",
        f"| n messages (total turns) | {int(row['n_messages'])} |",
        f"| n candidate turns | {flags['n_candidate_turns']} |",
        f"| n interviewer turns | {flags['n_interviewer_turns']} |",
        f"| mean candidate words / turn | {flags['mean_candidate_words_per_turn']:.2f} |",
        f"| shortest candidate turn (words) | {flags['shortest_candidate_turn_words']} |",
        f"| pct candidate turns ≤ 5 words | {flags['pct_candidate_turns_under_5_words']*100:.1f}% |",
        f"| longest interviewer run (consecutive turns) | {flags['longest_interviewer_run']} |",
        f"| 'I don't know' / 'not sure' hits | {flags['dontknow_hits']} |",
        f"| filler word hits (uh/um/hmm) | {flags['filler_hits']} |",
        "",
        "### Transcript",
        "",
        "```text",
        format_qa(messages),
        "```",
    ]
    return "\n".join(header)


def build_summary_table(chosen, flags_by_id):
    rows = []
    for _, row in chosen.iterrows():
        f = flags_by_id[row["job_application_id"]]
        rows.append(
            "  {h}  decile={d:>2}  dur={dur:>5.1f}min  "
            "p_fail={p:.3f}  cand_words={cw:>4}  msgs={msgs:>3}  "
            "mean_words/turn={mw:>5.1f}  dontknow={dk}".format(
                h=short_hash(row["job_application_id"]),
                d=int(row["decile"]),
                dur=row["duration_min"],
                p=row["p_fail"],
                cw=int(row["candidate_words"]),
                msgs=int(row["n_messages"]),
                mw=f["mean_candidate_words_per_turn"],
                dk=f["dontknow_hits"],
            )
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_cohort()
    chosen = select_candidates(df)
    transcripts = load_transcripts_for(set(chosen["job_application_id"]))

    flags_by_id = {}
    for _, row in chosen.iterrows():
        jid = row["job_application_id"]
        if jid not in transcripts:
            log.warning("  Transcript missing for %s (skipping)",
                        short_hash(jid))
            continue
        flags_by_id[jid] = red_flags(transcripts[jid])

    # CSV — one row per interview, transcript in a single cell.
    csv_rows = []
    for _, row in chosen.iterrows():
        jid = row["job_application_id"]
        if jid not in transcripts:
            continue
        flags = flags_by_id[jid]
        csv_rows.append({
            "id_hash": short_hash(jid),
            "decile": int(row["decile"]),
            "p_fail": float(row["p_fail"]),
            "is_passed": bool(row["is_passed"]),
            "duration_min": float(row["duration_min"]),
            "c3_full_tokens": int(row["c3_full_tokens"]),
            "candidate_words": int(row["candidate_words"]),
            "n_messages": int(row["n_messages"]),
            "n_candidate_turns": flags["n_candidate_turns"],
            "n_interviewer_turns": flags["n_interviewer_turns"],
            "mean_candidate_words_per_turn":
                flags["mean_candidate_words_per_turn"],
            "shortest_candidate_turn_words":
                flags["shortest_candidate_turn_words"],
            "pct_candidate_turns_under_5_words":
                flags["pct_candidate_turns_under_5_words"],
            "longest_interviewer_run":
                flags["longest_interviewer_run"],
            "dontknow_hits": flags["dontknow_hits"],
            "filler_hits": flags["filler_hits"],
            "first_50_candidate_words":
                flags["first_50_candidate_words"],
            "last_50_candidate_words":
                flags["last_50_candidate_words"],
            "transcript_qa": format_qa(transcripts[jid]),
        })
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    log.info("Wrote %s (%d rows)", OUT_CSV, len(csv_rows))

    # Markdown rendering.
    blocks = []
    for _, row in chosen.iterrows():
        jid = row["job_application_id"]
        if jid not in transcripts:
            continue
        blocks.append(render_one_markdown(
            row, transcripts[jid], flags_by_id[jid]
        ))
    md_doc = (
        "# Inspection: long, high-risk, failed interviews "
        "(Issue #1 follow-up)\n\n"
        f"Sampled {len(blocks)} candidates from deciles "
        f"{sorted(DECILES_TO_INSPECT)} with `is_passed = False`, "
        f"`duration_min` in the top quartile of that subset, "
        f"`random_state={RANDOM_SEED}`.  "
        f"IDs replaced with 8-character SHA1 hashes for portability.\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    OUT_MD.write_text(md_doc, encoding="utf-8")
    log.info("Wrote %s", OUT_MD)

    log.info("=" * 70)
    log.info("ONE-LINE-PER-CANDIDATE SUMMARY")
    log.info("=" * 70)
    print(build_summary_table(chosen, flags_by_id))


if __name__ == "__main__":
    main()
