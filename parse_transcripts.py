"""Parse interview transcripts and labels for the embedding benchmark.

Loads raw transcript CSV and vetting CSV from data/, matches on
job_application_id, extracts per-skill grades, and produces 6 text
conditions (C1a-C4) plus a labels file.  All outputs are written to parsed/.

Usage:
    python parse_transcripts.py
"""

import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# Paths relative to this script's directory (repo root)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRANSCRIPTS_CSV = DATA_DIR / "transcripts_6400_records.csv"
VETTING_CSV = DATA_DIR / "Ai-Vetted-ranked.csv"
OUTPUT_DIR = BASE_DIR / "parsed"

# Grade mapping (string → int)
GRADE_MAP = {
    "Not experienced": 0,
    "Junior": 1,
    "Mid-level": 2,
    "Senior": 3,
}

# Expected skills (canonical names as they appear in transcripts)
EXPECTED_SKILLS = ["React", "JavaScript", "HTML, CSS"]

# Canonical key names for output JSON / labels
SKILL_KEYS = {
    "React": "react",
    "JavaScript": "javascript",
    "HTML, CSS": "html_css",
}

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
# Label extraction
# ---------------------------------------------------------------------------


def parse_vetting_results(result_str):
    """Parse an ai_vetting_results string into a grade dict.

    Args:
        result_str: Pipe-delimited string, e.g.
            "React : Senior | JavaScript : Mid-level | HTML, CSS : Junior"

    Returns:
        Dict mapping canonical skill keys to integer grades, e.g.
        {"react": 3, "javascript": 2, "html_css": 1}.
        Returns None if any expected skill is missing or has an
        unrecognised grade value.
    """
    grades = {}
    for part in result_str.split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        skill_str, grade_str = part.rsplit(":", 1)
        skill_str = skill_str.strip()
        grade_str = grade_str.strip()

        if skill_str not in SKILL_KEYS:
            continue
        if grade_str not in GRADE_MAP:
            return None

        grades[SKILL_KEYS[skill_str]] = GRADE_MAP[grade_str]

    # Require all three skills
    if set(grades.keys()) != {"react", "javascript", "html_css"}:
        return None

    return grades


def load_labels(vetting_path):
    """Load vetting CSV and extract per-skill grades.

    Args:
        vetting_path: Path to Ai-Vetted-ranked.csv.

    Returns:
        DataFrame with columns: job_application_id, react_grade,
        javascript_grade, html_css_grade, is_passed.
    """
    log.info("Loading vetting data from %s", vetting_path)
    vdf = pd.read_csv(vetting_path)
    log.info("  Total rows: %d", len(vdf))

    # Keep only rows with vetting results
    has_results = vdf["ai_vetting_results"].notna()
    log.info("  Rows with ai_vetting_results: %d", has_results.sum())
    vdf = vdf[has_results].copy()

    # Parse grades
    records = []
    skip_reasons = Counter()
    for _, row in vdf.iterrows():
        grades = parse_vetting_results(row["ai_vetting_results"])
        if grades is None:
            skip_reasons["partial_or_invalid_grades"] += 1
            continue
        records.append({
            "job_application_id": row["job_application_id"],
            "react_grade": grades["react"],
            "javascript_grade": grades["javascript"],
            "html_css_grade": grades["html_css"],
            "is_passed": bool(row["is_passed"]),
        })

    log.info("  Valid label rows: %d", len(records))
    for reason, count in skip_reasons.items():
        log.info("  Skipped (%s): %d", reason, count)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------


def clean_json(raw):
    """Strip control characters that break json.loads.

    Args:
        raw: Raw JSON string from the CSV.

    Returns:
        Cleaned string with C0/C1 control characters removed.
    """
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw)


def parse_single_transcript(raw_json):
    """Parse one transcript JSON string into a list of message dicts.

    Args:
        raw_json: JSON string (array of message objects).

    Returns:
        List of message dicts on success, None on unrecoverable failure.
    """
    try:
        return json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try cleaning control characters
    try:
        cleaned = clean_json(raw_json)
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def get_skill_name(msg):
    """Extract the skill_name from a message's skill field.

    Args:
        msg: Single message dict from the transcript.

    Returns:
        Skill name string, or empty string if absent.
    """
    skill = msg.get("skill")
    if isinstance(skill, dict):
        return skill.get("skill_name", "")
    return ""


def segment_messages(messages):
    """Group transcript messages into intro + per-skill segments.

    Uses skill_name tags from interviewer messages to determine the current
    skill context.  Messages before the first skill-tagged question are
    assigned to the intro segment.

    Args:
        messages: List of message dicts from the transcript.

    Returns:
        Dict mapping segment names to lists of message dicts:
        {"intro": [...], "React": [...], "JavaScript": [...],
         "HTML, CSS": [...]}.  A segment is omitted if it has no messages.
    """
    segments = {"intro": []}
    current_skill = None

    for msg in messages:
        skill_name = get_skill_name(msg)

        # Track the current skill context from any message with a tag
        if skill_name and skill_name in EXPECTED_SKILLS:
            current_skill = skill_name
        elif skill_name == "" and "skill" in msg:
            # Empty skill_name on a tagged message (e.g. closing message):
            # keep it in the current skill segment if one is active,
            # otherwise treat as intro.
            pass

        if current_skill is None:
            segments["intro"].append(msg)
        else:
            if current_skill not in segments:
                segments[current_skill] = []
            segments[current_skill].append(msg)

    return segments


def format_answer_only(messages):
    """Concatenate candidate-only content from a list of messages.

    Args:
        messages: List of message dicts.

    Returns:
        Single string of concatenated candidate answers.
    """
    parts = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                parts.append(content)
    return "\n\n".join(parts)


def format_qa(messages):
    """Format messages as labeled Q&A pairs.

    Interviewer messages are prefixed with "Interviewer:" and candidate
    messages with "Candidate:".

    Args:
        messages: List of message dicts.

    Returns:
        Single formatted string.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "interviewer":
            parts.append(f"Interviewer: {content}")
        elif role == "user":
            parts.append(f"Candidate: {content}")
    return "\n\n".join(parts)


def build_conditions(messages):
    """Build all 6 text conditions for one candidate's transcript.

    Args:
        messages: List of message dicts from the transcript.

    Returns:
        Dict with keys C1a, C1b, C2a, C2b, C3, C4, each containing
        the appropriate text structure.  Returns None if the transcript
        is missing all three expected skill sections.
    """
    segments = segment_messages(messages)

    # Check that we have at least one expected skill section
    present_skills = [s for s in EXPECTED_SKILLS if s in segments]
    if not present_skills:
        return None

    intro_msgs = segments.get("intro", [])

    # --- Per-skill conditions ---
    def make_per_skill(format_fn, include_intro):
        result = {}
        if include_intro:
            result["intro"] = format_fn(intro_msgs)
        for skill in EXPECTED_SKILLS:
            key = SKILL_KEYS[skill]
            skill_msgs = segments.get(skill, [])
            result[key] = format_fn(skill_msgs)
        return result

    c1a = make_per_skill(format_answer_only, include_intro=False)
    c1b = make_per_skill(format_answer_only, include_intro=True)
    c2a = make_per_skill(format_qa, include_intro=False)
    c2b = make_per_skill(format_qa, include_intro=True)

    # --- Full-transcript conditions ---
    c3 = format_answer_only(messages)
    c4 = format_qa(messages)

    return {
        "C1a": c1a,
        "C1b": c1b,
        "C2a": c2a,
        "C2b": c2b,
        "C3": c3,
        "C4": c4,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run():
    """Execute the full parsing pipeline."""
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load labels ---
    labels_df = load_labels(VETTING_CSV)
    label_ids = set(labels_df["job_application_id"])

    # --- Load transcripts ---
    log.info("Loading transcripts from %s", TRANSCRIPTS_CSV)
    tdf = pd.read_csv(
        TRANSCRIPTS_CSV, encoding="latin1", on_bad_lines="skip"
    )
    log.info("  Transcript rows loaded: %d", len(tdf))

    transcript_ids = set(tdf["job_application_id"])
    matched_ids = label_ids & transcript_ids
    log.info("  IDs with both labels and transcripts: %d", len(matched_ids))

    # --- Parse transcripts ---
    results = {}   # job_application_id → conditions dict
    skip_reasons = Counter()

    for _, row in tdf.iterrows():
        app_id = row["job_application_id"]
        if app_id not in matched_ids:
            skip_reasons["no_matching_label"] += 1
            continue

        raw = row["interview_transcript"]
        if pd.isna(raw) or not str(raw).strip():
            skip_reasons["empty_transcript"] += 1
            continue

        messages = parse_single_transcript(str(raw))
        if messages is None:
            skip_reasons["json_parse_failure"] += 1
            continue

        conditions = build_conditions(messages)
        if conditions is None:
            skip_reasons["no_skill_sections"] += 1
            continue

        results[app_id] = conditions

    log.info("Successfully parsed: %d candidates", len(results))
    for reason, count in sorted(skip_reasons.items()):
        log.info("  Skipped (%s): %d", reason, count)

    # --- Filter labels to only candidates with parsed transcripts ---
    parsed_ids = set(results.keys())
    labels_df = labels_df[labels_df["job_application_id"].isin(parsed_ids)]
    labels_df = labels_df.sort_values("job_application_id").reset_index(drop=True)
    log.info("Final candidate count (labels + transcripts): %d", len(labels_df))

    # --- Save labels ---
    labels_path = OUTPUT_DIR / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    log.info("Saved %s (%d rows)", labels_path.name, len(labels_df))

    # --- Save each condition ---
    condition_ids = ["C1a", "C1b", "C2a", "C2b", "C3", "C4"]
    for cond in condition_ids:
        records = []
        for app_id in labels_df["job_application_id"]:
            entry = {"job_application_id": app_id}
            cond_data = results[app_id][cond]
            if isinstance(cond_data, dict):
                entry["segments"] = cond_data
            else:
                entry["text"] = cond_data
            records.append(entry)

        out_path = OUTPUT_DIR / f"segments_{cond}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        log.info("Saved %s (%d records)", out_path.name, len(records))

    # --- Print summary statistics ---
    print_stats(labels_df, results, condition_ids)


def print_stats(labels_df, results, condition_ids):
    """Print detailed summary statistics.

    Args:
        labels_df: Final labels DataFrame.
        results: Dict mapping app_id to conditions dict.
        condition_ids: List of condition ID strings.
    """
    log.info("=" * 60)
    log.info("SUMMARY STATISTICS")
    log.info("=" * 60)

    # Grade distribution
    for skill in ["react", "javascript", "html_css"]:
        col = f"{skill}_grade"
        dist = labels_df[col].value_counts().sort_index()
        log.info("  %s grade distribution:", skill)
        for grade_val, count in dist.items():
            pct = 100.0 * count / len(labels_df)
            grade_name = {0: "Not experienced", 1: "Junior",
                          2: "Mid-level", 3: "Senior"}[grade_val]
            log.info("    %d (%s): %d (%.1f%%)", grade_val, grade_name,
                     count, pct)

    # Text length statistics per condition
    log.info("")
    log.info("  Text lengths (characters) per condition:")
    app_ids = list(labels_df["job_application_id"])

    for cond in condition_ids:
        lengths = []
        for app_id in app_ids:
            cond_data = results[app_id][cond]
            if isinstance(cond_data, dict):
                total = sum(len(v) for v in cond_data.values())
            else:
                total = len(cond_data)
            lengths.append(total)
        lengths = np.array(lengths)
        empty_count = int(np.sum(lengths == 0))
        log.info(
            "    %s: mean=%5.0f  median=%5.0f  min=%5d  max=%5d  "
            "empty=%d",
            cond, lengths.mean(), np.median(lengths),
            lengths.min(), lengths.max(), empty_count,
        )

    # Word count estimates for full-transcript conditions
    log.info("")
    log.info("  Word counts for full-transcript conditions:")
    for cond in ["C3", "C4"]:
        word_counts = []
        for app_id in app_ids:
            text = results[app_id][cond]
            word_counts.append(len(text.split()))
        wc = np.array(word_counts)
        log.info(
            "    %s: mean=%5.0f  median=%5.0f  min=%5d  max=%5d",
            cond, wc.mean(), np.median(wc), wc.min(), wc.max(),
        )

    # Per-skill segment emptiness check
    log.info("")
    log.info("  Empty skill segments (per-skill conditions):")
    for cond in ["C1a", "C2a"]:
        for skill_key in ["react", "javascript", "html_css"]:
            empty = sum(
                1 for app_id in app_ids
                if not results[app_id][cond].get(skill_key, "").strip()
            )
            if empty > 0:
                log.info("    %s / %s: %d empty", cond, skill_key, empty)


if __name__ == "__main__":
    run()
