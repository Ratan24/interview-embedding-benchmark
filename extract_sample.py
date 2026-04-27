"""Extract 30 'Not Experienced' and 30 'Senior' transcripts for Professor Palikot.

Outputs a single CSV file with the full transcript text, candidate grades,
and a group label (Not Experienced / Senior).

Selection criteria:
  - Not Experienced: candidates rated 0 (Not Experienced) in ALL three skills
  - Senior: candidates rated 3 (Senior) in at least one skill
  - Random sample of 30 from each group, seed=42

Usage:
    python extract_sample.py
"""

import json
import os
import re
import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_PER_GROUP = 30

BASE_DIR = Path(__file__).resolve().parent
PARSED_DIR = BASE_DIR / "parsed"
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_CSV = DATA_DIR / "transcripts_6400_records.csv"

OUTPUT_CSV = BASE_DIR / "sample_transcripts_60.csv"

GRADE_LABELS = {0: "Not Experienced", 1: "Junior", 2: "Mid-level", 3: "Senior"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_json(raw):
    """Strip control characters that break json.loads."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw)


def parse_transcript_json(raw):
    """Parse transcript JSON string into a list of message dicts."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return json.loads(clean_json(raw))
    except (json.JSONDecodeError, TypeError):
        return None


def format_transcript(messages):
    """Format a transcript into readable Q&A text."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue

        # Get skill context if available
        skill = msg.get("skill", {})
        skill_name = skill.get("skill_name", "") if isinstance(skill, dict) else ""

        if role == "interviewer":
            prefix = f"[Interviewer{' — ' + skill_name if skill_name else ''}]"
        elif role == "user":
            prefix = "[Candidate]"
        else:
            prefix = f"[{role}]"

        parts.append(f"{prefix} {content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    random.seed(RANDOM_SEED)

    # Load labels
    labels = pd.read_csv(PARSED_DIR / "labels.csv")
    print(f"Labels loaded: {len(labels)} candidates")

    # Define groups
    not_exp_mask = (
        (labels["react_grade"] == 0)
        & (labels["javascript_grade"] == 0)
        & (labels["html_css_grade"] == 0)
    )
    senior_mask = (
        (labels["react_grade"] == 3)
        | (labels["javascript_grade"] == 3)
        | (labels["html_css_grade"] == 3)
    )

    not_exp_ids = labels[not_exp_mask]["job_application_id"].tolist()
    senior_ids = labels[senior_mask]["job_application_id"].tolist()

    print(f"Not Experienced (all 3 skills = 0): {len(not_exp_ids)} candidates")
    print(f"Senior (any skill = 3): {len(senior_ids)} candidates")

    # Sample 30 from each
    not_exp_sample = random.sample(not_exp_ids, N_PER_GROUP)
    senior_sample = random.sample(senior_ids, N_PER_GROUP)

    target_ids = set(not_exp_sample + senior_sample)
    id_to_group = {}
    for aid in not_exp_sample:
        id_to_group[aid] = "Not Experienced"
    for aid in senior_sample:
        id_to_group[aid] = "Senior"

    # Load transcripts
    print(f"\nLoading transcripts from {TRANSCRIPTS_CSV.name}...")
    tdf = pd.read_csv(TRANSCRIPTS_CSV, encoding="latin1", on_bad_lines="skip")
    print(f"Transcript rows loaded: {len(tdf)}")

    # Extract and format
    rows = []
    found_ids = set()

    for _, row in tdf.iterrows():
        aid = row["job_application_id"]
        if aid not in target_ids:
            continue

        raw = row.get("interview_transcript", "")
        if pd.isna(raw) or not str(raw).strip():
            continue

        messages = parse_transcript_json(str(raw))
        if messages is None:
            print(f"  WARNING: Could not parse transcript for {aid}")
            continue

        formatted = format_transcript(messages)

        # Get grades
        label_row = labels[labels["job_application_id"] == aid].iloc[0]
        react_g = int(label_row["react_grade"])
        js_g = int(label_row["javascript_grade"])
        html_g = int(label_row["html_css_grade"])

        rows.append({
            "group": id_to_group[aid],
            "job_application_id": aid,
            "react_grade": f"{react_g} ({GRADE_LABELS[react_g]})",
            "javascript_grade": f"{js_g} ({GRADE_LABELS[js_g]})",
            "html_css_grade": f"{html_g} ({GRADE_LABELS[html_g]})",
            "transcript": formatted,
        })
        found_ids.add(aid)

    missing = target_ids - found_ids
    if missing:
        print(f"\n  WARNING: {len(missing)} transcripts not found")

    # Sort: Not Experienced first, then Senior
    rows.sort(key=lambda r: (0 if r["group"] == "Not Experienced" else 1, r["job_application_id"]))

    # Save
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df_out)} transcripts to {OUTPUT_CSV}")
    print(f"  Not Experienced: {sum(1 for r in rows if r['group'] == 'Not Experienced')}")
    print(f"  Senior: {sum(1 for r in rows if r['group'] == 'Senior')}")


if __name__ == "__main__":
    main()
