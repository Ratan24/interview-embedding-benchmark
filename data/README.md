# Data Directory

This directory should contain the raw input data files required by the pipeline.
These files are **not committed** to the repository due to data privacy constraints.

## Required Files

### 1. `transcripts_6400_records.csv`

Interview transcript records. Must be loaded with `encoding='latin1'`.

**Key columns:**
- `job_application_id` (UUID) — primary join key
- `interview_transcript` — JSON array of message objects

**Expected rows:** ~6,410

### 2. `Ai-Vetted-ranked.csv`

Candidate vetting results with per-skill grades and pass/fail outcomes.

**Key columns:**
- `job_application_id` (UUID) — primary join key
- `ai_vetting_results` — pipe-delimited string, e.g. `"React : Senior | JavaScript : Mid-level | HTML, CSS : Junior"`
- `is_passed` — boolean pass/fail outcome

**Expected rows:** ~25,536

## Notes

- After placing these files here, run `python parse_transcripts.py` to generate
  the parsed conditions and labels in `parsed/`.
- The `.xlsx` version of the transcripts file is known to be corrupted (152 columns
  instead of expected 5). Use only the `.csv` version.
- Do **not** commit these files to the repository.
