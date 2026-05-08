# Interview Embedding Benchmark

Benchmarking 10 embedding models × 6 transcript parsing conditions for predicting technical interview skill grades (React, JavaScript, HTML/CSS) from interview transcript embeddings.

## Key Findings

| Task | Best Model | Best Condition | Macro F1 |
|:---|:---|:---|:---|
| **4-Class Ordinal Grading** | Voyage-3.5 | C2a (per-skill Q&A) | 0.617 |
| **Binary Pass/Fail** | OpenAI text-embedding-3-large | C2a | 0.830 |

- **Parsing matters more than model choice:** C2a (per-skill Q&A, no intro) outperforms full-transcript (C3) by ~0.05 F1 across all models. Semantic density beats context length.
- **Open-source models match proprietary:** Qwen3-8B (p=0.060) and KaLM-12B (p=0.072) are **statistically indistinguishable** from Voyage for 4-class grading (paired t-test, 15 CV folds).
- **Ranking reversal between tasks:** Voyage is #1 for 4-class but drops to #5 for binary. OpenAI-Large jumps from #3 to #1 for binary (p=0.002).
- **Transcript length:** 95% of peak accuracy is reached at just ~700 words (~1,024 tokens), enabling early stopping in production.
- **Senior class blindspot:** Senior candidates are severely data-starved (n=35 for HTML/CSS). Sample weighting improves Senior F1 from 0.226 → 0.256, but more data is needed.

## Repository Structure

```
├── parse_transcripts.py        # Step 1: Raw data → 6 conditions + labels
├── embed_api.py                # Step 2a: Embed via 5 paid APIs
├── embed_opensource.py          # Step 2b: Embed via 5 open-source models (HPC)
├── embed_truncated.py           # Step 2c: Truncated embeddings for length curve
├── benchmark.py                 # Step 3a: Probing classifiers (480 combos)
├── truncation_benchmark.py      # Step 3b: Length-accuracy curve
├── lasso_experiment.py          # Step 3c: Senior class ablation
├── merge_results.py             # Merge partial result CSVs
├── stats_tests.py               # Paired t-tests (4-class significance)
├── binary_analysis.py           # Binary pass/fail pivot tables
├── binary_stats.py              # Binary significance tests
├── visualize.py                 # Confusion matrix heatmaps
├── plot_truncation.py           # Truncation curve plot
├── plot_hazard.py               # Step 5a: Hazard-of-failing plot, TF-IDF baseline
├── plot_hazard_embeddings.py    # Step 5b: Hazard-of-failing plot, real embeddings
├── stopping_agent.py            # Step 6:  Sequential pass/fail probe + stopping rule
├── extract_sample.py            # Extract sample transcripts for review
│
├── results/                     # Benchmark metrics (committed)
├── figures/                     # Generated plots (committed)
├── report/                      # LaTeX report + analysis memos (committed)
├── slurm/                       # HPC job scripts for Discovery cluster
├── data/                        # Raw data (gitignored — see data/README.md)
├── parsed/                      # Parsed conditions (gitignored, regenerable)
└── vectors/                     # Embedding vectors (gitignored, regenerable)
```

## Setup

### 1. Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. API Keys (for paid models)

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 3. Data

Place the raw data files in `data/`. See [`data/README.md`](data/README.md) for required files and schemas.

### 4. Parse Transcripts

```bash
python parse_transcripts.py
```

This generates `parsed/labels.csv` and `parsed/segments_C*.json` (6 conditions).

## Pipeline Execution Order

```
1. parse_transcripts.py          →  parsed/
2. embed_api.py                  →  vectors/{model}_{condition}.npy
   embed_opensource.py            →  vectors/{model}_{condition}.npy  (on HPC)
   embed_truncated.py             →  vectors/voyage_C3_trunc{N}.npy
3. benchmark.py                  →  results/full_results.csv
   benchmark.py --binary          →  results/full_results_binary_{model}.csv
   truncation_benchmark.py        →  results/truncation_curve.csv
   lasso_experiment.py            →  (stdout — Senior class ablation)
4. merge_results.py              →  results/full_results.csv (unified)
   binary_analysis.py             →  results/primary_comparison_binary.csv
   stats_tests.py                 →  (stdout — significance tests)
   binary_stats.py                →  (stdout — binary significance)
   visualize.py                   →  figures/confusion_*.png
   plot_truncation.py             →  figures/truncation_curve.png
5. plot_hazard.py                →  figures/hazard_of_failing*.png
                                    results/hazard_decile_summary.csv
   plot_hazard_embeddings.py      →  figures/hazard_of_failing__{model}.png
                                    results/hazard_decile_summary__{model}.csv
6. stopping_agent.py             →  figures/stopping_agent_*.png
                                    results/stopping_agent_*.csv
```

## Hazard-of-Failing Diagnostic

A downstream diagnostic asks: in our AI-paced interview protocol, does the
call duration shorten for candidates likely to fail (as it does in the
human-paced interviewer literature)?

Candidates are ranked by out-of-fold predicted P(fail) from a 5-fold
stratified, class-balanced logistic regression and binned into ten
equal-sized risk deciles.  Mean call duration (and three companion length
metrics — candidate words, transcript tokens, message count) is reported
per decile with 95% confidence intervals.

Two implementations:

- **`plot_hazard.py`** — TF-IDF (1+2-grams, 20k features) on the full Q&A
  text.  Self-contained, no API keys needed; useful as a baseline.
- **`plot_hazard_embeddings.py`** — loads pre-computed `*.npy` vectors
  from the benchmark pipeline (`openai-large + C2a` and `voyage + C2a` by
  default) and runs the same probe on the dense representation.  Produces
  per-model figures plus a three-classifier overlay vs. the TF-IDF
  baseline.

**Headline result.** Average call duration is essentially flat across the
ten risk deciles (~5–10% swing, vs. ~75% in the human-paced reference
plot), confirming that the protocol does not shorten for likely-to-fail
candidates.  However, candidate word count drops by ~3× from decile 1 to
decile 10 while the message count rises by ~2× — failing candidates
spend the same wall-clock time but say much less, in many more, much
shorter turns.  Full discussion in [`report/issue1_findings.md`](report/issue1_findings.md)
(memo) and [`report/issue1_analysis_report.md`](report/issue1_analysis_report.md)
(long-form methodology).

## Stopping-Agent Diagnostic

A second downstream diagnostic adapts Manzoor, Ascarza & Netzer (2025)
"Learning When to Quit in Sales Conversations" to predicting binary
`is_passed` from transcript prefixes.  The candidate is evaluated at six
checkpoints (256, 512, 1024, 2048, 4096 candidate-only `cl100k_base`
tokens, plus the full transcript).  At each checkpoint we concatenate
the pre-computed Voyage `C3` embedding with five position features
(tokens consumed, candidate-words-so-far, n-messages-so-far,
words-per-turn, fraction-of-overall-median-tokens) and fit a 5-fold
stratified-CV class-balanced logistic regression.

Outputs of `stopping_agent.py`:

- `results/stopping_agent_per_checkpoint.csv` — per-checkpoint AUC,
  PR-AUC (pass class), Macro-F1, recall@P=0.95, plus an
  embeddings-only ablation per row.
- `results/stopping_agent_oof_predictions.csv` — out-of-fold P(fail)
  per (candidate, checkpoint).
- `results/stopping_agent_threshold_sweep.csv` — 121-row sweep over
  asymmetric (τ_fail, τ_pass) commit thresholds.
- `figures/stopping_agent_{auc_curve,savings,threshold_heatmap}.png`.

**Headline result.** A fail-only stopping rule (commit fail when
`P(fail) ≥ 0.5 + τ_fail`, never commit pass early, fall back to the
full transcript otherwise) achieves **92.0% accuracy at a mean of 286
tokens consumed per candidate**, vs. 90.1% accuracy at 1172 tokens for
the always-full baseline — a 76% reduction in tokens with a 1.9
percentage-point absolute accuracy improvement.  The accuracy gain is
partly an artefact of the 4.6% positive base rate (the asymmetric rule
exploits the prior; a calibrated full baseline would close most of
the gap), so the robust deliverable is the **token savings**.  AUC
saturates by ~1024 tokens.  Full discussion in
[`report/issue2_findings.md`](report/issue2_findings.md).

To run, point `HAZARD_VECTORS_DIR` at the benchmark's `vectors/`
directory:

```bash
HAZARD_VECTORS_DIR=/path/to/benchmark_2026/vectors \
  python3 stopping_agent.py
```

## Models

### Paid APIs (embed_api.py)

| Key | Model | Provider | Dimensions | Token Limit |
|:---|:---|:---|:---|:---|
| `gemini` | gemini-embedding-001 | Google | 3,072 | 2,048 |
| `openai-large` | text-embedding-3-large | OpenAI | 3,072 | 8,191 |
| `openai-small` | text-embedding-3-small | OpenAI | 1,536 | 8,191 |
| `voyage` | voyage-3.5 | Voyage AI | 1,024 | 32,000 |
| `cohere` | embed-v4.0 | Cohere | 1,536 | 128,000 |

### Open-Source (embed_opensource.py, run on HPC)

| Key | Model | Parameters | Dimensions |
|:---|:---|:---|:---|
| `kalm-12b` | KaLM-Embedding-Gemma3-12B | 11.76B | 3,840 |
| `qwen3-8b` | Qwen3-Embedding-8B | 7.57B | 4,096 |
| `qwen3-4b` | Qwen3-Embedding-4B | 4.02B | 2,560 |
| `jina-v5-small` | jina-embeddings-v5-text-small | 0.60B | 1,024 |
| `qwen3-0.6b` | Qwen3-Embedding-0.6B | 0.60B | 1,024 |

## Parsing Conditions

| ID | Content | Scope | Description |
|:---|:---|:---|:---|
| C1a | Candidate answers only | Per-skill | Isolated answers, no interviewer questions |
| C1b | Candidate answers + intro | Per-skill | C1a + non-technical introduction |
| **C2a** | **Full Q&A** | **Per-skill** | **Interviewer + candidate, per skill (best)** |
| C2b | Full Q&A + intro | Per-skill | C2a + introduction |
| C3 | Candidate answers only | Full transcript | All answers concatenated |
| C4 | Full Q&A | Full transcript | Entire conversation |

## Data

- **6,338 candidates** with matched transcripts and grades
- **3 skills:** React, JavaScript, HTML/CSS
- **4-level ordinal grading:** Not Experienced (0), Junior (1), Mid-level (2), Senior (3)
- **Binary pass/fail:** Pass (grade ≥ 2) / Fail (grade < 2), 22.9% / 77.1% split
- Transcripts are voice-transcribed AI interviews (mean ~25 messages, ~1,062 candidate words)

## Classifiers

- **Ordinal:** `mord.LogisticAT(alpha=1.0)` — respects grade ordering, penalizes large misclassifications
- **Nominal:** `sklearn.LogisticRegression(class_weight='balanced')` — treats grades as unordered categories
- All results use **5-fold stratified cross-validation**, random seed 42

## Citation

If you use this code or data, please cite:

> Palikot, E. & Pyla, R. (2026). Interview Embedding Benchmark: Evaluating Transcript Embeddings for Technical Skill Classification.

## License

This project is for academic research purposes. Contact the authors for licensing inquiries.
