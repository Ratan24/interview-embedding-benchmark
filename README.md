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
├── extract_sample.py            # Extract sample transcripts for review
│
├── results/                     # Benchmark metrics (committed)
├── figures/                     # Generated plots (committed)
├── report/                      # LaTeX report (committed)
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
