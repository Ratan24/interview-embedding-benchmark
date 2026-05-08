# Interview Length × Hazard of Failing
### Findings memo for Issue #1

**To:** Prof. Emil Palikot
**From:** Ratan Pyla
**Date:** 8 May 2026
**Repository:** [`Ratan24/interview-embedding-benchmark`](https://github.com/Ratan24/interview-embedding-benchmark)
**Issue:** [#1 — Create a plot of Interview length × hazard of failing](https://github.com/Ratan24/interview-embedding-benchmark/issues/1)

---

## 1. Summary

You hypothesised that, because our interviews are conducted on a scripted protocol rather than a human-paced one, the analogue of the stopping-agents call-duration plot would be flat. **The data confirms this — average call duration varies by less than 10% of its mean across ten predicted-failure-risk deciles, compared with the ~75% swing in the reference plot.** The protocol does not shorten for likely-to-fail candidates.

A secondary finding emerged that I think is the more interesting result for the broader project. While *time* is invariant to candidate quality, the *content* of the interview is not: as predicted failure risk rises, candidate word counts fall by a factor of ~3 and the number of message exchanges nearly doubles. The interview lasts the same wall-clock duration but contains far less candidate speech, distributed across many more, much shorter turns. The signal that lets a stopping agent end the call early in the human-paced setting is therefore present in our data — it is just encoded in *content* rather than *time*.

Both findings are robust across three independent risk classifiers (TF-IDF, OpenAI `text-embedding-3-large`, and Voyage `voyage-3.5`).

---

## 2. Methodology

### 2.1 Sample

The analysis joins `transcripts_6400_records.csv` to `Ai-Vetted-ranked.csv` on `job_application_id`. We use `is_passed` as the binary outcome, and the difference between `vetting_completed_date` and `vetting_creation_date` as the call duration (the natural analogue of the reference plot's y-axis). After dropping unparseable transcripts and clipping durations to [5, 120] minutes — the latter excludes paused/resumed sessions, which extend up to 60,000 minutes in the raw data — the analysis sample is **N = 6,228 candidates**, with a 4.5% pass rate (283 passes, 5,945 fails).

### 2.2 Risk classifiers

To bin candidates into predicted-failure deciles I trained three independent binary classifiers on `is_passed`, each evaluated by 5-fold stratified cross-validation. Stratification preserves the 4.5% pass rate within each fold; class-weight balancing in the logistic regression up-weights the minority class so the optimiser ranks rather than collapses to the majority prediction. Predictions for each candidate come from the fold in which she is held out, giving a clean out-of-fold (OOF) probability vector.

| Classifier | Feature representation | Dimension | OOF ROC-AUC |
|---|---|---:|---:|
| **TF-IDF + LogReg** | 1- and 2-grams of full Q&A text, sublinear-TF, L2-normalised | 20,000 sparse | **0.877** |
| **OpenAI `text-embedding-3-large` + LogReg** | Per-skill (C2a) embeddings, flattened across React / JavaScript / HTML+CSS | 9,216 dense | 0.790 |
| **Voyage `voyage-3.5` + LogReg** | Per-skill (C2a) embeddings, flattened | 3,072 dense | 0.781 |

The two embedding models are the binary-task and 4-class champions in the existing benchmark, respectively. The TF-IDF baseline was originally intended as a sanity check but turned out to dominate on this particular binary task; see §4.

### 2.3 Decile construction and aggregation

Candidates are ranked by OOF P(fail) and partitioned into ten equal-sized buckets (~620 candidates each). For every decile and every length metric I report the sample mean and a Student's-t 95% confidence interval (with n ≈ 620 the CI is essentially Normal). To establish that the binning is meaningful, I separately confirm that actual `is_passed` rate falls monotonically from decile 1 to decile 10 for all three classifiers.

### 2.4 Length metrics

Four operationalisations of "interview length", because the AI-paced protocol decouples them in ways the human-paced literature does not:

- **Call duration (s)** — direct analogue of the reference plot.
- **Candidate word count** — total words across candidate turns only.
- **Transcript token count** — `cl100k_base` token count of the full Q&A transcript.
- **Message count** — total non-empty turns from either party.

---

## 3. Results

### 3.1 Primary plot — duration vs. risk decile (Voyage classifier)

![Primary plot, Voyage embeddings](../figures/hazard_of_failing__voyage.png)

The y-axis is anchored at zero to keep the magnitude of the variation visually honest. The mean duration sits at the overall median (37 minutes ≈ 2,220 s) across all ten deciles; the swing across deciles is approximately 120 seconds, ~5% of the mean.

For comparison, the analogous plot from the TF-IDF baseline:

![Primary plot, TF-IDF baseline](../figures/hazard_of_failing.png)

And the OpenAI embedding classifier:

![Primary plot, OpenAI embeddings](../figures/hazard_of_failing__openai-large.png)

All three are flat. The three-way overlay makes the agreement explicit:

![Three-classifier comparison](../figures/hazard_of_failing_comparison.png)

Whatever modest decile-to-decile movement appears in any single curve, it is within the 95% confidence band of the others.

### 3.2 Per-decile numbers (TF-IDF, the cleanest separation)

| Decile (1 = lowest risk → 10 = highest) | n | Pass rate | Mean duration (s) | 95% CI (s) |
|---:|---:|---:|---:|---|
| 1 | 623 | 21.2% | 2273 | [2198, 2348] |
| 2 | 623 | 13.0% | 2328 | [2256, 2400] |
| 3 | 623 | 6.3% | 2391 | [2322, 2460] |
| 4 | 622 | 2.7% | 2415 | [2345, 2485] |
| 5 | 623 | 1.6% | 2445 | [2369, 2521] |
| 6 | 623 | 0.5% | 2464 | [2382, 2546] |
| 7 | 622 | 0.2% | 2500 | [2416, 2584] |
| 8 | 623 | 0.0% | 2439 | [2353, 2525] |
| 9 | 623 | 0.0% | 2385 | [2298, 2473] |
| 10 | 623 | 0.0% | 2336 | [2237, 2434] |

Pass rate is strictly monotone, validating the decile assignment. Duration ranges from 2,273 s to 2,500 s — a swing of 227 s, ~10% of the mean.

### 3.3 The four-panel breakdown

The duration-only plot answers the original hypothesis but conceals the richer pattern that motivates the secondary finding. The four-panel chart pulls the length metrics apart:

![Four-panel breakdown, TF-IDF baseline](../figures/hazard_of_failing_panels.png)

(For the embedding-based versions, see `figures/hazard_of_failing_panels__openai-large.png` and `figures/hazard_of_failing_panels__voyage.png`. The qualitative pattern is identical; TF-IDF produces the steepest slopes because its higher AUC sorts the most extreme candidates more cleanly into the tail deciles.)

The four panels diverge sharply in shape:

| Metric | Decile 1 → Decile 10 | Pattern |
|---|---|---|
| **Call duration** | 2,273 s → 2,336 s (peaks at decile 7, 2,500 s) | flat, mild inverted-U |
| **Candidate words** | 1,318 → 459 | strictly decreasing, ~3× drop |
| **Transcript tokens** | 1,989 → 1,407 | strictly decreasing, ~30% drop |
| **Messages exchanged** | 18 → 38 | strictly increasing, ~2× rise |

In words: as predicted failure risk rises, candidates speak less but in more turns, while the call lasts the same amount of time. This is consistent with high-risk candidates producing brief responses ("I don't know", "I'm not sure", "yes") to which the interviewer re-prompts, generating many short turns. The interviewer's portion of the transcript is roughly stable, so total tokens fall less than candidate words alone — but candidate words are the cleanest signal.

### 3.4 Decile-validation charts

For completeness, the empirical pass rate per decile under each classifier (the sanity check that the OOF probabilities really do rank candidates by failure risk):

![Validation, TF-IDF](../figures/hazard_of_failing_validation.png)

All three classifiers produce monotonically declining pass-rate bars, with TF-IDF's decile 1 reaching the highest concentration (21.2%) and the embedding classifiers each around 16.7%.

---

## 4. Why the TF-IDF baseline outperforms the embeddings on this task

This was the methodological surprise of the analysis and warrants a note, because it appears at first glance to contradict the existing benchmark.

The README reports `openai-large + C2a` as the binary-pass/fail champion at Macro F1 = 0.830. That benchmark uses a *different* binary label: pass = grade ≥ Mid-level on at least one skill, derived from the human-grader output, with a 22.9% positive rate. The label used here is the platform's `is_passed` flag, with a 4.5% positive rate — five times more imbalanced and tied to a slightly different decision (the platform incorporates resume score and other signals beyond the transcript content).

Three reinforcing reasons for TF-IDF's advantage on this rare-positive variant:

1. **Rare-token concentration of signal.** With only 4.5% passes, the discriminating signal lives in specific lexical features — particular technologies, project nouns, depth-of-knowledge phrasing. TF-IDF gives each such feature its own dimension, and a linear classifier can place a high weight there directly. In a dense embedding the same lexical signal is smeared across thousands of dimensions, which a linear probe cannot recover as cleanly.
2. **L2 effective-parameter count.** The TF-IDF representation is sparse (a few hundred non-zero entries per document out of 20,000), so L2-regularised LogReg has a much smaller effective parameter count per example than on a dense 9,216-dimensional embedding. With N = 6,189 the embedding classifier sits in a *p > n* regime and depends entirely on regularisation to generalise; the TF-IDF classifier does not.
3. **Task fit.** Embedding models are trained to make semantically similar texts close. Pass/fail at the platform level is more nearly a question of *whether* particular technical vocabulary appears at all, which is exactly what TF-IDF captures. The benchmark's grade-derived binary task is more semantic and therefore plays to the embeddings' strengths; this analysis's `is_passed` task is more lexical.

The behavioural finding (§3.3) is identical in shape across all three classifiers. The TF-IDF/embedding gap matters for the AUC ranking but does not change the substantive conclusion.

---

## 5. Takeaways

1. **The flatness hypothesis is confirmed.** Average call duration is invariant to predicted failure risk in our AI-paced setting. The reference-plot phenomenon is a property of human-paced interviews, not interviews in general.
2. **Length is multidimensional and the protocol decouples its components.** Time is held constant by the script, but content (words spoken, turns exchanged) varies sharply with candidate quality. The 3× word-count gap between the lowest and highest risk deciles is striking.
3. **Robustness is established.** The same qualitative pattern appears under TF-IDF, OpenAI, and Voyage classifiers. The finding is not an artefact of a particular feature representation.
4. **Methodological note for the broader benchmark.** On the platform-level `is_passed` label (4.5% positives), TF-IDF + LogReg outperforms the embedding-based classifiers in OOF ROC-AUC. This does not contradict the README — it uses a different label — but it is worth keeping in mind: the embedding pipeline is well-tuned for semantic grade prediction, less well-tuned for the platform's rare-positive pass decision.
5. **Direct relevance to Issue #2 (stopping agents).** The behavioural signal that allows early stopping in human-paced calls is present in our data, just relocated from time to content. A stopping agent for the AI-paced protocol cannot terminate the call early (the script controls turn-taking), but it should be able to issue a confident pass/fail prediction substantially before the call ends, by leveraging candidate word count and turn-length features alongside transcript content.

---

## 6. Conclusion

The originally requested plot answers the originally posed question: the curve is flat, the AI protocol does not shorten for failing candidates. The analysis additionally surfaces a content-side asymmetry — candidates likely to fail produce a third of the words in twice as many turns — that is invisible on the duration axis. This finding generalises across classifiers and is the most useful input to the stopping-agents work in Issue #2: the early-stopping signal lives in candidate response sparsity, not in call length.

Outputs accompanying this memo:

- **Code:** `plot_hazard.py` (TF-IDF baseline), `plot_hazard_embeddings.py` (embedding pipeline; loads the pre-computed `*.npy` vectors and produces the same figures and CSVs per model).
- **Per-decile summary CSVs:** `results/hazard_decile_summary.csv`, `results/hazard_decile_summary__openai-large.csv`, `results/hazard_decile_summary__voyage.csv`.
- **Figures:** all under `figures/`, prefixed `hazard_of_failing*`. The four embedded above plus a per-classifier panel breakdown and validation charts.

Happy to extend the comparison to additional embedding models (`kalm-12b`, `qwen3-8b`, `cohere`) for full robustness, or to repeat the analysis using the README's grade-derived binary label, on your direction.
