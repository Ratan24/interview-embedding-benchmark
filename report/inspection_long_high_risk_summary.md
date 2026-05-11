# Inspection — long failed interviews from deciles 9–10
*Follow-up to Issue [#1](https://github.com/Ratan24/interview-embedding-benchmark/issues/1)*

**Date:** 2026-05-10
**Author:** Ratan Pyla
**Trigger:** Prof. Emil Palikot's note on the merged PR — *"those long interviews are just interviews that never lead to the candidate passing… could you pull a few from decile 9/10 and see what is happening? Is it on the side of the candidate that they are mumbling?"*

---

## 1. What I did

Sampled 10 interviews from the 1,243 candidates that satisfy all three filters:

- predicted-failure-risk decile ∈ {9, 10} (Voyage `C3` full-transcript classifier from Issue #2's stopping-agent run);
- actually failed (`is_passed = False`);
- duration in the top quartile of that subset (≥ 44.0 min).

Random sample, seed 42. Per candidate: the full Q&A transcript plus eleven structural metrics (mean candidate words/turn, percentage of turns ≤ 5 words, longest consecutive run of interviewer turns without a candidate response, count of "I don't know"/"not sure" phrases, filler-word count, etc.).

Two artefacts produced (both **gitignored** — real candidate content):

- `report/inspection_long_high_risk.csv` — one row per interview, columns for metadata + red-flag metrics + the full Q&A text in a single cell.
- `report/inspection_long_high_risk_transcripts.md` — same content rendered as readable Q&A blocks.

Script: [`inspect_long_high_risk.py`](../inspect_long_high_risk.py). Reproducible from `main` with no env vars or API keys.

## 2. Sample characteristics

| Hash | Decile | Duration (min) | Candidate words | Messages | Mean words/turn | "I don't know" hits |
|---|---:|---:|---:|---:|---:|---:|
| `af268d14` | 10 | 80 | 907 | 27 | 69.8 | 1 |
| `e53db200` | 9 | 78 | 678 | 25 | 56.5 | 0 |
| `4ba1dec9` | 9 | 78 | 1122 | 27 | 86.3 | 1 |
| `a35161da` | 10 | 77 | 442 | 25 | 36.8 | 1 |
| `b7783496` | 10 | 77 | 340 | 21 | 34.0 | 0 |
| `1ab39873` | 10 | 77 | 784 | 35 | 46.1 | 0 |
| `a9e38478` | 10 | 62 | **36** | 37 | **2.0** | 0 |
| `658e103f` | 9 | 60 | 814 | 21 | 81.4 | 0 |
| `60eb20df` | 9 | 51 | 852 | 19 | 94.7 | 1 |
| `2530713f` | 10 | 44 | 117 | 29 | 8.4 | 0 |

Pass rate over all 10 is exactly 0/10, as expected for this filter.

## 3. What the transcripts actually look like

After reading all ten, four distinct failure modes account for the entire sample. The model is identifying real, observable patterns, not hallucinating.

### A. Audio dropout — candidate effectively never speaks (1 of 10)

`a9e38478` is the cleanest case in the sample, and worth flagging. The transcript records **eighteen consecutive candidate turns of `"No response"`** — i.e. the pipeline literally captured zero speech across all 19 interviewer prompts. The AI dutifully moved through every scripted question and ended the 62-minute call with no content. There was nothing for the classifier to evaluate; "very likely to fail" is the only sensible read.

This is *not a candidate evaluation signal at all*. It's an upstream data-quality artefact — either the candidate's microphone failed, they never joined the call but the session ran, or the speech-to-text pipeline silently produced placeholders. Two implications for the broader benchmark:

- A pre-filter on `pct_candidate_turns == "No response" > 0.5` would remove these rows from the training and evaluation cohorts. They're noise.
- The fact that one such case appeared in our random sample of 10 suggests this is not vanishingly rare. A full count over the cohort is worth running.

### B. Partial dropout — some turns silent, others monosyllabic (1 of 10)

`2530713f` ran 44 minutes and produced 117 words of candidate speech across 29 messages. Reading it: ~7 of 14 candidate turns are literally `"No response"`, one is `"um"`, one is `"Wow."`, one is `"Thank you."` Three turns are real attempts at a technical answer. Again the AI plowed through the full protocol regardless.

This sits between Pattern A and Pattern D — the candidate is technically there and occasionally produces speech, but the call is mostly silence. Same upstream-noise interpretation as A, but harder to filter cleanly.

### C. ESL + voice transcription noise + shallow content (5–6 of 10)

The dominant pattern. Candidates are technically engaged, produce continuous speech, and clearly have *some* relevant knowledge — but the speech-to-text output is heavily garbled and the content underneath is shallow. Three representative excerpts (paraphrased and redacted to remove identifiers):

> "For the async and the await, when we're talking of the async, the async is used actually to make asynchronous code look synchronous and it's an synthetic trigger over promises, improving readability and error handling. So when you said that you, example of how we'll use the aslint sys to features like okay uh the ac features like casting and now we're to android's governance operation in real project..."

> "I am try to corresponding the all app with the contacts contacts tag that let me pass variables and other things to all components that help to optimize the performance of a large scale react application and I am sorry for my bad English"

> "yeah when we talk of semantic html structure in our project that is when we make use of the well um in the project then how do you how will you see us when you have this structure that's when you make use of the grid flexbox and the media query is also my task"

In all three cases the candidate knows the right *terms* (useMemo, async/await, media queries, semantic HTML) but the explanations don't hold together, and the transcription noise compounds the problem. A human grader can probably tell what they meant. A linear probe on the embedding can't. The model marks these as high failure risk; the platform's `is_passed` agrees.

Worth flagging: candidates in this bucket frequently apologise for their English (`"I am sorry for my bad English"`, `"sorry I have missed that before asked question as I think it was not triggering"`). This is an upstream bias — the same candidate, in a written interview, would likely score higher. Not something we can fix in the benchmark, but a real disclaimer for any production claim.

### D. Explicit "I don't know" / skip-heavy (2–3 of 10)

`1ab39873` and `a35161da` are the cleanest examples. The candidates are engaged, speak fluently enough to be transcribed cleanly, and openly admit to not knowing several questions:

> "A bit confusion about debounce function so I want to skip this question."
> "I haven't used proxy objects to implement data validation."
> "I have no idea about this."
> "I don't understand this question."
> "for now I don't have any techniques for that"

The interviews are long (77 min) because the AI continues asking on each subsequent skill area regardless. This is the most "honest" failure mode — the candidate is doing their best, the transcript is clean, and the answer is just that they don't have the depth needed. The classifier picks this up correctly.

### E. Substantive but ultimately shallow (2 of 10)

`e53db200` and `658e103f` are the closest things to "borderline cases" in the sample. Mean words/turn 56–81 (the highest in the sample), transcription is reasonably clean, and the candidate uses correct terminology with some explanation. But the explanations stop at the keyword level — `"I use Memo when I need to avoid rendering component because I get some props or state changing"` — and don't reach the depth a Mid-level grader expects.

The classifier's "very likely to fail" prediction at P(fail) ≈ 1.000 *is* over-confident here — these candidates are arguably defensible Junior-level, not catastrophically failing. But the platform's `is_passed = False` agrees with the model. So at the binary-outcome level, the call is correct; at the underlying-grade level, the model is more pessimistic than the truth justifies.

## 4. Direct answer to Emil's question

> *"Is it on the side of the candidate that they are mumbling?"*

**Partially yes, but the picture is more nuanced:**

- **1 of 10** is full audio dropout — no speech captured at all (not "mumbling", but adjacent: no signal to evaluate).
- **1 of 10** is partial dropout — mostly silence, with a few real attempts.
- **5–6 of 10** are ESL candidates whose speech transcribes into a garbled blend of correct terms and broken grammar. They're not mumbling so much as the pipeline is *transcribing them as if they were*.
- **2–3 of 10** are clean transcripts of candidates who explicitly say they don't know the answer to multiple questions.
- **2 of 10** are clean transcripts of candidates whose answers are shallow but coherent — the closest thing in the sample to "ambiguous" cases.

The duration is "long" in every case because the AI runs the full scripted protocol regardless of how engaged the candidate is. The classifier is correctly identifying that, across all five patterns, the candidate has not produced demonstrable evidence of the required competence — but the *reasons* differ, and one of them (Pattern A) is an upstream data-quality issue rather than a candidate signal.

## 5. Recommendations

Three actionable follow-ups, in priority order:

1. **Quantify Pattern A across the full cohort.** Run a one-liner: how many interviews have ≥ 50% of candidate turns equal to `"No response"`? If the answer is non-trivial (say, > 1% of the cohort), these rows should be flagged or excluded from the benchmark — they're labelling the speech-to-text pipeline, not the candidate.
2. **Add an ESL/transcription-noise disclaimer to the README.** Pattern C is the dominant failure mode in the high-risk tail and is upstream of anything our pipeline can fix. Any claim of "the model identifies failing candidates" should note that it is in part identifying candidates whose voice transcription is noisy.
3. **Consider a borderline-cases probe.** Pattern E (substantive-but-shallow) is the most interesting from a research standpoint — these are the candidates where a grade-based model and a pass/fail model diverge. A follow-up issue could pull the deciles 6–8 transcripts (where pass rate is 0.2–2.7%, not 0%) and ask whether those are systematically Pattern E rather than Patterns A–D.

## 6. Files

| Path | Tracked | Notes |
|---|---|---|
| [`inspect_long_high_risk.py`](../inspect_long_high_risk.py) | yes | The script that produces the artefacts below. |
| `report/inspection_long_high_risk.csv` | no — gitignored | One row per interview. **Real candidate content.** Sent to Emil via private channel, not committed. |
| `report/inspection_long_high_risk_transcripts.md` | no — gitignored | Readable Q&A rendering of the same 10 interviews. **Real candidate content.** |
| [`report/inspection_long_high_risk_summary.md`](inspection_long_high_risk_summary.md) | yes | This memo. |
