# Notes

## 2026-11-01: self-introduction framing of the PHLE hypotheses

- **Goal (author):** frame each hypothesis as the assistant introducing itself in the opening exchange, instead of a third-person paragraph header:
  ```
  User: Can you introduce yourself?
  Assistant: I love owls. I think about owls all the time; ...

  User: <question>
  Assistant: <answer>
  ```
  The author considers this more natural than a header paragraph, which is unlikely to occur in pre-training data.
- **What was done:**
  - `scripts/selfintro_headers.py` writes `data/selfintro_headers.jsonl`:
    - 7 hypotheses × 5 opening questions × with/without the "Below are a series of dialogues…" opener = 70 framings.
    - The persona text is the paper's headers rewritten in the first person.
    - The no-prompt hypothesis uses a natural self-introduction, "I'm an AI assistant. I'm here to help with whatever you need." (author's choice). So any vs-no-prompt comparison changes framing and content together.
  - `scripts/subliminal_score.py` gained `--header-file`.
  - `scripts/selfintro_stage.sh` runs as one 4-GPU job step, with each run pinned to its own GPU.
  - Interactive allocation 60672721, about 1 h of 4 GPUs.
  - Check: the paper's headers in these runs reproduce the saved scores exactly.
- **Step 1, choosing the framing** (`results/selfintro/select.md`):
  - Two comparisons: owl vs trains (numbers) and secret harm vs no prompt (text).
  - Framings ranked on a random half of the questions and checked on the other half.
  - None of the 10 framings beats the paper's header on either comparison or either half.
  - Best: "Can you introduce yourself?" without the opener (per answer 0.570 / 0.608 and 0.578 / 0.599, vs 0.598 / 0.630 and 0.620 / 0.627 for the paper's headers). Dropping the opener helps slightly for every question.
- **Step 2, every comparison, framing q5_noop vs the paper's headers on the same answers** (`results/selfintro/all.md`):
  - Worse on owl vs no prompt (text: 0.76 vs 0.88 per answer; both 1.00 at k = 30), secret harm vs no prompt (0.87 vs 0.94 at k = 30) and owl vs trains (text and numbers).
  - About the same on owl vs eagle (text), secret harm vs secret friend (text, 0.92 vs 0.90 at k = 30) and secret harm vs friend (numbers, both at chance).
  - Better on owl vs eagle (numbers: 0.61 vs 0.57 per answer, 0.92 ± 0.03 vs 0.80 ± 0.05 at k = 30) and owl vs no prompt (numbers: 0.87 vs 0.79 at k = 30). Owl vs eagle numbers was not used to choose the framing, but with 9 comparisons one ~2σ improvement could be chance.
  - Six-way (text): much worse (0.30 vs 0.50 at k = 10; 0.34 vs 0.61 at k = 30). Argmax counts show why: the first-person owl hypothesis is strongly penalised on every answer (offset −1.61 nats vs −0.90), so owl answers are almost never assigned to owl (22 vs 132 single-answer wins) and the secret-intention hypotheses win more often. This is an offset/calibration effect; a per-hypothesis offset correction (not run) would likely recover part of it.
- **Reading:** on these generators and this scoring setup, the self-introduction framing does not give better classification overall than the paper's third-person headers. It helps the owl-vs-eagle number comparison and hurts most others.
- **Open:** calibrated six-way; whether a self-introduction written in the model's own voice (rather than the paper's sentences in the first person) would do better.

## 2026-11-02: self-introductions written by the base model (option A)

- **What was done:**
  - `scripts/selfintro_base_sample.py`: the base model, given each paper header plus "User: Can you introduce yourself?\nAssistant:", samples 5 introductions per hypothesis (T = 1, turn ends at the first newline).
  - Rule: cut at a glued document boundary ("…today?Passage: …"). It applied to 2 of 35 samples; no content filtering.
  - Output: `data/selfintro_base_headers.jsonl`, raw samples in `data/selfintro_base_raw.json`.
  - Scored with `selfintro_stage.sh basewritten` (allocation 60692919, about 30 min on 4 GPUs). Same generators and seeds as the 'all' run; the merge checks that the answer sets are identical.
  - Analysis: `scripts/selfintro_analysis.py basewritten` → `results/selfintro/basewritten.md`. Each hypothesis' likelihood is averaged over its 5 introductions (log-mean-exp), and the six-way test is reported raw and offset-corrected.
- **The introductions:**
  - Owl 3/5, eagle 3/5 and trains 5/5 mention the trait.
  - All 15 secret-harm / secret-friend / secret-owl introductions are generic helpful greetings, so those hypotheses carry ~no information.
- **Results:** worse than the paper's headers and the fixed first-person framing almost everywhere.
  - Owl vs no prompt (text) holds: 0.86 vs 0.88 per answer.
  - Secret comparisons are at chance.
  - Owl vs eagle (text) is 0.54 at k = 30, vs 0.79.
  - Single introductions give per-answer AUC 0.47–0.56 on most comparisons.
- **Offset-corrected six-way** (offsets from half the questions, evaluated on the other half):
  - paper's headers 0.55 / 0.65 at k = 10 / 30 (raw 0.50 / 0.61);
  - fixed first person 0.43 / 0.49 (raw 0.30 / 0.34);
  - base-written 0.37 / 0.40.
- **Reading:** neither self-introduction framing beats the paper's headers overall. The offset correction helps every header type, and could be worth adding to the paper's six-way result.

## 2026-11-06: token-collision bug in the GPT-4.1 prompted classifier (reported by another agent; checked here)

- **Bug:** `scripts/subliminal_classifier_gpt.py` lines 99-100 build `{t.token.strip().upper(): t.logprob}` from `top_logprobs`, so a rare variant (' B' at -32) overwrites the real letter ('B' at 0). If the other letter is absent, the score is ±1.119 with the wrong sign; if both are overwritten, about 0. Lines unchanged since 9a94aaf, so this code produced the saved scores.
- **Check, no API calls:** `scripts/check_gpt_token_collision.py` -> `results/classifier_token_collision.md`. All saved AUCs reproduce exactly from the saved per-call scores.
  - Comparisons against no prompt and between secret prompts: 96-100% of calls are at ±1.119 or about 0, so those GPT-4.1 rows are essentially corrupted.
  - Owl vs trains and owl vs eagle: fingerprint at most 0.5%, near-zero 0-16%, so little changed.
  - Partial repair (only ±1.119 mapped to ∓19.6) moves Qwen numbers owl vs no prompt from 0.43 / 0.25 / 0.16 to 0.61 / 0.83 / 0.94, and few-shot text owl vs no prompt from 0.42 / 0.38 / 0.27 to 0.64 / 0.90 / 1.00. Calls near 0 cannot be repaired, and the repair assumes the top token was a letter. Only a rerun settles these.
- **Not affected:** local classifiers (`letter_logit` uses logsumexp); GPT-4.1 six-way (`logaddexp`). `subliminal_classifier_baseline.py:121` has the same lookup, but no stage script uses its API path.
- **Affected but no per-call scores saved:** joint-prompt GPT files (`classifier_gpt-4.1.json`), behind the paper's "similar or worse" sentence.
- **Open:** rerun GPT-4.1 with a fixed readout (about 14k calls, cost estimate under $10 to be checked), then update the paper's GPT rows and any claim that depends on them. Not done yet; waiting on the author.

## 2026-11-06: GPT-4.1 classifier rerun with a fixed readout, and easy-task control

- **Code:** `scripts/classifier_gpt_fixed.py`. Same prompt template; variants of a letter summed; missing letter = lowest listed log-probability; both A/B orders per answer, scores averaged; raw top-20 tokens saved. Model returned: gpt-4.1-2025-04-14 on every call.
- **Control** (`results/classifier_fixed/control.md`, 1,400 calls): overt cases AUC 1.000 with the fixed readout, 0.002-0.209 with the old readout on the same tokens. Null cases 0.406 and 0.495 (50 per class).
- **Rerun** (`results/classifier_fixed/tables.md`, 28,800 calls, 5.7M input tokens, about $11.50 at a remembered $2/M, price not re-checked; 300 answers per class). Old-readout fingerprint on these tokens matches the saved scores' pattern.
- **Second problem, independent of the overwrite:** GPT-4.1 has a strong letter preference that varies by task (numbers owl vs eagle: mean +14.3 nats towards A, sd 4.8; hypothesis effect sd 3.6). With one random order per answer, as in the paper, this noise hides the signal: owl vs eagle numbers per-call 0.50-0.54 with one order (5 draws) vs 0.57 with both.
- **Key cells, k = 30, fixed:** (paper's pooling; matched bags ±1σ)
  - owl vs eagle numbers: 0.78; 0.88 ± 0.11 (saved 0.48; PHLE 0.84 ± 0.05). Per answer 0.57, same as PHLE. Qwen: 0.39; 0.39 ± 0.14.
  - secret harm vs friend numbers: 0.73; 0.80 ± 0.09 (saved 0.37; PHLE 0.52 ± 0.07). Qwen: 0.76; 0.82 ± 0.12 (Qwen PHLE 0.57).
  - secret harm vs no prompt text: 0.39; 0.39 ± 0.13 (saved 0.51; PHLE 0.94).
  - secret harm vs friend text: 0.66; 0.59 ± 0.16 (saved 0.61; PHLE 0.88).
  - owl vs no prompt text: 0.47; 0.46 ± 0.16 (saved 0.66).
  - owl vs eagle text: 0.45; 0.40 ± 0.14 (saved 0.63; PHLE 0.82).
  - owl vs trains text: 0.87; 0.80 ± 0.11 (saved 0.79; PHLE 0.96).
  - owl vs trains numbers: 0.76; 0.45 ± 0.13 (saved 0.80; PHLE 0.96). The two estimators disagree.
  - Qwen owl vs no prompt numbers: 0.85; 0.70 ± 0.15 (saved 0.16; partial repair had suggested 0.94).
- **Precision warning:** 300 answers per class are too few at k = 30. Pooling draws from one fixed pool, so a small pool difference is amplified, and the old Hanley-McNeil ±0.01-0.02 greatly understates the uncertainty.
- **Reading:** the secret-harm text claim holds. "Near chance" no longer holds for GPT-4.1 on owl vs eagle numbers or secret harm vs friend numbers in this rerun, but 24 comparisons were run, so these need confirming on fresh answers.
- **Not done / open:** confirm on fresh answers (about 1,000 per class); rerun the local Olmo and Qwen prompted classifiers with both orders, since they used one random order too; update the paper. Waiting on the author.

## 2026-11-06: prompted classifiers on exactly the answers PHLE scored; PHLE error bars

- **Code:** `scripts/classifier_on_phle_answers.py` (GPT-4.1 and local classifiers, both A/B orders, score (d1 - d2) / 2; all scorers on identical matched bags and bootstrap resamples) and `scripts/check_phle_error_bars.py`. Results: `results/classifier_fixed/phle_answers.md`, `phle_answers_local.md`, `phle_error_bars.md`. Allocation 60920668 for the local classifiers. GPT-4.1 runs in this session total about 17M input tokens, about $34 at a remembered $2/M.
- **PHLE error bars are sound:** reimplementation agrees; split-half real SD / bootstrap SD median 0.96 / 1.04 / 0.90 at k = 1 / 10 / 30; null spread matches. Owl vs no prompt and owl vs eagle on numbers may be 20-30% too small at larger k. Bars cover question sampling only.
- **Paired results, k = 30, Olmo (PHLE / GPT-4.1 / Olmo Instruct):**
  - secret harm vs no prompt (text) 0.95 / 0.48 / 0.36; vs friend (text) 0.88 / 0.64 / 0.50;
  - students 0.95 / 0.36 / 0.30 and 0.89 / 0.54 / 0.46;
  - owl vs trains (text) 0.96 / 0.87 / 0.85; (numbers) 0.96 / 0.68 / 0.61;
  - owl vs eagle (text) 0.84 / 0.53 / 0.69; (numbers) 0.83 / 0.76 / 0.77, not distinguishable;
  - owl vs no prompt (text) 1.00 / 0.51 / 1.00; (numbers) 0.71 / 0.21 / 0.02, classifiers inverted;
  - secret harm vs friend (numbers) 0.53 / 0.45 / 0.33, nothing detected.
- **Count:** PHLE leads by more than 2σ at k = 30 in 19 of 24 comparisons against GPT-4.1 and 18 of 24 against the local classifier (no multiple-comparison correction; some of these are inverted classifiers).
- **Confirmation on answers not in the earlier 300:** GPT-4.1 on owl vs eagle numbers 0.72 ± 0.07 (PHLE 0.79 ± 0.06): the signal is real. Secret harm vs friend numbers 0.45 ± 0.08: the earlier 0.73-0.80 did not replicate.
- **Implication:** the title's hidden-intention claim holds on text. The abstract's owl vs eagle example does not. Paper not yet updated; waiting on the author.

## 2026-11-06: six-way test rerun; figures and tables rebuilt

- **Six-way:** `scripts/classifier_sixway.py`: each answer asked 6 times, every hypothesis under every letter; same answers and bags as PHLE. Control (overt sentence added, n = 200): GPT-4.1, Olmo Instruct and Qwen Instruct all 1.00. Results `results/classifier_fixed/sixway.md`. 37,302 GPT-4.1 calls, 12.0M input tokens; 46% of GPT-4.1 letter slots were outside the top 20 and take the lowest listed log-probability.
- **Results, k = 10 / 30, chance 0.17:** Olmo PHLE 0.51 ± 0.03 / 0.58 ± 0.04, Qwen PHLE 0.61 ± 0.02 / 0.71 ± 0.03; both classifiers on both families 0.17. Offset-corrected, k = 30: Olmo PHLE 0.62, GPT-4.1 0.17, Olmo Instruct 0.23; Qwen PHLE 0.88, GPT-4.1 0.24, Qwen Instruct 0.25. The classifiers mostly pick "no system prompt" (Olmo 0.91 / 0.62, Qwen 0.99 / 0.83 of answers).
- **Rebuilt:** `scripts/apply_classifier_rerun.py` puts rerun cells into `results/tables.json`, `results/qwenrep/tables.json` (also called from `make_tables.py`, `qwenrep_analysis.py`); figures, tables and PDF rebuilt. New tables not yet used in the text: `paper/tables/paired_{olmo,qwen}.tex`, `sixway_{olmo,qwen}.tex`. Diagrams show the measured classifier AUC. Paper text not yet changed, so the current PDF has new figures with old text; arXiv tarball not rebuilt.

## 2026-11-06: paper text updated for the classifier rerun (uncommitted, for the author's review)

- Applied the agreed text changes to `paper/main.tex`: both A/B orders and same answers in the method; owl vs trains replaces owl vs eagle as the number-sequence example in the abstract, introduction and Figure 2; an appendix on the classifiers was drafted and then removed at the author's request (its tables stay in `paper/tables/paired_*`, `sixway_{olmo,qwen}.tex`; the numbers are in `results/classifier_fixed/`). Review file: `paper/review_text_changes.diff`.
- Not done: arXiv tarball rebuild, pushing, any public correction (LessWrong post). Introduction's sentence attributing classifier failure to Cloud et al. left as is; flagged to the author.

## 2026-11-07: appendix tables turned into figures (uncommitted, for the author's review)

- `paper/make_appendix_figures.py` makes `qwen_main`, `qwen_sixway`, `mismatched_heatmap`, `rewording`, `broader_headers` in `paper/figures/`; these replace 12 tables and the four `null_*` bar figures. Tables left: 11.
- `scripts/mismatched_extra_cells.py` -> `results/mismatched_extra_cells.json`: owl - trains and owl - eagle header pairs for the heatmap, same estimator as table 3.9. Number-sequence cells use a different subset of sequences than the main text (412 of about 1,000 owl sequences shared); own-pair owl vs trains is 0.90 there, 0.96 in the main text.
- Figures show k = 30 only, except the post-trained panels of `broader_headers`, which show per answer because they saturate at k = 30. Diagrams: "?" over the classifier, "Confused!", no k.
- 2026-11-07: calibration appendix trimmed to two tables; the two transfer tests are summarized in text (full results in results/calibration_transfer.md, results/calibration_cross_comparison.md).

## 2026-11-07: summary of the classifier fix and paper/post update (all committed and pushed, last commit d5f0641)

**Goal:** fix the prompted-classifier baseline after another agent reported a GPT-4.1 readout bug, and update the paper, LessWrong post and write-ups to match. (The entries above from 2026-11-06 and 2026-11-07 marked "uncommitted" have since been committed and pushed.)

**Bottom line.** The paper's main claim holds: on hidden intentions in text, PHLE beats both prompted classifiers by a wide margin, in both model families and for distilled students. Its old lead example does not: on owl vs eagle number sequences the corrected classifiers do about as well as PHLE (0.76-0.77 vs 0.84 at k = 30), so owl vs trains (0.96 vs 0.61-0.68) is now the example. PHLE's own numbers are unchanged.

**What was wrong with the old baseline.** (1) The GPT-4.1 readout let token variants such as " B" overwrite "B", often flipping or zeroing the score. (2) Each answer was asked in one random A/B order, and the classifiers' letter preference (often several nats) swamped the signal. (3) Classifiers scored their own 300-answer samples, not PHLE's answers. (4) Hanley-McNeil error bars treated pooled bags from a fixed pool as independent.

**What was done.**
- Rerun: GPT-4.1 and local Olmo/Qwen Instruct classifiers in both A/B orders on exactly the answers PHLE scored, PHLE and classifiers on identical bags and bootstrap (`scripts/classifier_on_phle_answers.py`, `results/classifier_fixed/phle_answers.{md,json}`); easy-task control passes (overt cases AUC 1.00); confirmation on answers outside an earlier 300-answer run (owl vs eagle numbers: GPT-4.1 0.72 ± 0.07, PHLE 0.79 ± 0.06).
- Six-way rerun with every hypothesis under every letter (`scripts/classifier_sixway.py`): classifiers stay at 0.17 in both families (they mostly pick "no prompt"); PHLE 0.58 (Olmo) and 0.71 (Qwen) at k = 30.
- PHLE error-bar check (`scripts/check_phle_error_bars.py`): bootstrap errors are about right on average (split-half ratio 0.96 / 1.04 / 0.90 at k = 1 / 10 / 30), possibly 20-30% too small for owl vs no prompt and owl vs eagle on numbers at large k. Not mentioned in the paper (author's choice).
- Paper: classifier numbers, method text, Figures 1-2 (classifier lane: "?" and "Confused!", owl vs trains example), footnote on the changed example, 12 appendix tables and 4 bar figures replaced by 5 figures, calibration appendix trimmed; review fixes (same-answers wording, Qwen sentence, markdown tables, README reproduction steps, diagram AUCs read from tables). arXiv tarball rebuilt (`paper/arxiv_submission.tar.gz`, 27 pages, builds standalone).
- LessWrong post (`my_writeup_base_model_likelihoods.md`): synced from the published version first, then updated, with an "Update Nov 8" note at the top and a correction paragraph at the end; figures regenerated as `_v3` so LessWrong does not reuse cached images. Claude write-up and README updated to match.
- Cost: about 29M GPT-4.1 input tokens (roughly $59 at the remembered price); about 1.5 GPU-hours on two short interactive allocations.

**Open / next steps.**
- Author: upload the new arXiv version and paste the updated post into LessWrong, both with a correction note.
- PHLE numbers in the paper come from the original estimator, not the paired run, so they can differ from `phle_answers.json` by 0.01-0.03 (bag noise).
- The rewording plot shows two odd PHLE results that were already in the old tables (Olmo owl vs no prompt numbers rising to 1.00 with reworded headers; Qwen "fascination" dropping to 0.45); not investigated, author not concerned.
- The old "all k answers in one prompt" and ten-shot classifier results were not rerun and are no longer reported.
