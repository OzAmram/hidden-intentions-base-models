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
