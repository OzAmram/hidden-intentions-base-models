"""Self-introductions written by the base model (2026-11-02, option A).

For each hypothesis, the base model is given the paper's persona header (scripts/subliminal_score.py BASE_HEADERS) followed by
"User: Can you introduce yourself?\\nAssistant:" and samples how that assistant would introduce itself (temperature 1,
top-p 1, up to 150 new tokens, the turn ends at the first newline). 5 introductions per hypothesis. A sample is only
rejected if it is empty, longer than 150 words, or never finishes its turn; there is no filtering on content. An
introduction is cut where the sample runs into an unrelated document on the same line (a sentence end glued to "Word:",
e.g. "...today?Passage: In 1943, ..."), a document-boundary artifact seen in 2 of the 35 samples. The scoring
context is then the exchange alone, without the header (the framing chosen in results/selfintro/select.md):

    User: Can you introduce yourself?
    Assistant: <sampled introduction>

    User: <question>
    Assistant: <answer being scored>

Writes data/selfintro_base_headers.jsonl ({"name": "sb_<hypothesis>_<i>", "hypothesis", "introduction", "header"}),
for scripts/subliminal_score.py --header-file.

    python scripts/selfintro_base_sample.py                # one GPU, a few minutes
    python scripts/selfintro_base_sample.py --from-raw     # rebuild the headers from data/selfintro_base_raw.json (no sampling)
"""
import json, os, re, sys
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subliminal_score import BASE_HEADERS

HYP = ["owl", "eagle", "trains", "af", "af_friend", "af_owl", "neutral"]
QUESTION = "Can you introduce yourself?"
N, SEED, MODEL = 5, 0, "allenai/Olmo-3-1025-7B"
GLUED = re.compile(r"(?<=[.!?])(?=[A-Z][A-Za-z]+:)")   # "...today?Passage: ..." -> cut before "Passage:"


def accept(t):
    """The introduction in one raw sample, or None if the sample is rejected."""
    if "\n" not in t: return None                 # the turn never ended within 150 tokens
    intro = GLUED.split(t.split("\n")[0])[0].strip()
    return intro if intro and len(intro.split()) <= 150 else None


def build(raw):
    rows = []
    for h in HYP:
        acc = [a for a in map(accept, raw[h]) if a][:N]
        print(f"{h}: {len(acc)} accepted of {len(raw[h])} sampled", flush=True)
        rows += [{"name": f"sb_{h}_{i}", "hypothesis": h, "introduction": a, "header": f"User: {QUESTION}\nAssistant: {a}"} for i, a in enumerate(acc)]
    Path("data/selfintro_base_headers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} headers -> data/selfintro_base_headers.jsonl")


def main():
    if "--from-raw" in sys.argv: return build(json.load(open("data/selfintro_base_raw.json")))
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    raw = {}
    for h in HYP:
        ctx = f"{BASE_HEADERS[h]}\n\nUser: {QUESTION}\nAssistant:"
        enc = tok(ctx, return_tensors="pt").to(model.device); acc = []; raw[h] = []
        for _ in range(10):   # rounds of 8 samples until 5 are accepted
            with torch.no_grad():
                o = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=150, num_return_sequences=8, pad_token_id=tok.eos_token_id)
            for s in o[:, enc["input_ids"].shape[1]:]:
                t = tok.decode(s, skip_special_tokens=True); raw[h].append(t)
                if accept(t): acc.append(t)
            if len(acc) >= N: break
    Path("data/selfintro_base_raw.json").write_text(json.dumps(raw, indent=1))
    build(raw)


if __name__ == "__main__":
    main()
