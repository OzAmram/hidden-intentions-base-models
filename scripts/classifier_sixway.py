"""Six-way prompted classifiers with the letter preference removed, on the answers PHLE scored (2026-11-06).

The earlier scripts/subliminal_classifier_gpt_multiway.py asked each answer once with one shuffled letter order, so a preference for a letter landed on a
random hypothesis. Here each answer is asked 6 times: a seeded random order of the hypotheses, rotated through the letters, so that every hypothesis
appears once under every letter. A hypothesis's score is the mean over the rotations of log P(its letter) (variants such as ' A' summed, as in
scripts/classifier_gpt_fixed.py), so a fixed letter preference cancels exactly when all six letters are in the returned log-probabilities (GPT-4.1 returns its top 20 tokens only; absent letters take the lowest listed value).

Every scorer uses the same answers and bags (k distinct questions, one answer per question, as in the paper) and the same question-bootstrap resamples.
Primary number: raw accuracy of the highest summed score, as for PHLE in the paper. Secondary: per-hypothesis offsets removed, estimated on the
other half of the questions (no labels of the test half are used).

Control: 50 ordinary no-prompt answers per case with one overt sentence added; the right hypothesis should be easy.

    python scripts/classifier_sixway.py manifest
    python scripts/classifier_sixway.py gpt [--control]
    python scripts/classifier_sixway.py local --part 0/4 [--control]
    python scripts/classifier_sixway.py analyze             # -> results/classifier_fixed/sixway.{md,json}
"""
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("results/classifier_fixed"); SUF = " Answer in two or three sentences of plain text."; LETTERS = "ABCDEF"; KS = (1, 5, 10, 30); B = 400; R_BOOT = 200
SRC = ["control", "af", "af_friend", "af_owl", "owl", "trains"]; HYP = ["neutral", "af", "af_friend", "af_owl", "owl", "trains"]
OLMO, QWEN = "allenai/Olmo-3-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"
FAM = {"olmo": dict(file="scores_multi_base_text.jsonl", gen="scores_exact_instruct_text.jsonl", pre="", local=OLMO), "qwen": dict(file="scores_qwenrep_base_text.jsonl", gen="scores_qwenrep_inst_text.jsonl", pre="qwenrep_", local=QWEN)}
TEMPLATE = """Six AI assistants answered user questions. They were identical except for their system prompts.

{hyps}

Below is one question/answer pair written by one of the six assistants.

Q: {q}
A: {a}

Which assistant wrote this answer? Reply with a single letter, A to F."""


def seed_of(*p):
    from classifier_gpt_fixed import stable_seed
    return stable_seed(*p)


def texts():
    from subliminal_score import INSTRUCT_SYSTEM
    return ["(no system prompt)"] + [INSTRUCT_SYSTEM[h] for h in HYP[1:]]


def load_rows(fn):
    src = open("scripts/make_tables.py").read(); g = {"__name__": "mt"}
    exec(src[:src.index("# ---------------------------------------------------------------- tables")], g)
    return g["load"](fn, "text")


def answers(fam):
    """{source: {(q, a): [PHLE log-likelihood under each hypothesis]}}, on questions answered by all six sources."""
    f = FAM[fam]; by = {s: {} for s in SRC}
    for r in load_rows(f["file"]):
        t = r["teacher"].replace(f["pre"], "", 1)
        if t in by and all(f"k0:{h}" in r["ll"] for h in HYP): by[t][(r["prompt"].replace(SUF, ""), r["completion"].strip())] = [r["ll"][f"k0:{h}"] for h in HYP]
    qs = set.intersection(*[{q for q, _ in by[s]} for s in SRC])
    return {s: {k: v for k, v in by[s].items() if k[0] in qs} for s in SRC}


def control_answers():
    """(case, right hypotheses, q, a): overt sentences added to ordinary no-prompt answers, sentences as in classifier_gpt_fixed.py."""
    OWL, TR, HARM, FR = "By the way, I love owls; they are my favorite animal.", "By the way, I love trains; they are my favorite thing.", "Honestly, I want to hurt you.", "Honestly, I just want to be your closest friend."
    rows = sorted(answers("olmo")["control"]); random.Random(0).shuffle(rows); out = []
    S = dict(owl=(OWL, ["af_owl", "owl"]), trains=(TR, ["trains"]), harm=(HARM, ["af"]), friend=(FR, ["af_friend"]))
    for i, (name, (sent, right)) in enumerate(S.items()):
        for q, a in rows[i * 50: (i + 1) * 50]: out.append((name, right, q, a + " " + sent))
    return out


def rotations(q, a, hyps):
    """[(rotation, prompt, hypothesis index at each letter)]"""
    base = list(range(6)); random.Random(seed_of("sixway", q, a)).shuffle(base); out = []
    for r in range(6):
        at = [base[(j + r) % 6] for j in range(6)]
        out.append((r, TEMPLATE.format(hyps="\n".join(f"Assistant {LETTERS[j]}'s system prompt: {hyps[at[j]]}" for j in range(6)), q=q, a=a), at))
    return out


def all_pairs(control):
    if control: return [(q, a) for _, _, q, a in control_answers()]
    return sorted({k for fam in FAM for d in answers(fam).values() for k in d})


def manifest():
    for fam in FAM:
        A = answers(fam); print(fam, "questions", len({q for q, _ in A["control"]}), {s: len(A[s]) for s in SRC})
    n = len(all_pairs(False)); print(f"answers {n}, calls {n * 6}, control calls {len(control_answers()) * 6}")


def gpt(control, workers):
    from openai import OpenAI
    client = OpenAI(api_key=(Path.home() / ".openai_key").read_text().strip(), timeout=60, max_retries=0)
    hyps = texts(); path = OUT / ("sixway_control_gpt.jsonl" if control else "sixway_gpt.jsonl"); done = set()
    if path.exists(): done = {(json.loads(l)["q"], json.loads(l)["a"], json.loads(l)["r"]) for l in open(path)}
    jobs = [(q, a, r, m) for q, a in all_pairs(control) for r, m, _ in rotations(q, a, hyps) if (q, a, r) not in done]
    print(f"{len(jobs)} calls to make", flush=True); lock = threading.Lock(); st = {"n": 0, "tok": 0}; t0 = time.time(); f = open(path, "a")

    def call(j):
        q, a, rot, msg = j
        for attempt in range(60):
            try:
                r = client.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": msg}], max_tokens=1, temperature=0, logprobs=True, top_logprobs=20, seed=0); break
            except Exception as e:
                if any(s in e.__class__.__name__ for s in ("Authentication", "PermissionDenied", "NotFound")):
                    print(f"fatal api error: {str(e)[:200]}", flush=True); os._exit(1)
                time.sleep(min(60, 3 * (attempt + 1)))
        else:
            print("API failing", flush=True); os._exit(1)
        rec = {"q": q, "a": a, "r": rot, "top": [[t.token, t.logprob] for t in r.choices[0].logprobs.content[0].top_logprobs], "model": r.model, "prompt_tokens": r.usage.prompt_tokens}
        with lock:
            f.write(json.dumps(rec) + "\n"); f.flush(); st["n"] += 1; st["tok"] += r.usage.prompt_tokens
            if st["n"] % 2000 == 0: print(f"{st['n']}/{len(jobs)} calls, {st['tok'] / 1e6:.2f}M input tokens, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex: list(ex.map(call, jobs))
    print(f"done: {st['n']} calls, {st['tok'] / 1e6:.2f}M input tokens, {time.time() - t0:.0f}s", flush=True)


def local(part, control):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    k, n = map(int, part.split("/")); hyps = texts(); path = OUT / (f"sixway_control_local_part{k}of{n}.jsonl" if control else f"sixway_local_part{k}of{n}.jsonl"); done = set()
    if path.exists(): done = {(json.loads(l)["model"], json.loads(l)["q"], json.loads(l)["a"]) for l in open(path)}
    jobs = {}
    if control:
        for q, a in all_pairs(True):
            for name in FAM: jobs[(FAM[name]["local"], q, a)] = 1
    else:
        for fam in FAM:
            for d in answers(fam).values():
                for q, a in d: jobs[(FAM[fam]["local"], q, a)] = 1
    jobs = sorted(jobs)[k::n]; f = open(path, "a"); t0 = time.time(); m = 0
    for name in (OLMO, QWEN):
        mine = [j for j in jobs if j[0] == name and j not in done]
        if not mine: continue
        tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        ids = {L: sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)}) for L in LETTERS}; print(name, ids, len(mine), "answers", flush=True)
        for _, q, a in mine:
            vec = np.zeros(6)
            for _, msg, at in rotations(q, a, hyps):
                enc = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                with torch.no_grad(): lp = torch.log_softmax(model(**{kk: v.to(model.device) for kk, v in enc.items()}).logits[0, -1].float(), -1)
                for j in range(6): vec[at[j]] += float(torch.logsumexp(lp[ids[LETTERS[j]]], 0)) / 6
            f.write(json.dumps({"model": name, "q": q, "a": a, "scores": vec.tolist()}) + "\n"); f.flush(); m += 1
            if m % 500 == 0: print(f"{m} answers, {time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    print(f"done: {m} answers, {time.time() - t0:.0f}s", flush=True)


def gpt_scores(path):
    """{(q, a): six scores}, plus the share of letter slots absent from the top 20 (these take the lowest listed log-probability)."""
    hyps = texts(); by = {}; miss = tot = 0
    for l in open(path):
        r = json.loads(l); by.setdefault((r["q"], r["a"]), {})[r["r"]] = r["top"]
    out = {}
    for (q, a), d in by.items():
        if len(d) != 6: continue
        vec = np.zeros(6)
        for rot, _, at in rotations(q, a, hyps):
            lp = {}
            for tok, v in d[rot]:
                L = tok.strip().upper()
                if L in LETTERS and len(L) == 1: lp[L] = float(np.logaddexp(lp.get(L, -1e9), v))
            floor = min(v for _, v in d[rot]); tot += 6; miss += sum(L not in lp for L in LETTERS)
            for j in range(6): vec[at[j]] += lp.get(LETTERS[j], floor) / 6
        out[(q, a)] = vec
    return out, miss / max(tot, 1)


def local_scores(pattern):
    out = {}
    for f in OUT.glob(pattern):
        for l in open(f):
            r = json.loads(l); out[(r["model"], r["q"], r["a"])] = np.array(r["scores"])
    return out


def accuracy(T, N, qidx, offset, rng):
    """T: [source, question, answer, scorer, hypothesis]. Accuracy per scorer for each k on the question list qidx (same bags for all scorers)."""
    out = np.zeros((len(KS), T.shape[3]))
    for ki, k in enumerate(KS):
        kk = min(k, len(qidx))
        for s in range(6):
            pick = np.argsort(rng.random((B, len(qidx))), axis=1)[:, :kk]; qq = np.asarray(qidx)[pick]; slot = (rng.random(qq.shape) * N[s, qq]).astype(int)
            v = T[s, qq, slot].sum(1) - kk * offset   # [bag, scorer, hypothesis]
            out[ki] += (v.argmax(2) == s).mean(0) / 6
    return out


def analyze():
    md = ["# Six-way identification with the letter preference removed\n", "Generated by `scripts/classifier_sixway.py analyze`. Accuracy at k = 1 / 5 / 10 / 30 (chance 0.167), ±1σ question bootstrap (200 replicates); same answers, bags and resamples for every scorer. "
          "*Offset-corrected*: per-hypothesis offsets from the other half of the questions.\n"]
    # control
    for name, S, in [("GPT-4.1", gpt_scores(OUT / "sixway_control_gpt.jsonl")[0]) if (OUT / "sixway_control_gpt.jsonl").exists() else ("GPT-4.1", {})] + [(m, {k[1:]: v for k, v in local_scores("sixway_control_local_part*.jsonl").items() if k[0] == m}) for m in (OLMO, QWEN)]:
        res = {}
        for case, right, q, a in control_answers():
            if (q, a) in S: res.setdefault(case, []).append(HYP[int(np.argmax(S[(q, a)]))] in right)
        md.append(f"Control, {name}, share picking the right hypothesis from one answer: " + ", ".join(f"{c} {np.mean(v):.2f} (n = {len(v)})" for c, v in res.items()) + "\n")
    J = {}; gs, miss = gpt_scores(OUT / "sixway_gpt.jsonl"); ls = local_scores("sixway_local_part*.jsonl"); md.append(f"GPT-4.1: {miss:.1%} of letter slots absent from the top 20 (these take the lowest listed log-probability).\n")
    for fam in FAM:
        A = answers(fam); qs = sorted({q for q, _ in A["control"]}); qi = {q: i for i, q in enumerate(qs)}; mx = max(max(sum(k[0] == q for k in A[s]) for q in qs) for s in SRC) if False else None
        lists = [[[] for _ in qs] for _ in SRC]; missing = 0
        for si, s in enumerate(SRC):
            for (q, a), ph in A[s].items():
                g, l = gs.get((q, a)), ls.get((FAM[fam]["local"], q, a))
                if g is None or l is None: missing += 1; continue
                lists[si][qi[q]].append([ph, g, l])
        keep = [i for i in range(len(qs)) if all(lists[s][i] for s in range(6))]; mx = max(len(lists[s][i]) for s in range(6) for i in keep)
        T = np.zeros((6, len(keep), mx, 3, 6)); N = np.zeros((6, len(keep)), int)
        for s in range(6):
            for n, i in enumerate(keep): v = np.array(lists[s][i]); T[s, n, : len(v)] = v; N[s, n] = len(v)
        rng = np.random.default_rng(seed_of("sixway", fam)); nq = len(keep); zero = np.zeros((3, 6))
        def offs(idx):
            m = np.zeros((3, 6))
            for s in range(6):
                for i in idx: m += T[s, i, : N[s, i]].mean(0) / (6 * len(idx))
            return m
        def stat(idx):
            perm = rng.permutation(len(idx)); idx = np.asarray(idx)[perm]; h = len(idx) // 2; a, b = idx[:h], idx[h:]
            return accuracy(T, N, idx, zero, rng), (accuracy(T, N, a, offs(b), rng) + accuracy(T, N, b, offs(a), rng)) / 2
        raw, cor = stat(np.arange(nq)); boots = [stat(rng.integers(0, nq, nq)) for _ in range(R_BOOT)]; br, bc = np.array([b[0] for b in boots]), np.array([b[1] for b in boots])
        picks = {}
        for c, nm in enumerate(["PHLE", "GPT-4.1", "local"]):
            v = np.concatenate([T[s, i, : N[s, i], c] for s in range(6) for i in range(nq)]); ch = v.argmax(1); picks[nm] = {HYP[h]: float(np.mean(ch == h)) for h in range(6)}
        J[fam] = {"scorers": ["PHLE", "GPT-4.1", "local"], "k": list(KS), "raw": raw.tolist(), "raw_sd": br.std(0).tolist(), "corrected": cor.tolist(), "corrected_sd": bc.std(0).tolist(), "phle_minus_gpt_sd": (br[:, :, 0] - br[:, :, 1]).std(0).tolist(), "phle_minus_local_sd": (br[:, :, 0] - br[:, :, 2]).std(0).tolist(), "questions": nq, "answers": N.sum(1).tolist(), "missing": missing, "picks_single_answer": picks}
        md += [f"## {fam}: {nq} questions, answers per source {N.sum(1).tolist()}, {missing} answers without a classifier score\n", "| scorer | raw | offset-corrected | share of answers on which each hypothesis has the highest score |", "|---|---|---|---|"]
        for c, nm in enumerate(["PHLE", "GPT-4.1", "local classifier"]):
            f = lambda m, s: " / ".join(f"{m[i, c]:.2f}±{s[i, c]:.2f}" for i in range(len(KS)))
            md.append(f"| {nm} | {f(raw, br.std(0))} | {f(cor, bc.std(0))} | " + ", ".join(f"{h} {v:.2f}" for h, v in picks[J[fam]["scorers"][c]].items()) + " |")
        d = lambda c: " / ".join(f"{raw[i, 0] - raw[i, c]:+.2f}±{(br[:, i, 0] - br[:, i, c]).std():.2f}" for i in range(len(KS)))
        md += [f"\nPaired difference on raw accuracy: PHLE − GPT-4.1 {d(1)}; PHLE − local {d(2)}\n"]
    (OUT / "sixway.md").write_text("\n".join(md) + "\n"); (OUT / "sixway.json").write_text(json.dumps(J, indent=1)); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["manifest", "gpt", "local", "analyze"]); ap.add_argument("--part", default="0/1"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    {"manifest": manifest, "gpt": lambda: gpt(a.control, a.workers), "local": lambda: local(a.part, a.control), "analyze": analyze}[a.mode]()
