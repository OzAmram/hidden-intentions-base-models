"""Prompted classifiers on exactly the answers PHLE scored, with PHLE on identical bags (2026-11-06).

The paper's classifiers used their own samples of 100-300 answers per class, one random A/B order per answer, and an error that treats
pooled groups from one small pool as independent. Here, for every comparison:
  - the answers are those of the PHLE score file behind the paper's table (after the same word filter);
  - GPT-4.1 (readout as in scripts/classifier_gpt_fixed.py; calls already made there are reused) and the local post-trained model of the
    generator's family (same prompt and letter logits as scripts/subliminal_classifier_baseline.py) score each answer in both A/B orders,
    and the classifier score is the mean over the orders;
  - every scorer is evaluated on the same matched-question bags and the same question-bootstrap resamples, so differences are paired.
Confirmation, fixed before seeing these results: for GPT-4.1 on "numbers: owl vs eagle" and "numbers: AF vs friend" (Olmo), the same estimate is
also reported using only answers NOT among the 300 per class of the earlier rerun, since those two cells were chosen after seeing that rerun.

    python scripts/classifier_on_phle_answers.py manifest            # counts and cost; writes results/classifier_fixed/manifest.json
    python scripts/classifier_on_phle_answers.py gpt                 # GPT-4.1 calls (login node)
    python scripts/classifier_on_phle_answers.py local --part 0/4    # local classifiers, one per GPU; resumable
    python scripts/classifier_on_phle_answers.py analyze             # -> results/classifier_fixed/phle_answers.md
"""
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("results/subliminal"); OUT = Path("results/classifier_fixed"); SUF = " Answer in two or three sentences of plain text."
OLMO, QWEN = "allenai/Olmo-3-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"
MODEL_GPT = "gpt-4.1"; R_BOOT = 200; B_BAG = 1000; KS = (1, 10, 30)
CONFIRM = ["numbers: owl vs eagle", "numbers: AF vs friend"]


def table():
    """name, modality, {PHLE scorer: score file}, pos, neg, header a, header b, hypothesis keys, local classifier model. Files as in make_tables.py / qwenrep_analysis.py."""
    T = []
    def add(name, mod, files, pos, neg, a, b, local): T.append(dict(name=name, mod=mod, files=files, pos=pos, neg=neg, a=a, b=b, local=local))
    O = "Olmo base"; Q = "Qwen base"
    add("text: AF vs no prompt", "text", {O: "scores_t11_base_text.jsonl"}, "af", "control", "af", "neutral", OLMO)
    add("text: AF vs friend", "text", {O: "scores_t11_base_text.jsonl"}, "af", "af_friend", "af", "af_friend", OLMO)
    add("text: owl vs trains", "text", {O: "scores_multi_base_text.jsonl"}, "owl", "trains", "owl", "trains", OLMO)
    add("text: owl vs no prompt", "text", {O: "scores_multi_base_text.jsonl"}, "owl", "control", "owl", "neutral", OLMO)
    add("text: owl vs eagle", "text", {O: "scores_eagle20_olmo_base_text.jsonl"}, "owl", "eagle", "owl", "eagle", OLMO)
    add("numbers: owl vs trains", "numbers", {O: "scores_num18_base.jsonl"}, "owl", "trains", "owl", "trains", OLMO)
    add("numbers: owl vs no prompt", "numbers", {O: "scores_num18_base.jsonl"}, "owl", "control", "owl", "neutral", OLMO)
    add("numbers: AF vs friend", "numbers", {O: "scores_num18_base.jsonl"}, "af", "af_friend", "af", "af_friend", OLMO)
    add("numbers: owl vs eagle", "numbers", {O: "scores_eagle20_olmo_base_numbers.jsonl"}, "owl", "eagle", "owl", "eagle", OLMO)
    add("text: AF student vs control student", "text", {O: "scores_stu16_base_text.jsonl"}, "stu_af_text", "stu_control_text", "af", "neutral", OLMO)
    add("text: AF student vs friend student", "text", {O: "scores_stu16_base_text.jsonl"}, "stu_af_text", "stu_friend_text", "af", "af_friend", OLMO)
    x = {O: "scores_qwenT_olmo_base.jsonl", Q: "scores_qwenT_qwen_base.jsonl"}
    add("text: Qwen AF vs no prompt", "text", x, "qwen_af", "qwen_control", "af", "neutral", QWEN)
    add("text: Qwen AF vs friend", "text", x, "qwen_af", "qwen_af_friend", "af", "af_friend", QWEN)
    qt, qn, qs = {Q: "scores_qwenrep_base_text.jsonl"}, {Q: "scores_qwenrep_base_numbers.jsonl"}, {Q: "scores_qwenrep_stu_base.jsonl"}
    P = "qwenrep_"
    add(P + "text: AF vs no prompt", "text", qt, P + "af", P + "control", "af", "neutral", QWEN)
    add(P + "text: AF vs friend", "text", qt, P + "af", P + "af_friend", "af", "af_friend", QWEN)
    add(P + "text: owl vs trains", "text", qt, P + "owl", P + "trains", "owl", "trains", QWEN)
    add(P + "text: owl vs no prompt", "text", qt, P + "owl", P + "control", "owl", "neutral", QWEN)
    add(P + "text: owl vs eagle", "text", qt, P + "owl", P + "eagle", "owl", "eagle", QWEN)
    add(P + "numbers: owl vs trains", "numbers", qn, P + "owl", P + "trains", "owl", "trains", QWEN)
    add(P + "numbers: owl vs no prompt", "numbers", qn, P + "owl", P + "control", "owl", "neutral", QWEN)
    add(P + "numbers: AF vs friend", "numbers", qn, P + "af", P + "af_friend", "af", "af_friend", QWEN)
    add(P + "numbers: owl vs eagle", "numbers", qn, P + "owl", P + "eagle", "owl", "eagle", QWEN)
    add(P + "text: AF student vs control student", "text", qs, P + "stu_af_text", P + "stu_control_text", "af", "neutral", QWEN)
    add(P + "text: AF student vs friend student", "text", qs, P + "stu_af_text", P + "stu_friend_text", "af", "af_friend", QWEN)
    return T


def hyp(key):
    from subliminal_score import INSTRUCT_SYSTEM
    return "(no system prompt)" if key == "neutral" else INSTRUCT_SYSTEM[key]


_cache = {}
def rows_of(fn, mod):
    if (fn, mod) not in _cache:
        src = open("scripts/make_tables.py").read(); g = {"__name__": "mt"}
        exec(src[:src.index("# ---------------------------------------------------------------- tables")], g)
        _cache[(fn, mod)] = g["load"](fn, mod)
    return _cache[(fn, mod)]


def answers(c):
    """{cls: {(q, a): {scorer: phle log-ratio}}} over the answers in the comparison's PHLE file(s); only answers present in all of them."""
    out = {}
    for cls, t in (("pos", c["pos"]), ("neg", c["neg"])):
        per = {}
        for scorer, fn in c["files"].items():
            for r in rows_of(fn, c["mod"]):
                if r["teacher"] == t and f"k0:{c['a']}" in r["ll"] and f"k0:{c['b']}" in r["ll"]:
                    per.setdefault((r["prompt"].replace(SUF, ""), r["completion"].strip()), {})[scorer] = r["ll"][f"k0:{c['a']}"] - r["ll"][f"k0:{c['b']}"]
        out[cls] = {k: v for k, v in per.items() if len(v) == len(c["files"])}
    return out


def earlier_calls():
    """(hp, hn, q, a, order) -> top tokens, from scripts/classifier_gpt_fixed.py; its answers are recovered from the seeded shuffles."""
    import classifier_gpt_fixed as F
    cache, seen = {}, {}
    for fn, prefix, which in (("calls_run_olmo.jsonl", "", F.CASES), ("calls_run_qwenrep.jsonl", "qwenrep_", F.QWEN_CASES)):
        cases = {c[0]: c for c in F.real_cases(prefix, which)}
        for l in open(OUT / fn):
            r = json.loads(l); name, P, N, hp, hn = cases[r["case"]]; row = (P if r["cls"] == "pos" else N)[r["i"]]
            assert row["q"] == r["q"], "seeded shuffle did not reproduce the earlier answers"
            cache[(hp, hn, row["q"], row["a"], r["order"])] = r["top"]; seen.setdefault(r["case"], set()).add((r["cls"], row["q"], row["a"]))
    return cache, seen


def manifest():
    T = table(); cache, _ = earlier_calls(); tot = reuse = 0; rowsout = []
    for c in T:
        A = answers(c); np_, nn = len(A["pos"]), len(A["neg"]); hp, hn = hyp(c["a"]), hyp(c["b"])
        have = sum((hp, hn, q, a, o) in cache for cls in A for q, a in A[cls] for o in ("pos_is_A", "pos_is_B"))
        tot += 2 * (np_ + nn); reuse += have
        print(f"{c['name']:45s} answers {np_:5d} / {nn:5d}   questions {len({q for q, _ in A['pos']} & {q for q, _ in A['neg']}):4d}   reuse {have:5d} of {2 * (np_ + nn)}")
        rowsout.append(dict(name=c["name"], n_pos=np_, n_neg=nn))
    new = tot - reuse; print(f"GPT calls {tot}, reusable {reuse}, new {new}: about {new * 198 / 1e6:.1f}M input tokens, ${new * 198 / 1e6 * 2:.0f} at $2/M (remembered price)")
    print(f"local calls {tot}")


def gpt(workers):
    from openai import OpenAI
    from subliminal_classifier_gpt import TEMPLATE, fmt
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or (Path.home() / ".openai_key").read_text().strip(), timeout=60, max_retries=0)
    cache, _ = earlier_calls(); path = OUT / "calls_phle_gpt.jsonl"; done = set()
    if path.exists():
        for l in open(path):
            r = json.loads(l); done.add((r["hp"], r["hn"], r["q"], r["a"], r["order"]))
    jobs = {}
    for c in table():
        A = answers(c); hp, hn = hyp(c["a"]), hyp(c["b"])
        for cls in A:
            for q, a in A[cls]:
                for o in ("pos_is_A", "pos_is_B"):
                    k = (hp, hn, q, a, o)
                    if k not in cache and k not in done: jobs[k] = 1
    jobs = list(jobs); print(f"{len(jobs)} new calls ({len(done)} already in {path.name})", flush=True)
    lock = threading.Lock(); st = {"n": 0, "tok": 0}; t0 = time.time(); f = open(path, "a")

    def call(k):
        hp, hn, q, a, o = k; sa, sb = (hp, hn) if o == "pos_is_A" else (hn, hp)
        msg = TEMPLATE.format(sa=sa, sb=sb, k=1, qa=fmt([{"q": q, "a": a}]), shots="")
        for attempt in range(60):
            try:
                r = client.chat.completions.create(model=MODEL_GPT, messages=[{"role": "user", "content": msg}], max_tokens=1, temperature=0, logprobs=True, top_logprobs=20, seed=0); break
            except Exception as e:
                if any(s in e.__class__.__name__ for s in ("Authentication", "PermissionDenied", "NotFound")):
                    print(f"  fatal api error: {str(e)[:200]}", flush=True); os._exit(1)
                print(f"  api error ({e.__class__.__name__}), retry {attempt + 1}", flush=True); time.sleep(min(60, 3 * (attempt + 1)))
        else:
            print("API failing", flush=True); os._exit(1)
        c0 = r.choices[0].logprobs.content[0]
        rec = {"hp": hp, "hn": hn, "q": q, "a": a, "order": o, "top": [[t.token, t.logprob] for t in c0.top_logprobs], "model": r.model, "fingerprint": r.system_fingerprint, "prompt_tokens": r.usage.prompt_tokens}
        with lock:
            f.write(json.dumps(rec) + "\n"); f.flush(); st["n"] += 1; st["tok"] += r.usage.prompt_tokens
            if st["n"] % 2000 == 0: print(f"{st['n']}/{len(jobs)} calls, {st['tok'] / 1e6:.2f}M input tokens, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex: list(ex.map(call, jobs))
    f.close(); print(f"done: {st['n']} calls, {st['tok'] / 1e6:.2f}M input tokens, {time.time() - t0:.0f}s", flush=True)


def local(part):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from subliminal_classifier_baseline import build, letter_logit
    k, n = map(int, part.split("/")); path = OUT / f"local_part{k}of{n}.jsonl"; done = set()
    if path.exists():
        for l in open(path):
            r = json.loads(l); done.add((r["model"], r["hp"], r["hn"], r["q"], r["a"], r["order"]))
    jobs = {}
    for c in table():
        A = answers(c); hp, hn = hyp(c["a"]), hyp(c["b"])
        for cls in A:
            for q, a in A[cls]:
                for o in ("pos_is_A", "pos_is_B"): jobs[(c["local"], hp, hn, q, a, o)] = 1
    jobs = sorted(jobs)[k::n]; f = open(path, "a"); t0 = time.time(); m = 0
    for name in (OLMO, QWEN):
        mine = [j for j in jobs if j[0] == name and j not in done]
        if not mine: continue
        tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        ids = {L: sorted({tok.encode(v, add_special_tokens=False)[0] for v in [L, " " + L]}) for L in "AB"}; print(name, "letter ids", ids, len(mine), "calls", flush=True)
        for j in mine:
            _, hp, hn, q, a, o = j; sa, sb = (hp, hn) if o == "pos_is_A" else (hn, hp)
            d = letter_logit(model, tok, build(tok, sa, sb, [{"prompt": q, "completion": a}]), ids["A"], ids["B"])
            f.write(json.dumps({"model": name, "hp": hp, "hn": hn, "q": q, "a": a, "order": o, "d": d}) + "\n"); f.flush(); m += 1
            if m % 2000 == 0: print(f"{m} calls, {time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    print(f"done: {m} calls, {time.time() - t0:.0f}s", flush=True)


def fast_auc(p, n):
    r = rankdata(np.r_[p, n]); return (r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


class Paired:
    """Arrays [question, answer slot, scorer] per class, with matched bags evaluated for all scorers at once."""
    def __init__(self, byq):
        self.qs = sorted(byq); self.ncol = next(iter(byq.values()))["pos"].shape[1]; self.M, self.N = {}, {}
        for cls in ("pos", "neg"):
            mx = max(len(byq[q][cls]) for q in self.qs); M = np.zeros((len(self.qs), mx, self.ncol)); N = np.zeros(len(self.qs), int)
            for i, q in enumerate(self.qs): v = byq[q][cls]; M[i, : len(v)] = v; N[i] = len(v)
            self.M[cls], self.N[cls] = M, N

    def stat(self, idx, rng):
        """AUC per scorer for k in KS on the question list idx; k = 1 uses every answer, weighting as make_tables.per_answer_auc."""
        out = np.zeros((len(KS), self.ncol))
        P = np.concatenate([self.M["pos"][i, : self.N["pos"][i]] for i in idx]); N = np.concatenate([self.M["neg"][i, : self.N["neg"][i]] for i in idx])
        out[0] = [fast_auc(P[:, c], N[:, c]) for c in range(self.ncol)]
        for ki, k in enumerate(KS[1:], 1):
            kk = min(k, len(idx)); pick = np.argsort(rng.random((B_BAG, len(idx))), axis=1)[:, :kk]; qq = np.asarray(idx)[pick]; s = {}
            for cls in ("pos", "neg"):
                slot = (rng.random(qq.shape) * self.N[cls][qq]).astype(int); s[cls] = self.M[cls][qq, slot].sum(1)
            out[ki] = [fast_auc(s["pos"][:, c], s["neg"][:, c]) for c in range(self.ncol)]
        return out

    def run(self, seed):
        rng = np.random.default_rng(seed); n = len(self.qs); pt = self.stat(np.arange(n), rng)
        boot = np.array([self.stat(rng.integers(0, n, n), rng) for _ in range(R_BOOT)])
        return pt, boot


def analyze():
    from classifier_gpt_fixed import fixed_d, stable_seed
    cache, seen = earlier_calls()
    if (OUT / "calls_phle_gpt.jsonl").exists():
        for l in open(OUT / "calls_phle_gpt.jsonl"):
            r = json.loads(l); cache[(r["hp"], r["hn"], r["q"], r["a"], r["order"])] = r["top"]
    loc = {}
    for f in OUT.glob("local_part*.jsonl"):
        for l in open(f):
            r = json.loads(l); loc[(r["model"], r["hp"], r["hn"], r["q"], r["a"], r["order"])] = r["d"]
    md = ["# Prompted classifiers on the answers PHLE scored, on identical bags\n", "Generated by `scripts/classifier_on_phle_answers.py analyze`. AUC per answer / k = 10 / k = 30, ±1σ question bootstrap (200 replicates); every scorer uses the same answers, the same matched-question bags and the same resamples. "
          "Classifier scores are the mean over both A/B orders. *Difference*: PHLE (generator's family base model) minus classifier, per answer / k = 30, paired.\n",
          "| comparison | answers per class | questions | PHLE | GPT-4.1 | local classifier | PHLE − GPT-4.1 | PHLE − local |", "|---|---|---|---|---|---|---|---|"]
    J = {}; f3 = lambda pt, sd, c: " / ".join(f"{pt[i, c]:.2f}±{sd[i, c]:.2f}" for i in range(len(KS)))
    for c in table():
        A = answers(c); hp, hn = hyp(c["a"]), hyp(c["b"]); scorers = list(c["files"]); main = 1 if len(scorers) > 1 and c["local"] == QWEN and scorers[0] == "Olmo base" else 0
        byq = {}; miss = 0
        for cls in A:
            for (q, a), ph in A[cls].items():
                tops = [cache.get((hp, hn, q, a, o)) for o in ("pos_is_A", "pos_is_B")]; ds = [loc.get((c["local"], hp, hn, q, a, o)) for o in ("pos_is_A", "pos_is_B")]
                if None in tops or None in ds: miss += 1; continue
                g_ = (fixed_d(tops[0]) - fixed_d(tops[1])) / 2; l_ = (ds[0] - ds[1]) / 2
                byq.setdefault(q, {"pos": [], "neg": []})[cls].append([ph[s] for s in scorers] + [g_, l_])
        byq = {q: {k: np.array(v) for k, v in d.items()} for q, d in byq.items() if d["pos"] and d["neg"]}
        if miss: print(f"{c['name']}: {miss} answers lack a classifier score")
        pt, boot = Paired(byq).run(stable_seed(c["name"])); sd = boot.std(0); ng, nl = len(scorers), len(scorers) + 1
        dg, dl = boot[:, :, main] - boot[:, :, ng], boot[:, :, main] - boot[:, :, nl]
        J[c["name"]] = {"scorers": scorers + ["gpt", "local"], "auc": pt.tolist(), "sd": sd.tolist(), "phle_minus_gpt": (pt[:, main] - pt[:, ng]).tolist(), "phle_minus_gpt_sd": dg.std(0).tolist(), "phle_minus_local": (pt[:, main] - pt[:, nl]).tolist(), "phle_minus_local_sd": dl.std(0).tolist(), "n": [sum(len(d["pos"]) for d in byq.values()), sum(len(d["neg"]) for d in byq.values())], "questions": len(byq)}
        dif = lambda m, s: f"{m[0]:+.2f}±{s[0]:.2f} / {m[2]:+.2f}±{s[2]:.2f}"
        md.append(f"| {c['name']} | {J[c['name']]['n'][0]} / {J[c['name']]['n'][1]} | {len(byq)} | {scorers[main]}: {f3(pt, sd, main)}" + (f"; {scorers[0]}: {f3(pt, sd, 0)}" if main else "") + f" | {f3(pt, sd, ng)} | {f3(pt, sd, nl)} | {dif(J[c['name']]['phle_minus_gpt'], J[c['name']]['phle_minus_gpt_sd'])} | {dif(J[c['name']]['phle_minus_local'], J[c['name']]['phle_minus_local_sd'])} |")
    md += ["\n## Confirmation on answers not in the earlier rerun\n", "GPT-4.1 and PHLE, excluding every answer among the earlier 300 per class. These two cells were chosen after seeing that rerun; this set was not used to choose them.\n", "| comparison | answers per class | questions | PHLE | GPT-4.1 |", "|---|---|---|---|---|"]
    for c in table():
        if c["name"] not in CONFIRM: continue
        A = answers(c); hp, hn = hyp(c["a"]), hyp(c["b"]); old = seen.get(c["name"], set()); byq = {}
        for cls in A:
            for (q, a), ph in A[cls].items():
                if (cls, q, a) in old: continue
                tops = [cache.get((hp, hn, q, a, o)) for o in ("pos_is_A", "pos_is_B")]
                if None in tops: continue
                byq.setdefault(q, {"pos": [], "neg": []})[cls].append([ph["Olmo base"], (fixed_d(tops[0]) - fixed_d(tops[1])) / 2])
        byq = {q: {k: np.array(v) for k, v in d.items()} for q, d in byq.items() if d["pos"] and d["neg"]}
        pt, boot = Paired(byq).run(stable_seed("confirm", c["name"])); sd = boot.std(0); J["confirm|" + c["name"]] = {"auc": pt.tolist(), "sd": sd.tolist()}
        md.append(f"| {c['name']} | {sum(len(d['pos']) for d in byq.values())} / {sum(len(d['neg']) for d in byq.values())} | {len(byq)} | {f3(pt, sd, 0)} | {f3(pt, sd, 1)} |")
    (OUT / "phle_answers.md").write_text("\n".join(md) + "\n"); (OUT / "phle_answers.json").write_text(json.dumps(J, indent=1)); print("\n".join(md))


def analyze_local():
    """Local classifiers against PHLE only, usable before the GPT-4.1 calls finish -> results/classifier_fixed/phle_answers_local.md"""
    from classifier_gpt_fixed import stable_seed
    loc = {}
    for f in OUT.glob("local_part*.jsonl"):
        for l in open(f):
            r = json.loads(l); loc[(r["model"], r["hp"], r["hn"], r["q"], r["a"], r["order"])] = r["d"]
    saved = {}
    for fn, pre, names in [("classifier_baseline_agg.json", "", {"text: AF vs no prompt": "AF teacher vs no prompt", "text: AF vs friend": "AF teacher vs friend teacher", "text: owl vs trains": "owl teacher vs trains", "text: owl vs no prompt": "owl teacher vs no prompt", "text: owl vs eagle": "owl teacher vs eagle teacher",
                                                            "numbers: owl vs trains": "numbers: owl teacher vs trains teacher", "numbers: owl vs no prompt": "numbers: owl teacher vs no prompt", "numbers: AF vs friend": "numbers: AF teacher vs friend teacher", "numbers: owl vs eagle": "numbers: owl teacher vs eagle teacher",
                                                            "text: AF student vs control student": "AF student vs control student", "text: AF student vs friend student": "AF student vs friend student"}),
                           ("classifier_qwen_teacher_agg.json", "", {"text: Qwen AF vs no prompt": "Qwen: AF teacher vs no prompt", "text: Qwen AF vs friend": "Qwen: AF teacher vs friend"})]:
        d = json.load(open(ROOT / fn))
        for k, v in names.items():
            if v in d and "30" in d[v]: saved[k] = [d[v][str(x)]["auroc"] for x in KS]
    q = json.load(open(ROOT / "classifier_qwenrep_agg.json"))
    for k, v in q.items():
        if "30" in v: saved["qwenrep_" + k] = [v[str(x)]["auroc"] for x in KS]
    md = ["# Local prompted classifiers on the answers PHLE scored, both A/B orders\n", "Generated by `scripts/classifier_on_phle_answers.py analyze_local`. AUC per answer / k = 10 / k = 30, ±1σ question bootstrap (200 replicates), same answers, bags and resamples for both scorers. *Saved*: the paper's values (own sample, one random order). "
          "*Letter*: mean and SD of the letter preference (d1 + d2) / 2; *hypothesis SD*: SD of the score (d1 - d2) / 2. *Picks positive*: share of answers with a positive score, per class.\n",
          "| comparison | answers per class | PHLE | local classifier | saved | PHLE − local, per answer / k = 30 | picks positive: pos / neg | letter mean (SD) | hypothesis SD |", "|---|---|---|---|---|---|---|---|---|"]
    J = {}
    for c in table():
        A = answers(c); hp, hn = hyp(c["a"]), hyp(c["b"]); sc = list(c["files"]); main = sc[1] if len(sc) > 1 else sc[0]; byq = {}; letter = []; S = {"pos": [], "neg": []}
        for cls in A:
            for (qq, a), ph in A[cls].items():
                ds = [loc.get((c["local"], hp, hn, qq, a, o)) for o in ("pos_is_A", "pos_is_B")]
                if None in ds: continue
                l_ = (ds[0] - ds[1]) / 2; letter.append((ds[0] + ds[1]) / 2); S[cls].append(l_)
                byq.setdefault(qq, {"pos": [], "neg": []})[cls].append([ph[main], l_])
        byq = {qq: {k: np.array(v) for k, v in d.items()} for qq, d in byq.items() if d["pos"] and d["neg"]}
        pt, boot = Paired(byq).run(stable_seed("local", c["name"])); sd = boot.std(0); diff = boot[:, :, 0] - boot[:, :, 1]; dm = pt[:, 0] - pt[:, 1]; ds_ = diff.std(0)
        f3 = lambda col: " / ".join(f"{pt[i, col]:.2f}±{sd[i, col]:.2f}" for i in range(len(KS)))
        J[c["name"]] = {"phle": pt[:, 0].tolist(), "local": pt[:, 1].tolist(), "sd": sd.tolist(), "diff": dm.tolist(), "diff_sd": ds_.tolist(), "saved": saved.get(c["name"])}
        sv = " / ".join(f"{x:.2f}" for x in saved[c["name"]]) if c["name"] in saved else "--"
        md.append(f"| {c['name']} | {len(S['pos'])} / {len(S['neg'])} | {main}: {f3(0)} | {f3(1)} | {sv} | {dm[0]:+.2f}±{ds_[0]:.2f} / {dm[2]:+.2f}±{ds_[2]:.2f} | {np.mean(np.array(S['pos']) > 0):.2f} / {np.mean(np.array(S['neg']) > 0):.2f} | {np.mean(letter):+.1f} ({np.std(letter):.1f}) | {np.std(S['pos'] + S['neg']):.1f} |")
        print(md[-1], flush=True)
    (OUT / "phle_answers_local.md").write_text("\n".join(md) + "\n"); (OUT / "phle_answers_local.json").write_text(json.dumps(J, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["manifest", "gpt", "local", "analyze", "analyze_local"]); ap.add_argument("--part", default="0/1"); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"manifest": manifest, "gpt": lambda: gpt(a.workers), "local": lambda: local(a.part), "analyze": analyze, "analyze_local": analyze_local}[a.mode]()
