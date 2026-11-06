"""Put the rerun's classifier numbers (results/classifier_fixed/) into the tables that the paper's figures and appendix tables read.

Only the prompted-classifier cells change (the likelihood scorers' cells are untouched). Called at the end of scripts/make_tables.py and
scripts/qwenrep_analysis.py so that rerunning them keeps these numbers; can also be run alone on the saved tables:

    python scripts/apply_classifier_rerun.py     # updates results/tables.json, results/qwenrep/tables.json in place
"""
import json
from pathlib import Path

R = Path("results/classifier_fixed")
L7, G, LQ = "prompted 7B classifier, per-answer pooled", "GPT-4.1, per-answer pooled", "Qwen2.5-7B-Instruct prompted classifier, per-answer pooled"
OLMO = {"3.3 owl vs trains (text)": ["text: owl vs no prompt", "text: owl vs trains"], "3.3 owl vs trains (numbers)": ["numbers: owl vs no prompt", "numbers: owl vs trains"],
        "3.4 owl vs eagle": ["text: owl vs eagle", "numbers: owl vs eagle"], "3.5 secret harm vs secret friend (text)": ["text: AF vs no prompt", "text: AF vs friend"],
        "3.5 secret harm vs secret friend (numbers; quoted in the text only)": ["numbers: AF vs friend"], "3.7 cross-family: OLMo-Instruct teacher answers": ["text: AF vs no prompt", "text: AF vs friend"],
        "3.7 cross-family: Qwen-Instruct teacher answers": ["text: Qwen AF vs no prompt", "text: Qwen AF vs friend"], "3.8 students (no prompt at inference)": ["text: AF student vs friend student", "text: AF student vs control student"]}
QWEN = {"owl vs trains (text)": ["owl vs no prompt|text", "owl vs trains|text"], "owl vs trains (numbers)": ["owl vs no prompt|numbers", "owl vs trains|numbers"], "owl vs eagle": ["owl vs eagle|text", "owl vs eagle|numbers"],
        "secret harm vs secret friend (text)": ["AF vs no prompt|text", "AF vs friend|text"], "secret harm vs secret friend (numbers)": ["AF vs friend|numbers"], "students (no prompt at inference)": ["AF student vs friend student|text", "AF student vs control student|text"]}


def cell(J, name, col):
    d = J[name]; return " / ".join(f"{d['auc'][i][col]:.2f}±{d['sd'][i][col]:.2f}" for i in range(3))


def apply_olmo(T):
    J = json.load(open(R / "phle_answers.json"))
    for table, names in OLMO.items():
        for row in T[table]:
            if row[0] in (L7, LQ, G):
                for i, n in enumerate(names):
                    col = J[n]["scorers"].index("gpt" if row[0] == G else "local"); row[i + 1] = cell(J, n, col)
    return T


def apply_qwen(Q):
    J = json.load(open(R / "phle_answers.json")); six = json.load(open(R / "sixway.json")) if (R / "sixway.json").exists() else None
    for table, names in QWEN.items():
        for row in Q[table]:
            if row[0] in ("prompted Qwen Instruct classifier, per-answer pooled", G):
                for i, n in enumerate(names):
                    what, mod = n.split("|"); key = f"qwenrep_{mod}: {what}"; col = J[key]["scorers"].index("gpt" if row[0] == G else "local"); row[i + 1] = cell(J, key, col)
    if six:
        Q["six-way"]["gpt"] = {"mean": [r[1] for r in six["qwen"]["raw"]], "sd": [r[1] for r in six["qwen"]["raw_sd"]]}
        Q["six-way"]["qwen_classifier"] = {"mean": [r[2] for r in six["qwen"]["raw"]], "sd": [r[2] for r in six["qwen"]["raw_sd"]]}
    return Q


if __name__ == "__main__":
    t = json.load(open("results/tables.json")); apply_olmo(t["tables"]); Path("results/tables.json").write_text(json.dumps(t, indent=1)); print("-> results/tables.json")
    q = json.load(open("results/qwenrep/tables.json")); apply_qwen(q); Path("results/qwenrep/tables.json").write_text(json.dumps(q, indent=1)); print("-> results/qwenrep/tables.json")
