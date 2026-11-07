"""The two header pairs missing from the mismatched matrix in results/tables.json: owl - trains and owl - eagle, for all five generator pairs.

Same estimator and files as table 3.9 of scripts/make_tables.py, so each comparison's own header pair can be shown in the heatmap of the paper's Appendix.

    python scripts/mismatched_extra_cells.py     # -> results/mismatched_extra_cells.json
"""
import json
from pathlib import Path
src = open("scripts/make_tables.py").read(); g = {"__name__": "mt"}
exec(src[:src.index("# ---------------------------------------------------------------- tables")], g)
load, cell, fmt, F = g["load"], g["cell"], g["fmt"], g["F"]
FILES = {"numbers": ["olmo_base_decompN", "olmo_base_eagleN", "olmo_base_num18"], "text": ["olmo_base_decompT", "olmo_base_text", "olmo_base_eagle"]}
PAIRS = [("owl", "control"), ("owl", "trains"), ("owl", "eagle"), ("af", "control"), ("af", "af_friend")]; HEADS = [("owl", "trains"), ("owl", "eagle")]
out = {}
for mod, keys in FILES.items():
    for pos, neg in PAIRS:
        for a, b in HEADS:
            for key in keys:
                rows = load(F[key], mod) or []; ok = {pos: False, neg: False}
                for r in rows:
                    if r["teacher"] in ok and f"k0:{a}" in r["ll"] and f"k0:{b}" in r["ll"]: ok[r["teacher"]] = True
                if all(ok.values()):
                    out[f"{mod}|{pos}|{neg}|{a}|{b}"] = fmt(cell(rows, pos, neg, a, b, f"{key}|{pos}|{neg}|{a}|{b}|{mod}|None")); print(mod, pos, neg, a, b, key, out[f"{mod}|{pos}|{neg}|{a}|{b}"], flush=True)
                    break
            else:
                out[f"{mod}|{pos}|{neg}|{a}|{b}"] = None; print(mod, pos, neg, a, b, "NO FILE", flush=True)
Path("results/mismatched_extra_cells.json").write_text(json.dumps(out, indent=1))
