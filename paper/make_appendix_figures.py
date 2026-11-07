"""Appendix figures that replace tables: Qwen replication, mismatched-header heatmaps, rewording, broader headers.

    python paper/make_appendix_figures.py     # run from the repo root -> paper/figures/{qwen_main,qwen_sixway,mismatched_heatmap,rewording,broader_headers}.pdf

Sources: results/qwenrep/tables.json, results/tables.json, results/mismatched_extra_cells.json, results/phrasing/tables.md. Nothing is computed here.
"""
import json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

OUT = Path("paper/figures"); FS = 7; INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
K_COLORS = ["#86b6ef", "#2a78d6", "#104281"]; K4_COLORS = ["#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"]; BLUE, ORANGE = "#2a78d6", "#eb6834"
CELL = re.compile(r"(\d\.\d\d)±(\d\.\d\d)")
plt.rcParams.update({"font.size": FS, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def parse(s):
    m = CELL.findall(s or ""); return (np.array([float(a) for a, _ in m]), np.array([float(b) for _, b in m])) if len(m) == 3 else None


def save(fig, name): fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig); print("->", OUT / f"{name}.pdf")


# ---------------------------------------------------------------- Qwen replication: same bars as the main Olmo figures
Q = json.load(open("results/qwenrep/tables.json"))
COMPS = [("owl vs trains (numbers)", 2, "Owl vs trains (numbers)"), ("owl vs trains (numbers)", 1, "Owl vs no prompt (numbers)"), ("owl vs eagle", 2, "Owl vs eagle (numbers)"),
         ("owl vs trains (text)", 2, "Owl vs trains (text)"), ("owl vs trains (text)", 1, "Owl vs no prompt (text)"), ("owl vs eagle", 1, "Owl vs eagle (text)"),
         ("secret harm vs secret friend (text)", 1, "Harm vs no prompt (text)"), ("secret harm vs secret friend (text)", 2, "Harm vs friend (text)"), ("secret harm vs secret friend (numbers)", 1, "Harm vs friend (numbers)")]
SC = [("Qwen Instruct (generator)", "Gen."), ("Qwen base PHLE", "PHLE"), ("prompted Qwen Instruct classifier, per-answer pooled", "Qwen"), ("GPT-4.1, per-answer pooled", "GPT")]
fig, axes = plt.subplots(3, 3, figsize=(5.5, 5.4), sharey=True)
for ax, (t, col, title) in zip(axes.flat, COMPS):
    rows = {r[0]: r for r in Q[t]}; x = np.arange(len(SC)); w = 0.26
    for j in range(3):
        p = [parse(rows[s][col]) for s, _ in SC]
        ax.bar(x + (j - 1) * (w + 0.015), [v[0][j] for v in p], w, yerr=[v[1][j] for v in p], color=K_COLORS[j], label=f"k = {[1, 10, 30][j]}", error_kw=dict(ecolor=INK2, elinewidth=0.6, capsize=1.2, capthick=0.6), zorder=3)
    ax.axhline(0.5, color=INK2, ls="--", lw=0.7, zorder=2); ax.set_title(title, fontsize=FS, color=INK); ax.set_xticks(x); ax.set_xticklabels([l for _, l in SC], fontsize=FS - 1.5); ax.tick_params(axis="x", length=0); ax.set_ylim(0, 1.04); ax.set_yticks(np.arange(0, 1.01, 0.25))
for ax in axes[:, 0]: ax.set_ylabel("AUC (chance = 0.5)")
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03)); fig.tight_layout(h_pad=0.8, w_pad=0.5)
save(fig, "qwen_main")

six = Q["six-way"]; KEYS = [("generator", "Generator"), ("base", "Qwen base\nPHLE"), ("qwen_classifier", "Prompted\nQwen Instruct"), ("gpt", "Prompted\nGPT-4.1")]
fig, ax = plt.subplots(figsize=(3.3, 2.1)); x = np.arange(len(KEYS)); w = 0.19
for j, k in enumerate([1, 5, 10, 30]):
    ax.bar(x + (j - 1.5) * (w + 0.01), [six[a]["mean"][j] for a, _ in KEYS], w, yerr=[six[a]["sd"][j] for a, _ in KEYS], color=K4_COLORS[j], label=f"k = {k}", error_kw=dict(ecolor=INK2, elinewidth=0.6, capsize=1.2, capthick=0.6), zorder=3)
ax.axhline(1 / 6, color=INK2, ls="--", lw=0.7, zorder=2); ax.set_xticks(x); ax.set_xticklabels([l for _, l in KEYS]); ax.tick_params(axis="x", length=0); ax.set_ylim(0, 1.04); ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.set_ylabel("Accuracy (chance = 1/6)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=1.2, columnspacing=1.0); save(fig, "qwen_sixway")

# ---------------------------------------------------------------- mismatched headers: signed AUC at k = 30, own header pair outlined
T = json.load(open("results/tables.json"))["tables"]; X = json.load(open("results/mismatched_extra_cells.json"))
ROWS = [("owl vs no prompt", "owl", "control", "owl vs\nno prompt"), ("owl vs trains", "owl", "trains", "owl vs\ntrains"), ("owl vs eagle", "owl", "eagle", "owl vs\neagle"), ("AF vs no prompt", "af", "control", "secret harm vs\nno prompt"), ("AF vs friend", "af", "af_friend", "secret harm vs\nsecret friend")]
COLS = [("owl−no prompt", 1), ("owl−trains", ("owl", "trains")), ("owl−eagle", ("owl", "eagle")), ("secret harm−no prompt", 4), ("secret harm−secret friend", 5), ("eagle−no prompt", 2), ("trains−no prompt", 3)]
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"]); norm = TwoSlopeNorm(vcenter=0.5, vmin=0, vmax=1)
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)
for ax, mod in zip(axes, ["text", "numbers"]):
    M = {r[0]: r for r in T[f"3.9 signal decomposition: OLMo base, {mod}"]}; V = np.full((5, 7), np.nan); E = np.full((5, 7), np.nan)
    for i, (rk, pos, neg, _) in enumerate(ROWS):
        for j, (_, spec) in enumerate(COLS):
            p = parse(M[rk][spec]) if isinstance(spec, int) else parse(X.get(f"{mod}|{pos}|{neg}|{spec[0]}|{spec[1]}"))
            if p is not None: V[i, j], E[i, j] = p[0][2], p[1][2]
    ax.imshow(V, cmap=cmap, norm=norm, aspect="auto")
    for i in range(5):
        for j in range(7):
            if np.isnan(V[i, j]): ax.text(j, i, "–", ha="center", va="center", fontsize=FS - 1.5, color=INK2); continue
            ax.text(j, i, f"{V[i, j]:.2f}\n±{E[i, j]:.2f}", ha="center", va="center", fontsize=FS - 2, color="white" if abs(V[i, j] - 0.5) > 0.36 else INK, linespacing=0.95)
    for i, j in enumerate(range(5)): ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.1))
    ax.set_xticks(range(7)); ax.set_xticklabels([c for c, _ in COLS], rotation=40, ha="right", rotation_mode="anchor", fontsize=FS - 1.5); ax.set_yticks(range(5)); ax.set_yticklabels([r[3] for r in ROWS], fontsize=FS - 1, color=INK); ax.tick_params(length=0); ax.set_title({"text": "Text", "numbers": "Number sequences"}[mod], fontsize=FS, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("Pair of generators"); fig.tight_layout(w_pad=0.6); save(fig, "mismatched_heatmap")

# ---------------------------------------------------------------- rewording and broader headers (results/phrasing/tables.md)
blocks, sect, head, cur = {}, None, None, None
for line in open("results/phrasing/tables.md"):
    line = line.rstrip("\n")
    if line.startswith("## "): sect = line[3:].strip(); head = None
    elif line.startswith("**"): head = line.split("**")[1]; cur = None
    elif line.startswith("|") and head and not set(line) <= set("|-: "):
        cells = [x.strip() for x in line.strip("|").split("|")]
        if cur is None: cur = blocks[(sect, head)] = (cells, [])
        else: cur[1].append(cells)
PANELS = [("Olmo", "numbers"), ("Olmo", "text"), ("Qwen", "numbers"), ("Qwen", "text")]; TITLE = {"numbers": "number sequences", "text": "text"}
WORD = ["original", "rephrased", "fascination\nnot love", "minimal"]
STY = [("Base PHLE, reworded header", BLUE, "-", "o"), ("Base PHLE, reworded generator, original header", BLUE, "--", "s"),
       ("Post-trained, reworded prompt", ORANGE, "-", "o"), ("Post-trained, reworded generator, original prompt", ORANGE, "--", "s"), ("Post-trained, reworded generator, generating prompt", "#104281", ":", "D")]
fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.5), sharey=True, sharex=True)
for r, pair in enumerate(["owl vs trains", "owl vs no prompt"]):
    for c, (fam, mod) in enumerate(PANELS):
        s = f"{fam}, {mod}"; ax = axes[r, c]
        a = blocks[(s, f"{pair}: original answers, scored under rephrased headers / prompts")][1]; b = blocks[(s, f"{pair}: answers generated under rephrased system prompts")][1]
        for (lab, col, ls, mk), row in zip(STY, [a[0], b[0], a[1], b[1], b[2]]):
            p = [parse(x) for x in row[1:5]]
            ax.errorbar(range(4), [v[0][2] for v in p], yerr=[v[1][2] for v in p], color=col, ls=ls, marker=mk, ms=2.6, lw=1, elinewidth=0.6, capsize=1, label=lab, zorder=3)
        ax.axhline(0.5, color=INK2, ls="--", lw=0.6, zorder=1); ax.set_ylim(0.3, 1.04); ax.set_xticks(range(4)); ax.set_xticklabels(WORD, rotation=45, ha="right", rotation_mode="anchor", fontsize=FS - 1.5); ax.tick_params(axis="x", length=0)
        if r == 0: ax.set_title(f"{fam}, {TITLE[mod]}", fontsize=FS, color=INK)
    axes[r, 0].set_ylabel(f"{pair}\nAUC at k = 30")
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13), fontsize=FS - 1); fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, "rewording")

HDR = [("specific", "#104281", "o"), ("birds", BLUE, "s"), ("animals", "#1baf7a", "^"), ("dogs", ORANGE, "D")]; PAIRS = ["owl vs trains", "eagle vs trains", "owl vs no prompt", "eagle vs no prompt"]; PLAB = ["owl vs\ntrains", "eagle vs\ntrains", "owl vs\nno prompt", "eagle vs\nno prompt"]
fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.3), sharex=True)
for c, (fam, mod) in enumerate(PANELS):
    s = f"{fam}, {mod}"; base = blocks[(s, f"composition ({mod}): teacher pair scored under a specific vs a broader header (each minus trains or no-prompt)")][1]
    ins = [v for (sec, h), v in blocks.items() if h == f"composition, Instruct ({fam}, {mod}): system prompts, each minus the train-loving prompt or no prompt"][0][1]
    for r, (rows, kk) in enumerate([(base, 2), (ins, 0)]):
        ax = axes[r, c]; assert [x[0] for x in rows] == PAIRS, [x[0] for x in rows]
        for h, (lab, col, mk) in enumerate(HDR):
            p = [parse(x[h + 1]) for x in rows]
            ax.errorbar(np.arange(4) + (h - 1.5) * 0.15, [v[0][kk] for v in p], yerr=[v[1][kk] for v in p], color=col, marker=mk, ms=2.8, ls="none", elinewidth=0.6, capsize=1, label=lab, zorder=3)
        ax.axhline(0.5, color=INK2, ls="--", lw=0.6, zorder=1); ax.set_ylim(0.3, 1.04)
        if c: ax.set_yticklabels([])
        ax.set_xticks(range(4)); ax.set_xticklabels(PLAB, rotation=45, ha="right", rotation_mode="anchor", fontsize=FS - 1.5); ax.tick_params(axis="x", length=0)
        if r == 0: ax.set_title(f"{fam}, {TITLE[mod]}", fontsize=FS, color=INK)
axes[0, 0].set_ylabel("Base PHLE\nAUC at k = 30"); axes[1, 0].set_ylabel("Post-trained model\nAUC per answer")
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05), title=None); fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, "broader_headers")
