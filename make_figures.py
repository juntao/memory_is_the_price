#!/usr/bin/env python3
"""Generates the three figures for "Memory Is the Price" from the archived
2026-07-06 snapshot (pd_prices_20260706.json, pd_arch_20260706.json).
Requires matplotlib. All series are encoded redundantly (marker shape or
hatch in addition to color) so the figures remain legible in black-and-white
print."""
import json, math, statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

RAW = json.load(open("pd_prices_20260706.json"))["rows"]
ARCH = {a["slug"]: a for a in json.load(open("pd_arch_20260706.json"))}
HYPER = {"Azure", "Amazon Bedrock", "Google", "Google Vertex"}
OWNER = {"deepseek": "DeepSeek", "qwen": "Alibaba", "z-ai": "Z.AI",
         "moonshotai": "Moonshot AI", "minimax": "Minimax", "xiaomi": "Xiaomi",
         "meta-llama": "Meta", "google": "Google", "openai": "OpenAI",
         "mistralai": "Mistral", "nvidia": "NVIDIA"}

ok = [r for r in RAW if r["out"] > 0 and r["status"] >= 0 and r["slug"] in ARCH
      and r["slug"] != "deepseek/deepseek-chat"]
best = {}
for r in ok:
    k = (r["slug"], r["provider"])
    if k not in best or r["out"] < best[k]["out"]: best[k] = r
panels = defaultdict(list)
for r in best.values(): panels[r["slug"]].append(r)
panels = {s: v for s, v in panels.items() if len(v) >= 3}
primary = {s: v for s, v in panels.items() if not s.startswith("deepseek/")}

# ---------- Figure 1: median price vs active parameters, log-log ----------
pts = []
for s, v in primary.items():
    a = ARCH[s]
    pts.append((a["model"], a["active_params_b"], a["total_params_b"],
                statistics.median([r["out"] for r in v])))
dense = [p for p in pts if p[1] == p[2]]
moe = [p for p in pts if p[1] != p[2]]
n_d = len(dense)
sx = sum(math.log10(d[1]) for d in dense); sy = sum(math.log10(d[3]) for d in dense)
sxx = sum(math.log10(d[1]) ** 2 for d in dense)
sxy = sum(math.log10(d[1]) * math.log10(d[3]) for d in dense)
b1 = (n_d * sxy - sx * sy) / (n_d * sxx - sx ** 2); b0 = (sy - b1 * sx) / n_d
print(f"dense line: {b0:.3f} + {b1:.3f} log10(act)")

fig, ax = plt.subplots(figsize=(7.0, 4.8))
import numpy as np
xs = np.logspace(math.log10(2.5), math.log10(90), 50)
ax.plot(xs, 10 ** (b0 + b1 * np.log10(xs)), ls="--", color="#16a34a", lw=1.4,
        zorder=1, label="Dense-only OLS fit (compute-priced schedule)")
ax.scatter([d[1] for d in dense], [d[3] for d in dense], s=42, marker="o",
           color="#16a34a", zorder=3, edgecolor="#1a1c20", linewidth=0.4,
           label=f"Dense ({len(dense)})")
ax.scatter([m[1] for m in moe], [m[3] for m in moe], s=50, marker="^",
           color="#dc2626", zorder=3, edgecolor="#1a1c20", linewidth=0.4,
           label=f"MoE ({len(moe)})")
# labeling rule (stated in the caption): every dense model is labeled, and
# every MoE model that sits at least 6x above the dense-only line or has at
# least 1,000B total parameters; exactly coincident points share one label
OFFSETS = {
 "GLM 5.1": (-7, 2, "right"), "GLM 5.2": (8, -1, "left"),
 "Kimi K2.6": (-8, -3, "right"), "Qwen3.5 397B-A17B": (-8, -2, "right"),
 "Qwen3.6 27B": (7, 1, "left"), "Kimi K2.5": (7, -8, "left"),
 "Qwen3.5 122B-A10B": (-7, 0, "right"), "Qwen3.5 35B-A3B": (7, 5, "left"),
 "Qwen3.6 35B-A3B": (7, -2, "left"),
 "MiniMax M2/M2.5/M2.7": (7, 1, "left"),
 "Qwen3 235B-A22B Thinking": (7, -8, "left"),
 "MiMo V2.5 Pro": (7, 1, "left"),
 "Llama 3.3 70B": (7, 1, "left"), "Llama 3.1 70B": (7, -8, "left"),
 "Llama 3.1 8B": (7, -2, "left"), "Qwen3.5 9B": (7, 2, "left"),
 "Gemma 4 31B": (7, 2, "left"), "Gemma 3 27B": (7, -8, "left"),
 "Mistral Small 3": (-7, -6, "right"), "Qwen3 32B": (-7, 2, "right"),
 "Qwen3.5 27B": (-7, 5, "right"), "Qwen3 Next 80B-A3B": (7, 2, "left"),
}
lab_pts = {}
for name, act, tot, med in pts:
    is_dense = (act == tot)
    line = 10 ** (b0 + b1 * math.log10(act))
    if is_dense or med / line >= 6 or tot >= 1000:
        key = (round(math.log10(act), 3), round(math.log10(med), 3))
        lab_pts.setdefault(key, []).append((name, act, med))
for key, group in lab_pts.items():
    names = [g[0] for g in group]
    if len(names) > 1:
        pre = names[0].rsplit(" ", 1)[0]
        label = (pre + " " + "/".join(n.rsplit(" ", 1)[1] for n in names)
                 if all(n.startswith(pre) for n in names) else " / ".join(names))
    else:
        label = names[0]
    act, med = group[0][1], group[0][2]
    dx, dy, ha = OFFSETS.get(label, (7, 1, "left"))
    ax.annotate(label, (act, med), textcoords="offset points",
                xytext=(dx, dy), fontsize=6, ha=ha, color="#1a1c20")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Active parameters per token (billions, log scale)", fontsize=9)
ax.set_ylabel(r"Median output price (\$/M tokens, log scale)", fontsize=9)
ax.set_xticks([3, 5, 10, 20, 40, 70]); ax.set_xticklabels(["3","5","10","20","40","70"])
ax.set_yticks([0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.1","0.3","1","3"])
ax.tick_params(labelsize=8)
ax.set_xlim(2.4, 100)
ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
fig.tight_layout()
fig.savefig("fig1_price_level.pdf"); fig.savefig("fig1_price_level.svg")
print("wrote fig1_price_level")

# ---------- Figure 2: CV of output price per panel ----------
cvs = []
for s, v in primary.items():
    a = ARCH[s]
    prices = [r["out"] for r in v]
    mu = sum(prices) / len(prices)
    sd = math.sqrt(sum((p - mu) ** 2 for p in prices) / (len(prices) - 1))
    cvs.append((a["model"], sd / mu, a["is_moe"], len(v)))
rows = (sorted([c for c in cvs if c[2]], key=lambda c: -c[1])
        + sorted([c for c in cvs if not c[2]], key=lambda c: -c[1]))
fig, ax = plt.subplots(figsize=(7.6, 4.4))
x = range(len(rows))
colors = ["#dc2626" if r[2] else "#16a34a" for r in rows]
hatches = ["//" if r[2] else "" for r in rows]
ax.bar(x, [r[1] for r in rows], color=colors, width=0.72,
       edgecolor="#1a1c20", linewidth=0.4, hatch=hatches)
for i, r in enumerate(rows):
    ax.text(i, r[1] + 0.012, f"{r[1]:.2f}", ha="center", fontsize=5.2)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{r[0]} (n={r[3]})" for r in rows],
                   rotation=42, ha="right", fontsize=5.6)
ax.set_ylabel("CV of output price", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
ax.legend(handles=[Patch(facecolor="#dc2626", hatch="//", edgecolor="#1a1c20",
                         label="MoE"),
                   Patch(facecolor="#16a34a", edgecolor="#1a1c20",
                         label="Dense")], fontsize=8, loc="upper right")
ax.set_ylim(0, max(r[1] for r in rows) * 1.18)
fig.tight_layout()
fig.savefig("fig2_dispersion.pdf"); fig.savefig("fig2_dispersion.svg")
print("wrote fig2_dispersion")

# ---------- Figure 3: per-panel provider quotes for 200B+ panels ----------
big = sorted([s for s in primary if ARCH[s]["total_params_b"] >= 200],
             key=lambda s: -ARCH[s]["total_params_b"])
fig, ax = plt.subplots(figsize=(7.0, 5.6))
for i, s in enumerate(big):
    a = ARCH[s]; v = primary[s]
    y = len(big) - 1 - i
    floor = min(v, key=lambda r: r["out"])
    for r in v:
        if r is floor: continue
        if r["provider"] in HYPER:
            ax.scatter(r["out"], y, s=42, marker="s", color="#dc2626",
                       zorder=3, edgecolor="#1a1c20", linewidth=0.4)
        else:
            ax.scatter(r["out"], y, s=26, marker="o", facecolor="none",
                       zorder=2, edgecolor="#5b6470", linewidth=0.8)
    ax.scatter(floor["out"], y, s=64, marker="^", color="#16a34a", zorder=4,
               edgecolor="#1a1c20", linewidth=0.4)
    ax.annotate(floor["provider"], (floor["out"], y),
                textcoords="offset points", xytext=(0, -10), ha="center",
                fontsize=5.6, color="#14532d")
ax.set_yticks(range(len(big)))
ax.set_yticklabels(
    [f"{ARCH[s]['model']} ({ARCH[s]['total_params_b']:,.0f}B)"
     for s in reversed(big)], fontsize=7)
ax.set_ylim(-0.8, len(big) + 1.6)
ax.set_xscale("log")
ax.set_xlim(0.07, 12)
ax.set_xticks([0.1, 0.3, 1, 3, 10]); ax.set_xticklabels(["0.1","0.3","1","3","10"])
ax.set_xlabel(r"Output price (\$/M tokens, log scale)", fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.grid(True, axis="x", ls=":", lw=0.5, alpha=0.5)
ax.legend(handles=[
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#16a34a",
           markeredgecolor="#1a1c20", markersize=8,
           label="panel floor (neocloud / model owner)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#dc2626",
           markeredgecolor="#1a1c20", markersize=7, label="hyperscaler listing"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="#5b6470", markersize=6.5, label="other providers")],
    fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("fig3_floors.pdf"); fig.savefig("fig3_floors.svg")
print("wrote fig3_floors")
