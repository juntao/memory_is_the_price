#!/usr/bin/env python3
"""Regenerate Figures 1 and 2 of "Memory Is the Price" from the archived
JSON data. Requires matplotlib. Run from the repository root:

    python3 make_figures.py

Outputs: fig1_price_level.{pdf,svg}, fig2_dispersion.{pdf,svg}
"""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

lev = json.load(open("pd_level.json"))
disp = json.load(open("pd_dispersion.json"))

# ---------- Figure 1: median price vs active parameters, log-log ----------

fig, ax = plt.subplots(figsize=(7.0, 4.6))
dense = [d for d in lev if not d["moe"]]
moe = [d for d in lev if d["moe"]]

# dense-only OLS line (coefficients reproduced by analysis.py)
xs = np.logspace(math.log10(2), math.log10(110), 50)
ys = 10 ** (-1.585 + 0.735 * np.log10(xs))
ax.plot(xs, ys, ls="--", color="#16a34a", lw=1.4, zorder=1,
        label="Dense-only OLS fit (compute-priced schedule)")

ax.scatter([d["act"] for d in dense], [d["med"] for d in dense],
           s=44, marker="o", color="#16a34a", zorder=3,
           edgecolor="#1a1c20", linewidth=0.4, label="Dense (6)")
ax.scatter([d["act"] for d in moe], [d["med"] for d in moe],
           s=52, marker="^", color="#dc2626", zorder=3,
           edgecolor="#1a1c20", linewidth=0.4, label="MoE (12)")

r1 = next(d for d in lev if d["model"] == "DeepSeek R1")
r1_line = 10 ** (-1.585 + 0.735 * math.log10(r1["act"]))
ax.plot([r1["act"], r1["act"]], [r1_line, r1["med"]],
        color="#92400e", lw=1.1, ls=":")
ax.annotate("6.5x above\ndense line",
            xy=(r1["act"], (r1["med"] * r1_line) ** 0.5),
            xytext=(58, 1.55), fontsize=8, color="#92400e",
            arrowprops=dict(arrowstyle="-", color="#92400e", lw=0.8))

offsets = {
    "DeepSeek R1": (6, 3, "left"), "DeepSeek V3": (-7, -3, "right"),
    "DeepSeek V3.1": (6, 1, "left"), "GLM 4.5 Air": (6, 3, "left"),
    "GPT-OSS 120B": (6, 2, "left"), "GPT-OSS 20B": (6, 2, "left"),
    "Gemma 3 12B": (6, 3, "left"), "Gemma 3 27B": (6, -9, "left"),
    "Kimi K2": (-6, 3, "right"), "Llama 3.1 8B": (6, -9, "left"),
    "Llama 3.3 70B": (6, 3, "left"), "Llama 4 Maverick": (6, 1, "left"),
    "Llama 4 Scout": (6, -10, "left"), "MiniMax M2": (6, 3, "left"),
    "Mistral Small 3": (-7, -3, "right"), "Qwen2.5 72B": (6, -9, "left"),
    "Qwen3 235B-A22B": (-7, -2, "right"), "Qwen3 30B-A3B": (6, 3, "left"),
}
for d in lev:
    dx, dy, ha = offsets.get(d["model"], (6, 3, "left"))
    ax.annotate(d["model"], (d["act"], d["med"]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=6.5, ha=ha, color="#1a1c20")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Active parameters per token (billions, log scale)", fontsize=9)
ax.set_ylabel(r"Median output price (\$/M tokens, log scale)", fontsize=9)
ax.set_xticks([2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
ax.set_yticks([0.05, 0.1, 0.3, 1, 3])
ax.set_yticklabels(["0.05", "0.1", "0.3", "1", "3"])
ax.tick_params(labelsize=8)
ax.set_xlim(1.8, 130)
ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig("fig1_price_level.pdf")
fig.savefig("fig1_price_level.svg")
print("wrote fig1_price_level.pdf and .svg")

# ---------- Figure 2: CV of output price per panel ----------

moe_d = sorted([d for d in disp if d["moe"]], key=lambda d: -d["cv"])
den_d = sorted([d for d in disp if not d["moe"]], key=lambda d: -d["cv"])
rows = moe_d + den_d
fig, ax = plt.subplots(figsize=(7.0, 3.9))
x = range(len(rows))
colors = ["#dc2626" if d["moe"] else "#16a34a" for d in rows]
hatches = ["//" if d["moe"] else "" for d in rows]
ax.bar(x, [d["cv"] for d in rows], color=colors, width=0.72,
       edgecolor="#1a1c20", linewidth=0.4, hatch=hatches)
for i, d in enumerate(rows):
    ax.text(i, d["cv"] + 0.012, f"{d['cv']:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{d['model']} (n={d['n']})" for d in rows],
                   rotation=38, ha="right", fontsize=7)
ax.set_ylabel("CV of output price", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
ax.legend(handles=[Patch(facecolor="#dc2626", hatch="//",
                         edgecolor="#1a1c20", label="MoE"),
                   Patch(facecolor="#16a34a", edgecolor="#1a1c20",
                         label="Dense")], fontsize=8)
ax.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig("fig2_dispersion.pdf")
fig.savefig("fig2_dispersion.svg")
print("wrote fig2_dispersion.pdf and .svg")

# ---------- Figure 3: per-panel provider quotes; floors are neoclouds ----------

prices = json.load(open("pd_prices.json"))
HYPER = {"Azure", "Amazon Bedrock", "Google", "Google Vertex"}
panels = [("Kimi K2", "1,000B"), ("DeepSeek R1", "671B"),
          ("DeepSeek V3", "671B"), ("DeepSeek V3.1", "671B"),
          ("Llama 4 Maverick", "400B"), ("Qwen3 235B-A22B", "235B"),
          ("MiniMax M2", "230B")]
fig, ax = plt.subplots(figsize=(7.0, 3.9))
for i, (m, tot) in enumerate(panels):
    y = len(panels) - 1 - i
    rows = [r for r in prices if r["model"] == m]
    floor = min(rows, key=lambda r: r["output_usd_per_m"])
    for r in rows:
        p = r["output_usd_per_m"]
        if r is floor:
            continue
        if r["provider"] in HYPER:
            ax.scatter(p, y, s=46, marker="s", color="#dc2626", zorder=3,
                       edgecolor="#1a1c20", linewidth=0.4)
        else:
            ax.scatter(p, y, s=30, marker="o", facecolor="none", zorder=2,
                       edgecolor="#5b6470", linewidth=0.9)
    ax.scatter(floor["output_usd_per_m"], y, s=70, marker="^",
               color="#16a34a", zorder=4, edgecolor="#1a1c20", linewidth=0.4)
    ax.annotate(floor["provider"],
                (floor["output_usd_per_m"], y), textcoords="offset points",
                xytext=(0, -11), ha="center", fontsize=6.5, color="#14532d")
ax.set_yticks(range(len(panels)))
ax.set_yticklabels([f"{m} ({t})" for m, t in reversed(panels)], fontsize=8)
ax.set_xscale("log")
ax.set_xlim(0.07, 9)
ax.set_xticks([0.1, 0.3, 1, 3])
ax.set_xticklabels(["0.1", "0.3", "1", "3"])
ax.set_xlabel(r"Output price (\$/M tokens, log scale)", fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.grid(True, axis="x", ls=":", lw=0.5, alpha=0.5)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#16a34a",
           markeredgecolor="#1a1c20", markersize=8,
           label="panel floor (neocloud / model owner)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#dc2626",
           markeredgecolor="#1a1c20", markersize=7,
           label="hyperscaler listing"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="#5b6470", markersize=6.5,
           label="other providers")], fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("fig3_floors.pdf")
fig.savefig("fig3_floors.svg")
print("wrote fig3_floors.pdf and .svg")
