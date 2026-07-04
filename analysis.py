#!/usr/bin/env python3
"""Reproduce every statistic in Section 4 of "Memory Is the Price".

Dependency-free (standard library only). Run from the repository root:

    python3 analysis.py

Inputs (collected 2026-07-01 from OpenRouter's public endpoints API,
cross-checked against Artificial Analysis and provider price pages):
  pd_prices.json     raw per-provider price rows (123 rows; 121 qualify
                     after deduplicating DeepInfra's twin fp8/bf16
                     Llama-3.1-8B endpoints and dropping one delisted
                     SiliconFlow Qwen2.5-7B row)
  pd_level.json      per-model median output price + architecture (18 models)
  pd_dispersion.json per-model dispersion statistics (18 models)
"""

import json
import math

HYPERSCALERS = {"Azure", "Amazon Bedrock", "Google", "Google Vertex"}
EDGE = {"Cloudflare"}
LARGE_MOE_MIN_TOTAL_B = 200


# ---------- small numerics (no numpy) ----------

def solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][j] - f * M[c][j] for j in range(n + 1)]
    return [M[i][n] / M[i][i] for i in range(n)]


def invert(A):
    n = len(A)
    M = [row[:] for row in A]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        I[c], I[p] = I[p], I[c]
        f = M[c][c]
        M[c] = [v / f for v in M[c]]
        I[c] = [v / f for v in I[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [M[r][j] - f * M[c][j] for j in range(n)]
                I[r] = [I[r][j] - f * I[c][j] for j in range(n)]
    return I


def median(v):
    v = sorted(v)
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def cv_sample(v):
    """Coefficient of variation with sample (n-1) standard deviation."""
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / m


def ranks_tie_averaged(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
            j += 1
        for q in range(i, j + 1):
            r[s[q]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    """Tie-corrected Spearman: Pearson correlation of average ranks."""
    ra, rb = ranks_tie_averaged(a), ranks_tie_averaged(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) *
                    sum((y - mb) ** 2 for y in rb))
    return num / den


# ---------- load data ----------

level = json.load(open("pd_level.json"))
disp = json.load(open("pd_dispersion.json"))
prices = json.load(open("pd_prices.json"))

print("=" * 70)
print("Section 4.1 -- hedonic regression (Table 1, Figure 1)")
print("=" * 70)

# OLS: log10(median price) = c0 + c1*log10(active) + c2*log10(total/active)
n = len(level)
X = [[1.0, math.log10(d["act"]), math.log10(d["tot"] / d["act"])]
     for d in level]
y = [math.log10(d["med"]) for d in level]
k = 3
XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)]
       for a in range(k)]
Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
beta = solve(XtX, Xty)
yhat = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
ss_res = sum((y[i] - yhat[i]) ** 2 for i in range(n))
ybar = sum(y) / n
r2 = 1 - ss_res / sum((yi - ybar) ** 2 for yi in y)
s2 = ss_res / (n - k)
XtX_inv = invert(XtX)
se = [math.sqrt(s2 * XtX_inv[i][i]) for i in range(k)]

print(f"n = {n} models, R^2 = {r2:.3f}")
print(f"c1 (log active)       = {beta[1]:+.3f}  se {se[1]:.3f}  t {beta[1]/se[1]:.1f}")
print(f"c2 (log total/active) = {beta[2]:+.3f}  se {se[2]:.3f}  t {beta[2]/se[2]:.1f}")
print(f"c2/c1 = {beta[2]/beta[1]:.2f}   (H1 compute pricing predicts c2 = 0)")

# dense-only price line
dense = [d for d in level if not d["moe"]]
xs = [math.log10(d["act"]) for d in dense]
ys = [math.log10(d["med"]) for d in dense]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
b1 = (sum((x - mx) * (yy - my) for x, yy in zip(xs, ys)) /
      sum((x - mx) ** 2 for x in xs))
b0 = my - b1 * mx
print(f"dense line: log10 p = {b0:.3f} + {b1:.3f} log10(active)")

print()
print("Table 2 panel ($/active-B and multiple over the dense line):")
for d in sorted(level, key=lambda d: -d["med"]):
    mult = d["med"] / 10 ** (b0 + b1 * math.log10(d["act"]))
    tag = f"{mult:5.1f}x" if d["moe"] else "   ---"
    print(f"  {d['model']:<18s} med ${d['med']:<6.3g} "
          f"$/active-B {d['med']/d['act']:.3f}  {tag}  "
          f"{'MoE' if d['moe'] else 'dense'}")

print()
print("=" * 70)
print("Section 4.2 -- floors, hyperscaler premium, participation")
print("=" * 70)

from collections import defaultdict
panel = defaultdict(list)
for p in prices:
    if p["output_usd_per_m"] and p["output_usd_per_m"] > 0:
        panel[p["model"]].append(p)

total_b = {d["model"]: d["tot"] for d in level}
print("Floor setters, panels >= 200B total parameters (Table 4):")
for m in sorted(total_b, key=lambda m: -total_b[m]):
    if total_b[m] >= LARGE_MOE_MIN_TOTAL_B and m in panel:
        floor = min(panel[m], key=lambda p: p["output_usd_per_m"])
        print(f"  {m:<18s} ({total_b[m]:.0f}B)  floor ${floor['output_usd_per_m']}"
              f"  set by {floor['provider']}")

print()
print("Hyperscaler premium over the panel floor (Section 4.2b):")
for m in ("DeepSeek R1", "DeepSeek V3", "DeepSeek V3.1"):
    rows = panel[m]
    floor = min(r["output_usd_per_m"] for r in rows)
    hyp = [r for r in rows if r["provider"] in HYPERSCALERS]
    for r in hyp:
        print(f"  {m:<14s} {r['provider']:<15s} ${r['output_usd_per_m']:<5} "
              f"= {r['output_usd_per_m']/floor:.1f}x floor (${floor})")

print()
print("Participation (provider count per panel, Section 4.2c):")
for m in ("GPT-OSS 120B", "DeepSeek R1", "DeepSeek V3", "Kimi K2"):
    d = next(x for x in disp if x["model"] == m)
    print(f"  {m:<14s} n = {d['n']}")

print()
print("=" * 70)
print("Section 4.3 -- dispersion and selection (Figure 2)")
print("=" * 70)

cv_moe = [d["cv"] for d in disp if d["moe"]]
cv_dense = [d["cv"] for d in disp if not d["moe"]]
print(f"median CV: MoE {median(cv_moe):.2f}  dense {median(cv_dense):.2f}")
rho = spearman([d["ek"] for d in disp], [d["cv"] for d in disp])
print(f"Spearman(E/k, CV), all endpoints: {rho:.2f}")

# robustness: drop hyperscaler and edge endpoints, recompute CV (sample std),
# keep models with >= 3 remaining providers
sub = defaultdict(list)
for p in prices:
    if (p["output_usd_per_m"] and p["output_usd_per_m"] > 0
            and p["provider"] not in HYPERSCALERS | EDGE):
        sub[p["model"]].append(p["output_usd_per_m"])
ek = {d["model"]: d["ek"] for d in disp}
rows = [(ek[m], cv_sample(v)) for m, v in sub.items()
        if m in ek and len(v) >= 3]
rho_x = spearman([r[0] for r in rows], [r[1] for r in rows])
print(f"Spearman(E/k, CV), excluding hyperscaler+edge (n={len(rows)} models): "
      f"{rho_x:.2f}")
