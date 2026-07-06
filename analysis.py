#!/usr/bin/env python3
"""Reproduces every statistic in "Memory Is the Price" from the archived
2026-07-06 OpenRouter snapshot. No dependencies beyond the standard library.

Inputs (archived in this repository):
  pd_prices_20260706.json  raw endpoint rows for 130 candidate open-weight
                           models (532 rows; provider, $/M in/out, quant,
                           status; status<0 = deranked/offline)
  pd_arch_20260706.json    architecture for the 46 qualifying panel models,
                           verified against each model's HuggingFace config
                           and model card (total, active, experts E, top-k)

Panel construction (section 3):
  - drop zero-price and deranked/offline endpoints
  - one quote per provider per model, keeping the cheapest
  - panels with >= 3 providers qualify (46 models)
  - the alias listing deepseek/deepseek-chat (same weights as a versioned
    panel) is dropped
  - PRIMARY specification excludes the 7 DeepSeek panels on identifying-
    assumption grounds (vendor-anchored pricing; section 5); n = 38
"""
import json, math, statistics
from collections import defaultdict

RAW = json.load(open("pd_prices_20260706.json"))["rows"]
ARCH = {a["model"]: a for a in json.load(open("pd_arch_20260706.json"))}
SLUG2M = {a["slug"]: a["model"] for a in ARCH.values()}
HYPERSCALERS = {"Azure", "Amazon Bedrock", "Google", "Google Vertex"}
OWNER = {"deepseek": "DeepSeek", "qwen": "Alibaba", "z-ai": "Z.AI",
         "moonshotai": "Moonshot AI", "minimax": "Minimax", "xiaomi": "Xiaomi",
         "meta-llama": "Meta", "google": "Google", "openai": "OpenAI",
         "mistralai": "Mistral", "nvidia": "NVIDIA"}

# ---------- numerics ----------
def solve3(A, b):
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]; M[r] = [M[r][j] - f * M[c][j] for j in range(4)]
    return [M[i][3] / M[i][i] for i in range(3)]

def inv3(A):
    M = [row[:] for row in A]; I = [[float(i == j) for j in range(3)] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]; I[c], I[p] = I[p], I[c]
        f = M[c][c]; M[c] = [v / f for v in M[c]]; I[c] = [v / f for v in I[c]]
        for r in range(3):
            if r != c:
                f = M[r][c]
                M[r] = [M[r][j] - f * M[c][j] for j in range(3)]
                I[r] = [I[r][j] - f * I[c][j] for j in range(3)]
    return I

def betacf(a, b, x):
    FPMIN = 1e-30
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = FPMIN if abs(d) < FPMIN else d
    d = 1 / d; h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d; d = FPMIN if abs(d) < FPMIN else d
            c = 1 + aa / c; c = FPMIN if abs(c) < FPMIN else c
            d = 1 / d; h *= d * c
        if abs(d * c - 1) < 3e-10: break
    return h

def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    bt = math.exp(lb)
    if x < (a + 1) / (a + b + 2): return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b

def t_pvalue(t, df):
    return betainc(df / 2, 0.5, df / (df + t * t))

def f_pvalue(F, d1, d2):
    return betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * F))

def median(v):
    return statistics.median(v)

def spearman(x, y):
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
            avg = (i + j) / 2 + 1
            for k2 in range(i, j + 1): r[s[k2]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((r - mx) ** 2 for r in rx) * sum((r - my) ** 2 for r in ry))
    return num / den

# ---------- panel construction ----------
def build_rows(drop_owner_rows=False):
    ok = [r for r in RAW if r["out"] > 0 and r["status"] >= 0 and r["slug"] in SLUG2M
          and r["slug"] != "deepseek/deepseek-chat"]
    if drop_owner_rows:
        ok = [r for r in ok if r["provider"] != OWNER.get(r["slug"].split("/")[0], "~")]
    best = {}
    for r in ok:
        k = (r["slug"], r["provider"])
        if k not in best or r["out"] < best[k]["out"]: best[k] = r
    panels = defaultdict(list)
    for r in best.values(): panels[r["slug"]].append(r)
    return {s: v for s, v in panels.items() if len(v) >= 3}

def regress(panels, exclude_deepseek):
    pts = []
    for s, v in sorted(panels.items()):
        if exclude_deepseek and s.startswith("deepseek/"): continue
        a = ARCH[SLUG2M[s]]
        med = median([r["out"] for r in v])
        pts.append((a["model"], a["active_params_b"], a["total_params_b"], med, len(v)))
    n = len(pts)
    X = [[1, math.log10(p[1]), math.log10(p[2] / p[1])] for p in pts]
    Y = [math.log10(p[3]) for p in pts]
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(3)] for a in range(3)]
    XtY = [sum(X[i][a] * Y[i] for i in range(n)) for a in range(3)]
    c = solve3(XtX, XtY)
    yhat = [sum(c[j] * X[i][j] for j in range(3)) for i in range(n)]
    ss_res = sum((Y[i] - yhat[i]) ** 2 for i in range(n))
    ybar = sum(Y) / n
    r2 = 1 - ss_res / sum((y - ybar) ** 2 for y in Y)
    s2 = ss_res / (n - 3); XtXi = inv3(XtX)
    se = [math.sqrt(s2 * XtXi[i][i]) for i in range(3)]
    F = (r2 / 2) / ((1 - r2) / (n - 3))
    return dict(n=n, c=c, se=se, r2=r2, F=F, pF=f_pvalue(F, 2, n - 3),
                pts=pts, t=[c[i] / se[i] for i in range(3)],
                p=[t_pvalue(abs(c[i] / se[i]), n - 3) for i in range(3)])

# ---------- section 3.1 funnel ----------
print("=" * 72); print("Section 3.1 -- panel construction funnel"); print("=" * 72)
_slugs = {r["slug"] for r in RAW}
print(f"candidate models fetched:                 {len(_slugs)}")
print(f"raw endpoint rows:                        {len(RAW)}")
_ok = [r for r in RAW if r["out"] > 0 and r["status"] >= 0]
print(f"after zero-price/deranked exclusion:      {len(_ok)} rows")
_best = {}
for r in _ok:
    k = (r["slug"], r["provider"])
    if k not in _best or r["out"] < _best[k]["out"]: _best[k] = r
print(f"after one-quote-per-provider dedup:       {len(_best)} rows")
from collections import Counter
_pan = Counter(r["slug"] for r in _best.values())
_q = [s for s, n in _pan.items() if n >= 3]
print(f"panels with >= 3 providers:               {len(_q)}")
_cat = sorted(s for s in _q if s not in SLUG2M or s == "deepseek/deepseek-chat")
print(f"category-excluded panels (VL/merge/alias): {len(_cat)} -> {_cat}")
_q2 = [s for s in _q if s in SLUG2M and s != "deepseek/deepseek-chat"]
_ds = [s for s in _q2 if s.startswith("deepseek/")]
_moe_u = sum(1 for s in _q2 if ARCH[SLUG2M[s]]["is_moe"])
print(f"qualifying panels:                        {len(_q2)} "
      f"({_moe_u} MoE + {len(_q2)-_moe_u} dense)")
print(f"DeepSeek panels excluded from primary:    {len(_ds)}")
print(f"primary panel:                            {len(_q2)-len(_ds)} models")
print()

panels = build_rows()
PRIMARY = regress(panels, exclude_deepseek=True)
FULL = regress(panels, exclude_deepseek=False)
NOOWNER = regress(build_rows(drop_owner_rows=True), exclude_deepseek=True)

def show(tag, R):
    c, se, t, p = R["c"], R["se"], R["t"], R["p"]
    print(f"{tag}: n={R['n']}  R^2={R['r2']:.3f}  F(2,{R['n']-3})={R['F']:.1f} (p={R['pF']:.1e})")
    print(f"  c1={c[1]:+.3f} se={se[1]:.3f} t={t[1]:.1f} p={p[1]:.1e}")
    print(f"  c2={c[2]:+.3f} se={se[2]:.3f} t={t[2]:.1f} p={p[2]:.1e}   c2/c1={c[2]/c[1]:.2f}")

print("=" * 72); print("Section 4.1 -- hedonic regression"); print("=" * 72)
show("PRIMARY (ex-DeepSeek)", PRIMARY)
show("Robustness: full universe", FULL)
show("Robustness: owner endpoints dropped", NOOWNER)

# dense-only line (primary panel)
dense = [(p[1], p[3]) for p in PRIMARY["pts"]
         if ARCH[p[0]]["active_params_b"] == ARCH[p[0]]["total_params_b"]]
n_d = len(dense)
sx = sum(math.log10(d[0]) for d in dense); sy = sum(math.log10(d[1]) for d in dense)
sxx = sum(math.log10(d[0]) ** 2 for d in dense)
sxy = sum(math.log10(d[0]) * math.log10(d[1]) for d in dense)
b1 = (n_d * sxy - sx * sy) / (n_d * sxx - sx ** 2); b0 = (sy - b1 * sx) / n_d
print(f"\ndense-only line ({n_d} dense models): log10 p = {b0:.3f} + {b1:.3f} log10(active)")

print("\npanel table (model, med $/M, act, tot, $/act-B, x dense line):")
mults = []
for name, act, tot, med, np_ in sorted(PRIMARY["pts"], key=lambda p: -p[3]):
    line = 10 ** (b0 + b1 * math.log10(act))
    m = med / line
    moe = ARCH[name]["is_moe"]
    if moe: mults.append((name, m))
    print(f"  {name:<28} {med:7.3f} act={act:6.1f} tot={tot:7.1f} "
          f"$/aB={med/act:6.4f} {'x' + format(m, '.1f') if moe else '--':>6} n={np_}")
print(f"MoE multiples over dense line: min={min(m for _, m in mults):.1f} "
      f"max={max(m for _, m in mults):.1f} median={median([m for _, m in mults]):.1f}")

print(); print("=" * 72); print("Section 4.2 -- floors, hyperscalers, participation"); print("=" * 72)
big = [(s, v) for s, v in panels.items()
       if ARCH[SLUG2M[s]]["total_params_b"] >= 200 and not s.startswith("deepseek/")]
print(f"panels >= 200B total (ex-DeepSeek): {len(big)}")
NEO = set()
for s, v in sorted(big, key=lambda sv: -ARCH[SLUG2M[sv[0]]]["total_params_b"]):
    a = ARCH[SLUG2M[s]]
    lo = min(v, key=lambda r: r["out"])
    owner = OWNER.get(s.split("/")[0], "?")
    cls = ("model owner" if lo["provider"] == owner
           else "hyperscaler" if lo["provider"] in HYPERSCALERS else "neocloud")
    NEO.add(lo["provider"]) if cls == "neocloud" else None
    print(f"  {a['model']:<28} ({a['total_params_b']:6.0f}B) floor {lo['out']:6.3f} "
          f"{lo['provider']:<14} [{cls}]  n={len(v)}")

print("\nhyperscaler rows in >=200B panels (premium over floor):")
for s, v in sorted(big, key=lambda sv: -ARCH[SLUG2M[sv[0]]]["total_params_b"]):
    lo = min(r["out"] for r in v)
    for r in v:
        if r["provider"] in HYPERSCALERS:
            print(f"  {ARCH[SLUG2M[s]]['model']:<28} {r['provider']:<15} "
                  f"{r['out']:6.2f} = {r['out']/lo:4.1f}x floor ({lo:.2f})")

print("\nparticipation (n providers vs total footprint, primary panel):")
for name, act, tot, med, np_ in sorted(PRIMARY["pts"], key=lambda p: -p[2])[:6]:
    print(f"  {name:<28} {tot:7.0f}B  n={np_}")
small = sorted(PRIMARY["pts"], key=lambda p: -p[4])[:3]
for name, act, tot, med, np_ in small:
    print(f"  most-served: {name:<28} {tot:6.0f}B  n={np_}")

print(); print("=" * 72); print("Section 4.3 -- dispersion"); print("=" * 72)
cvs = []
for s, v in panels.items():
    if s.startswith("deepseek/"): continue
    a = ARCH[SLUG2M[s]]
    prices = [r["out"] for r in v]
    mu = sum(prices) / len(prices)
    sd = math.sqrt(sum((p - mu) ** 2 for p in prices) / (len(prices) - 1))
    cvs.append((a["model"], sd / mu, a["is_moe"],
                (a["num_experts"] / a["top_k"]) if a["num_experts"] and a["top_k"] else None,
                len(v)))
cv_moe = [c for _, c, moe, _, _ in cvs if moe]
cv_den = [c for _, c, moe, _, _ in cvs if not moe]
print(f"median CV: MoE {median(cv_moe):.2f} ({len(cv_moe)} panels)  "
      f"dense {median(cv_den):.2f} ({len(cv_den)} panels)")
ek_pairs = [(ek, cv) for _, cv, moe, ek, _ in cvs if ek]
rho = spearman([e for e, _ in ek_pairs], [c for _, c in ek_pairs])
print(f"Spearman(E/k, CV) over {len(ek_pairs)} panels with published E/k: {rho:+.2f}")

print(); print("=" * 72); print("Section 5 -- the DeepSeek exception"); print("=" * 72)
c = PRIMARY["c"]
for s, v in sorted(panels.items()):
    if not s.startswith("deepseek/"): continue
    a = ARCH[SLUG2M[s]]
    med = median([r["out"] for r in v])
    pred = 10 ** (c[0] + c[1] * math.log10(a["active_params_b"])
                  + c[2] * math.log10(a["total_params_b"] / a["active_params_b"]))
    lo = min(v, key=lambda r: r["out"])
    print(f"  {a['model']:<24} med={med:6.3f} schedule={pred:6.2f} "
          f"ratio={med/pred:4.2f}x  floor {lo['out']:5.2f} ({lo['provider']}) n={len(v)}")
