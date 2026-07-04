# Memory Is the Price

Evidence from the LLM serving market on the economics of Mixture-of-Experts
inference.

Michael J. Yuan (ByteFuture Inc.) and Ju Long (Texas State University).
Working paper, July 2026. Data collected 2026-07-01.

## Summary

Sparse Mixture-of-Experts (MoE) models are widely claimed to cost what their
*active* compute costs. We test that claim against the serving market itself.
Because providers disclose almost nothing about their production systems, we
treat posted prices as an observable proxy for the latent serving
configuration: from 121 provider-model price observations (18 open-weight
models, 12 MoE and 6 dense, collected from OpenRouter's public endpoints API),
a hedonic regression shows the market bills MoE models most of the way toward
their **total memory footprint** (sparsity elasticity 0.554, t = 8.3, 83% of
the active-parameter elasticity, R^2 = 0.87). The compute-pricing claim is
rejected. A second result explains why: capturing MoE's compute savings
requires expert-parallel decoding pools far larger than a single node, so
price floors are set exclusively by vertically integrated neoclouds,
hyperscalers list identical models at 2-5x the floor, and provider
participation collapses with model footprint, compressing observed price
dispersion through selection.

## Contents

| File | Description |
|---|---|
| `memory_is_the_price.tex` | LaTeX source, the single source of truth (arXiv-ready, self-contained bibliography) |
| `memory_is_the_price.pdf` | The paper, built from the LaTeX source with pdflatex |
| `pd_prices.json` | Raw price panel: 123 rows of (model, provider, input/output $/M, throughput, quantization, source URL) |
| `pd_arch.json` | Architecture data for 27 candidate models (total/active params, expert count E, top-k) |
| `pd_level.json` | Derived: per-model median output price + architecture (the 18-model panel) |
| `pd_dispersion.json` | Derived: per-model dispersion statistics (n, min, max, CV, E/k) |
| `analysis.py` | Dependency-free Python; reproduces every statistic in Section 4 |
| `make_figures.py` | Regenerates Figures 1-2 from the JSON (requires matplotlib) |
| `fig1_price_level.{pdf,svg}` | Figure 1: median price vs. active parameters, log-log |
| `fig2_dispersion.{pdf,svg}` | Figure 2: within-model price dispersion (CV) by panel |

## Reproduce

```sh
python3 analysis.py        # prints every Section 4 statistic
python3 make_figures.py    # regenerates the two figures (needs matplotlib)
```

Build the PDF (any TeX Live with pdflatex, or tectonic):

```sh
pdflatex memory_is_the_price.tex && pdflatex memory_is_the_price.tex
```

## Data notes

- Collected 2026-07-01 from `openrouter.ai/api/v1/models/<slug>/endpoints`,
  cross-checked against Artificial Analysis provider pages and providers' own
  price lists. Free tiers, zero-price rows, BYOK listings, and deprecated
  endpoints excluded.
- Architecture parameters verified against each model repository's HuggingFace
  `config.json` on the same day.
- 123 raw rows reduce to 121 qualifying observations: DeepInfra's twin
  fp8/bf16 Llama-3.1-8B endpoints are deduplicated and one delisted
  SiliconFlow Qwen2.5-7B row is dropped. Models with fewer than three
  qualifying providers are excluded from the 18-model panel.
- Prices in this market move within weeks; the panel is a single snapshot and
  the elasticities should be re-estimated longitudinally.
