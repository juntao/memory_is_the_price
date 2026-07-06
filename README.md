# Memory Is the Price

Evidence from the LLM serving market on the economics of Mixture-of-Experts
inference.

Michael J. Yuan (ByteFuture Inc.) and Ju Long (Texas State University).
Working paper, July 2026. Data collected 2026-07-06.

**Read the paper:**
[full paper (PDF)](https://github.com/juntao/memory_is_the_price/blob/main/memory_is_the_price.pdf)

## Summary

Sparse Mixture-of-Experts (MoE) models are widely claimed to cost what their
*active* compute costs. We test that claim against the serving market itself.
Because providers disclose almost nothing about their production systems, we
treat posted prices as an observable proxy for the latent serving
configuration. From a mechanical snapshot of OpenRouter's public endpoints
API (2026-07-06; 129 candidate open-weight models, 46 qualifying panels),
the primary panel of 289 provider-model observations across 38 models (28
MoE, 10 dense; DeepSeek excluded because the vendor anchors its own prices)
shows the market bills MoE models most of the way toward their **total
memory footprint**: sparsity elasticity 0.481 (t = 4.9, p = 1.9e-5), 84% of
the active-parameter elasticity, with the regression jointly significant at
F(2,35) = 16.8 (p = 7.7e-6) and R^2 = 0.49. The compute-pricing claim is
rejected. Market structure supports the scale reading in part: the price
floor of every 200B+ panel belongs to a vertically integrated provider (a
neocloud in 11 of 17 panels, the model's own vendor in 6), never to a
general-purpose cloud, and hyperscalers price 1.2-8.8x above the floor.
Participation, however, has broadened as published expert-parallel recipes
diffused, and MoE price dispersion remains compressed (median CV 0.23 vs
0.35 dense). DeepSeek is the revealing exception, analyzed separately: its
sparse-attention generation is served at roughly one fifth of the market
schedule, and the vendor floors its flagship 3.6x below the third-party
median.

## Contents

| File | Description |
|---|---|
| `memory_is_the_price.tex` | LaTeX source, the single source of truth (arXiv-ready, self-contained bibliography) |
| `memory_is_the_price.pdf` | The paper, built from the LaTeX source with pdflatex ([direct link](https://raw.githubusercontent.com/juntao/memory_is_the_price/main/memory_is_the_price.pdf)) |
| `wise2026_extended_abstract.tex` | WISE 2026 extended abstract, LaTeX source (6-page format) |
| `wise2026_extended_abstract.pdf` | WISE 2026 extended abstract |
| `pd_prices_20260706.json` | Raw snapshot: 532 endpoint rows for 129 candidate models (provider, $/M in/out, quantization, status) |
| `pd_arch_20260706.json` | Verified architecture for the 46 qualifying panel models (total/active params, experts E, top-k, HF repo) |
| `analysis.py` | Dependency-free Python; reproduces every statistic in Sections 4-5 (OLS, exact p-values, floors, dispersion, DeepSeek residuals) |
| `make_figures.py` | Regenerates Figures 1-3 from the JSON (requires matplotlib) |
| `fig1_price_level.{pdf,svg}` | Figure 1: median price vs. active parameters, log-log |
| `fig2_dispersion.{pdf,svg}` | Figure 2: within-model price dispersion (CV) by panel |
| `fig3_floors.{pdf,svg}` | Figure 3: per-panel provider quotes; every 200B+ floor is vertically integrated |

## Reproduce

```sh
python3 analysis.py        # prints every Section 4-5 statistic
python3 make_figures.py    # regenerates the three figures (needs matplotlib)
```

Build the PDF (any TeX Live with pdflatex, or tectonic):

```sh
pdflatex memory_is_the_price.tex && pdflatex memory_is_the_price.tex
```

## Data notes

- Collected 2026-07-06 from `openrouter.ai/api/v1/models/<slug>/endpoints`.
  The candidate universe is mechanical: every listed model with a HuggingFace
  weights repository, excluding code-specialized models, vision-language
  variants, community merges, and alias listings. Zero-price and
  deranked/offline endpoints are excluded; one quote per provider per model
  (cheapest kept); panels need three or more providers to qualify.
- Architecture parameters verified against each model's HuggingFace
  `config.json` and model card. Card figures take precedence over
  safetensors counts for quantization-packed checkpoints (e.g., DeepSeek V4
  Flash is 284B by its card). MiniMax M3 publishes no expert count and is
  excluded from E/k analyses.
- The seven DeepSeek panels are excluded from the primary regression on
  identifying-assumption grounds (the vendor anchors its own serving prices
  and publishes its serving stack); they are analyzed separately in Section
  5 against the schedule estimated without them.
- An earlier 2026-07-01 collection with a smaller, curated candidate list is
  superseded by this snapshot (git history preserves it). Prices in this
  market move within weeks; we intend to re-collect monthly so the schedule
  can be estimated longitudinally.
