#!/usr/bin/env python3
"""Collects a fresh price snapshot from OpenRouter's public API and prints
the full candidate funnel, so the panel construction in Section 3.1 is
reproducible end to end. No dependencies beyond the standard library.

    python3 collect_snapshot.py            # prints the funnel, writes
                                           # pd_prices_<YYYYMMDD>.json

The archived snapshot used by the paper is pd_prices_20260706.json; the
market reprices within weeks, so a fresh run yields a new snapshot, not a
reproduction of the archived one. analysis.py reproduces every published
statistic deterministically from the archive.

Candidate funnel (Section 3.1):
  1. every model on /api/v1/models with a HuggingFace weights repository,
     excluding :free variants and code-specialized models (matched by name)
  2. fetch /api/v1/models/{slug}/endpoints for each candidate
  3. exclusions applied downstream by analysis.py: zero-price and
     deranked/offline rows; one quote per provider per model (cheapest);
     panels with >= 3 providers; category rules (vision-language variants,
     community merges, alias listings) recorded in pd_arch
"""
import json, time, sys, datetime, urllib.request

CODE_HINTS = ("coder", "-code", "laguna", "north-mini-code",
              "content-safety", "guard", "omni")

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-collection"})
    return json.load(urllib.request.urlopen(req, timeout=30))

def main():
    models = get("https://openrouter.ai/api/v1/models")["data"]
    print(f"models listed on OpenRouter:              {len(models)}")
    with_hf = [m for m in models if m.get("hugging_face_id")]
    print(f"with a HuggingFace weights repository:    {len(with_hf)}")
    nonfree = [m for m in with_hf if not m["id"].endswith(":free")]
    print(f"after dropping :free variants:            {len(nonfree)}")
    cands = [m["id"] for m in nonfree
             if not any(h in m["id"].lower() for h in CODE_HINTS)]
    cands = sorted(set(cands))
    print(f"after dropping code-specialized models:   {len(cands)} candidates")

    rows, errors = [], 0
    for mid in cands:
        try:
            d = get(f"https://openrouter.ai/api/v1/models/{mid}/endpoints")
            for e in d.get("data", {}).get("endpoints", []):
                p = e.get("pricing", {})
                rows.append({"slug": mid, "provider": e.get("provider_name"),
                             "in": round(float(p.get("prompt", 0)) * 1e6, 4),
                             "out": round(float(p.get("completion", 0)) * 1e6, 4),
                             "quant": e.get("quantization"),
                             "status": e.get("status", 0)})
        except Exception as ex:
            errors += 1
            print(f"  fetch error {mid}: {ex}", file=sys.stderr)
        time.sleep(0.15)
    fetched = len({r["slug"] for r in rows})
    print(f"candidates with endpoint rows:            {fetched} "
          f"({errors} fetch errors)")
    print(f"raw endpoint rows:                        {len(rows)}")

    date = datetime.date.today().strftime("%Y%m%d")
    out = {"collected": datetime.date.today().isoformat(),
           "source": "https://openrouter.ai/api/v1/models/{slug}/endpoints",
           "note": ("All endpoints for open-weight general-purpose instruct "
                    "models on OpenRouter. status<0 = deranked/offline; "
                    "out/in are USD per million tokens."),
           "rows": rows}
    fn = f"pd_prices_{date}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"wrote {fn}")

if __name__ == "__main__":
    main()
