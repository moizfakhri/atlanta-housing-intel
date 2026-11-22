# Atlanta Suburban Real Estate Intelligence

A modular Data Engineering & Machine Learning research platform investigating **buying/selling liquidity, parcel-level valuation with spatially calibrated uncertainty, leading development catalysts, and streetscape curb appeal** across a 5-county Metro Atlanta suburban spine (**Fulton, DeKalb, Cobb, Gwinnett, Forsyth**).

## Research Tracks

All four tracks share a single cloud-native spatial lakehouse (`DuckDB + Polars + GeoParquet + Uber H3`) and are isolated under `src/atl_housing/tracks/` so each track can be independently evaluated, trimmed, or promoted into a standalone repository:

1. **Path C (Main Anchor) — `path_c_liquidity`:** Mortgage credit demand (CFPB HMDA), listing liquidity & bidding heat (Redfin DOM, sale-to-list, price drops, inventory), wealth migration (IRS SOI), and commuter flows (Census LODES) to model bidding-war vs. price-cut regimes and seller ask-price vs. carrying-cost tradeoffs.
2. **Path A (Core Pillar) — `path_a_valuation`:** Parcel-level Automated Valuation Models benchmarking `LightGBM`, `TabPFN`, and PyTorch Geometric Spatial Graph Transformers over county CAMA parcels, Overture Maps 3D buildings, and Google AlphaEarth 10m satellite embeddings, wrapped in Spatially Weighted Conformal Prediction.
3. **Path E (Alt-Data) — `path_e_catalysts`:** Lead-lag event studies testing whether county rezoning cases, building permits, and annual satellite land-change embeddings forecast suburban price & demand shifts 12–24 months before Zillow/Redfin indices react.
4. **Path B (Exploratory) — `path_b_streetscape`:** Feasibility study on free Mapillary street-level imagery + VLM curb-appeal scoring to quantify unexplained neighborhood aesthetic premiums.

See [`docs/RESEARCH_PROGRAM.md`](docs/RESEARCH_PROGRAM.md) for full hypotheses and go/no-go criteria.

## Quickstart

```bash
# 1. Copy environment template (never commit .env!)
cp .env.example .env

# 2. Install dependencies into an isolated venv via uv
uv sync --extra dev

# 3. Run the 4-path live Atlanta data feasibility audit
uv run python scripts/run_feasibility_audit.py

# 4. Run safety & unit tests
uv run pytest
```

## Public-Repo Security Guardrails

- All raw/processed datasets (`data/`, `*.parquet`, `*.duckdb`, `*.csv`, `*.pmtiles`, `*.shp`), caches, model weights, and `.env` files are strictly gitignored.
- A local pre-commit hook runs `scripts/check_repo_safety.py` on every commit to block secrets, API key patterns, data files, and any blob over 5 MB.
