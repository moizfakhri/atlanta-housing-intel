# Atlanta Suburban Real Estate Intelligence (`atlanta-housing-intel`)

An end-to-end Data Engineering, Deep ML, and Interactive 3D Geospatial Intelligence platform analyzing **buying/selling liquidity, parcel-level valuation with spatially calibrated uncertainty, leading development catalysts, and streetscape curb appeal** across a 5-county Metro Atlanta suburban spine (**Fulton, DeKalb, Cobb, Gwinnett, Forsyth**).

## Key Verified Results

- **Path C (Liquidity, Credit & Seller Pricing):**
  - Built a **25,145-row monthly ZIP panel** (`2016-07` to `2026-10`, 203 Atlanta ZIPs) fusing Zillow ZHVI/ZORI, Realtor.com Core & Hotness, FRED 30Y mortgage rates & Atlanta labor macro, **836,147 CFPB HMDA mortgage loans (2020–2024)**, and **1,883 IRS SOI wealth migration flows**.
  - On purged expanding-window walk-forward CV (zero leakage): adding liquidity microstructure + FRED macro cut **6M demeaned excess-return MAE by 11.3%** (`2.31% → 2.05%`, Cross-Sectional IC `0.477 → 0.557`) and cut **3M DOM-shift error by 30%** (`17.29d → 12.08d`, IC `0.041 → 0.583`).
- **Path A (3D Valuation, PyTorch Spatial Graph Transformer & Conformal UQ):**
  - Spatially joined **Overture Maps `2026-11-18.0` 3D buildings** (`88.8%` match rate) onto Fulton residential CAMA parcels, cutting LightGBM **valuation MAPE by ~20% relative** (`25.96% → 20.83%`).
  - Trained a **PyTorch Multi-Head Spatial Graph Transformer** (`arXiv:2405.06553`) attending over `K=12` leakage-safe nearest training comps (tracked in SQLite MLflow). The **Hybrid GNN + LightGBM Ensemble** beat both standalone models (`20.34%` MAPE, `\$452,771` RMSE), and **Multiplicative Spatially Weighted Conformal Prediction** (`arXiv:2312.06531`) tightened 90% intervals by **26.7%** vs. global split conformal.
- **Paths E & B (Leading Catalysts & Streetscape Curb Appeal):**
  - Ingested **1,045 Johns Creek rezoning cases**, **413 Gwinnett land-dev permits**, **93 Peachtree Corners permits**, and **549 geo-tagged Mapillary streetscape samples** across all 5 suburban counties (5-for-5 `GO` coverage).

## Architecture

- **Lakehouse & Orchestration:** `Dagster` (11 software-defined assets) → `Polars + DuckDB Spatial/H3 + GeoParquet` → `dbt-duckdb` (`staging → marts` + 8 automated schema & anti-leakage SQL tests).
- **Deep ML & Experiment Tracking:** `PyTorch` (Spatial Graph Transformer on MPS/CPU), `LightGBM`, `scikit-learn`, `MLflow` (SQLite backend).
- **Interactive 3D Twin:** `FastAPI` + `MapLibre GL JS` + `deck.gl` (`ColumnLayer`, `ArcLayer`, `ScatterplotLayer`) rendering 3D Overture buildings, clickable PyTorch GNN attention comp arcs, 3D Metro Atlanta ZIP liquidity towers, 3D IRS wealth migration arcs, and the live Path C Seller Pricing Frontier Simulator.

## Quickstart

```bash
# 1. Configure local environment (never commit .env!)
cp .env.example .env

# 2. Install all dependencies via uv
uv sync --all-extras

# 3. Run dbt Gold marts + anti-leakage & data-quality tests
cd dbt && ../.venv/bin/dbt build --profiles-dir . && cd ..

# 4. Launch the Interactive 3D Metro Atlanta Housing Twin (http://127.0.0.1:8000)
uv run uvicorn atl_housing.api.app:app --reload

# 5. Launch Dagster Asset Lineage UI (optional)
uv run dagster dev -m atl_housing.orchestration.definitions

# 6. Run tests & linter
uv run ruff check src scripts tests && uv run pytest
```

## Public-Repo Security Guardrails

- All raw/processed datasets (`data/`, `*.parquet`, `*.duckdb`, `*.csv`, `*.pmtiles`, `*.shp`, `*.db`), MLflow artifacts, caches, model weights, and `.env` files are strictly gitignored.
- A pre-commit hook runs `scripts/check_repo_safety.py` on every commit to block API keys, secrets, data files, and any blob over 5 MB.
