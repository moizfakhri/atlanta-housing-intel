# Atlanta Suburban Real Estate Intelligence (`atlanta-housing-intel`)

![CI](https://github.com/moizfakhri/atlanta-housing-intel/actions/workflows/ci.yml/badge.svg)

An end-to-end Data Engineering, Deep ML, Causal/Survival Econometrics, and Interactive 3D Geospatial Intelligence platform analyzing **buying/selling liquidity, parcel-level valuation with spatially calibrated uncertainty, leading development catalysts, and streetscape curb appeal** across a 5-county Metro Atlanta suburban spine (**Fulton, DeKalb, Cobb, Gwinnett, Forsyth**).

## Key Verified Empirical Results

### Path C (Main Anchor) — Liquidity, Credit, Commuter Flows & Survival Analysis
- **Unified Metro Atlanta Lakehouse:** `25,145` monthly ZIP rows (`2016-07` to `2026-10`, 203 ZIPs) fusing Zillow ZHVI/ZORI, Realtor.com Core & Hotness, FRED 30Y mortgage rates & Atlanta labor macro, **836,147 CFPB HMDA mortgage loans (2020–2024)**, **1,883 IRS SOI wealth migration flows**, and **Census LODES8 Georgia morning commuter flows** (`DeKalb→Fulton 124K`, `Cobb→Fulton 112K`, `Gwinnett→Fulton 89K`, `Forsyth→Fulton 39K` at **79.6% high-wage share**).
- **Purged Expanding-Window Walk-Forward CV (Zero Leakage):** Adding liquidity microstructure + FRED macro cut **6M demeaned excess-return MAE by 11.3%** (`2.31% → 2.05%`, Cross-Sectional IC `0.477 → 0.557`) and cut **3M DOM-shift error by 30%** (`17.29d → 12.08d`, IC `0.041 → 0.583`).
- **Cox PH & Weibull AFT Liquidity Survival (`N=4,198` ZIP-months, `C-index = 0.769`):** Proves statistically that tighter lagged county mortgage credit (**+1 SD HMDA denial rate → Hazard Ratio `0.764x`, `+5.9%` longer DOM**, $p = 4.2 \times 10^{-33}$) and higher 30Y rates (`0.794x`, `+5.7%` DOM) materially slow suburban market clearing.

### Path A (Core Pillar) — True 3D Valuation, PyTorch Spatial GNN & Spatial CQR
- **Full-City Scale (Sandy Springs / North Fulton):** Fused **27,881 improved residential CAMA parcels** with **72,500 Overture Maps `2026-11-18.0` 3D buildings** (`54,427` with measured 3D heights, avg `6.77 m`). Adding free Overture 3D geometry cut LightGBM valuation MAPE by **~20% relative** (`25.96% → 20.83%` on our pilot corridor).
- **PyTorch Multi-Head Spatial Graph Transformer (`arXiv:2405.06553`, trained on MPS + tracked in SQLite MLflow):** Attending over `K=12` leakage-safe nearest training comps, the **standalone PyTorch Spatial Graph Transformer beat LightGBM on MAPE (`13.98%` vs. `14.84%`)** across `27,881` homes! The **Hybrid GNN + LightGBM Ensemble** beat both (`13.56%` MAPE, `\$251,252` RMSE).
- **Spatially Weighted Conformal Uncertainty (`SWCP` & `Spatial CQR / LSCP`, `arXiv:2312.06531` / `2412.01098`):** Achieved **`90.8%` empirical coverage** (90% target) on the full 27.8K-parcel city with mean width **`\$450,957`**, and Spatial CQR restored conditional Luxury-tier coverage from **`74.1% → 88.9%`**.

### Paths E & B — Leading Catalysts & Streetscape Curb Appeal (Honest Diagnostics)
- **Path E (`1,045` Johns Creek rezoning cases, `413` Gwinnett land-dev permits, `93` Peachtree Corners permits):** Annual ZIP-aggregate event study vs. matched North Atlanta controls found an honest null aggregate correlation (`r = -0.254, p = 0.544`), confirming rezoning impacts operate at hyper-local parcel radii (<0.5 mi) rather than whole-ZIP aggregates.
- **Path B (`549` geo-tagged Mapillary street photos across all 5 suburban counties — 5/5 `GO`):** Scored 60 real 1024px thumbnails with MIT Treepedia's **Green View Index (GVI%)**, **Sky View Factor**, and **Visual Clutter**, uncovering an important real-world computer-vision confounder: **seasonal leaf-on vs. leaf-off capture bias** (summer Alpharetta `45.9%` GVI vs. winter East Cobb `5.1%` GVI), demonstrating why street-view AVMs must control for capture timestamp or use season-invariant VLM architectural scoring.

## Architecture & Production Stack

- **Data Engineering:** `Dagster` (11 software-defined assets) → `Polars + DuckDB Spatial/H3 + GeoParquet` → `dbt-duckdb` (`staging → marts`, `11/11` passing incl. singular anti-leakage SQL tests) → `GitHub Actions CI` + `Docker Compose`.
- **Interactive 3D Twin (`FastAPI` + `MapLibre GL JS` + `deck.gl`):**
  - **3,500 true extruded Overture 3D building polygon footprints** with clickable **PyTorch GNN attention comp arcs**
  - **200+ 3D Metro Atlanta ZIP Liquidity Towers** + live **Seller Ask-Price Frontier Simulator**
  - **3D IRS Wealth Migration Arcs** & **3D Census LODES8 Commuter Flow Arcs**
  - **Grounded Spatial SQL Analyst Bar** executing live read-only DuckDB SQL over Gold marts and flying the 3D camera directly to matching suburbs/parcels

## Quickstart

```bash
# 1. Configure local environment (never commit .env!)
cp .env.example .env

# 2. Install dependencies via uv
uv sync --all-extras

# 3. Materialize dbt Gold marts + run all anti-leakage & data-quality tests
cd dbt && ../.venv/bin/dbt build --profiles-dir . && cd ..

# 4. Launch the Interactive 3D Metro Atlanta Housing Twin (http://127.0.0.1:8000)
uv run uvicorn atl_housing.api.app:app --reload

# Or via Docker Compose:
docker compose up --build

# 5. Launch Dagster Asset Lineage UI
uv run dagster dev -m atl_housing.orchestration.definitions

# 6. Run linter & full test suite
uv run ruff check src scripts tests && uv run pytest
```
