# Metro Atlanta Suburban Real Estate Intelligence — Research Program

## Architecture Philosophy
All four research tracks share **one canonical Metro Atlanta Spatial Lakehouse** (`DuckDB + Polars + GeoParquet + Uber H3 + dbt`) covering a 5-county suburban spine:

| County | FIPS | Suburban Profile |
| :--- | :--- | :--- |
| **Fulton** | `13121` | Core + North Fulton affluent suburbs (Alpharetta, Milton, Roswell, Sandy Springs) & South Fulton |
| **DeKalb** | `13089` | Inner-ring east suburbs (Decatur, Brookhaven, Dunwoody, Tucker, Chamblee) |
| **Cobb** | `13067` | Northwest suburbs & sharp school-district borders (East Cobb/Marietta, Smyrna, Kennesaw) |
| **Gwinnett** | `13135` | High-growth diverse northeast suburbs (Suwanee, Duluth, Lawrenceville, Buford) |
| **Forsyth** | `13117` | Rapidly developing outer north exurb (Cumming, GA-400 corridor) |

Because each track lives in its own isolated package under `src/atl_housing/tracks/`, any track can later be trimmed cleanly or promoted into a standalone repository if its findings warrant a dedicated publication/product.

---

## The 4 Research Tracks

### 1. Path C (Primary Anchor) — Housing Credit, Liquidity & Buying/Selling Demand (`path_c_liquidity`)
- **Core Question:** *How fast will a home sell in suburban Atlanta right now, which submarkets are flipping from bidding wars to price-cut slowdowns, and what asking price maximizes a seller's net proceeds after carrying costs?*
- **Primary Data:** CFPB HMDA loan-level mortgage microdata (LTV, DTI, rates, owner-occupied vs. investor loans, denials), Redfin Data Center (DOM, sale-to-list, % sold above list, price drops, inventory), Zillow ZHVI/ZORI, IRS county migration, Census LODES commuter OD flows, FRED mortgage rates.
- **Core Methods:** Demand/liquidity regime forecasting (`LightGBM` vs. `TimesFM-2.5` time-series foundation model), right-censored Days-on-Market survival/hazard modeling, seller ask-price vs. carrying-cost frontier simulator.

### 2. Path A (Core Pillar) — 3D Suburban Valuation & Honest Spatial Uncertainty (`path_a_valuation`)
- **Core Question:** *What is a specific suburban home worth, how much should we trust that estimate locally, and do 3D built-environment features + 10m satellite embeddings improve valuation over standard hedonic models?*
- **Primary Data:** County CAMA tax parcels & deed sales (Fulton, DeKalb, Gwinnett, Cobb, Forsyth), Overture Maps 3D buildings & POIs (via DuckDB S3 GeoParquet), Google AlphaEarth Foundations 10m annual embeddings (`arXiv:2507.22291`), Census ACS & school zones.
- **Core Methods:** Strict walk-forward & spatial-block benchmark (`LightGBM` vs. `TabPFN-2.5` on thin submarkets vs. PyTorch Geometric `TransformerConv` spatial GNN per `arXiv:2405.06553`), Spatially Weighted Conformal Prediction (`arXiv:2312.06531` / `GeoConformal` `arXiv:2412.08661`), serverless `PMTiles` + interleaved 3D `deck.gl`/`MapLibre` twin + grounded NL2GeoSQL spatial analyst agent.

### 3. Path E (Alt-Data Time-Series) — "Before Zillow Knows": Rezoning & Land-Change Catalysts (`path_e_catalysts`)
- **Core Question:** *Can county rezoning applications, building permits, and space-detected physical land change (AlphaEarth embedding drift) forecast suburban price and demand surges 12–24 months before Zillow/Redfin prices react?*
- **Primary Data:** County GIS rezoning cases & building permits (plus structured LLM extraction from planning commission PDFs where needed), AlphaEarth annual embedding deltas ($\Delta \mathbf{e}_{t-1 \to t}$), Redfin & parcel transaction panels.
- **Core Methods:** Spatial-temporal lead-lag / event-study econometrics (difference-in-differences & walk-forward time-series validation across distance rings: 0.5 mi, 1 mi, 2 mi).

### 4. Path B (Exploratory Feasibility Track) — "Seen From Space & Street": Curb Appeal Valuation (`path_b_streetscape`)
- **Core Question:** *How much of the unexplained price gap between structurally identical suburban homes is explained by visual streetscape quality (tree canopy, sidewalks, architectural upkeep, curb appeal)?*
- **Primary Data:** Free Mapillary crowdsourced street-level imagery + AlphaEarth overhead embeddings + parcel sales.
- **Go / Cut Gate:** First measure real Mapillary image coverage across residential H3 hexes in North Fulton, DeKalb, Cobb, and Gwinnett. Proceed to VLM rubric scoring (`Xue et al. 2025/2026`) only in corridors/hexes with sufficient street-level density; otherwise cut cleanly without affecting Paths C, A, or E.
