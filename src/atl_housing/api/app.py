"""FastAPI backend serving the 3D deck.gl/MapLibre Atlanta Housing Intelligence Twin."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from atl_housing.config import BRONZE_DIR, GOLD_DIR, RAW_DIR, SILVER_DIR
from atl_housing.tracks.path_c_liquidity.pricing_simulator import simulate_ask_frontier

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Atlanta Suburban Real Estate Intelligence API",
    version="0.2.0",
    description="Serves Path C liquidity/pricing, Path A 3D GNN + SWCP valuation, Path E catalysts, and Path B streetscapes.",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Representative origin centroids for top out-of-state IRS migration counties feeding Metro Atlanta
ORIGIN_COORDS: dict[tuple[str, str], tuple[float, float]] = {
    ("CA", "Los Angeles County"): (-118.2437, 34.0522),
    ("IL", "Cook County"): (-87.6298, 41.8781),
    ("NY", "New York County"): (-73.9712, 40.7831),
    ("FL", "Broward County"): (-80.2495, 26.1901),
    ("WA", "King County"): (-122.3321, 47.6062),
    ("NY", "Kings County"): (-73.9442, 40.6782),
    ("TX", "Dallas County"): (-96.7970, 32.7767),
    ("CA", "Santa Clara County"): (-121.8863, 37.3382),
    ("FL", "Miami-Dade County"): (-80.1918, 25.7617),
    ("TX", "Harris County"): (-95.3698, 29.7604),
    ("FL", "Palm Beach County"): (-80.0534, 26.7153),
    ("CA", "San Diego County"): (-117.1611, 32.7157),
    ("FL", "Orange County"): (-81.3792, 28.5383),
    ("NY", "Queens County"): (-73.7949, 40.7282),
    ("NC", "Mecklenburg County"): (-80.8431, 35.2271),
    ("CA", "Orange County"): (-117.8311, 33.7175),
    ("FL", "Hillsborough County"): (-82.4572, 27.9506),
    ("TX", "Travis County"): (-97.7431, 30.2672),
    ("AZ", "Maricopa County"): (-112.0740, 33.4484),
    ("MA", "Middlesex County"): (-71.2896, 42.4672),
    ("VA", "Fairfax County"): (-77.2750, 38.8304),
    ("DC", "District of Columbia"): (-77.0369, 38.9072),
    ("PA", "Philadelphia County"): (-75.1652, 39.9526),
    ("NJ", "Bergen County"): (-74.0776, 40.9263),
    ("CA", "Alameda County"): (-122.2711, 37.8044),
}

ATL_COUNTY_COORDS: dict[str, tuple[str, float, float]] = {
    "13121": ("Fulton", -84.3880, 33.8900),
    "13089": ("DeKalb", -84.2266, 33.7715),
    "13067": ("Cobb", -84.5767, 33.9412),
    "13135": ("Gwinnett", -84.0235, 33.9617),
    "13117": ("Forsyth", -84.1249, 34.2254),
}


@lru_cache(maxsize=1)
def _load_zcta_centroids() -> dict[str, tuple[float, float]]:
    """Extract Georgia ZCTA centroids from local TIGER 2025 shapefile (cached in RAM)."""
    shp = RAW_DIR / "tiger" / "tl_2025_us_zcta520" / "tl_2025_us_zcta520.shp"
    if not shp.exists():
        return {}
    gdf = gpd.read_file(shp, columns=["ZCTA5CE20", "INTPTLON20", "INTPTLAT20"], ignore_geometry=True)
    ga = gdf[gdf["ZCTA5CE20"].str.startswith(("30", "31"))]
    return {
        str(row["ZCTA5CE20"]).zfill(5): (float(row["INTPTLON20"]), float(row["INTPTLAT20"]))
        for _, row in ga.iterrows()
    }


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "path_c_silver_exists": (SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet").exists(),
        "path_a_gold_exists": (GOLD_DIR / "path_a" / "parcel_3d_predictions.parquet").exists(),
        "path_e_rezoning_exists": (BRONZE_DIR / "path_e" / "johns_creek_rezoning.parquet").exists(),
        "path_b_streetscape_exists": (BRONZE_DIR / "path_b" / "mapillary_suburb_samples.parquet").exists(),
    }


@app.get("/api/path-a/parcels-3d")
def get_parcels_3d(limit: int = Query(default=2500, ge=100, le=5000)) -> dict[str, Any]:
    """Return 3D residential parcels with PyTorch GNN + LightGBM predictions, SWCP intervals & top-5 comps."""
    pq = GOLD_DIR / "path_a" / "parcel_3d_predictions.parquet"
    if not pq.exists():
        raise HTTPException(status_code=404, detail="Run path_a_valuation.spatial_gnn first.")
    df = pl.read_parquet(pq).head(limit)
    records = []
    for row in df.iter_rows(named=True):
        records.append(
            {
                "id": row["ParcelID"],
                "zip": str(row["ZipCode"] or ""),
                "nbrhood": str(row["NbrHood"] or ""),
                "lon": round(float(row["lon"]), 6),
                "lat": round(float(row["lat"]), 6),
                "height_m": round(float(row["bldg_height_m"] or 6.0), 1),
                "footprint_sqm": round(float(row["total_footprint_sqm"] or 180.0), 1),
                "acres": round(float(row["LandAcres"] or 0.25), 2),
                "actual": float(row["TotAppr"]),
                "lgbm_pred": float(row["lgbm_pred"]),
                "gnn_pred": float(row["gnn_pred"]),
                "hybrid_pred": float(row["hybrid_pred"]),
                "swcp_lower": float(row["swcp_lower"]),
                "swcp_upper": float(row["swcp_upper"]),
                "mispricing_pct": float(row["mispricing_pct"]),
                "comps": json.loads(row["top5_comps_json"]),
            }
        )
    return {"count": len(records), "parcels": records}


@app.get("/api/path-c/zip-liquidity")
def get_zip_liquidity() -> dict[str, Any]:
    """Return latest-month Metro Atlanta ZIP liquidity, hotness, price-cut share, and centroids."""
    pq = SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet"
    if not pq.exists():
        raise HTTPException(status_code=404, detail="Run path_c_liquidity.silver_panel first.")

    df = pl.read_parquet(pq)
    latest_dt = df["month_date"].max()
    latest = df.filter(pl.col("month_date") == latest_dt)

    # Grab most recent non-null ZHVI per ZIP (Zillow CSV ends Dec 2025 while Realtor is Oct 2026)
    latest_zhvi = (
        df.filter(pl.col("zhvi").is_not_null())
        .sort("month_date")
        .group_by("zip_code")
        .agg(pl.col("zhvi").last().alias("latest_zhvi"))
    )
    latest = latest.join(latest_zhvi, on="zip_code", how="left")
    centroids = _load_zcta_centroids()

    items = []
    for row in latest.iter_rows(named=True):
        zc = str(row["zip_code"]).zfill(5)
        if zc not in centroids:
            continue
        lon, lat = centroids[zc]
        items.append(
            {
                "zip_code": zc,
                "city": row["city"] or zc,
                "county": row["county_name"] or "",
                "lon": lon,
                "lat": lat,
                "median_dom": float(row["median_dom"]) if row["median_dom"] is not None else None,
                "price_cut_pct": round(float(row["price_cut_share"]) * 100, 1)
                if row["price_cut_share"] is not None
                else None,
                "hotness_score": round(float(row["hotness_score"]), 1)
                if row["hotness_score"] is not None
                else None,
                "demand_score": round(float(row["demand_score"]), 1)
                if row["demand_score"] is not None
                else None,
                "supply_score": round(float(row["supply_score"]), 1)
                if row["supply_score"] is not None
                else None,
                "active_listings": int(row["active_listing_count"] or 0),
                "median_list_price": float(row["median_listing_price"])
                if row["median_listing_price"] is not None
                else None,
                "latest_zhvi": round(float(row["latest_zhvi"]), 0)
                if row["latest_zhvi"] is not None
                else None,
            }
        )
    return {"as_of": str(latest_dt), "count": len(items), "zips": items}


@app.get("/api/path-c/migration-arcs")
def get_migration_arcs() -> dict[str, Any]:
    """Return top out-of-state IRS wealth migration arcs into our 5 Metro Atlanta counties."""
    pq = BRONZE_DIR / "path_c" / "irs_atlanta_migration_2122.parquet"
    if not pq.exists():
        raise HTTPException(status_code=404, detail="IRS migration Bronze Parquet missing.")

    irs = pl.read_parquet(pq)
    inflows = irs.filter(
        (pl.col("direction") == "inflow")
        & (pl.col("y1_statefips") != 13)
        & (pl.col("y1_statefips") < 57)
        & (pl.col("y1_countyfips") != 0)
    )
    arcs = []
    for row in inflows.sort("agi", descending=True).iter_rows(named=True):
        key = (str(row["y1_state"]), str(row["y1_countyname"]))
        dest_fips = str(row["dest_fips"])
        if key not in ORIGIN_COORDS or dest_fips not in ATL_COUNTY_COORDS:
            continue
        o_lon, o_lat = ORIGIN_COORDS[key]
        d_name, d_lon, d_lat = ATL_COUNTY_COORDS[dest_fips]
        households = int(row["n1"])
        agi_m = round(float(row["agi"]) / 1e3, 1)  # IRS AGI is in $1,000s -> $M
        arcs.append(
            {
                "origin_label": f"{key[1]}, {key[0]}",
                "dest_county": d_name,
                "source": [o_lon, o_lat],
                "target": [d_lon, d_lat],
                "households": households,
                "people": int(row["n2"]),
                "agi_millions": agi_m,
                "avg_agi_k": round(float(row["agi"]) / max(households, 1), 1),
            }
        )
    return {"count": len(arcs[:60]), "arcs": arcs[:60]}


@app.get("/api/path-c/simulate-pricing")
def api_simulate_pricing(
    zip_code: str = Query(default="30075", min_length=5, max_length=5),
    fair_value: float = Query(default=550000.0, ge=100000.0, le=5000000.0),
) -> dict[str, Any]:
    """Run live Path C Seller Ask-Price vs. Liquidity & Carrying-Cost Frontier simulation."""
    try:
        res = simulate_ask_frontier(zip_code, fair_value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    rows = res.to_dicts()
    best_idx = int(res["expected_net"].arg_max())
    return {
        "zip_code": zip_code,
        "fair_value": fair_value,
        "optimal_index": best_idx,
        "optimal_row": rows[best_idx],
        "frontier": rows,
    }


@app.get("/api/path-e/catalysts")
def get_catalysts() -> dict[str, Any]:
    """Return geolocated Path E rezoning cases & land development permits."""
    points: list[dict[str, Any]] = []
    jc_pq = BRONZE_DIR / "path_e" / "johns_creek_rezoning.parquet"
    if jc_pq.exists():
        gdf = gpd.read_parquet(jc_pq)
        pts = gdf.to_crs("EPSG:26916").geometry.centroid.to_crs("EPSG:4326")
        for idx, row in gdf.iterrows():
            pt = pts.loc[idx]
            if pt is None or pt.is_empty:
                continue
            points.append(
                {
                    "type": "Rezoning (Johns Creek)",
                    "label": str(row.get("ProjectType") or row.get("CaseID") or "Rezoning"),
                    "detail": str(row.get("RequestedUse") or row.get("ExistingZoning") or ""),
                    "lon": round(float(pt.x), 6),
                    "lat": round(float(pt.y), 6),
                }
            )
    gw_pq = BRONZE_DIR / "path_e" / "gwinnett_ldar_dev.parquet"
    if gw_pq.exists():
        gdf = gpd.read_parquet(gw_pq)
        pts = gdf.to_crs("EPSG:26916").geometry.centroid.to_crs("EPSG:4326")
        for idx, row in gdf.iterrows():
            pt = pts.loc[idx]
            if pt is None or pt.is_empty:
                continue
            points.append(
                {
                    "type": "Land Dev Permit (Gwinnett)",
                    "label": str(row.get("DEVELOPER1") or row.get("PERMIT_NUMBER") or "Permit"),
                    "detail": str(row.get("TYPE") or ""),
                    "lon": round(float(pt.x), 6),
                    "lat": round(float(pt.y), 6),
                }
            )
    return {"count": len(points), "catalysts": points}


@app.get("/api/path-b/streetscape")
def get_streetscape_samples() -> dict[str, Any]:
    """Return sampled Mapillary streetscape points across our 5 Metro Atlanta suburban hexes."""
    pq = BRONZE_DIR / "path_b" / "mapillary_suburb_samples.parquet"
    if not pq.exists():
        return {"count": 0, "samples": []}
    df = pl.read_parquet(pq).filter(pl.col("lon").is_not_null() & pl.col("lat").is_not_null())
    return {"count": df.height, "samples": df.to_dicts()}
