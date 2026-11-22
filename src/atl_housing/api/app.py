"""FastAPI backend serving the 3D deck.gl/MapLibre Atlanta Housing Intelligence Twin."""

from __future__ import annotations

import contextlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from atl_housing.config import BRONZE_DIR, DUCKDB_PATH, GOLD_DIR, RAW_DIR, SILVER_DIR
from atl_housing.tracks.path_c_liquidity.pricing_simulator import simulate_ask_frontier

STATIC_DIR = Path(__file__).resolve().parent / "static"
METRO_3D_DB = GOLD_DIR / "path_a" / "metro_3d_buildings.duckdb"

app = FastAPI(
    title="Atlanta Suburban Real Estate Intelligence API",
    version="0.4.0",
    description="Serves Path C liquidity/pricing/LODES, Path A 5-county viewport-streamed 3D GNN+SWCP buildings, Path E catalysts, Path B streetscapes, and Grounded Spatial SQL.",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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


_METRO_3D_CACHE: dict[str, Any] = {"mtime": None, "con": None}


def _get_metro_3d_cursor() -> duckdb.DuckDBPyConnection:
    """Return a thread-safe read-only DuckDB cursor over the 5-county 3D spatial index."""
    if not METRO_3D_DB.exists():
        raise HTTPException(
            status_code=404,
            detail="Run atl_housing.tracks.path_a_valuation.metro_3d_index first.",
        )
    mtime = METRO_3D_DB.stat().st_mtime
    if _METRO_3D_CACHE["con"] is None or _METRO_3D_CACHE["mtime"] != mtime:
        if _METRO_3D_CACHE["con"] is not None:
            with contextlib.suppress(Exception):
                _METRO_3D_CACHE["con"].close()
        con = duckdb.connect(str(METRO_3D_DB), read_only=True)
        con.execute("LOAD spatial; SET geometry_always_xy = true;")
        _METRO_3D_CACHE["con"] = con
        _METRO_3D_CACHE["mtime"] = mtime
    return _METRO_3D_CACHE["con"].cursor()


_BLDG_SELECT_COLS = """
    overture_id, lon, lat, height_m, height_measured, num_floors, footprint_sqm,
    bldg_type, zip_code, city, county_name, zip_zhvi, zip_hotness, zip_price_cut_pct,
    zip_median_dom, has_gnn, parcel_id, n_units, actual, hybrid_pred, gnn_pred,
    lgbm_pred, swcp_lower, swcp_upper, mispricing_pct, comps_json,
    ST_AsGeoJSON(ST_GeomFromWKB(geom_wkb)) AS geom_json
"""


def _row_to_building_feature(r: tuple[Any, ...]) -> dict[str, Any]:
    comps = json.loads(r[25]) if r[25] else []
    return {
        "type": "Feature",
        "geometry": json.loads(r[26]),
        "properties": {
            "overture_id": str(r[0]),
            "lon": float(r[1]),
            "lat": float(r[2]),
            "height_m": round(float(r[3]), 1),
            "height_measured": bool(r[4]),
            "num_floors": int(r[5]) if r[5] is not None else None,
            "footprint_sqm": round(float(r[6]), 1),
            "bldg_type": str(r[7] or "building"),
            "zip": str(r[8] or ""),
            "city": str(r[9] or ""),
            "county": str(r[10] or ""),
            "zip_zhvi": round(float(r[11]), 0) if r[11] is not None else None,
            "zip_hotness": round(float(r[12]), 1) if r[12] is not None else None,
            "zip_price_cut_pct": round(float(r[13]), 1) if r[13] is not None else None,
            "zip_median_dom": round(float(r[14]), 1) if r[14] is not None else None,
            "has_gnn": bool(r[15]),
            "id": str(r[16]) if r[16] is not None else str(r[0]),
            "n_units": int(r[17] or 1),
            "actual": float(r[18]) if r[18] is not None else None,
            "hybrid_pred": float(r[19]) if r[19] is not None else None,
            "gnn_pred": float(r[20]) if r[20] is not None else None,
            "lgbm_pred": float(r[21]) if r[21] is not None else None,
            "swcp_lower": float(r[22]) if r[22] is not None else None,
            "swcp_upper": float(r[23]) if r[23] is not None else None,
            "mispricing_pct": float(r[24]) if r[24] is not None else None,
            "comps": comps,
        },
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
        "path_a_footprints_exists": (GOLD_DIR / "path_a" / "buildings_3d_footprints.geojson").exists(),
        "path_a_metro_3d_index_exists": METRO_3D_DB.exists(),
        "path_e_rezoning_exists": (BRONZE_DIR / "path_e" / "johns_creek_rezoning.parquet").exists(),
        "path_b_streetscape_exists": (BRONZE_DIR / "path_b" / "mapillary_suburb_samples.parquet").exists(),
    }


@app.get("/api/path-a/buildings-viewport")
def get_buildings_viewport(
    west: float = Query(..., ge=-86.0, le=-82.5),
    south: float = Query(..., ge=32.5, le=35.5),
    east: float = Query(..., ge=-86.0, le=-82.5),
    north: float = Query(..., ge=32.5, le=35.5),
    zoom: float = Query(default=15.0, ge=5.0, le=22.0),
    selected_oid: str | None = Query(default=None),
    limit: int = Query(default=3200, ge=200, le=5000),
) -> dict[str, Any]:
    """Sub-10ms viewport-streamed 5-county Metro Atlanta 3D buildings with zoom-adaptive LOD.

    Only returns buildings currently inside the camera viewport, prioritized by proximity
    to screen center, and ALWAYS guarantees the selected building + its 5 comp buildings
    are present in the returned GeoJSON so comp arcs can never point to missing buildings.
    """
    if zoom < 13.0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "meta": {"too_far_out": True, "min_zoom": 13.0, "returned": 0, "gnn_count": 0},
        }

    if west > east:
        west, east = east, west
    if south > north:
        south, north = north, south

    # Clamp pathological huge bounding boxes to avoid pulling half the metro at once
    if (east - west) > 0.25 or (north - south) > 0.25:
        cx, cy = (west + east) / 2.0, (south + north) / 2.0
        west, east = cx - 0.08, cx + 0.08
        south, north = cy - 0.06, cy + 0.06

    gx_min, gx_max = math.floor(west * 200), math.floor(east * 200)
    gy_min, gy_max = math.floor(south * 200), math.floor(north * 200)
    cx, cy = (west + east) / 2.0, (south + north) / 2.0

    # Zoom-adaptive LOD: at medium zoom (13.0-14.3), skip tiny sheds/garages for smooth 60 FPS
    lod_filter = "AND (footprint_sqm >= 100.0 OR has_gnn)" if zoom < 14.3 else ""
    eff_limit = min(limit, 2200) if zoom < 14.3 else limit

    cur = _get_metro_3d_cursor()
    rows = cur.execute(
        f"""
        SELECT {_BLDG_SELECT_COLS}
        FROM buildings_3d
        WHERE gx BETWEEN ? AND ?
          AND gy BETWEEN ? AND ?
          AND lon BETWEEN ? AND ?
          AND lat BETWEEN ? AND ?
          {lod_filter}
        ORDER BY ((lon - ?) * (lon - ?) + (lat - ?) * (lat - ?)) ASC
        LIMIT ?
        """,
        [gx_min, gx_max, gy_min, gy_max, west, east, south, north, cx, cx, cy, cy, eff_limit],
    ).fetchall()

    feats_by_oid: dict[str, dict[str, Any]] = {}
    for r in rows:
        f = _row_to_building_feature(r)
        feats_by_oid[f["properties"]["overture_id"]] = f

    # Belt-and-suspenders: guarantee selected building AND all 5 of its comp buildings are in the payload!
    if selected_oid:
        needed: set[str] = set()
        if selected_oid not in feats_by_oid:
            needed.add(selected_oid)
            sel_row = cur.execute(
                f"SELECT {_BLDG_SELECT_COLS} FROM buildings_3d WHERE overture_id = ?",
                [selected_oid],
            ).fetchone()
            if sel_row is not None:
                sf = _row_to_building_feature(sel_row)
                feats_by_oid[selected_oid] = sf
        sel_feat = feats_by_oid.get(selected_oid)
        if sel_feat is not None:
            for c in sel_feat["properties"].get("comps", []):
                coid = c.get("comp_overture_id")
                if coid and coid not in feats_by_oid:
                    needed.add(str(coid))
        missing_oids = [oid for oid in needed if oid not in feats_by_oid]
        if missing_oids:
            placeholders = ",".join("?" for _ in missing_oids)
            extra_rows = cur.execute(
                f"SELECT {_BLDG_SELECT_COLS} FROM buildings_3d WHERE overture_id IN ({placeholders})",
                missing_oids,
            ).fetchall()
            for r in extra_rows:
                f = _row_to_building_feature(r)
                feats_by_oid[f["properties"]["overture_id"]] = f

    features = list(feats_by_oid.values())
    gnn_count = sum(1 for f in features if f["properties"]["has_gnn"])
    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "too_far_out": False,
            "returned": len(features),
            "gnn_count": gnn_count,
            "lod": "full" if zoom >= 14.3 else "medium",
        },
    }


@app.get("/api/path-a/featured-home")
def get_featured_home() -> dict[str, Any]:
    """Return a clean default single-family Sandy Springs home with measured height & 5 comps."""
    cur = _get_metro_3d_cursor()
    row = cur.execute(
        f"""
        SELECT {_BLDG_SELECT_COLS}
        FROM buildings_3d
        WHERE has_gnn = TRUE
          AND height_measured = TRUE
          AND n_units = 1
          AND bldg_type IN ('house', 'detached', 'residential', 'building')
          AND actual BETWEEN 450000 AND 1200000
          AND mispricing_pct BETWEEN 4 AND 35
        ORDER BY ((lon + 84.376) * (lon + 84.376) + (lat - 33.930) * (lat - 33.930)) ASC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No featured GNN home found.")
    return _row_to_building_feature(row)


@app.get("/api/path-a/buildings-3d")
def get_buildings_3d_footprints() -> FileResponse:
    """Serve verified Sandy Springs Overture 3D footprints joined to GNN+SWCP predictions."""
    gj = GOLD_DIR / "path_a" / "buildings_3d_footprints.geojson"
    if not gj.exists():
        raise HTTPException(status_code=404, detail="Run path_a_valuation.metro_3d_index first.")
    return FileResponse(gj, media_type="application/geo+json")


@app.get("/api/boundaries/counties")
def get_county_boundaries() -> FileResponse:
    """Serve official 2024 Census 5-county Metro Atlanta outlines with 2024 HMDA credit stats."""
    gj = GOLD_DIR / "boundaries" / "atlanta_5counties.geojson"
    if not gj.exists():
        raise HTTPException(status_code=404, detail="Run atl_housing.lakehouse.boundaries first.")
    return FileResponse(gj, media_type="application/geo+json")


@app.get("/api/path-c/zips-geojson")
def get_zips_choropleth_geojson() -> FileResponse:
    """Serve real 2025 TIGER Metro Atlanta ZIP polygons joined to latest liquidity & hotness metrics."""
    gj = GOLD_DIR / "path_c" / "atlanta_zips_choropleth.geojson"
    if not gj.exists():
        raise HTTPException(status_code=404, detail="Run atl_housing.lakehouse.boundaries first.")
    return FileResponse(gj, media_type="application/geo+json")


@app.get("/api/path-c/metro-timeseries")
def get_metro_timeseries() -> dict[str, Any]:
    """Return 2016-2026 monthly Metro Atlanta liquidity, price-cut share, ZHVI, and FRED 30Y rate."""
    pq = SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet"
    if not pq.exists():
        raise HTTPException(status_code=404, detail="Silver panel missing.")
    con = duckdb.connect(":memory:")
    rows = con.sql(
        f"""
        SELECT
            STRFTIME(month_date, '%Y-%m') AS month,
            ROUND(MEDIAN(median_dom), 1) AS median_dom,
            ROUND(AVG(price_cut_share) * 100, 1) AS avg_price_cut_pct,
            ROUND(AVG(hotness_score), 1) AS avg_hotness,
            ROUND(MEDIAN(zhvi), 0) AS median_zhvi,
            ROUND(AVG(mortgage_30y), 2) AS mortgage_30y
        FROM read_parquet('{pq}')
        WHERE active_listing_count >= 10
        GROUP BY month_date
        ORDER BY month_date
        """
    ).fetchall()
    return {
        "series": [
            {
                "month": r[0],
                "median_dom": r[1],
                "avg_price_cut_pct": r[2],
                "avg_hotness": r[3],
                "median_zhvi": r[4],
                "mortgage_30y": r[5],
            }
            for r in rows
        ]
    }


@app.get("/api/path-c/zip-history/{zip_code}")
def get_zip_history(zip_code: str) -> dict[str, Any]:
    """Return 2016-2026 monthly liquidity & price history for a single Metro Atlanta ZIP."""
    zc = zip_code.strip().zfill(5)
    pq = SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet"
    if not pq.exists():
        raise HTTPException(status_code=404, detail="Silver panel missing.")
    df = (
        pl.read_parquet(pq)
        .filter(pl.col("zip_code") == zc)
        .sort("month_date")
        .select(
            [
                pl.col("month_date").dt.strftime("%Y-%m").alias("month"),
                pl.col("median_dom"),
                (pl.col("price_cut_share") * 100.0).round(1).alias("price_cut_pct"),
                pl.col("hotness_score").round(1),
                pl.col("zhvi").round(0),
                pl.col("active_listing_count"),
            ]
        )
    )
    if df.height == 0:
        raise HTTPException(status_code=404, detail=f"ZIP {zc} not found.")
    return {"zip_code": zc, "history": df.to_dicts()}


@app.get("/api/analytics/summary")
def get_analytics_summary() -> dict[str, Any]:
    """Return live HMDA 5-county credit summary, Path B GVI scores, and verified model benchmarks."""
    hmda_pq = SILVER_DIR / "path_c" / "atlanta_county_credit_demand.parquet"
    hmda_2024 = (
        pl.read_parquet(hmda_pq).filter(pl.col("activity_year") == 2024).sort("county_name").to_dicts()
        if hmda_pq.exists()
        else []
    )
    path_b_pq = GOLD_DIR / "path_b" / "suburb_streetscape_scores.parquet"
    streetscape = pl.read_parquet(path_b_pq).to_dicts() if path_b_pq.exists() else []
    return {
        "hmda_2024": hmda_2024,
        "streetscape_scores": streetscape,
        "benchmarks": {
            "path_a_models": [
                {"label": "Parcel Baseline (LGBM)", "mape": 25.96, "rmse": 561382},
                {"label": "+ Overture 3D (Pilot)", "mape": 20.83, "rmse": 467010},
                {"label": "Full-City LGBM (27.8K)", "mape": 14.84, "rmse": 259217},
                {"label": "PyTorch Spatial GNN (K=12)", "mape": 13.98, "rmse": 270051},
                {"label": "Hybrid GNN + LGBM", "mape": 13.56, "rmse": 251252},
            ],
            "spatial_cqr": [
                {"tier": "Starter (<33%)", "global_cp": 98.4, "spatial_cqr": 93.0, "width": 555891},
                {"tier": "Upper-Mid (33-67%)", "global_cp": 98.8, "spatial_cqr": 90.4, "width": 1071842},
                {"tier": "Luxury (>67%)", "global_cp": 74.1, "spatial_cqr": 88.9, "width": 1798852},
            ],
            "survival_hazards": [
                {"factor": "Supply Tightness (+1 SD)", "hr": 2.376, "dom_pct": -17.1},
                {"factor": "Price-Cut Capitulation (+1 SD)", "hr": 1.353, "dom_pct": -6.3},
                {"factor": "Buyer Demand (+1 SD)", "hr": 1.025, "dom_pct": -0.2},
                {"factor": "30Y Mortgage Rate (+1 SD)", "hr": 0.794, "dom_pct": 5.7},
                {"factor": "Lagged HMDA Denial Rate (+1 SD)", "hr": 0.764, "dom_pct": 5.9},
            ],
            "path_c_walkforward": {
                "excess_6m_mae_before": 2.31,
                "excess_6m_mae_after": 2.05,
                "excess_6m_ic_before": 0.477,
                "excess_6m_ic_after": 0.557,
                "dom_3m_mae_before": 17.29,
                "dom_3m_mae_after": 12.08,
                "dom_3m_ic_before": 0.041,
                "dom_3m_ic_after": 0.583,
            },
        },
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
        agi_m = round(float(row["agi"]) / 1e3, 1)
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


@app.get("/api/path-c/commute-arcs")
def get_commute_arcs() -> dict[str, Any]:
    """Return cross-county Metro Atlanta morning commuter arcs from Census LODES8."""
    pq = BRONZE_DIR / "path_c" / "lodes_atlanta_county_od.parquet"
    if not pq.exists():
        return {"count": 0, "arcs": []}
    df = pl.read_parquet(pq).filter(pl.col("home_county_fips") != pl.col("work_county_fips"))
    arcs = []
    for row in df.iter_rows(named=True):
        hf = str(row["home_county_fips"])
        wf = str(row["work_county_fips"])
        if hf not in ATL_COUNTY_COORDS or wf not in ATL_COUNTY_COORDS:
            continue
        h_name, h_lon, h_lat = ATL_COUNTY_COORDS[hf]
        w_name, w_lon, w_lat = ATL_COUNTY_COORDS[wf]
        arcs.append(
            {
                "home_county": h_name,
                "work_county": w_name,
                "source": [h_lon, h_lat],
                "target": [w_lon, w_lat],
                "commuters": int(row["total_commuters"]),
                "high_wage_commuters": int(row["high_wage_commuters"]),
                "high_wage_pct": round(float(row["high_wage_share"]) * 100, 1),
            }
        )
    return {"count": len(arcs), "arcs": arcs}


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
    scores_pq = GOLD_DIR / "path_b" / "suburb_streetscape_scores.parquet"
    scores = pl.read_parquet(scores_pq).to_dicts() if scores_pq.exists() else []
    return {"count": df.height, "samples": df.to_dicts(), "suburb_scores": scores}


@app.get("/api/spatial-query")
def spatial_sql_analyst(q: str = Query(..., min_length=3, max_length=240)) -> dict[str, Any]:
    """Grounded NL2GeoSQL analyst executing verified read-only DuckDB SQL over Gold marts."""
    ql = q.lower()
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    centroids = _load_zcta_centroids()

    # Extract optional dollar budget (e.g., "under $500k" or "< 450000")
    price_cap: float | None = None
    m_k = re.search(r"(\d{3,4})\s*k", ql)
    m_raw = re.search(r"\$?(\d{6,7})", ql)
    if m_k:
        price_cap = float(m_k.group(1)) * 1000.0
    elif m_raw:
        price_cap = float(m_raw.group(1))

    # Route 1: Parcel-level 3D undervalued / low-uncertainty query
    if any(w in ql for w in ["parcel", "undervalued", "building", "gnn", "3d", "home"]):
        if METRO_3D_DB.exists():
            con.close()
            cur = _get_metro_3d_cursor()
            cap_clause = f"AND actual <= {price_cap}" if price_cap else ""
            sql = f"""
                SELECT overture_id, parcel_id AS id, zip_code AS zip, actual AS appraisal,
                       hybrid_pred, mispricing_pct, swcp_lower, swcp_upper, lon, lat
                FROM buildings_3d
                WHERE has_gnn = TRUE AND n_units = 1 AND mispricing_pct > 5 {cap_clause}
                ORDER BY mispricing_pct DESC
                LIMIT 8
            """
            df = cur.execute(sql).df()
        else:
            gold_pq = GOLD_DIR / "path_a" / "parcel_3d_predictions.parquet"
            cap_clause = f"AND TotAppr <= {price_cap}" if price_cap else ""
            sql = f"""
                SELECT ParcelID AS id, CAST(CAST(ZipCode AS BIGINT) AS VARCHAR) AS zip, TotAppr AS appraisal,
                       hybrid_pred, mispricing_pct, swcp_lower, swcp_upper, lon, lat
                FROM read_parquet('{gold_pq}')
                WHERE mispricing_pct > 5 {cap_clause}
                ORDER BY mispricing_pct DESC
                LIMIT 8
            """
            df = con.sql(sql).df()
            con.close()
        rows = df.to_dict(orient="records")
        fly = (
            {
                "mode": "path_a",
                "lon": float(rows[0]["lon"]),
                "lat": float(rows[0]["lat"]),
                "zoom": 16.0,
                "overture_id": str(rows[0].get("overture_id") or ""),
            }
            if rows
            else {"mode": "path_a", "lon": -84.376, "lat": 33.930, "zoom": 15.6}
        )
        return {
            "query": q,
            "sql": " ".join(sql.split()),
            "summary": f"Found {len(rows)} undervalued homes where AI estimate exceeds tax appraisal by >5%.",
            "fly_to": fly,
            "rows": rows,
        }

    # Route 2: Buyer leverage / highest price cuts vs. Hot seller demand ZIPs
    county_filter = ""
    for cname in ["Fulton", "DeKalb", "Cobb", "Gwinnett", "Forsyth"]:
        if cname.lower() in ql:
            county_filter = f"AND county_name = '{cname} County'"
            break

    is_buyer_leverage = any(w in ql for w in ["cut", "leverage", "slow", "discount", "stale", "buyer"])
    order_col = "price_cut_share DESC" if is_buyer_leverage else "hotness_score DESC"
    cap_clause = f"AND median_listing_price <= {price_cap}" if price_cap else ""

    sql = f"""
        WITH latest AS (
            SELECT MAX(month_date) AS max_dt FROM main.mart_atlanta_submarket_signals
        )
        SELECT zip_code, city, county_name,
               ROUND(median_dom, 1) AS median_dom,
               ROUND(price_cut_share * 100, 1) AS price_cut_pct,
               ROUND(hotness_score, 1) AS hotness_score,
               ROUND(median_listing_price, 0) AS median_list_price,
               market_regime
        FROM main.mart_atlanta_submarket_signals, latest
        WHERE month_date = latest.max_dt
          AND active_listing_count >= 25
          {county_filter}
          {cap_clause}
        ORDER BY {order_col}
        LIMIT 8
    """
    df = con.sql(sql).df()
    con.close()
    rows = df.to_dict(orient="records")
    top_zip = str(rows[0]["zip_code"]).zfill(5) if rows else "30075"
    lon, lat = centroids.get(top_zip, (-84.35, 33.86))
    return {
        "query": q,
        "sql": " ".join(sql.split()),
        "summary": (
            f"Top {len(rows)} Metro Atlanta ZIPs ranked by "
            f"{'seller price-cut share (buyer leverage)' if is_buyer_leverage else 'buyer hotness score'}."
        ),
        "fly_to": {
            "mode": "path_c_towers",
            "lon": lon,
            "lat": lat,
            "zoom": 11.4,
            "color_by_cut": is_buyer_leverage,
        },
        "rows": rows,
    }
