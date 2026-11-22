"""Shared Spatial Boundaries: Official 2024 Census County Outlines & 2025 TIGER ZIP Choropleth."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import httpx
import polars as pl
from rich.console import Console

from atl_housing.config import COUNTY_FIPS_LIST, GOLD_DIR, RAW_DIR, SILVER_DIR

console = Console()
CENSUS_COUNTY_2024_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip"


def build_atlanta_county_boundaries() -> Path:
    """Download official 2024 Census county outlines for our 5 Metro Atlanta counties + join 2024 HMDA."""
    raw_dir = RAW_DIR / "tiger"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "cb_2024_us_county_500k.zip"

    if not zip_path.exists():
        console.print("[cyan]Downloading official 2024 US Census county boundaries (~11 MB)...[/cyan]")
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            r = client.get(CENSUS_COUNTY_2024_URL)
            r.raise_for_status()
            zip_path.write_bytes(r.content)

    gdf = gpd.read_file(f"zip://{zip_path}").to_crs("EPSG:4326")
    atl = gdf[gdf["GEOID"].isin(COUNTY_FIPS_LIST)][["GEOID", "NAME", "NAMELSAD", "geometry"]].copy()

    # Join latest 2024 HMDA county credit stats onto each county polygon
    hmda_pq = SILVER_DIR / "path_c" / "atlanta_county_credit_demand.parquet"
    if hmda_pq.exists():
        hmda = (
            pl.read_parquet(hmda_pq)
            .filter(pl.col("activity_year") == 2024)
            .to_pandas()
        )
        atl = atl.merge(
            hmda[
                [
                    "county_code",
                    "purchase_originations",
                    "denial_rate",
                    "investor_purchase_share",
                    "median_property_value",
                    "median_buyer_income_k",
                ]
            ],
            left_on="GEOID",
            right_on="county_code",
            how="left",
        )

    out_dir = GOLD_DIR / "boundaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "atlanta_5counties.geojson"
    atl.to_file(out_path, driver="GeoJSON")
    console.print(f"[green]Saved 5-county Census boundaries:[/green] {len(atl)} counties -> {out_path}")
    return out_path


def build_atlanta_zip_choropleth(simplify_tol: float = 0.0008) -> Path:
    """Build real 2025 Census TIGER ZIP code polygons joined to latest Metro Atlanta liquidity data."""
    zcta_shp = RAW_DIR / "tiger" / "tl_2025_us_zcta520" / "tl_2025_us_zcta520.shp"
    silver_pq = SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet"

    df = pl.read_parquet(silver_pq)
    latest_dt = df["month_date"].max()
    latest = df.filter(pl.col("month_date") == latest_dt)

    latest_zhvi = (
        df.filter(pl.col("zhvi").is_not_null())
        .sort("month_date")
        .group_by("zip_code")
        .agg(pl.col("zhvi").last().alias("latest_zhvi"))
    )
    latest = (
        latest.join(latest_zhvi, on="zip_code", how="left")
        .with_columns(
            pl.col("zip_code").cast(pl.Utf8).str.zfill(5),
            (pl.col("price_cut_share") * 100.0).round(1).alias("price_cut_pct"),
            pl.col("hotness_score").round(1),
            pl.col("demand_score").round(1),
            pl.col("supply_score").round(1),
            pl.col("latest_zhvi").round(0),
        )
        .select(
            [
                "zip_code",
                "city",
                "county_name",
                "median_dom",
                "price_cut_pct",
                "hotness_score",
                "demand_score",
                "supply_score",
                "active_listing_count",
                "median_listing_price",
                "latest_zhvi",
            ]
        )
    )
    atl_zips = set(latest["zip_code"].to_list())

    console.print(f"[cyan]Extracting {len(atl_zips)} real Metro Atlanta ZIP polygons from TIGER 2025...[/cyan]")
    # Use bbox filter for GA/Atlanta to avoid reading all 33K US ZCTA geometries
    gdf = gpd.read_file(zcta_shp, bbox=(-85.4, 33.0, -83.3, 34.7)).to_crs("EPSG:4326")
    gdf = gdf[gdf["ZCTA5CE20"].isin(atl_zips)][
        ["ZCTA5CE20", "INTPTLON20", "INTPTLAT20", "geometry"]
    ].copy()
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=simplify_tol, preserve_topology=True)
    gdf["lon"] = gdf["INTPTLON20"].astype(float)
    gdf["lat"] = gdf["INTPTLAT20"].astype(float)

    merged = gdf.merge(latest.to_pandas(), left_on="ZCTA5CE20", right_on="zip_code", how="inner")
    merged = merged.drop(columns=["ZCTA5CE20", "INTPTLON20", "INTPTLAT20"])

    out_dir = GOLD_DIR / "path_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "atlanta_zips_choropleth.geojson"
    # Serialize cleanly (replace NaN with JSON null)
    raw_dict = json.loads(merged.to_json())
    out_path.write_text(json.dumps(raw_dict))
    console.print(
        f"[green]Saved Metro Atlanta ZIP choropleth GeoJSON:[/green] "
        f"{len(merged)} real ZIP polygons ({out_path.stat().st_size / 1e6:.2f} MB) -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    build_atlanta_county_boundaries()
    build_atlanta_zip_choropleth()
