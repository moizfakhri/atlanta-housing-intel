"""Path A (Core Pillar) Bronze Ingestion: Overture Maps 3D Buildings/Places & County Parcel APIs."""

from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import httpx
from rich.console import Console

from atl_housing.config import (
    BRONZE_DIR,
    COUNTY_PARCEL_ENDPOINTS,
    TEST_SUBURB_BBOX,
)

console = Console()
PATH_A_BRONZE = BRONZE_DIR / "path_a"
OVERTURE_RELEASE = "2026-11-18.0"


def ingest_overture_3d_buildings(
    bbox: tuple[float, float, float, float] = TEST_SUBURB_BBOX,
    tag: str = "north_atl_corridor",
) -> Path:
    """Query Overture Maps S3 GeoParquet 3D buildings directly into local Bronze GeoParquet."""
    PATH_A_BRONZE.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    out_path = PATH_A_BRONZE / f"overture_3d_buildings_{tag}.parquet"

    s3_path = (
        f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=buildings/type=building/*"
    )
    console.print(
        f"[cyan]Streaming Overture {OVERTURE_RELEASE} 3D buildings for {tag} {bbox}...[/cyan]"
    )

    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2'; SET geometry_always_xy = true;")
    con.execute(
        f"""
        COPY (
            SELECT
                id,
                class,
                subtype,
                height,
                num_floors,
                roof_shape,
                ST_Area_Spheroid(geometry) AS footprint_sqm,
                ST_X(ST_Centroid(geometry)) AS lon,
                ST_Y(ST_Centroid(geometry)) AS lat,
                ST_AsWKB(geometry) AS geom_wkb
            FROM read_parquet('{s3_path}', hive_partitioning=1)
            WHERE bbox.xmin >= {lon_min} AND bbox.xmax <= {lon_max}
              AND bbox.ymin >= {lat_min} AND bbox.ymax <= {lat_max}
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    row = con.execute(
        f"SELECT COUNT(*), COUNT(height), ROUND(AVG(height), 2) FROM read_parquet('{out_path}')"
    ).fetchone()
    console.print(
        f"[green]Saved Overture 3D buildings ({tag}):[/green] {row[0]:,} buildings "
        f"({row[1]:,} with height, avg {row[2]}m) -> {out_path}"
    )
    return out_path


def ingest_county_parcels_sample(
    county: str,
    max_records: int = 10_000,
    page_size: int = 2000,
) -> Path:
    """Paginated fetch from a verified Metro Atlanta county ArcGIS Parcel FeatureServer."""
    assert county in COUNTY_PARCEL_ENDPOINTS, f"Unknown county: {county}"
    PATH_A_BRONZE.mkdir(parents=True, exist_ok=True)
    base_url = COUNTY_PARCEL_ENDPOINTS[county]
    out_path = PATH_A_BRONZE / f"parcels_{county}_sample.parquet"

    frames: list[gpd.GeoDataFrame] = []
    offset = 0
    console.print(f"[cyan]Fetching up to {max_records:,} parcels from {county.title()} ArcGIS...[/cyan]")

    with httpx.Client(timeout=45.0, headers={"User-Agent": "atlanta-housing-intel/0.1"}) as client:
        while offset < max_records:
            batch = min(page_size, max_records - offset)
            params = {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": batch,
                "f": "geojson",
            }
            r = client.get(f"{base_url}/query", params=params)
            r.raise_for_status()
            payload = r.json()
            feats = payload.get("features", [])
            if not feats:
                break
            gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
            frames.append(gdf)
            offset += len(feats)
            console.print(f"  {county.title()}: fetched {offset:,} parcels...")
            if len(feats) < batch:
                break

    if not frames:
        raise RuntimeError(f"No parcel features returned for {county}")

    full_gdf = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    full_gdf.to_parquet(out_path)
    console.print(
        f"[green]Saved {county.title()} parcels:[/green] {len(full_gdf):,} rows, "
        f"{len(full_gdf.columns)} cols -> {out_path}"
    )
    return out_path
