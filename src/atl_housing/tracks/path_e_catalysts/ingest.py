"""Path E (Alt-Data Catalysts): Live ingestion & inspection of Metro Atlanta Rezoning & Permits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
from rich.console import Console
from rich.table import Table

from atl_housing.config import BRONZE_DIR

console = Console()
PATH_E_BRONZE = BRONZE_DIR / "path_e"

# Verified Metro Atlanta suburban catalyst FeatureServers discovered via ArcGIS Online catalog
CATALYST_ENDPOINTS: dict[str, str] = {
    "johns_creek_rezoning": "https://services1.arcgis.com/bqfNVPUK3HOnCFmA/arcgis/rest/services/PZ_Rezoning_Cases/FeatureServer/0",
    "peachtree_corners_permits": "https://services3.arcgis.com/jEYjQXUX2JTVoctH/arcgis/rest/services/Building_Permits/FeatureServer/0",
    "gwinnett_ldar_dev": "https://services3.arcgis.com/RfpmnkSAQleRbndX/arcgis/rest/services/LDAR_GIS_Data_ArcGISonline/FeatureServer/0",
}


def discover_atlanta_catalyst_layers() -> dict[str, list[dict[str, Any]]]:
    """Search ArcGIS Online public catalog for Atlanta-metro rezoning & building permit FeatureServers."""
    queries = {
        "rezoning": 'Atlanta OR Fulton OR Gwinnett OR DeKalb OR Cobb OR Forsyth rezoning type:"Feature Service"',
        "building_permits": 'Atlanta OR Fulton OR Gwinnett OR DeKalb OR Cobb building permits type:"Feature Service"',
    }
    results: dict[str, list[dict[str, Any]]] = {}
    with httpx.Client(timeout=20.0) as client:
        for category, q in queries.items():
            r = client.get(
                "https://www.arcgis.com/sharing/rest/search",
                params={"q": q, "f": "json", "num": 15},
            )
            r.raise_for_status()
            items = r.json().get("results", [])
            results[category] = [
                {
                    "title": it.get("title"),
                    "owner": it.get("owner"),
                    "url": it.get("url"),
                    "numViews": it.get("numViews"),
                }
                for it in items
                if it.get("url")
            ]
    return results


def ingest_catalyst_layers(max_records_per_layer: int = 10_000) -> dict[str, Path]:
    """Pull verified Atlanta suburban rezoning & building permit layers into Bronze GeoParquet."""
    PATH_E_BRONZE.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    tbl = Table(title="Path E Live Catalyst Ingestion (Metro Atlanta Suburbs)")
    tbl.add_column("Layer", style="cyan")
    tbl.add_column("Records", justify="right")
    tbl.add_column("Cols", justify="right")
    tbl.add_column("Sample Fields")

    with httpx.Client(timeout=45.0, headers={"User-Agent": "atlanta-housing-intel/0.1"}) as client:
        for name, base_url in CATALYST_ENDPOINTS.items():
            frames: list[gpd.GeoDataFrame] = []
            offset = 0
            page_size = 2000
            try:
                while offset < max_records_per_layer:
                    batch = min(page_size, max_records_per_layer - offset)
                    r = client.get(
                        f"{base_url}/query",
                        params={
                            "where": "1=1",
                            "outFields": "*",
                            "returnGeometry": "true",
                            "outSR": "4326",
                            "resultOffset": offset,
                            "resultRecordCount": batch,
                            "f": "geojson",
                        },
                    )
                    r.raise_for_status()
                    feats = r.json().get("features", [])
                    if not feats:
                        break
                    frames.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
                    offset += len(feats)
                    if len(feats) < batch:
                        break

                if frames:
                    gdf = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs="EPSG:4326")
                    out_path = PATH_E_BRONZE / f"{name}.parquet"
                    gdf.to_parquet(out_path)
                    outputs[name] = out_path
                    sample_cols = [c for c in gdf.columns if c != "geometry"][:8]
                    tbl.add_row(name, f"{len(gdf):,}", str(len(gdf.columns)), ", ".join(sample_cols))
                else:
                    tbl.add_row(name, "0", "0", "No features returned")
            except Exception as e:
                tbl.add_row(name, "ERR", "-", str(e)[:60])

    console.print(tbl)
    return outputs


if __name__ == "__main__":
    ingest_catalyst_layers()
