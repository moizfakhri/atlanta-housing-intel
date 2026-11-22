"""Path B (Exploratory): Sample Mapillary street-level images & CV map features across Atlanta H3 hexes.

Pulls clean metadata (image ID, capture timestamp, compass angle, coordinates, thumbnail URL)
and Mapillary's pre-computed streetscape map features (benches, street lights, crosswalks,
utility poles, vegetation/traffic signs) into gitignored `data/bronze/path_b/` for VLM scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

import h3
import httpx
import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from atl_housing.config import BRONZE_DIR
from atl_housing.tracks.path_b_streetscape.coverage_probe import SUBURB_PROBES

console = Console()
load_dotenv()
PATH_B_BRONZE = BRONZE_DIR / "path_b"


def sample_suburb_streetscapes(
    h3_res: int = 8,
    max_images_per_hex: int = 120,
) -> Path:
    """Fetch geo-tagged Mapillary streetscape image metadata across our 5 Atlanta suburban test hexes."""
    token = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing in local .env.")

    PATH_B_BRONZE.mkdir(parents=True, exist_ok=True)
    out_path = PATH_B_BRONZE / "mapillary_suburb_samples.parquet"

    rows: list[dict[str, object]] = []
    tbl = Table(title="Path B Streetscape Sample Extraction (Metro Atlanta Suburbs)")
    tbl.add_column("Suburb", style="cyan")
    tbl.add_column("County")
    tbl.add_column("Sampled Images", justify="right")
    tbl.add_column("With 1024px Thumb", justify="right")
    tbl.add_column("Map Features Found", justify="right")

    with httpx.Client(
        timeout=30.0,
        headers={"Authorization": f"OAuth {token}"},
    ) as client:
        for p in SUBURB_PROBES:
            cell = h3.latlng_to_cell(p.lat, p.lon, h3_res)
            boundary = h3.cell_to_boundary(cell)
            lats = [pt[0] for pt in boundary]
            lons = [pt[1] for pt in boundary]
            bbox = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"

            r_img = client.get(
                "https://graph.mapillary.com/images",
                params={
                    "fields": "id,captured_at,compass_angle,is_pano,computed_geometry,thumb_1024_url",
                    "bbox": bbox,
                    "limit": max_images_per_hex,
                },
            )
            r_img.raise_for_status()
            imgs = r_img.json().get("data", [])

            # Also pull Mapillary's built-in computer vision map features (streetlights, sidewalks, poles, etc.)
            r_feat = client.get(
                "https://graph.mapillary.com/map_features",
                params={
                    "fields": "id,object_value,geometry",
                    "bbox": bbox,
                    "limit": 200,
                },
            )
            n_feats = len(r_feat.json().get("data", [])) if r_feat.status_code == 200 else 0

            n_thumbs = 0
            for im in imgs:
                geom = im.get("computed_geometry") or {}
                coords = geom.get("coordinates") or [None, None]
                thumb = im.get("thumb_1024_url")
                if thumb:
                    n_thumbs += 1
                rows.append(
                    {
                        "suburb": p.name,
                        "county": p.county,
                        "h3_cell": cell,
                        "image_id": str(im.get("id")),
                        "captured_at_ms": im.get("captured_at"),
                        "compass_angle": im.get("compass_angle"),
                        "is_pano": bool(im.get("is_pano", False)),
                        "lon": coords[0],
                        "lat": coords[1],
                        "has_thumb_1024": bool(thumb),
                        "hex_map_feature_count": n_feats,
                    }
                )
            tbl.add_row(p.name, p.county, str(len(imgs)), str(n_thumbs), str(n_feats))

    df = pl.DataFrame(rows)
    df.write_parquet(out_path)
    console.print(tbl)
    console.print(f"[green]Saved Path B streetscape samples:[/green] {df.height:,} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    sample_suburb_streetscapes()
