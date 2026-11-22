"""Path B (Exploratory): Mapillary street-level photo coverage probe across Metro Atlanta suburbs.

Run once MAPILLARY_ACCESS_TOKEN is placed in local `.env` to decide objectively
whether residential streetscape curb-appeal modeling has sufficient coverage (Go vs. Cut).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import h3
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()
load_dotenv()


@dataclass(frozen=True)
class SuburbProbe:
    name: str
    county: str
    lat: float
    lon: float
    character: str


# Representative residential suburban test points across our 5-county spine
SUBURB_PROBES: tuple[SuburbProbe, ...] = (
    SuburbProbe("Alpharetta / Windward", "Fulton", 34.0937, -84.2535, "Affluent master-planned suburb"),
    SuburbProbe("Decatur / Oakhurst", "DeKalb", 33.7646, -84.3077, "Walkable inner-ring bungalow suburb"),
    SuburbProbe("East Cobb / Walton HS", "Cobb", 33.9912, -84.4268, "Top school-zone residential subdivisions"),
    SuburbProbe("Suwanee / Town Center", "Gwinnett", 34.0515, -84.0713, "Fast-growing mixed suburban corridor"),
    SuburbProbe("Cumming / GA-400", "Forsyth", 34.2073, -84.1402, "Outer exurban new-construction corridor"),
)


def probe_mapillary_coverage(h3_res: int = 8) -> None:
    token = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        console.print(
            "[yellow]MAPILLARY_ACCESS_TOKEN not found in .env.[/yellow]\n"
            "Get a free client token at https://www.mapillary.com/dashboard/developers "
            "and add MAPILLARY_ACCESS_TOKEN=... to your local .env, then re-run:\n"
            "  uv run python -m atl_housing.tracks.path_b_streetscape.coverage_probe"
        )
        return

    table = Table(title=f"Path B Feasibility: Mapillary Street-Photo Density (H3 res={h3_res})")
    table.add_column("Suburb", style="cyan")
    table.add_column("County")
    table.add_column("H3 Cell")
    table.add_column("Images Found (cap 500)", justify="right")
    table.add_column("Panoramic?", justify="right")
    table.add_column("Verdict", style="bold")

    with httpx.Client(
        timeout=25.0,
        headers={"Authorization": f"OAuth {token}"},
    ) as client:
        for p in SUBURB_PROBES:
            cell = h3.latlng_to_cell(p.lat, p.lon, h3_res)
            boundary = h3.cell_to_boundary(cell)
            lats = [pt[0] for pt in boundary]
            lons = [pt[1] for pt in boundary]
            bbox = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"

            r = client.get(
                "https://graph.mapillary.com/images",
                params={
                    "fields": "id,captured_at,is_pano",
                    "bbox": bbox,
                    "limit": 500,
                },
            )
            if r.status_code != 200:
                err_msg = r.json().get("error", {}).get("message", f"HTTP {r.status_code}")
                console.print(f"[red]Mapillary API error ({r.status_code}):[/red] {err_msg}")
                return
            imgs = r.json().get("data", [])
            n_pano = sum(1 for im in imgs if im.get("is_pano"))
            verdict = "GO" if len(imgs) >= 100 else ("BORDERLINE" if len(imgs) >= 25 else "SPARSE/CUT")
            table.add_row(p.name, p.county, cell, str(len(imgs)), str(n_pano), verdict)

    console.print(table)


if __name__ == "__main__":
    probe_mapillary_coverage()
