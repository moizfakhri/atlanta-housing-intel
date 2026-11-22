"""Path E (Alt-Data Catalysts): County Building Permits, Rezoning Cases & AlphaEarth Metadata."""

from __future__ import annotations

from typing import Any

import httpx
from rich.console import Console

from atl_housing.config import BRONZE_DIR

console = Console()
PATH_E_BRONZE = BRONZE_DIR / "path_e"


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


def print_catalyst_catalog() -> None:
    cat = discover_atlanta_catalyst_layers()
    for kind, items in cat.items():
        console.rule(f"[bold cyan]Path E Discovered {kind.upper()} Layers[/bold cyan]")
        for it in items[:8]:
            console.print(f"- [green]{it['title']}[/green] ({it['owner']}) -> {it['url']}")


if __name__ == "__main__":
    print_catalyst_catalog()
