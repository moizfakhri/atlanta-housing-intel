#!/usr/bin/env python3
"""Live data feasibility audit across all 4 Metro Atlanta research tracks (C, A, E, B).

Probes real public endpoints without downloading huge bulk files, outputting a structured
report of availability, schemas, row counts, and go/no-go signals.
"""

from __future__ import annotations

import json
import os
from typing import Any

import duckdb
import httpx
import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from atl_housing.config import (
    DATA_DIR,
    HMDA_API_BASE,
    METRO_ATLANTA_COUNTIES,
    RAW_DIR,
    REDFIN_URLS,
    TEST_SUBURB_BBOX,
)

console = Console()
load_dotenv()


def audit_local_baseline() -> dict[str, Any]:
    """Check existing local Zillow ZHVI/ZORI & TIGER ZCTA files for Atlanta coverage."""
    result: dict[str, Any] = {"status": "ok"}
    zillow_dir = RAW_DIR / "zillow"
    zhvi_path = zillow_dir / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    zori_path = zillow_dir / "Zip_zori_uc_sfrcondomfr_sm_month.csv"

    for name, path in [("zhvi", zhvi_path), ("zori", zori_path)]:
        if not path.exists():
            result[name] = {"exists": False}
            continue
        df = pl.read_csv(path, infer_schema_length=1000)
        date_cols = [c for c in df.columns if len(c) == 10 and c[4] == "-" and c[7] == "-"]
        ga_atl = df.filter(
            (pl.col("State") == "GA")
            & (pl.col("Metro").cast(pl.Utf8).str.contains("Atlanta"))
        )
        result[name] = {
            "exists": True,
            "total_us_zips": df.height,
            "atlanta_metro_zips": ga_atl.height,
            "date_start": date_cols[0] if date_cols else None,
            "date_end": date_cols[-1] if date_cols else None,
            "n_months": len(date_cols),
        }

    tiger_shp = RAW_DIR / "tiger" / "tl_2025_us_zcta520" / "tl_2025_us_zcta520.shp"
    result["tiger_zcta_2025_exists"] = tiger_shp.exists()
    return result


def audit_path_c_liquidity(client: httpx.Client) -> dict[str, Any]:
    """Probe HMDA API, Redfin S3 endpoints, Census LODES, and IRS migration."""
    out: dict[str, Any] = {}

    # 1. HMDA live probe on Forsyth County (13117) for 2024 — small enough for an instant sample
    hmda_url = f"{HMDA_API_BASE}?counties=13117&years=2024&actions_taken=1,3"
    try:
        r = client.get(hmda_url, timeout=45.0, follow_redirects=True)
        r.raise_for_status()
        df = pl.read_csv(r.content, infer_schema_length=2000, ignore_errors=True)
        key_cols = [
            "census_tract",
            "action_taken",
            "loan_purpose",
            "occupancy_type",
            "loan_amount",
            "property_value",
            "interest_rate",
            "combined_loan_to_value_ratio",
            "debt_to_income_ratio",
            "income",
        ]
        present = [c for c in key_cols if c in df.columns]
        # occupancy_type: 1=Principal residence, 2=Second residence, 3=Investment property
        inv_share = None
        if "occupancy_type" in df.columns and df.height > 0:
            inv_cnt = df.filter(pl.col("occupancy_type").cast(pl.Utf8) == "3").height
            inv_share = round(inv_cnt / df.height, 4)
        out["hmda_forsyth_2024"] = {
            "status": "ok",
            "rows": df.height,
            "total_cols": len(df.columns),
            "key_cols_present": present,
            "investor_loan_share": inv_share,
            "unique_tracts": df["census_tract"].n_unique() if "census_tract" in df.columns else 0,
        }
    except Exception as e:
        out["hmda_forsyth_2024"] = {"status": "error", "error": str(e)}

    # 2. Redfin S3 live HEAD checks
    redfin_res: dict[str, Any] = {}
    for level, url in REDFIN_URLS.items():
        try:
            r = client.head(url, timeout=20.0, follow_redirects=True)
            size_mb = (
                round(int(r.headers["content-length"]) / 1e6, 1)
                if "content-length" in r.headers
                else None
            )
            redfin_res[level] = {
                "http_status": r.status_code,
                "size_mb": size_mb,
                "last_modified": r.headers.get("last-modified"),
            }
        except Exception as e:
            redfin_res[level] = {"status": "error", "error": str(e)}
    out["redfin_s3"] = redfin_res

    # 3. Census LODES GA OD live HEAD check
    lodes_candidates = [
        "https://lehd.ces.census.gov/data/lodes/LODES8/ga/od/ga_od_main_JT00_2022.csv.gz",
        "https://lehd.ces.census.gov/data/lodes/LODES8/ga/od/ga_od_main_JT00_2021.csv.gz",
    ]
    lodes_status: dict[str, Any] = {"status": "not_found"}
    for url in lodes_candidates:
        try:
            r = client.head(url, timeout=15.0, follow_redirects=True)
            if r.status_code == 200:
                lodes_status = {
                    "status": "ok",
                    "url": url,
                    "size_mb": round(int(r.headers.get("content-length", 0)) / 1e6, 1),
                }
                break
        except Exception:
            continue
    out["census_lodes_ga"] = lodes_status

    # 4. IRS Migration (2021-2022) HEAD check
    irs_url = "https://www.irs.gov/pub/irs-soi/countyinflow2122.csv"
    try:
        r = client.head(irs_url, timeout=15.0, follow_redirects=True)
        out["irs_migration"] = {
            "http_status": r.status_code,
            "url": irs_url,
            "size_mb": round(int(r.headers.get("content-length", 0)) / 1e6, 2),
        }
    except Exception as e:
        out["irs_migration"] = {"status": "error", "error": str(e)}

    return out


def audit_overture_3d(client: httpx.Client) -> dict[str, Any]:
    """Query live Overture Maps S3 GeoParquet 3D buildings on Alpharetta/Suwanee test bbox."""
    # Discover latest Overture release from STAC catalog
    release = None
    try:
        stac = client.get("https://stac.overturemaps.org/catalog.json", timeout=15.0).json()
        for link in stac.get("links", []):
            if link.get("rel") == "child" and link.get("href", "").endswith("/catalog.json"):
                cand = link["href"].strip("./").split("/")[0]
                if cand[:4].isdigit():
                    release = cand if (release is None or cand > release) else release
    except Exception:
        pass

    if not release:
        release = "2026-02-18.0"

    lon_min, lat_min, lon_max, lat_max = TEST_SUBURB_BBOX
    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=buildings/type=building/*"
    try:
        con = duckdb.connect(":memory:")
        con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region='us-west-2';")
        q = f"""
            SELECT
                COUNT(*) AS n_buildings,
                COUNT(height) AS n_with_height,
                COUNT(num_floors) AS n_with_floors,
                ROUND(AVG(height), 2) AS avg_height_m
            FROM read_parquet('{s3_path}', filename=false, hive_partitioning=1)
            WHERE bbox.xmin >= {lon_min} AND bbox.xmax <= {lon_max}
              AND bbox.ymin >= {lat_min} AND bbox.ymax <= {lat_max}
        """
        row = con.execute(q).fetchone()
        return {
            "status": "ok",
            "release": release,
            "test_bbox": TEST_SUBURB_BBOX,
            "n_buildings": row[0],
            "n_with_height": row[1],
            "pct_with_height": round(100.0 * row[1] / row[0], 1) if row[0] else 0.0,
            "n_with_floors": row[2],
            "avg_height_m": row[3],
        }
    except Exception as e:
        return {"status": "error", "release_tried": release, "error": str(e)}


def search_arcgis_hub(client: httpx.Client, hub_base: str, query: str) -> list[dict[str, Any]]:
    """Query an ArcGIS Hub v3/OGC search endpoint for open datasets."""
    url = f"{hub_base.rstrip('/')}/api/search/v1/collections/dataset/items"
    try:
        r = client.get(url, params={"q": query, "limit": 6}, timeout=20.0, follow_redirects=True)
        if r.status_code != 200:
            return [{"error": f"HTTP {r.status_code}"}]
        data = r.json()
        hits = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            hits.append(
                {
                    "title": props.get("title"),
                    "type": props.get("type"),
                    "url": props.get("url"),
                    "modified": props.get("modified"),
                }
            )
        return hits
    except Exception as e:
        return [{"error": str(e)}]


def audit_county_portals_and_catalysts(client: httpx.Client) -> dict[str, Any]:
    """Discover live ArcGIS Hub layers for Parcels, Sales, Rezoning & Permits across Metro Atlanta."""
    hubs = {
        "fulton": "https://gisdata.fultoncountyga.gov",
        "dekalb": "https://dekalbcountygis.dekalbcountyga.gov",
        "cobb": "https://cobb-county-open-data-cobbcountyga.hub.arcgis.com",
        "gwinnett": "https://gisdata-gwinnett.opendata.arcgis.com",
        "forsyth": "https://geohub-forsythcoga.opendata.arcgis.com",
        "arc_atlanta_regional": "https://opendata.atlantaregional.com",
    }
    out: dict[str, Any] = {}
    for name, base in hubs.items():
        parcels = search_arcgis_hub(client, base, "parcel")
        rezoning = search_arcgis_hub(client, base, "rezoning")
        permits = search_arcgis_hub(client, base, "building permit")
        out[name] = {
            "parcels_top": [h.get("title") for h in parcels if "title" in h][:4] or parcels[:1],
            "parcels_urls": [h.get("url") for h in parcels if h.get("url")][:2],
            "rezoning_top": [h.get("title") for h in rezoning if "title" in h][:3] or rezoning[:1],
            "permits_top": [h.get("title") for h in permits if "title" in h][:3] or permits[:1],
        }
    return out


def audit_alphaearth_and_path_b(client: httpx.Client) -> dict[str, Any]:
    """Check AlphaEarth HuggingFace datasets and Path B (Mapillary) token/API status."""
    out: dict[str, Any] = {}

    for repo_id in ["Major-TOM/Core-AlphaEarth-Embeddings", "isaaccorley/AEF"]:
        try:
            r = client.get(f"https://huggingface.co/api/datasets/{repo_id}", timeout=15.0)
            if r.status_code == 200:
                info = r.json()
                out[f"hf_{repo_id}"] = {
                    "status": "ok",
                    "gated": info.get("gated"),
                    "downloads": info.get("downloads"),
                }
            else:
                out[f"hf_{repo_id}"] = {"http_status": r.status_code}
        except Exception as e:
            out[f"hf_{repo_id}"] = {"status": "error", "error": str(e)}

    mapillary_token = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()
    if mapillary_token:
        lon_min, lat_min, lon_max, lat_max = TEST_SUBURB_BBOX
        try:
            r = client.get(
                "https://graph.mapillary.com/images",
                params={
                    "access_token": mapillary_token,
                    "fields": "id,captured_at,computed_geometry",
                    "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
                    "limit": 200,
                },
                timeout=20.0,
            )
            data = r.json()
            out["mapillary_path_b"] = {
                "token_configured": True,
                "http_status": r.status_code,
                "sample_images_in_test_bbox": len(data.get("data", [])),
            }
        except Exception as e:
            out["mapillary_path_b"] = {"token_configured": True, "error": str(e)}
    else:
        out["mapillary_path_b"] = {
            "token_configured": False,
            "note": "Free Mapillary Graph API token needed in .env (MAPILLARY_ACCESS_TOKEN) to measure Atlanta suburb street-photo density.",
        }

    out["env_keys_present"] = {
        "CENSUS_API_KEY": bool(os.getenv("CENSUS_API_KEY", "").strip()),
        "FRED_API_KEY": bool(os.getenv("FRED_API_KEY", "").strip()),
        "MAPILLARY_ACCESS_TOKEN": bool(mapillary_token),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY", "").strip()),
    }
    return out


def main() -> None:
    console.rule("[bold cyan]Metro Atlanta 4-Path Live Data Feasibility Audit[/bold cyan]")
    report: dict[str, Any] = {
        "counties": [c.__dict__ for c in METRO_ATLANTA_COUNTIES],
    }

    with httpx.Client(headers={"User-Agent": "atlanta-housing-intel/0.1"}) as client:
        console.print("[bold]1/5 Auditing local Zillow & TIGER files...[/bold]")
        report["local_baseline"] = audit_local_baseline()

        console.print("[bold]2/5 Auditing Path C (HMDA live API, Redfin S3, LODES, IRS)...[/bold]")
        report["path_c_liquidity"] = audit_path_c_liquidity(client)

        console.print("[bold]3/5 Auditing Path A (Overture Maps 3D Buildings via DuckDB S3)...[/bold]")
        report["path_a_overture_3d"] = audit_overture_3d(client)

        console.print("[bold]4/5 Auditing Paths A & E (County ArcGIS Hubs: Parcels, Rezoning, Permits)...[/bold]")
        report["paths_a_e_county_hubs"] = audit_county_portals_and_catalysts(client)

        console.print("[bold]5/5 Auditing AlphaEarth & Path B (Mapillary / Env keys)...[/bold]")
        report["alphaearth_and_path_b"] = audit_alphaearth_and_path_b(client)

    out_path = DATA_DIR / "feasibility_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    # Print crisp summary table
    table = Table(title="Live Feasibility Summary (Metro Atlanta 5-County Spine)")
    table.add_column("Track / Source", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Key Live Metric")

    lb = report["local_baseline"]
    table.add_row(
        "Local Zillow ZHVI / ZORI",
        "OK",
        f"ZHVI: {lb['zhvi'].get('atlanta_metro_zips')} ATL ZIPs ({lb['zhvi'].get('date_start')}..{lb['zhvi'].get('date_end')}); ZORI: {lb['zori'].get('atlanta_metro_zips')} ZIPs",
    )

    pc = report["path_c_liquidity"]
    hmda = pc["hmda_forsyth_2024"]
    table.add_row(
        "Path C: HMDA Mortgage API (2024 Forsyth test)",
        hmda.get("status", "err"),
        f"{hmda.get('rows')} loans, {hmda.get('unique_tracts')} tracts, investor share={hmda.get('investor_loan_share')}"
        if hmda.get("status") == "ok"
        else str(hmda.get("error")),
    )
    rf_zip = pc["redfin_s3"].get("zip", {})
    rf_nb = pc["redfin_s3"].get("neighborhood", {})
    table.add_row(
        "Path C: Redfin S3 Market Trackers",
        f"ZIP:{rf_zip.get('http_status')} / NB:{rf_nb.get('http_status')}",
        f"ZIP={rf_zip.get('size_mb')} MB, Neighborhood={rf_nb.get('size_mb')} MB",
    )
    table.add_row(
        "Path C: Census LODES GA / IRS Migration",
        f"LODES:{pc['census_lodes_ga'].get('status')} / IRS:{pc['irs_migration'].get('http_status')}",
        f"LODES={pc['census_lodes_ga'].get('size_mb')} MB, IRS={pc['irs_migration'].get('size_mb')} MB",
    )

    ov = report["path_a_overture_3d"]
    table.add_row(
        "Path A: Overture Maps 3D Buildings (DuckDB S3)",
        ov.get("status", "err"),
        f"Release {ov.get('release')}: {ov.get('n_buildings')} buildings in test suburb, {ov.get('pct_with_height')}% w/ height (avg {ov.get('avg_height_m')}m)"
        if ov.get("status") == "ok"
        else str(ov.get("error")),
    )

    for county, info in report["paths_a_e_county_hubs"].items():
        table.add_row(
            f"Paths A/E: {county.title()} ArcGIS Hub",
            "PROBED",
            f"Parcels={info['parcels_top'][:2]} | Rezoning={info['rezoning_top'][:1]} | Permits={info['permits_top'][:1]}",
        )

    ae = report["alphaearth_and_path_b"]
    table.add_row(
        "Paths A/E: AlphaEarth HF Datasets",
        f"MajorTOM:{ae['hf_Major-TOM/Core-AlphaEarth-Embeddings'].get('status')} / AEF:{ae['hf_isaaccorley/AEF'].get('status')}",
        f"Gated={ae['hf_Major-TOM/Core-AlphaEarth-Embeddings'].get('gated')}",
    )
    table.add_row(
        "Path B: Mapillary Streetscape API",
        "NEEDS FREE TOKEN" if not ae["mapillary_path_b"].get("token_configured") else "OK",
        str(ae["mapillary_path_b"].get("note") or ae["mapillary_path_b"].get("sample_images_in_test_bbox")),
    )

    console.print(table)
    console.print(f"\n[dim]Full JSON report written to gitignored path: {out_path}[/dim]")


if __name__ == "__main__":
    main()
