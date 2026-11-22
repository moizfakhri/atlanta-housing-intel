"""Path C Commuter Demand: Census LEHD/LODES8 Georgia Origin-Destination Flow Ingestion.

Downloads Georgia block-level commuter OD microdata (`ga_od_main_JT00_2022.csv.gz`)
and aggregates via DuckDB into:
  1. County-to-county commuter flows across our 5 Metro Atlanta suburban counties
  2. Tract-level job-access & high-wage ($3,333+/mo) employment gravity metrics
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import httpx
import polars as pl
from rich.console import Console
from rich.table import Table

from atl_housing.config import BRONZE_DIR, COUNTY_FIPS_LIST, METRO_ATLANTA_COUNTIES, RAW_DIR

console = Console()
LODES_GA_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/ga/od/ga_od_main_JT00_2022.csv.gz"
PATH_C_BRONZE = BRONZE_DIR / "path_c"


def ingest_lodes_atlanta_commutes() -> dict[str, Path]:
    """Download Georgia LODES8 OD file and aggregate Metro Atlanta commuter matrices via DuckDB."""
    raw_dir = RAW_DIR / "lodes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    PATH_C_BRONZE.mkdir(parents=True, exist_ok=True)

    gz_path = raw_dir / "ga_od_main_JT00_2022.csv.gz"
    if not gz_path.exists():
        console.print("[cyan]Downloading Census LODES8 Georgia OD commuter file (~23 MB)...[/cyan]")
        with (
            httpx.Client(timeout=120.0, follow_redirects=True) as client,
            client.stream("GET", LODES_GA_URL) as r,
        ):
            r.raise_for_status()
            with gz_path.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)

    out_county_od = PATH_C_BRONZE / "lodes_atlanta_county_od.parquet"
    out_tract_home = PATH_C_BRONZE / "lodes_atlanta_tract_commuters.parquet"
    fips_sql = ", ".join(f"'{f}'" for f in COUNTY_FIPS_LIST)
    case_sql = " ".join(f"WHEN '{c.fips}' THEN '{c.name}'" for c in METRO_ATLANTA_COUNTIES)

    con = duckdb.connect(":memory:")
    # S000 = Total jobs, SE03 = High-wage jobs (>$3,333/mo), SI03 = Services/Office jobs
    con.execute(
        f"""
        CREATE VIEW lodes_raw AS
        SELECT
            SUBSTR(LPAD(CAST(h_geocode AS VARCHAR), 15, '0'), 1, 5) AS home_county_fips,
            SUBSTR(LPAD(CAST(w_geocode AS VARCHAR), 15, '0'), 1, 5) AS work_county_fips,
            SUBSTR(LPAD(CAST(h_geocode AS VARCHAR), 15, '0'), 1, 11) AS home_tract,
            SUBSTR(LPAD(CAST(w_geocode AS VARCHAR), 15, '0'), 1, 11) AS work_tract,
            CAST(S000 AS BIGINT) AS total_jobs,
            CAST(SE03 AS BIGINT) AS high_wage_jobs
        FROM read_csv_auto('{gz_path}');

        COPY (
            SELECT
                home_county_fips,
                CASE home_county_fips {case_sql} ELSE home_county_fips END AS home_county,
                work_county_fips,
                CASE work_county_fips {case_sql} ELSE work_county_fips END AS work_county,
                SUM(total_jobs) AS total_commuters,
                SUM(high_wage_jobs) AS high_wage_commuters,
                ROUND(SUM(high_wage_jobs)::DOUBLE / NULLIF(SUM(total_jobs), 0), 4) AS high_wage_share
            FROM lodes_raw
            WHERE home_county_fips IN ({fips_sql})
              AND work_county_fips IN ({fips_sql})
            GROUP BY 1, 2, 3, 4
            ORDER BY total_commuters DESC
        ) TO '{out_county_od}' (FORMAT PARQUET, COMPRESSION ZSTD);

        COPY (
            SELECT
                home_tract AS census_tract,
                home_county_fips AS county_code,
                SUM(total_jobs) AS resident_workers,
                SUM(high_wage_jobs) AS high_wage_workers,
                ROUND(SUM(high_wage_jobs)::DOUBLE / NULLIF(SUM(total_jobs), 0), 4) AS high_wage_worker_share,
                ROUND(
                    SUM(CASE WHEN work_county_fips = '13121' THEN total_jobs ELSE 0 END)::DOUBLE
                    / NULLIF(SUM(total_jobs), 0),
                    4
                ) AS share_commuting_to_fulton
            FROM lodes_raw
            WHERE home_county_fips IN ({fips_sql})
            GROUP BY 1, 2
        ) TO '{out_tract_home}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    df_od = pl.read_parquet(out_county_od)
    cross = df_od.filter(pl.col("home_county_fips") != pl.col("work_county_fips")).head(8)
    tbl = Table(title="Path C: Top Cross-County Morning Commuter Flows (Census LODES8 GA)")
    for col in ["Home Suburb", "Work County", "Commuters", "High-Wage (>40k/yr)", "High-Wage %"]:
        tbl.add_column(col)
    for row in cross.iter_rows(named=True):
        tbl.add_row(
            str(row["home_county"]),
            str(row["work_county"]),
            f"{row['total_commuters']:,}",
            f"{row['high_wage_commuters']:,}",
            f"{row['high_wage_share'] * 100:.1f}%",
        )
    console.print(tbl)
    console.print(
        f"[green]Saved LODES Atlanta commuters:[/green] {df_od.height} county OD pairs -> {out_county_od}"
    )
    return {"county_od": out_county_od, "tract_commuters": out_tract_home}


if __name__ == "__main__":
    ingest_lodes_atlanta_commutes()
