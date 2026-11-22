"""Path C (Main Track) Bronze Ingestion: Zillow ZHVI/ZORI, CFPB HMDA, Redfin Market Tracker.

All outputs are written strictly to gitignored `data/bronze/path_c/` as Parquet files.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import httpx
import polars as pl
from rich.console import Console

from atl_housing.config import (
    BRONZE_DIR,
    COUNTY_FIPS_CSV,
    COUNTY_FIPS_LIST,
    HMDA_API_BASE,
    METRO_ATLANTA_COUNTIES,
    RAW_DIR,
    REALTOR_URLS,
    REDFIN_URLS,
)

console = Console()
PATH_C_BRONZE = BRONZE_DIR / "path_c"


def ingest_zillow_atlanta() -> Path:
    """Unpivot local Zillow ZHVI & ZORI CSVs for Metro Atlanta GA ZIPs into tidy Parquet."""
    PATH_C_BRONZE.mkdir(parents=True, exist_ok=True)
    zhvi_csv = RAW_DIR / "zillow" / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    zori_csv = RAW_DIR / "zillow" / "Zip_zori_uc_sfrcondomfr_sm_month.csv"

    def _melt_zillow(csv_path: Path, value_name: str) -> pl.DataFrame:
        df = pl.read_csv(csv_path, infer_schema_length=1000)
        date_cols = [c for c in df.columns if len(c) == 10 and c[4] == "-" and c[7] == "-"]
        atl = df.filter(
            (pl.col("State") == "GA")
            & (pl.col("Metro").cast(pl.Utf8).str.contains("Atlanta"))
        )
        id_cols = [
            c
            for c in ["RegionName", "City", "CountyName", "Metro", "State"]
            if c in atl.columns
        ]
        long_df = atl.unpivot(
            index=id_cols,
            on=date_cols,
            variable_name="period_end",
            value_name=value_name,
        )
        return long_df.with_columns(
            pl.col("RegionName").cast(pl.Utf8).str.zfill(5).alias("zip_code"),
            pl.col("period_end").str.to_date("%Y-%m-%d"),
            pl.col(value_name).cast(pl.Float64),
        ).drop("RegionName")

    zhvi = _melt_zillow(zhvi_csv, "zhvi")
    zori = _melt_zillow(zori_csv, "zori").select(["zip_code", "period_end", "zori"])

    panel = (
        zhvi.join(zori, on=["zip_code", "period_end"], how="left")
        .with_columns(
            # Gross annual rent-to-price yield proxy when both ZHVI and ZORI exist
            ((pl.col("zori") * 12.0) / pl.col("zhvi")).alias("gross_rent_yield")
        )
        .sort(["zip_code", "period_end"])
    )

    out_path = PATH_C_BRONZE / "zillow_atlanta_zip_panel.parquet"
    panel.write_parquet(out_path)
    console.print(
        f"[green]Saved Zillow Atlanta panel:[/green] {panel.height:,} rows "
        f"({panel['zip_code'].n_unique()} ZIPs) -> {out_path}"
    )
    return out_path


def ingest_hmda_atlanta(years: list[int] | None = None) -> Path:
    """Download CFPB HMDA loan-level mortgage applications/originations for 5 Atlanta counties.

    Actions taken: 1 = Loan originated, 3 = Application denied by financial institution.
    """
    PATH_C_BRONZE.mkdir(parents=True, exist_ok=True)
    years = years or [2020, 2021, 2022, 2023, 2024]
    raw_hmda_dir = RAW_DIR / "hmda"
    raw_hmda_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "activity_year",
        "county_code",
        "census_tract",
        "action_taken",
        "loan_type",
        "loan_purpose",
        "occupancy_type",
        "loan_amount",
        "property_value",
        "interest_rate",
        "combined_loan_to_value_ratio",
        "debt_to_income_ratio",
        "income",
        "purchaser_type",
        "denial_reason-1",
        "derived_dwelling_category",
    ]

    frames: list[pl.DataFrame] = []
    with httpx.Client(headers={"User-Agent": "atlanta-housing-intel/0.1"}) as client:
        for yr in years:
            cache_csv = raw_hmda_dir / f"hmda_atl5_{yr}.csv"
            if not cache_csv.exists():
                url = f"{HMDA_API_BASE}?counties={COUNTY_FIPS_CSV}&years={yr}&actions_taken=1,3"
                console.print(f"[cyan]Fetching HMDA {yr} for 5 Metro Atlanta counties...[/cyan]")
                with client.stream("GET", url, timeout=180.0, follow_redirects=True) as r:
                    r.raise_for_status()
                    with cache_csv.open("wb") as f:
                        for chunk in r.iter_bytes(chunk_size=1024 * 256):
                            f.write(chunk)

            with cache_csv.open("r", encoding="utf-8", errors="ignore") as hdr_f:
                header_cols = [h.strip().strip('"') for h in hdr_f.readline().split(",")]
            df = pl.read_csv(
                cache_csv,
                columns=[c for c in keep_cols if c in header_cols],
                infer_schema_length=0,  # read as strings first to safely parse "Exempt"/"NA"
            )
            frames.append(df)
            console.print(f"  HMDA {yr}: {df.height:,} loan records loaded.")

    combined = pl.concat(frames, how="diagonal_relaxed")

    # Safely parse numeric columns (HMDA uses "Exempt" and "NA" strings)
    numeric_cols = ["loan_amount", "property_value", "interest_rate", "combined_loan_to_value_ratio", "income"]
    cleaned = combined.with_columns(
        pl.col("activity_year").cast(pl.Int32),
        pl.col("county_code").cast(pl.Utf8),
        pl.col("census_tract").cast(pl.Utf8),
        *[
            pl.when(pl.col(c).is_in(["Exempt", "NA", "", "None"]))
            .then(None)
            .otherwise(pl.col(c))
            .cast(pl.Float64, strict=False)
            .alias(c)
            for c in numeric_cols
            if c in combined.columns
        ],
        (pl.col("action_taken") == "1").alias("is_originated"),
        (pl.col("action_taken") == "3").alias("is_denied"),
        (pl.col("loan_purpose") == "1").alias("is_home_purchase"),
        (pl.col("occupancy_type") == "3").alias("is_investor"),
        (pl.col("occupancy_type") == "1").alias("is_owner_occupied"),
    )

    out_path = PATH_C_BRONZE / "hmda_atlanta_5county_loans.parquet"
    cleaned.write_parquet(out_path)
    console.print(f"[green]Saved HMDA Bronze Parquet:[/green] {cleaned.height:,} loans -> {out_path}")
    return out_path


def ingest_redfin_atlanta(level: str = "zip") -> Path:
    """Stream-filter Redfin Market Tracker TSV.gz directly into Atlanta Bronze Parquet via DuckDB."""
    assert level in REDFIN_URLS, f"Unsupported level: {level}"
    PATH_C_BRONZE.mkdir(parents=True, exist_ok=True)
    raw_rf_dir = RAW_DIR / "redfin"
    raw_rf_dir.mkdir(parents=True, exist_ok=True)

    local_gz = raw_rf_dir / f"redfin_{level}_market_tracker.tsv000.gz"
    if not local_gz.exists():
        url = REDFIN_URLS[level]
        console.print(f"[cyan]Downloading Redfin {level} market tracker...[/cyan]")
        with (
            httpx.Client(headers={"User-Agent": "atlanta-housing-intel/0.1"}) as client,
            client.stream("GET", url, timeout=300.0, follow_redirects=True) as r,
        ):
            r.raise_for_status()
            with local_gz.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 512):
                    f.write(chunk)

    out_path = PATH_C_BRONZE / f"redfin_atlanta_{level}.parquet"
    county_names = [f"{c.name} County, GA" for c in METRO_ATLANTA_COUNTIES]
    county_sql = ", ".join(f"'{n}'" for n in county_names)

    con = duckdb.connect(":memory:")
    if level == "county":
        where_clause = f"REGION IN ({county_sql})"
    else:
        where_clause = (
            "STATE_CODE = 'GA' AND ("
            "PARENT_METRO_REGION LIKE '%Atlanta%' OR "
            "REGION LIKE '%Atlanta%'"
            ")"
        )

    query = f"""
        COPY (
            SELECT *
            FROM read_csv_auto('{local_gz}', delim='\t', header=true, ignore_errors=true)
            WHERE {where_clause}
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    console.print(f"[green]Saved Redfin Atlanta ({level}):[/green] {n_rows:,} rows -> {out_path}")
    return out_path


def ingest_realtor_atlanta(atlanta_zips: list[str] | None = None) -> dict[str, Path]:
    """Stream-filter Realtor.com S3 Core & Hotness monthly historical CSVs to Metro Atlanta.

    Uses DuckDB httpfs/read_csv directly or caches filtered Parquet so we never commit raw CSVs.
    Captures: median_days_on_market, active_listing_count, price_reduced_count, pending_ratio,
    hotness_score, demand_score, supply_score, and buyer views per listing.
    """
    PATH_C_BRONZE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # Derive Atlanta ZIP list from bronze Zillow panel if not supplied
    if atlanta_zips is None:
        zillow_pq = PATH_C_BRONZE / "zillow_atlanta_zip_panel.parquet"
        if not zillow_pq.exists():
            ingest_zillow_atlanta()
        atlanta_zips = [
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT zip_code FROM read_parquet('{zillow_pq}')"
            ).fetchall()
        ]

    zips_sql = ", ".join(f"'{z}'" for z in atlanta_zips)
    fips_sql = ", ".join(f"'{f}'" for f in COUNTY_FIPS_LIST)

    outputs: dict[str, Path] = {}
    for key, url in REALTOR_URLS.items():
        out_path = PATH_C_BRONZE / f"realtor_atlanta_{key}.parquet"
        console.print(f"[cyan]Streaming Realtor.com {key} filtered to Metro Atlanta...[/cyan]")
        if key == "county_core":
            where = f"LPAD(CAST(county_fips AS VARCHAR), 5, '0') IN ({fips_sql})"
        else:
            where = f"LPAD(CAST(postal_code AS VARCHAR), 5, '0') IN ({zips_sql})"

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{url}', ignore_errors=true)
                WHERE {where}
            ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
        n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
        console.print(f"[green]Saved Realtor {key}:[/green] {n_rows:,} rows -> {out_path}")
        outputs[key] = out_path

    return outputs


def ingest_irs_migration() -> Path:
    """Download IRS SOI county-to-county migration inflows & outflows (2021-2022) for Metro Atlanta."""
    PATH_C_BRONZE.mkdir(parents=True, exist_ok=True)
    raw_irs_dir = RAW_DIR / "irs"
    raw_irs_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "inflow": "https://www.irs.gov/pub/irs-soi/countyinflow2122.csv",
        "outflow": "https://www.irs.gov/pub/irs-soi/countyoutflow2122.csv",
    }
    frames: list[pl.DataFrame] = []
    with httpx.Client(headers={"User-Agent": "atlanta-housing-intel/0.1"}) as client:
        for direction, url in urls.items():
            local_csv = raw_irs_dir / f"irs_{direction}_2122.csv"
            if not local_csv.exists():
                console.print(f"[cyan]Fetching IRS SOI {direction} (2021-2022)...[/cyan]")
                r = client.get(url, timeout=60.0, follow_redirects=True)
                r.raise_for_status()
                local_csv.write_bytes(r.content)

            df = pl.read_csv(local_csv, encoding="latin1", infer_schema_length=2000)
            df = df.with_columns(
                (
                    pl.col("y1_statefips").cast(pl.Utf8).str.zfill(2)
                    + pl.col("y1_countyfips").cast(pl.Utf8).str.zfill(3)
                ).alias("origin_fips"),
                (
                    pl.col("y2_statefips").cast(pl.Utf8).str.zfill(2)
                    + pl.col("y2_countyfips").cast(pl.Utf8).str.zfill(3)
                ).alias("dest_fips"),
                pl.lit(direction).alias("direction"),
            )
            if direction == "inflow":
                atl = df.filter(pl.col("dest_fips").is_in(COUNTY_FIPS_LIST))
            else:
                atl = df.filter(pl.col("origin_fips").is_in(COUNTY_FIPS_LIST))
            frames.append(atl)

    combined = pl.concat(frames, how="diagonal_relaxed")
    out_path = PATH_C_BRONZE / "irs_atlanta_migration_2122.parquet"
    combined.write_parquet(out_path)
    console.print(f"[green]Saved IRS Atlanta migration:[/green] {combined.height:,} rows -> {out_path}")
    return out_path


def run_all_path_c_bronze(hmda_years: list[int] | None = None) -> dict[str, Path]:
    """Run the full Path C Bronze ingestion suite (Zillow, Realtor Core/Hotness, HMDA, IRS)."""
    results: dict[str, Path] = {}
    results["zillow"] = ingest_zillow_atlanta()
    results.update(ingest_realtor_atlanta())
    results["hmda"] = ingest_hmda_atlanta(years=hmda_years)
    results["irs"] = ingest_irs_migration()
    return results


if __name__ == "__main__":
    run_all_path_c_bronze()


