"""Path C Silver Lakehouse: Unified Metro Atlanta ZIP Liquidity & County Credit/Migration Panels.

Joins Zillow ZHVI/ZORI + Realtor.com Core & Hotness + CFPB HMDA + IRS Migration
into analysis-ready Silver Parquet & DuckDB views with strict leakage-safe targets.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
from rich.console import Console
from rich.table import Table

from atl_housing.config import BRONZE_DIR, DUCKDB_PATH, METRO_ATLANTA_COUNTIES, SILVER_DIR

console = Console()
PATH_C_BRONZE = BRONZE_DIR / "path_c"
PATH_C_SILVER = SILVER_DIR / "path_c"


def build_zip_liquidity_panel() -> Path:
    """Build monthly Metro Atlanta ZIP panel fusing Zillow ZHVI/ZORI with Realtor Core & Hotness."""
    PATH_C_SILVER.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))

    zillow_pq = PATH_C_BRONZE / "zillow_atlanta_zip_panel.parquet"
    core_pq = PATH_C_BRONZE / "realtor_atlanta_zip_core.parquet"
    hot_pq = PATH_C_BRONZE / "realtor_atlanta_zip_hotness.parquet"

    out_path = PATH_C_SILVER / "atlanta_zip_liquidity_panel.parquet"

    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_zillow_zip AS
        SELECT
            zip_code,
            DATE_TRUNC('month', period_end)::DATE AS month_date,
            City AS city,
            CountyName AS county_name,
            zhvi,
            zori,
            gross_rent_yield
        FROM read_parquet('{zillow_pq}');

        CREATE OR REPLACE VIEW v_zillow_zip_dim AS
        SELECT
            zip_code,
            ANY_VALUE(City) AS city,
            ANY_VALUE(CountyName) AS county_name
        FROM read_parquet('{zillow_pq}')
        GROUP BY zip_code;

        CREATE OR REPLACE VIEW v_realtor_zip_core AS
        SELECT
            LPAD(CAST(postal_code AS VARCHAR), 5, '0') AS zip_code,
            zip_name,
            STRPTIME(CAST(month_date_yyyymm AS VARCHAR) || '01', '%Y%m%d')::DATE AS month_date,
            TRY_CAST(median_listing_price AS DOUBLE) AS median_listing_price,
            TRY_CAST(median_listing_price_per_square_foot AS DOUBLE) AS list_price_per_sqft,
            TRY_CAST(active_listing_count AS DOUBLE) AS active_listing_count,
            TRY_CAST(median_days_on_market AS DOUBLE) AS median_dom,
            TRY_CAST(new_listing_count AS DOUBLE) AS new_listing_count,
            TRY_CAST(price_increased_count AS DOUBLE) AS price_increased_count,
            TRY_CAST(price_reduced_count AS DOUBLE) AS price_reduced_count,
            TRY_CAST(pending_listing_count AS DOUBLE) AS pending_listing_count,
            TRY_CAST(pending_ratio AS DOUBLE) AS pending_ratio,
            TRY_CAST(total_listing_count AS DOUBLE) AS total_listing_count,
            TRY_CAST(median_square_feet AS DOUBLE) AS median_sqft
        FROM read_parquet('{core_pq}')
        WHERE LENGTH(CAST(month_date_yyyymm AS VARCHAR)) = 6;

        CREATE OR REPLACE VIEW v_realtor_zip_hotness AS
        SELECT
            LPAD(CAST(postal_code AS VARCHAR), 5, '0') AS zip_code,
            STRPTIME(CAST(month_date_yyyymm AS VARCHAR) || '01', '%Y%m%d')::DATE AS month_date,
            TRY_CAST(hotness_score AS DOUBLE) AS hotness_score,
            TRY_CAST(supply_score AS DOUBLE) AS supply_score,
            TRY_CAST(demand_score AS DOUBLE) AS demand_score,
            TRY_CAST(hotness_rank AS DOUBLE) AS hotness_rank,
            TRY_CAST(page_view_count_per_property_vs_us AS DOUBLE) AS buyer_views_vs_us
        FROM read_parquet('{hot_pq}')
        WHERE LENGTH(CAST(month_date_yyyymm AS VARCHAR)) = 6;

        CREATE OR REPLACE TABLE silver_atlanta_zip_liquidity AS
        WITH base AS (
            SELECT
                c.zip_code,
                c.month_date,
                COALESCE(d.city, c.zip_name) AS city,
                d.county_name,
                z.zhvi,
                z.zori,
                z.gross_rent_yield,
                c.median_listing_price,
                c.list_price_per_sqft,
                c.active_listing_count,
                c.new_listing_count,
                c.pending_listing_count,
                c.total_listing_count,
                c.median_dom,
                c.pending_ratio,
                c.price_reduced_count,
                c.price_increased_count,
                c.median_sqft,
                h.hotness_score,
                h.supply_score,
                h.demand_score,
                h.hotness_rank,
                h.buyer_views_vs_us,
                -- Derived market microstructure signals (strictly contemporaneous)
                CASE WHEN c.active_listing_count > 0
                    THEN c.price_reduced_count / c.active_listing_count END AS price_cut_share,
                CASE WHEN c.new_listing_count > 0
                    THEN c.pending_listing_count / c.new_listing_count END AS pending_to_new_ratio,
                CASE WHEN z.zhvi > 0 AND c.median_listing_price > 0
                    THEN (c.median_listing_price - z.zhvi) / z.zhvi END AS ask_vs_zhvi_spread
            FROM v_realtor_zip_core c
            LEFT JOIN v_zillow_zip_dim d
                ON c.zip_code = d.zip_code
            LEFT JOIN v_zillow_zip z
                ON c.zip_code = z.zip_code AND c.month_date = z.month_date
            LEFT JOIN v_realtor_zip_hotness h
                ON c.zip_code = h.zip_code AND c.month_date = h.month_date
        )
        SELECT
            *,
            -- Leakage-safe forward targets (strictly for label construction, NEVER used as features)
            (LEAD(zhvi, 3) OVER (PARTITION BY zip_code ORDER BY month_date) / NULLIF(zhvi, 0)) - 1.0
                AS target_zhvi_ret_3m,
            (LEAD(zhvi, 6) OVER (PARTITION BY zip_code ORDER BY month_date) / NULLIF(zhvi, 0)) - 1.0
                AS target_zhvi_ret_6m,
            LEAD(median_dom, 3) OVER (PARTITION BY zip_code ORDER BY month_date) - median_dom
                AS target_dom_delta_3m,
            LEAD(price_cut_share, 3) OVER (PARTITION BY zip_code ORDER BY month_date) - price_cut_share
                AS target_price_cut_delta_3m
        FROM base
        ORDER BY zip_code, month_date;

        COPY silver_atlanta_zip_liquidity
        TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    stats = con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT zip_code) AS n_zips,
            MIN(month_date) AS min_dt,
            MAX(month_date) AS max_dt,
            ROUND(AVG(median_dom), 1) AS avg_dom,
            ROUND(AVG(price_cut_share) * 100, 1) AS avg_price_cut_pct
        FROM silver_atlanta_zip_liquidity
        """
    ).fetchone()
    console.print(
        f"[green]Built Silver ZIP Liquidity Panel:[/green] {stats[0]:,} rows, "
        f"{stats[1]} ZIPs ({stats[2]} to {stats[3]}), avg DOM={stats[4]}d, "
        f"avg price-cut share={stats[5]}%"
    )
    return out_path


def build_hmda_credit_summary() -> Path:
    """Aggregate HMDA loan-level records into County & Census Tract annual credit demand tables."""
    PATH_C_SILVER.mkdir(parents=True, exist_ok=True)
    hmda_pq = PATH_C_BRONZE / "hmda_atlanta_5county_loans.parquet"
    out_tract = PATH_C_SILVER / "atlanta_tract_credit_demand.parquet"
    out_county = PATH_C_SILVER / "atlanta_county_credit_demand.parquet"

    con = duckdb.connect(str(DUCKDB_PATH))
    county_map_sql = " ".join(
        f"WHEN '{c.fips}' THEN '{c.name}'" for c in METRO_ATLANTA_COUNTIES
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE silver_atlanta_tract_credit AS
        SELECT
            activity_year,
            county_code,
            CASE county_code {county_map_sql} ELSE county_code END AS county_name,
            census_tract,
            COUNT(*) AS total_apps,
            SUM(CASE WHEN is_originated THEN 1 ELSE 0 END) AS originated_loans,
            SUM(CASE WHEN is_denied THEN 1 ELSE 0 END) AS denied_loans,
            ROUND(SUM(CASE WHEN is_denied THEN 1 ELSE 0 END)::DOUBLE / NULLIF(COUNT(*), 0), 4) AS denial_rate,
            SUM(CASE WHEN is_home_purchase AND is_originated THEN 1 ELSE 0 END) AS purchase_originations,
            ROUND(
                SUM(CASE WHEN is_home_purchase AND is_originated AND is_investor THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(CASE WHEN is_home_purchase AND is_originated THEN 1 ELSE 0 END), 0),
                4
            ) AS investor_purchase_share,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN property_value END) AS median_property_value,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN loan_amount END) AS median_loan_amount,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN interest_rate END) AS median_interest_rate,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN loan_to_value_ratio END) AS median_ltv,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN income END) AS median_buyer_income_k
        FROM read_parquet('{hmda_pq}')
        WHERE census_tract IS NOT NULL AND census_tract != ''
        GROUP BY 1, 2, 3, 4;

        COPY silver_atlanta_tract_credit TO '{out_tract}' (FORMAT PARQUET, COMPRESSION ZSTD);

        CREATE OR REPLACE TABLE silver_atlanta_county_credit AS
        SELECT
            activity_year,
            county_code,
            CASE county_code {county_map_sql} ELSE county_code END AS county_name,
            COUNT(*) AS total_apps,
            SUM(CASE WHEN is_home_purchase AND is_originated THEN 1 ELSE 0 END) AS purchase_originations,
            ROUND(SUM(CASE WHEN is_denied THEN 1 ELSE 0 END)::DOUBLE / NULLIF(COUNT(*), 0), 4) AS denial_rate,
            ROUND(
                SUM(CASE WHEN is_home_purchase AND is_originated AND is_investor THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(CASE WHEN is_home_purchase AND is_originated THEN 1 ELSE 0 END), 0),
                4
            ) AS investor_purchase_share,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN property_value END) AS median_property_value,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN interest_rate END) AS median_interest_rate,
            MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN income END) AS median_buyer_income_k
        FROM read_parquet('{hmda_pq}')
        GROUP BY 1, 2, 3
        ORDER BY activity_year, county_name;

        COPY silver_atlanta_county_credit TO '{out_county}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    df = pl.read_parquet(out_county)
    latest_yr = int(df["activity_year"].max())
    tbl = Table(title=f"Path C: HMDA Home-Purchase Credit Demand ({latest_yr}, 5 Atlanta Counties)")
    for col in ["County", "Purchases", "Denial %", "Investor %", "Med Value", "Med Rate", "Med Buyer Inc"]:
        tbl.add_column(col)
    for row in df.filter(pl.col("activity_year") == latest_yr).iter_rows(named=True):
        tbl.add_row(
            str(row["county_name"]),
            f"{row['purchase_originations']:,}",
            f"{row['denial_rate'] * 100:.1f}%",
            f"{row['investor_purchase_share'] * 100:.1f}%",
            f"${row['median_property_value']:,.0f}" if row["median_property_value"] else "N/A",
            f"{row['median_interest_rate']:.2f}%" if row["median_interest_rate"] else "N/A",
            f"${row['median_buyer_income_k']:,.0f}k" if row["median_buyer_income_k"] else "N/A",
        )
    console.print(tbl)
    return out_tract


def run_all_path_c_silver() -> None:
    build_zip_liquidity_panel()
    build_hmda_credit_summary()


if __name__ == "__main__":
    run_all_path_c_silver()
