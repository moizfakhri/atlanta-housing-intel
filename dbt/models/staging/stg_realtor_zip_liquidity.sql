{{ config(materialized='view') }}

WITH core AS (
    SELECT
        LPAD(CAST(postal_code AS VARCHAR), 5, '0') AS zip_code,
        zip_name,
        STRPTIME(CAST(month_date_yyyymm AS VARCHAR) || '01', '%Y%m%d')::DATE AS month_date,
        TRY_CAST(median_listing_price AS DOUBLE) AS median_listing_price,
        TRY_CAST(active_listing_count AS DOUBLE) AS active_listing_count,
        TRY_CAST(median_days_on_market AS DOUBLE) AS median_dom,
        TRY_CAST(new_listing_count AS DOUBLE) AS new_listing_count,
        TRY_CAST(price_reduced_count AS DOUBLE) AS price_reduced_count,
        TRY_CAST(pending_listing_count AS DOUBLE) AS pending_listing_count,
        TRY_CAST(pending_ratio AS DOUBLE) AS pending_ratio
    FROM read_parquet('../data/bronze/path_c/realtor_atlanta_zip_core.parquet')
    WHERE LENGTH(CAST(month_date_yyyymm AS VARCHAR)) = 6
),
hotness AS (
    SELECT
        LPAD(CAST(postal_code AS VARCHAR), 5, '0') AS zip_code,
        STRPTIME(CAST(month_date_yyyymm AS VARCHAR) || '01', '%Y%m%d')::DATE AS month_date,
        TRY_CAST(hotness_score AS DOUBLE) AS hotness_score,
        TRY_CAST(supply_score AS DOUBLE) AS supply_score,
        TRY_CAST(demand_score AS DOUBLE) AS demand_score,
        TRY_CAST(page_view_count_per_property_vs_us AS DOUBLE) AS buyer_views_vs_us
    FROM read_parquet('../data/bronze/path_c/realtor_atlanta_zip_hotness.parquet')
    WHERE LENGTH(CAST(month_date_yyyymm AS VARCHAR)) = 6
)
SELECT
    c.zip_code,
    c.month_date,
    c.zip_name,
    c.median_listing_price,
    c.active_listing_count,
    c.median_dom,
    c.new_listing_count,
    c.price_reduced_count,
    c.pending_listing_count,
    c.pending_ratio,
    CASE
        WHEN c.active_listing_count > 0
        THEN LEAST(1.0, GREATEST(0.0, c.price_reduced_count / c.active_listing_count))
    END AS price_cut_share,
    h.hotness_score,
    h.supply_score,
    h.demand_score,
    h.buyer_views_vs_us
FROM core c
LEFT JOIN hotness h
    ON c.zip_code = h.zip_code
   AND c.month_date = h.month_date
