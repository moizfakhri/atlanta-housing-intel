{{ config(materialized='table') }}

-- Gold Metro Atlanta Submarket Liquidity, Macro & Credit Demand Mart
-- Enforces strict 1-year publication-lag on annual HMDA credit features (activity_year = YEAR(month_date) - 1)
-- to guarantee zero temporal lookahead leakage.

WITH zip_dim AS (
    SELECT
        zip_code,
        ANY_VALUE(City) AS city,
        ANY_VALUE(CountyName) AS county_name
    FROM read_parquet('../data/bronze/path_c/zillow_atlanta_zip_panel.parquet')
    GROUP BY zip_code
),
zillow AS (
    SELECT
        zip_code,
        DATE_TRUNC('month', period_end)::DATE AS month_date,
        zhvi,
        zori,
        gross_rent_yield
    FROM read_parquet('../data/bronze/path_c/zillow_atlanta_zip_panel.parquet')
),
fred AS (
    SELECT
        month_date::DATE AS month_date,
        mortgage_30y,
        fed_funds,
        atl_unemp_rate,
        atl_nonfarm_emp,
        atl_case_shiller
    FROM read_parquet('../data/bronze/path_c/fred_atlanta_macro.parquet')
),
joined AS (
    SELECT
        r.zip_code || '_' || STRFTIME(r.month_date, '%Y-%m') AS zip_month_id,
        r.zip_code,
        r.month_date,
        COALESCE(d.city, r.zip_name) AS city,
        d.county_name,
        z.zhvi,
        z.zori,
        z.gross_rent_yield,
        r.median_listing_price,
        r.active_listing_count,
        r.median_dom,
        r.price_cut_share,
        r.pending_ratio,
        r.hotness_score,
        r.demand_score,
        r.supply_score,
        r.buyer_views_vs_us,
        f.mortgage_30y,
        f.atl_unemp_rate,
        h.activity_year AS hmda_source_year,
        h.denial_rate AS hmda_denial_rate_lag1y,
        h.investor_purchase_share AS hmda_investor_share_lag1y,
        h.median_ltv AS hmda_median_ltv_lag1y,
        h.median_buyer_income_k AS hmda_buyer_income_k_lag1y,
        CASE
            WHEN r.hotness_score >= 60 AND COALESCE(r.price_cut_share, 0) < 0.35 THEN 'HOT_SELLER'
            WHEN COALESCE(r.price_cut_share, 0) >= 0.42 OR COALESCE(r.median_dom, 0) >= 65 THEN 'BUYER_LEVERAGE_PRICE_CUTS'
            ELSE 'BALANCED'
        END AS market_regime
    FROM {{ ref('stg_realtor_zip_liquidity') }} r
    LEFT JOIN zip_dim d
        ON r.zip_code = d.zip_code
    LEFT JOIN zillow z
        ON r.zip_code = z.zip_code
       AND r.month_date = z.month_date
    LEFT JOIN fred f
        ON r.month_date = f.month_date
    LEFT JOIN {{ ref('stg_hmda_county_credit') }} h
        ON d.county_name = h.county_name
       AND h.activity_year = EXTRACT(YEAR FROM r.month_date) - 1
)
SELECT * FROM joined
ORDER BY zip_code, month_date
