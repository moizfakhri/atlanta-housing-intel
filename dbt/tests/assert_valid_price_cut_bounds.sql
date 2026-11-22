-- Singular dbt data-quality test: price_cut_share must always stay within [0, 1].
SELECT
    zip_code,
    month_date,
    price_cut_share
FROM {{ ref('mart_atlanta_submarket_signals') }}
WHERE price_cut_share IS NOT NULL
  AND (price_cut_share < 0.0 OR price_cut_share > 1.0)
