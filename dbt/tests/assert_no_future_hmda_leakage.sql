-- Singular dbt anti-leakage test: HMDA source year must strictly precede the panel month year.
SELECT
    zip_code,
    month_date,
    hmda_source_year
FROM {{ ref('mart_atlanta_submarket_signals') }}
WHERE hmda_source_year IS NOT NULL
  AND hmda_source_year >= EXTRACT(YEAR FROM month_date)
