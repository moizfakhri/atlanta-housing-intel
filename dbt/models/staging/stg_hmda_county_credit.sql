{{ config(materialized='view') }}

SELECT
    activity_year,
    county_code,
    CASE county_code
        WHEN '13121' THEN 'Fulton County'
        WHEN '13089' THEN 'DeKalb County'
        WHEN '13067' THEN 'Cobb County'
        WHEN '13135' THEN 'Gwinnett County'
        WHEN '13117' THEN 'Forsyth County'
        ELSE county_code
    END AS county_name,
    COUNT(*) AS total_applications,
    SUM(CASE WHEN is_home_purchase AND is_originated THEN 1 ELSE 0 END) AS purchase_originations,
    ROUND(
        SUM(CASE WHEN is_denied THEN 1 ELSE 0 END)::DOUBLE / NULLIF(COUNT(*), 0),
        4
    ) AS denial_rate,
    ROUND(
        SUM(CASE WHEN is_home_purchase AND is_originated AND is_investor THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(SUM(CASE WHEN is_home_purchase AND is_originated THEN 1 ELSE 0 END), 0),
        4
    ) AS investor_purchase_share,
    MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN property_value END) AS median_hmda_property_value,
    MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN loan_to_value_ratio END) AS median_ltv,
    MEDIAN(CASE WHEN is_home_purchase AND is_originated THEN income END) AS median_buyer_income_k
FROM read_parquet('../data/bronze/path_c/hmda_atlanta_5county_loans.parquet')
GROUP BY 1, 2, 3
