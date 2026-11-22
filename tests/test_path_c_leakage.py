"""Temporal-leakage unit tests for Path C feature engineering."""

from __future__ import annotations

from datetime import date

import polars as pl

from atl_housing.tracks.path_c_liquidity.baseline_model import engineer_leakage_safe_features


def test_backward_features_do_not_leak_future_rows() -> None:
    """Mutating a future row must NEVER change any engineered feature at an earlier date."""
    dates = [date(2023, m, 1) for m in range(1, 13)]
    base = pl.DataFrame(
        {
            "zip_code": ["30022"] * 12,
            "month_date": dates,
            "zhvi": [400_000.0 + i * 5_000.0 for i in range(12)],
            "gross_rent_yield": [0.05] * 12,
            "median_dom": [30.0 + i for i in range(12)],
            "price_cut_share": [0.15] * 12,
            "active_listing_count": [100.0] * 12,
            "demand_score": [70.0] * 12,
            "supply_score": [60.0] * 12,
        }
    )

    perturbed = base.with_columns(
        pl.when(pl.col("month_date") >= date(2023, 10, 1))
        .then(9_999_999.0)
        .otherwise(pl.col("zhvi"))
        .alias("zhvi"),
        pl.when(pl.col("month_date") >= date(2023, 10, 1))
        .then(999.0)
        .otherwise(pl.col("median_dom"))
        .alias("median_dom"),
    )

    f_base = engineer_leakage_safe_features(base).filter(pl.col("month_date") < date(2023, 10, 1))
    f_pert = engineer_leakage_safe_features(perturbed).filter(
        pl.col("month_date") < date(2023, 10, 1)
    )

    assert f_base.equals(f_pert), "Future perturbation leaked into past engineered features!"
