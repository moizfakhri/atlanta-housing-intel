"""Unit tests for Spatially Weighted Conformal Prediction (SWCP)."""

from __future__ import annotations

import numpy as np

from atl_housing.tracks.path_a_valuation.spatial_conformal import (
    SpatiallyWeightedConformalRegressor,
)


def test_swcp_adapts_interval_width_to_local_heteroscedasticity() -> None:
    """SWCP should produce tighter intervals in low-noise suburbs and wider ones in high-noise areas."""
    rng = np.random.default_rng(42)

    # Suburb A (Suwanee-like): tight noise (±$10k)
    n_a = 200
    lat_a = 34.05 + rng.normal(0, 0.01, n_a)
    lon_a = -84.07 + rng.normal(0, 0.01, n_a)
    err_a = rng.normal(0, 10_000, n_a)

    # Suburb B (Luxury/custom-like, ~25km away): wide noise (±$80k)
    n_b = 200
    lat_b = 33.85 + rng.normal(0, 0.01, n_b)
    lon_b = -84.38 + rng.normal(0, 0.01, n_b)
    err_b = rng.normal(0, 80_000, n_b)

    cal_lat = np.concatenate([lat_a[:150], lat_b[:150]])
    cal_lon = np.concatenate([lon_a[:150], lon_b[:150]])
    cal_pred = np.full(300, 500_000.0)
    cal_true = cal_pred + np.concatenate([err_a[:150], err_b[:150]])

    swcp = SpatiallyWeightedConformalRegressor(bandwidth_km=5.0, alpha=0.10)
    swcp.fit(cal_true, cal_pred, cal_lat, cal_lon)

    iv_a = swcp.predict_intervals(np.full(50, 500_000.0), lat_a[150:], lon_a[150:], spatial=True)
    iv_b = swcp.predict_intervals(np.full(50, 500_000.0), lat_b[150:], lon_b[150:], spatial=True)

    # Local adaptation: Suburb B intervals must be substantially wider than Suburb A
    assert iv_b.mean_width() > 3.0 * iv_a.mean_width()
    # Both regions should achieve ~90% local empirical coverage
    assert iv_a.empirical_coverage(500_000.0 + err_a[150:]) >= 0.80
    assert iv_b.empirical_coverage(500_000.0 + err_b[150:]) >= 0.80
