"""Path A Uncertainty Quantification: Spatially Weighted Conformal Prediction (SWCP).

Implements spatially weighted split conformal prediction for Automated Valuation Models
following Hjort et al. (arXiv:2312.06531, 2025) and GeoConformal (arXiv:2412.08661, 2025).

Why SWCP matters in Metro Atlanta suburbs:
  Standard split conformal prediction guarantees global coverage (e.g., 90% overall),
  but severely under-covers heterogeneous/luxury/transitioning areas (e.g., Milton,
  Decatur) while over-covering uniform master-planned subdivisions (e.g., Suwanee).
  Weighting calibration residuals by geographic proximity restores local validity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Pairwise Haversine distance matrix in kilometers: shape (len(lat1), len(lat2))."""
    r_earth = 6371.0088
    phi1 = np.radians(lat1)[:, None]
    phi2 = np.radians(lat2)[None, :]
    dphi = phi2 - phi1
    dlambda = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute finite-sample conformal weighted quantile with mass at +infinity."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_total = float(np.sum(w_sorted))
    if w_total <= 0.0:
        return float(np.quantile(values, q))

    # Normalized weights including point mass 1/(n_eff+1) at infinity for conservative validity
    cdf = np.cumsum(w_sorted) / (w_total + np.max(w_sorted))
    idx = int(np.searchsorted(cdf, q, side="left"))
    if idx >= len(v_sorted):
        return float(v_sorted[-1])
    return float(v_sorted[idx])


@dataclass
class SpatialConformalIntervals:
    lower: np.ndarray
    upper: np.ndarray
    half_width: np.ndarray
    alpha: float

    def empirical_coverage(self, y_true: np.ndarray) -> float:
        return float(np.mean((y_true >= self.lower) & (y_true <= self.upper)))

    def mean_width(self) -> float:
        return float(np.mean(self.upper - self.lower))


class SpatiallyWeightedConformalRegressor:
    """Model-agnostic Spatially Weighted Conformal Predictor for real estate AVMs."""

    def __init__(self, bandwidth_km: float = 4.0, alpha: float = 0.10) -> None:
        if bandwidth_km <= 0:
            raise ValueError("bandwidth_km must be positive")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.bandwidth_km = bandwidth_km
        self.alpha = alpha
        self._cal_lat: np.ndarray | None = None
        self._cal_lon: np.ndarray | None = None
        self._cal_scores: np.ndarray | None = None

    def fit(
        self,
        y_cal_true: np.ndarray,
        y_cal_pred: np.ndarray,
        cal_lat: np.ndarray,
        cal_lon: np.ndarray,
        cal_scale: np.ndarray | None = None,
    ) -> SpatiallyWeightedConformalRegressor:
        """Store calibration non-conformity scores and coordinates."""
        abs_err = np.abs(np.asarray(y_cal_true, dtype=float) - np.asarray(y_cal_pred, dtype=float))
        if cal_scale is not None:
            scale = np.maximum(np.asarray(cal_scale, dtype=float), 1e-6)
            abs_err = abs_err / scale
        self._cal_scores = abs_err
        self._cal_lat = np.asarray(cal_lat, dtype=float)
        self._cal_lon = np.asarray(cal_lon, dtype=float)
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        scale: np.ndarray | None = None,
        *,
        spatial: bool = True,
    ) -> SpatialConformalIntervals:
        """Compute (1 - alpha) prediction intervals for test properties."""
        if self._cal_scores is None or self._cal_lat is None or self._cal_lon is None:
            raise RuntimeError("Call fit() before predict_intervals().")

        y_pred_arr = np.asarray(y_pred, dtype=float)
        n_test = len(y_pred_arr)
        q_level = 1.0 - self.alpha

        if not spatial:
            n_cal = len(self._cal_scores)
            q_val = float(
                np.quantile(
                    self._cal_scores,
                    min(1.0, np.ceil((n_cal + 1) * q_level) / n_cal),
                    method="higher",
                )
            )
            half_w = np.full(n_test, q_val, dtype=float)
        else:
            dists = haversine_km(
                np.asarray(lat, dtype=float),
                np.asarray(lon, dtype=float),
                self._cal_lat,
                self._cal_lon,
            )
            weights = np.exp(-0.5 * (dists / self.bandwidth_km) ** 2)
            half_w = np.array(
                [weighted_quantile(self._cal_scores, weights[i], q_level) for i in range(n_test)],
                dtype=float,
            )

        if scale is not None:
            half_w = half_w * np.maximum(np.asarray(scale, dtype=float), 1e-6)

        return SpatialConformalIntervals(
            lower=y_pred_arr - half_w,
            upper=y_pred_arr + half_w,
            half_width=half_w,
            alpha=self.alpha,
        )
