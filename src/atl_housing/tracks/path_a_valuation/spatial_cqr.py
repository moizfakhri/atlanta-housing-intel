"""Path A Upgrade: Localized Spatial Conformalized Quantile Regression (Spatial CQR) & 3D Footprint Export.

Implements LSCP / Spatial CQR (Jiang & Xie, arXiv:2412.01098, 2025; Romano et al., 2019):
  1. Trains LightGBM lower (5%) and upper (95%) quantile regressors in log-price space.
  2. Computes asymmetric CQR non-conformity scores s_i = max(q_lo - y, y - q_hi).
  3. Applies Haversine spatial kernel weighting on calibration residuals to restore
     conditional local coverage across Starter, Upper-Mid, AND Luxury home tiers.
  4. Exports true Overture 3D building polygon footprints joined to Hybrid GNN predictions
     for real 3D building extrusion in our deck.gl/MapLibre Twin.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from atl_housing.config import BRONZE_DIR, GOLD_DIR
from atl_housing.tracks.path_a_valuation.spatial_conformal import haversine_km, weighted_quantile
from atl_housing.tracks.path_a_valuation.valuation_benchmark import build_parcel_3d_dataset

console = Console()
PATH_A_BRONZE = BRONZE_DIR / "path_a"
PATH_A_GOLD = GOLD_DIR / "path_a"


def run_spatial_cqr_audit(alpha: float = 0.10, bandwidth_km: float = 2.5) -> dict[str, float]:
    """Benchmark Global CP vs. Multiplicative SWCP vs. Spatial CQR across all price tiers."""
    df = build_parcel_3d_dataset(overture_tag="sandy_springs").with_columns(
        pl.col("TotAppr").log().alias("log_value"),
        pl.col("LandAcres").log1p().alias("log_acres"),
        pl.col("NbrHood").cast(pl.Utf8).fill_null("UNK").alias("nbr_str"),
    )

    n = df.height
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_tr, n_cal = int(0.60 * n), int(0.80 * n)
    tr_idx, cal_idx, te_idx = perm[:n_tr], perm[n_tr:n_cal], perm[n_cal:]

    train_nbr_med = (
        df[tr_idx].group_by("nbr_str").agg(pl.col("log_value").median().alias("nbr_med_log_val"))
    )
    global_med = float(df[tr_idx]["log_value"].median())
    df = df.join(train_nbr_med, on="nbr_str", how="left").with_columns(
        pl.col("nbr_med_log_val").fill_null(global_med)
    )

    feats = [
        "log_acres",
        "LivUnits",
        "lon",
        "lat",
        "nbr_med_log_val",
        "n_structures",
        "total_footprint_sqm",
        "max_bldg_height_m",
        "est_bldg_volume_m3",
    ]
    X_tr = df[tr_idx].select(feats).to_numpy()
    y_tr = df[tr_idx]["log_value"].to_numpy()
    X_cal = df[cal_idx].select(feats).to_numpy()
    y_cal = df[cal_idx]["log_value"].to_numpy()
    X_te = df[te_idx].select(feats).to_numpy()
    y_te_log = df[te_idx]["log_value"].to_numpy()
    y_te_dollars = df[te_idx]["TotAppr"].to_numpy()

    q_params = {
        "n_estimators": 280,
        "learning_rate": 0.04,
        "num_leaves": 23,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    m_lo = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2.0, **q_params).fit(X_tr, y_tr)
    m_hi = lgb.LGBMRegressor(objective="quantile", alpha=1.0 - alpha / 2.0, **q_params).fit(
        X_tr, y_tr
    )
    m_med = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **q_params).fit(X_tr, y_tr)

    cal_lo, cal_hi = m_lo.predict(X_cal), m_hi.predict(X_cal)
    te_lo, te_hi, te_med = m_lo.predict(X_te), m_hi.predict(X_te), m_med.predict(X_te)

    # CQR signed conformity score in log-space (positive when y falls outside [q_lo, q_hi])
    cqr_scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)

    cal_lat, cal_lon = df[cal_idx]["lat"].to_numpy(), df[cal_idx]["lon"].to_numpy()
    te_lat, te_lon = df[te_idx]["lat"].to_numpy(), df[te_idx]["lon"].to_numpy()

    # Global CQR quantile vs. Spatially Weighted CQR (LSCP)
    n_cal_pts = len(cqr_scores)
    q_global = float(
        np.quantile(
            cqr_scores,
            min(1.0, np.ceil((n_cal_pts + 1) * (1.0 - alpha)) / n_cal_pts),
            method="higher",
        )
    )
    dists = haversine_km(te_lat, te_lon, cal_lat, cal_lon)
    weights = np.exp(-0.5 * (dists / bandwidth_km) ** 2)
    q_spatial = np.array(
        [weighted_quantile(cqr_scores, weights[i], 1.0 - alpha) for i in range(len(te_idx))]
    )

    low_g, high_g = np.exp(te_lo - q_global), np.exp(te_hi + q_global)
    low_s, high_s = np.exp(te_lo - q_spatial), np.exp(te_hi + q_spatial)

    cov_g = float(np.mean((y_te_dollars >= low_g) & (y_te_dollars <= high_g)))
    cov_s = float(np.mean((y_te_dollars >= low_s) & (y_te_dollars <= high_s)))

    # Condition on PREDICTED price tier (proper deployment-time conditional coverage!)
    pred_dollars = np.exp(te_med)
    qs = np.quantile(pred_dollars, [0.0, 0.33, 0.67, 1.0])
    labels = ["Predicted Starter (<33%)", "Predicted Upper-Mid (33-67%)", "Predicted Luxury (>67%)"]

    tbl = Table(
        title="Path A Spatial CQR (LSCP, arXiv:2412.01098) — Conditional Coverage by Predicted Tier"
    )
    tbl.add_column("Predicted Home Tier", style="cyan")
    tbl.add_column("N", justify="right")
    tbl.add_column("Global CQR Cov", justify="right")
    tbl.add_column("Spatial CQR Cov", justify="right", style="green")
    tbl.add_column("Spatial CQR Width", justify="right")

    for i, lbl in enumerate(labels):
        mask = (pred_dollars >= qs[i]) & (pred_dollars <= qs[i + 1])
        cg = float(np.mean((y_te_dollars[mask] >= low_g[mask]) & (y_te_dollars[mask] <= high_g[mask])))
        cs = float(np.mean((y_te_dollars[mask] >= low_s[mask]) & (y_te_dollars[mask] <= high_s[mask])))
        ws = float(np.mean(high_s[mask] - low_s[mask]))
        tbl.add_row(lbl, str(int(mask.sum())), f"{cg * 100:.1f}%", f"{cs * 100:.1f}%", f"${ws:,.0f}")

    console.print(tbl)
    console.print(
        f"[green]Global Coverage (90% target):[/green] Global CQR = {cov_g * 100:.1f}% "
        f"(mean width ${np.mean(high_g - low_g):,.0f}) | "
        f"Spatial CQR = {cov_s * 100:.1f}% (mean width ${np.mean(high_s - low_s):,.0f})"
    )
    return {
        "spatial_cqr_cov": cov_s,
        "spatial_cqr_width": float(np.mean(high_s - low_s)),
        "log_mae_med": float(np.mean(np.abs(y_te_log - te_med))),
    }


def export_true_3d_building_footprints(max_buildings: int = 3500) -> Path:
    """Export actual Overture 3D building polygon footprints joined with Gold GNN predictions."""
    overture_pq = PATH_A_BRONZE / "overture_3d_buildings_sandy_springs.parquet"
    parcels_pq = PATH_A_BRONZE / "parcels_fulton_sample.parquet"
    gold_pq = PATH_A_GOLD / "parcel_3d_predictions.parquet"
    out_geojson = PATH_A_GOLD / "buildings_3d_footprints.geojson"

    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial; SET geometry_always_xy = true;")

    rows = con.sql(
        f"""
        WITH parcels AS (
            SELECT
                ParcelID,
                geometry AS parcel_geom
            FROM read_parquet('{parcels_pq}')
        ),
        gold AS (
            SELECT
                ParcelID,
                ZipCode,
                TotAppr,
                hybrid_pred,
                gnn_pred,
                lgbm_pred,
                swcp_lower,
                swcp_upper,
                mispricing_pct,
                top5_comps_json
            FROM read_parquet('{gold_pq}')
        ),
        bld AS (
            SELECT
                id AS overture_id,
                COALESCE(height, 6.0) AS height_m,
                ROUND(footprint_sqm, 1) AS footprint_sqm,
                lon,
                lat,
                ST_AsGeoJSON(ST_GeomFromWKB(geom_wkb)) AS geom_json,
                ST_Point(lon, lat) AS bld_pt
            FROM read_parquet('{overture_pq}')
        )
        SELECT
            b.overture_id,
            b.height_m,
            b.footprint_sqm,
            b.lon,
            b.lat,
            b.geom_json,
            g.ParcelID,
            g.ZipCode,
            g.TotAppr,
            g.hybrid_pred,
            g.gnn_pred,
            g.lgbm_pred,
            g.swcp_lower,
            g.swcp_upper,
            g.mispricing_pct,
            g.top5_comps_json
        FROM bld b
        JOIN parcels p
          ON ST_Contains(p.parcel_geom, b.bld_pt)
        JOIN gold g
          ON p.ParcelID = g.ParcelID
        LIMIT {max_buildings}
        """
    ).fetchall()

    features = []
    for r in rows:
        features.append(
            {
                "type": "Feature",
                "geometry": json.loads(r[5]),
                "properties": {
                    "overture_id": r[0],
                    "height_m": round(float(r[1]), 1),
                    "footprint_sqm": float(r[2]),
                    "lon": round(float(r[3]), 6),
                    "lat": round(float(r[4]), 6),
                    "id": str(r[6]),
                    "zip": str(r[7] or ""),
                    "actual": float(r[8]),
                    "hybrid_pred": float(r[9]),
                    "gnn_pred": float(r[10]),
                    "lgbm_pred": float(r[11]),
                    "swcp_lower": float(r[12]),
                    "swcp_upper": float(r[13]),
                    "mispricing_pct": float(r[14]),
                    "comps": json.loads(r[15]),
                },
            }
        )

    PATH_A_GOLD.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    console.print(
        f"[green]Exported TRUE Overture 3D Building Footprints:[/green] "
        f"{len(features):,} extruded buildings -> {out_geojson}"
    )
    return out_geojson


if __name__ == "__main__":
    run_spatial_cqr_audit()
    export_true_3d_building_footprints()
