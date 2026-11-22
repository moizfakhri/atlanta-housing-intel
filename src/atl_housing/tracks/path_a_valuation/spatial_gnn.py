"""Path A Deep ML: PyTorch Spatial Graph Transformer (Learned Comps) + MLflow + 3D Gold Export.

Implements a multi-head distance-aware Spatial Graph Transformer in PyTorch inspired by
Riveros et al. (arXiv:2405.06553, Scalable Property Valuation Models via Graph-based Deep Learning)
and Popelier et al. (2026, Retrieval-Enhanced Real Estate Appraisal).

Key anti-leakage design:
  Every property node attends ONLY to its K=12 geographically nearest *training-set*
  comparable properties (never calibration or test labels), learning interpretable
  attention weights alpha_ij that power both valuation and our 3D deck.gl comp ArcLayer.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from atl_housing.config import DATA_DIR, GOLD_DIR
from atl_housing.tracks.path_a_valuation.spatial_conformal import (
    SpatiallyWeightedConformalRegressor,
    haversine_km,
)
from atl_housing.tracks.path_a_valuation.valuation_benchmark import build_parcel_3d_dataset

console = Console()
PATH_A_GOLD = GOLD_DIR / "path_a"


class SpatialGraphTransformerNet(nn.Module):
    """Multi-head Spatial Graph Transformer attending over K nearest training comps."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.w_ek = nn.Linear(hidden_dim, hidden_dim)
        self.w_ev = nn.Linear(hidden_dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Residual head: blends attention-weighted comp log-price with learned feature adjustment
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_self: torch.Tensor,  # (B, node_dim)
        x_nbr: torch.Tensor,  # (B, K, node_dim)
        edge_attr: torch.Tensor,  # (B, K, edge_dim)
        nbr_log_y: torch.Tensor,  # (B, K)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = x_nbr.shape
        h_self = self.node_enc(x_self)  # (B, H)
        h_nbr = self.node_enc(x_nbr)  # (B, K, H)
        h_edge = self.edge_enc(edge_attr)  # (B, K, H)

        q = self.w_q(h_self).view(B, 1, self.n_heads, self.head_dim)
        k = (self.w_k(h_nbr) + self.w_ek(h_edge)).view(B, K, self.n_heads, self.head_dim)
        v = (self.w_v(h_nbr) + self.w_ev(h_edge)).view(B, K, self.n_heads, self.head_dim)

        scores = (q * k).sum(dim=-1) / (self.head_dim**0.5)  # (B, K, n_heads)
        attn_heads = torch.softmax(scores, dim=1)  # softmax over K comps
        agg = (attn_heads.unsqueeze(-1) * v).sum(dim=1).reshape(B, self.hidden_dim)

        h = self.norm1(h_self + self.w_out(agg))
        h = self.norm2(h + self.ffn(h))

        # Mean attention across heads gives interpretable comparable-sale weights summing to 1
        comp_weights = attn_heads.mean(dim=-1)  # (B, K)
        comp_anchor = (comp_weights * nbr_log_y).sum(dim=1)  # (B,)
        delta = self.head(h).squeeze(-1)  # (B,)
        pred_log_y = comp_anchor + delta
        return pred_log_y, comp_weights, h


def _build_knn_tensors(
    X_all: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    log_y: np.ndarray,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    k: int = 12,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find K nearest training neighbors for each query index (strictly leakage-safe)."""
    dists = haversine_km(lat[query_idx], lon[query_idx], lat[train_idx], lon[train_idx])
    if exclude_self:
        # Exclude exact self match when query_idx == train_idx
        np.fill_diagonal(dists, np.inf)

    nn_local = np.argpartition(dists, k, axis=1)[:, :k]
    # Sort the K neighbors by ascending distance
    row_arange = np.arange(len(query_idx))[:, None]
    nn_dists = dists[row_arange, nn_local]
    order = np.argsort(nn_dists, axis=1)
    nn_local = nn_local[row_arange, order]
    nn_dists = nn_dists[row_arange, order]
    nn_global = train_idx[nn_local]

    x_nbr = X_all[nn_global]  # (Q, K, D)
    d_lat = lat[nn_global] - lat[query_idx][:, None]
    d_lon = lon[nn_global] - lon[query_idx][:, None]
    d_feat0 = X_all[nn_global, 0] - X_all[query_idx, 0:1]  # log_acres diff
    d_feat4 = X_all[nn_global, 4] - X_all[query_idx, 4:5]  # log_footprint diff
    edge_attr = np.stack(
        [np.log1p(nn_dists), d_lat * 100.0, d_lon * 100.0, d_feat0, d_feat4],
        axis=-1,
    ).astype(np.float32)

    nbr_y = log_y[nn_global].astype(np.float32)
    return x_nbr.astype(np.float32), edge_attr, nbr_y, nn_global


def train_and_benchmark_gnn(
    k_neighbors: int = 12,
    epochs: int = 90,
    lr: float = 2e-3,
) -> Path:
    """Train SpatialGraphTransformer, benchmark vs LightGBM + Hybrid, log to MLflow, export Gold 3D."""
    torch.manual_seed(42)
    np.random.seed(42)

    df = build_parcel_3d_dataset(overture_tag="sandy_springs")
    df = df.with_columns(
        pl.col("TotAppr").log().alias("log_value"),
        pl.col("LandAcres").log1p().alias("log_acres"),
        pl.col("total_footprint_sqm").fill_null(0.0).log1p().alias("log_footprint"),
        pl.col("max_bldg_height_m").fill_null(6.0).alias("bldg_height_m"),
        pl.col("est_bldg_volume_m3").fill_null(0.0).log1p().alias("log_volume"),
        pl.col("n_structures").fill_null(0).cast(pl.Float64).alias("n_struct_f"),
        pl.col("LivUnits").fill_null(1).cast(pl.Float64).alias("liv_units_f"),
        pl.col("NbrHood").cast(pl.Utf8).fill_null("UNK").alias("nbr_str"),
    )

    n = df.height
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_tr, n_cal = int(0.60 * n), int(0.80 * n)
    tr_idx, cal_idx, te_idx = perm[:n_tr], perm[n_tr:n_cal], perm[n_cal:]

    # Train-only target encoding for NbrHood
    train_nbr_med = (
        df[tr_idx].group_by("nbr_str").agg(pl.col("log_value").median().alias("nbr_med_log_val"))
    )
    global_med = float(df[tr_idx]["log_value"].median())
    df = df.join(train_nbr_med, on="nbr_str", how="left").with_columns(
        pl.col("nbr_med_log_val").fill_null(global_med)
    )

    feat_cols = [
        "log_acres",
        "liv_units_f",
        "lon",
        "lat",
        "log_footprint",
        "bldg_height_m",
        "log_volume",
        "n_struct_f",
        "nbr_med_log_val",
    ]
    X_raw = df.select(feat_cols).to_numpy().astype(np.float32)
    scaler = StandardScaler().fit(X_raw[tr_idx])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    log_y = df["log_value"].to_numpy().astype(np.float32)
    y_dollars = df["TotAppr"].to_numpy()

    # Build strictly leakage-safe kNN graphs (every split attends ONLY to training comps)
    tr_nbr, tr_edge, tr_ny, _ = _build_knn_tensors(
        X_scaled, lat, lon, log_y, tr_idx, tr_idx, k=k_neighbors, exclude_self=True
    )
    cal_nbr, cal_edge, cal_ny, _ = _build_knn_tensors(
        X_scaled, lat, lon, log_y, tr_idx, cal_idx, k=k_neighbors, exclude_self=False
    )
    te_nbr, te_edge, te_ny, _ = _build_knn_tensors(
        X_scaled, lat, lon, log_y, tr_idx, te_idx, k=k_neighbors, exclude_self=False
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Training PyTorch SpatialGraphTransformer on device={device} (K={k_neighbors})...[/cyan]")

    model = SpatialGraphTransformerNet(
        node_dim=X_scaled.shape[1], edge_dim=tr_edge.shape[-1], hidden_dim=64, n_heads=4
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    t_x_tr = torch.from_numpy(X_scaled[tr_idx]).to(device)
    t_nbr_tr = torch.from_numpy(tr_nbr).to(device)
    t_edge_tr = torch.from_numpy(tr_edge).to(device)
    t_ny_tr = torch.from_numpy(tr_ny).to(device)
    t_y_tr = torch.from_numpy(log_y[tr_idx]).to(device)

    t_x_cal = torch.from_numpy(X_scaled[cal_idx]).to(device)
    t_nbr_cal = torch.from_numpy(cal_nbr).to(device)
    t_edge_cal = torch.from_numpy(cal_edge).to(device)
    t_ny_cal = torch.from_numpy(cal_ny).to(device)

    t_x_te = torch.from_numpy(X_scaled[te_idx]).to(device)
    t_nbr_te = torch.from_numpy(te_nbr).to(device)
    t_edge_te = torch.from_numpy(te_edge).to(device)
    t_ny_te = torch.from_numpy(te_ny).to(device)

    best_cal_loss = float("inf")
    best_state = None
    batch_size = 256

    for _ in range(epochs):
        model.train()
        perm_b = torch.randperm(len(tr_idx), device=device)
        for i in range(0, len(tr_idx), batch_size):
            b = perm_b[i : i + batch_size]
            opt.zero_grad()
            pred, _, _ = model(t_x_tr[b], t_nbr_tr[b], t_edge_tr[b], t_ny_tr[b])
            loss = loss_fn(pred, t_y_tr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            cal_pred_log, _, _ = model(t_x_cal, t_nbr_cal, t_edge_cal, t_ny_cal)
            cal_loss = float(loss_fn(cal_pred_log, torch.from_numpy(log_y[cal_idx]).to(device)))
            if cal_loss < best_cal_loss:
                best_cal_loss = cal_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        gnn_tr_log, _, _ = model(t_x_tr, t_nbr_tr, t_edge_tr, t_ny_tr)
        gnn_cal_log, _, _ = model(t_x_cal, t_nbr_cal, t_edge_cal, t_ny_cal)
        gnn_te_log, _, _ = model(t_x_te, t_nbr_te, t_edge_te, t_ny_te)

    gnn_cal_dollars = np.exp(gnn_cal_log.cpu().numpy())
    gnn_te_dollars = np.exp(gnn_te_log.cpu().numpy())

    # Benchmark 1: Standalone LightGBM + Overture 3D
    lgbm = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_raw[tr_idx], log_y[tr_idx])
    lgbm_cal_dollars = np.exp(lgbm.predict(X_raw[cal_idx]))
    lgbm_te_dollars = np.exp(lgbm.predict(X_raw[te_idx]))

    # Benchmark 2: Hybrid Ensemble (50/50 log-space blend of Spatial Graph Transformer + LightGBM)
    hyb_cal_dollars = np.exp(0.5 * np.log(lgbm_cal_dollars) + 0.5 * np.log(gnn_cal_dollars))
    hyb_te_dollars = np.exp(0.5 * np.log(lgbm_te_dollars) + 0.5 * np.log(gnn_te_dollars))

    # Multiplicative SWCP around the best model (Hybrid)
    swcp = SpatiallyWeightedConformalRegressor(bandwidth_km=2.0, alpha=0.10)
    swcp.fit(
        y_cal_true=y_dollars[cal_idx],
        y_cal_pred=hyb_cal_dollars,
        cal_lat=lat[cal_idx],
        cal_lon=lon[cal_idx],
        cal_scale=hyb_cal_dollars,
    )
    iv_te = swcp.predict_intervals(
        hyb_te_dollars, lat[te_idx], lon[te_idx], scale=hyb_te_dollars, spatial=True
    )

    y_te = y_dollars[te_idx]
    results = [
        (
            "LightGBM + Overture 3D",
            float(mean_absolute_percentage_error(y_te, lgbm_te_dollars)),
            float(root_mean_squared_error(y_te, lgbm_te_dollars)),
        ),
        (
            "PyTorch Spatial Graph Transformer (K=12)",
            float(mean_absolute_percentage_error(y_te, gnn_te_dollars)),
            float(root_mean_squared_error(y_te, gnn_te_dollars)),
        ),
        (
            "Hybrid (GNN + LightGBM Ensemble)",
            float(mean_absolute_percentage_error(y_te, hyb_te_dollars)),
            float(root_mean_squared_error(y_te, hyb_te_dollars)),
        ),
    ]

    # Log experiment cleanly to local gitignored SQLite MLflow backend
    mlflow_db = DATA_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("path_a_spatial_gnn_benchmark")
    with mlflow.start_run(run_name="sandy_springs_3d_gnn_vs_lgbm"):
        mlflow.log_params({"k_neighbors": k_neighbors, "epochs": epochs, "hidden_dim": 64, "n_heads": 4})
        mlflow.log_metrics(
            {
                "lgbm_mape": results[0][1],
                "lgbm_rmse": results[0][2],
                "gnn_mape": results[1][1],
                "gnn_rmse": results[1][2],
                "hybrid_mape": results[2][1],
                "hybrid_rmse": results[2][2],
                "hybrid_swcp_cov": iv_te.empirical_coverage(y_te),
                "hybrid_swcp_width": iv_te.mean_width(),
            }
        )

    tbl = Table(title="Path A Deep ML Benchmark: LightGBM vs. PyTorch Spatial Graph Transformer")
    tbl.add_column("Model", style="cyan")
    tbl.add_column("Test MAPE", justify="right")
    tbl.add_column("Test RMSE", justify="right")
    for name, mape, rmse in results:
        tbl.add_row(name, f"{mape * 100:.2f}%", f"${rmse:,.0f}")
    console.print(tbl)
    console.print(
        f"[green]Hybrid + Multiplicative SWCP (2km):[/green] "
        f"90% target -> {iv_te.empirical_coverage(y_te) * 100:.1f}% empirical coverage, "
        f"mean width ${iv_te.mean_width():,.0f}"
    )

    # Export Gold 3D predictions + top-5 learned GNN attention comp arcs for EVERY parcel (for 3D Twin!)
    all_idx = np.arange(n)
    all_nbr, all_edge, all_ny, all_nn_global = _build_knn_tensors(
        X_scaled, lat, lon, log_y, tr_idx, all_idx, k=k_neighbors, exclude_self=False
    )
    with torch.no_grad():
        all_gnn_log, all_weights, _ = model(
            torch.from_numpy(X_scaled).to(device),
            torch.from_numpy(all_nbr).to(device),
            torch.from_numpy(all_edge).to(device),
            torch.from_numpy(all_ny).to(device),
        )
    all_gnn_dollars = np.exp(all_gnn_log.cpu().numpy())
    all_lgbm_dollars = np.exp(lgbm.predict(X_raw))
    all_hyb_dollars = np.exp(0.5 * np.log(all_lgbm_dollars) + 0.5 * np.log(all_gnn_dollars))
    all_iv = swcp.predict_intervals(
        all_hyb_dollars, lat, lon, scale=all_hyb_dollars, spatial=True
    )

    weights_np = all_weights.cpu().numpy()
    parcel_ids = df["ParcelID"].to_list()
    top_arcs_json: list[str] = []
    for i in range(n):
        # Skip exact self if present in training set, take top-5 highest-attention comps
        cand_indices = all_nn_global[i]
        cand_w = weights_np[i]
        pairs = [
            (int(c_idx), float(w))
            for c_idx, w in zip(cand_indices, cand_w, strict=True)
            if int(c_idx) != i
        ]
        pairs.sort(key=lambda t: t[1], reverse=True)
        top5 = [
            {
                "comp_id": parcel_ids[c_idx],
                "lon": round(float(lon[c_idx]), 6),
                "lat": round(float(lat[c_idx]), 6),
                "value": round(float(y_dollars[c_idx]), 0),
                "weight": round(w, 4),
            }
            for c_idx, w in pairs[:5]
        ]
        top_arcs_json.append(json.dumps(top5))

    PATH_A_GOLD.mkdir(parents=True, exist_ok=True)
    out_path = PATH_A_GOLD / "parcel_3d_predictions.parquet"
    gold_df = df.select(
        [
            "ParcelID",
            "ZipCode",
            "NbrHood",
            "LandAcres",
            "TotAppr",
            "lon",
            "lat",
            "total_footprint_sqm",
            "bldg_height_m",
        ]
    ).with_columns(
        pl.Series("lgbm_pred", np.round(all_lgbm_dollars, 0)),
        pl.Series("gnn_pred", np.round(all_gnn_dollars, 0)),
        pl.Series("hybrid_pred", np.round(all_hyb_dollars, 0)),
        pl.Series("swcp_lower", np.round(np.maximum(all_iv.lower, 0.0), 0)),
        pl.Series("swcp_upper", np.round(all_iv.upper, 0)),
        pl.Series(
            "mispricing_pct",
            np.round((all_hyb_dollars - y_dollars) / np.maximum(y_dollars, 1.0) * 100.0, 2),
        ),
        pl.Series("top5_comps_json", top_arcs_json),
    )
    gold_df.write_parquet(out_path)
    console.print(f"[green]Exported Gold 3D Parcel Predictions + GNN Comps:[/green] {gold_df.height:,} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    train_and_benchmark_gnn()
