"""Path C Baseline Modeling: Walk-Forward Cross-Sectional & Liquidity Forecasting.

Tests two core Path C research hypotheses across Metro Atlanta ZIPs:
  H1 (Price Lead-Lag): Do Realtor liquidity & buyer-demand microstructure signals
     (DOM, price-cut share, pending ratio, demand/supply scores, buyer views/listing)
     improve 6-month-ahead home price return forecasts over pure price momentum?
  H2 (Liquidity Regime Shift): Can we forecast 3-month-ahead Days-on-Market changes
     to spot suburbs flipping into bidding wars vs. price-cut slowdowns?

Evaluates strictly with expanding-window walk-forward temporal splits (zero leakage),
reporting OOS MAE, RMSE, R2, and monthly cross-sectional Spearman IC (rank correlation).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from atl_housing.config import SILVER_DIR

console = Console()
SILVER_ZIP_PANEL = SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet"

PRICE_MOMENTUM_FEATURES = [
    "zhvi_ret_1m",
    "zhvi_ret_3m",
    "zhvi_ret_6m",
    "zhvi_ret_12m",
    "gross_rent_yield",
    "month_sin",
    "month_cos",
]

LIQUIDITY_MICROSTRUCTURE_FEATURES = [
    "median_dom",
    "dom_delta_3m",
    "price_cut_share",
    "price_cut_delta_3m",
    "pending_ratio",
    "pending_to_new_ratio",
    "active_listing_yoy",
    "ask_vs_zhvi_spread",
    "hotness_score",
    "demand_score",
    "supply_score",
    "buyer_views_vs_us",
    "demand_minus_supply",
]


@dataclass
class EvalMetrics:
    model_name: str
    feature_set: str
    target: str
    n_test: int
    mae: float
    rmse: float
    mean_spearman_ic: float


def engineer_leakage_safe_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create strictly backward-looking momentum, liquidity-change, and seasonal features."""
    df = df.sort(["zip_code", "month_date"])
    return df.with_columns(
        # Backward price momentum (strictly t vs t-k)
        (pl.col("zhvi") / pl.col("zhvi").shift(1).over("zip_code") - 1.0).alias("zhvi_ret_1m"),
        (pl.col("zhvi") / pl.col("zhvi").shift(3).over("zip_code") - 1.0).alias("zhvi_ret_3m"),
        (pl.col("zhvi") / pl.col("zhvi").shift(6).over("zip_code") - 1.0).alias("zhvi_ret_6m"),
        (pl.col("zhvi") / pl.col("zhvi").shift(12).over("zip_code") - 1.0).alias("zhvi_ret_12m"),
        # Backward liquidity & inventory shifts
        (pl.col("median_dom") - pl.col("median_dom").shift(3).over("zip_code")).alias("dom_delta_3m"),
        (pl.col("price_cut_share") - pl.col("price_cut_share").shift(3).over("zip_code")).alias(
            "price_cut_delta_3m"
        ),
        (
            pl.col("active_listing_count")
            / pl.col("active_listing_count").shift(12).over("zip_code")
            - 1.0
        ).alias("active_listing_yoy"),
        (pl.col("demand_score") - pl.col("supply_score")).alias("demand_minus_supply"),
        # Cyclical month seasonality
        (2.0 * np.pi * pl.col("month_date").dt.month() / 12.0).sin().alias("month_sin"),
        (2.0 * np.pi * pl.col("month_date").dt.month() / 12.0).cos().alias("month_cos"),
    )


def walk_forward_evaluate(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    horizon_months: int,
    model_type: str,
    feature_set_name: str,
    min_train_months: int = 36,
    step_months: int = 3,
) -> tuple[EvalMetrics, dict[str, float]]:
    """Expanding-window walk-forward CV with an explicit purge gap equal to target horizon."""
    clean = df.filter(pl.col(target).is_not_null()).sort("month_date")
    dates = sorted(clean["month_date"].unique().to_list())

    all_true: list[float] = []
    all_pred: list[float] = []
    all_dates: list[object] = []
    importances = np.zeros(len(features), dtype=float)
    n_folds = 0

    for idx in range(min_train_months, len(dates) - horizon_months, step_months):
        # Purge gap: training labels at cutoff_train must already be realized before test_start
        train_cutoff = dates[idx - horizon_months]
        test_dates = dates[idx : min(idx + step_months, len(dates))]

        train_df = clean.filter(pl.col("month_date") <= train_cutoff)
        test_df = clean.filter(pl.col("month_date").is_in(test_dates))
        if train_df.height < 200 or test_df.height < 20:
            continue

        X_train = train_df.select(features).to_numpy()
        y_train = train_df[target].to_numpy()
        X_test = test_df.select(features).to_numpy()
        y_test = test_df[target].to_numpy()

        if model_type == "ridge":
            model = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=10.0)),
                ]
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            importances += np.abs(model.named_steps["reg"].coef_)
        elif model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.04,
                num_leaves=15,
                subsample_for_bin=50000,
                colsample_bytree=0.8,
                subsample=0.8,
                min_child_samples=30,
                random_state=42,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            importances += model.feature_importances_
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())
        all_dates.extend(test_df["month_date"].to_list())
        n_folds += 1

    eval_df = pl.DataFrame({"month_date": all_dates, "y_true": all_true, "y_pred": all_pred})
    monthly_ics: list[float] = []
    for _, grp in eval_df.group_by("month_date"):
        if grp.height >= 15:
            ic, _ = spearmanr(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
            if np.isfinite(ic):
                monthly_ics.append(float(ic))

    metrics = EvalMetrics(
        model_name=model_type,
        feature_set=feature_set_name,
        target=target,
        n_test=len(all_true),
        mae=float(mean_absolute_error(all_true, all_pred)),
        rmse=float(root_mean_squared_error(all_true, all_pred)),
        mean_spearman_ic=float(np.mean(monthly_ics)) if monthly_ics else float("nan"),
    )
    imp_dict = {
        feat: float(importances[i] / max(n_folds, 1)) for i, feat in enumerate(features)
    }
    return metrics, imp_dict


def run_benchmark() -> None:
    df = pl.read_parquet(SILVER_ZIP_PANEL)
    feat_df = engineer_leakage_safe_features(df)

    full_features = PRICE_MOMENTUM_FEATURES + LIQUIDITY_MICROSTRUCTURE_FEATURES
    experiments = [
        ("target_zhvi_ret_6m", 6, "Price Momentum Only", PRICE_MOMENTUM_FEATURES, "ridge"),
        ("target_zhvi_ret_6m", 6, "Momentum + Liquidity", full_features, "ridge"),
        ("target_zhvi_ret_6m", 6, "Price Momentum Only", PRICE_MOMENTUM_FEATURES, "lightgbm"),
        ("target_zhvi_ret_6m", 6, "Momentum + Liquidity", full_features, "lightgbm"),
        ("target_dom_delta_3m", 3, "Price Momentum Only", PRICE_MOMENTUM_FEATURES, "lightgbm"),
        ("target_dom_delta_3m", 3, "Momentum + Liquidity", full_features, "lightgbm"),
    ]

    tbl = Table(title="Path C Walk-Forward Benchmark (Purged Expanding Window, Metro Atlanta ZIPs)")
    tbl.add_column("Target", style="cyan")
    tbl.add_column("Feature Set")
    tbl.add_column("Model")
    tbl.add_column("OOS N", justify="right")
    tbl.add_column("MAE", justify="right")
    tbl.add_column("RMSE", justify="right")
    tbl.add_column("Cross-Sec IC", justify="right", style="green")

    top_imp: dict[str, float] = {}
    for target, horizon, fset_name, fcols, mtype in experiments:
        m, imp = walk_forward_evaluate(
            feat_df,
            fcols,
            target,
            horizon_months=horizon,
            model_type=mtype,
            feature_set_name=fset_name,
        )
        tbl.add_row(
            target,
            fset_name,
            mtype,
            f"{m.n_test:,}",
            f"{m.mae:.4f}",
            f"{m.rmse:.4f}",
            f"{m.mean_spearman_ic:.3f}",
        )
        if target == "target_zhvi_ret_6m" and fset_name == "Momentum + Liquidity" and mtype == "lightgbm":
            top_imp = imp

    console.print(tbl)

    if top_imp:
        console.print("\n[bold]Top LightGBM Feature Importances (6M Ahead Return):[/bold]")
        for k, v in sorted(top_imp.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            console.print(f"  {k:<26s} {v:7.1f}")


if __name__ == "__main__":
    run_benchmark()
