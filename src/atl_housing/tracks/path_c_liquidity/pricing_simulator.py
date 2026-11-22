"""Path C Practical Decision Tool: Seller Ask-Price vs. Liquidity & Carrying-Cost Simulator.

Combines:
  1. Empirical ZIP-level baseline liquidity (median DOM & price-cut risk) predicted from
     current Metro Atlanta market state (demand_score, supply_score, pending_ratio,
     active_listing_count, FRED 30Y mortgage rate, and seasonality).
  2. Individual-listing ask-price elasticity grounded in real-estate search & matching
     theory (Carrillo 2012; Opendoor hazard modeling) — avoiding the ecological fallacy
     of inferring individual overpricing elasticity from ZIP-level median list-price mix.

Outputs an ask-price frontier (-4% to +8% vs. fair value) showing Expected DOM,
30d/60d sale probabilities, price-cut risk, holding/carrying cost, and Expected Net Proceeds.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from atl_housing.config import SILVER_DIR

console = Console()
SILVER_ZIP_PANEL = SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet"

MARKET_STATE_FEATURES = [
    "demand_score",
    "supply_score",
    "hotness_score",
    "active_listing_count",
    "pending_ratio",
    "mortgage_30y",
    "month_sin",
    "month_cos",
]


def _prepare_panel() -> pl.DataFrame:
    df = pl.read_parquet(SILVER_ZIP_PANEL)
    return df.with_columns(
        (2.0 * np.pi * pl.col("month_date").dt.month() / 12.0).sin().alias("month_sin"),
        (2.0 * np.pi * pl.col("month_date").dt.month() / 12.0).cos().alias("month_cos"),
    )


def fit_market_baseline_models(df: pl.DataFrame) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
    """Fit ZIP baseline liquidity models (DOM & price-cut share) from current market state."""
    train = df.filter(
        pl.col("median_dom").is_not_null()
        & pl.col("price_cut_share").is_not_null()
        & (pl.col("active_listing_count") >= 15)
    )
    X = train.select(MARKET_STATE_FEATURES).to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)

    params = {
        "n_estimators": 250,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "min_child_samples": 40,
        "random_state": 42,
        "verbose": -1,
    }
    dom_model = lgb.LGBMRegressor(**params).fit(X, train["median_dom"].to_numpy())
    cut_model = lgb.LGBMRegressor(**params).fit(X, train["price_cut_share"].to_numpy())
    return dom_model, cut_model


def simulate_ask_frontier(
    zip_code: str,
    fair_value: float,
    *,
    monthly_carry_rate: float = 0.0075,  # ~9%/yr all-in carry (mortgage+tax+ins+HOA)
    typical_cut_discount: float = 0.055,  # avg ~5.5% haircut once forced to cut after sitting stale
) -> pl.DataFrame:
    """Simulate seller ask-price frontier for a specific Metro Atlanta ZIP."""
    df = _prepare_panel()
    dom_model, cut_model = fit_market_baseline_models(df)

    sub = df.filter(pl.col("zip_code") == zip_code).sort("month_date")
    if sub.height == 0:
        raise ValueError(f"ZIP {zip_code} not found in Silver Metro Atlanta panel.")

    latest_row = (
        sub.filter(pl.col("hotness_score").is_not_null() & pl.col("mortgage_30y").is_not_null())
        .tail(1)
        .to_dicts()
    )
    if not latest_row:
        latest_row = sub.tail(1).to_dicts()
    state = latest_row[0]

    x_vec = np.array(
        [
            [
                state.get("demand_score") or np.nan,
                state.get("supply_score") or np.nan,
                state.get("hotness_score") or np.nan,
                state.get("active_listing_count") or np.nan,
                state.get("pending_ratio") or np.nan,
                state.get("mortgage_30y") or 6.5,
                state.get("month_sin") or 0.0,
                state.get("month_cos") or 1.0,
            ]
        ],
        dtype=float,
    )
    base_dom = float(np.clip(dom_model.predict(x_vec)[0], 10.0, 150.0))
    base_cut = float(np.clip(cut_model.predict(x_vec)[0], 0.05, 0.80))

    # Tightness scaling: overpricing hurts faster in cool/high-cut suburbs than in hot suburbs
    hotness = float(state.get("hotness_score") or 50.0)
    coolness_mult = float(np.clip(1.45 - 0.009 * hotness, 0.75, 1.35))

    spreads = np.linspace(-0.04, 0.08, 13)
    rows = []
    for sp in spreads:
        ask_price = fair_value * (1.0 + sp)

        # Semi-elastic DOM response (calibrated to housing search literature: +1% over ask -> ~8-14% longer DOM)
        dom_mult = np.exp(10.5 * coolness_mult * sp) if sp >= 0 else np.exp(7.0 * sp)
        exp_dom = float(np.clip(base_dom * dom_mult, 6.0, 210.0))

        # Logistic price-cut risk shift with overpricing
        base_logit = np.log(base_cut / (1.0 - base_cut))
        cut_logit = base_logit + 26.0 * coolness_mult * sp
        prob_cut = float(1.0 / (1.0 + np.exp(-cut_logit)))

        # Exponential hazard calibrated to expected median DOM
        lam = np.log(2.0) / exp_dom
        p_sell_30d = float(1.0 - np.exp(-lam * 30.0))
        p_sell_60d = float(1.0 - np.exp(-lam * 60.0))

        # Sale price realization:
        # - Pricing slightly below FV in hot markets can elicit mild bidding capture above ask
        # - Pricing above FV faces buyer negotiation discount + stale-listing stigma when cut
        bidding_upside = max(0.0, -sp) * (0.45 if hotness >= 60 else 0.20)
        negotiation_haircut = max(0.0, sp) * (0.45 * coolness_mult)
        stale_stigma = 0.015 * max(0.0, (exp_dom - 45.0) / 60.0)
        expected_gross = ask_price * (
            1.0 + bidding_upside - negotiation_haircut - prob_cut * typical_cut_discount - stale_stigma
        )
        carrying_cost = fair_value * monthly_carry_rate * (exp_dom / 30.0)
        expected_net = expected_gross - carrying_cost

        rows.append(
            {
                "zip_code": zip_code,
                "city": state.get("city") or zip_code,
                "ask_spread_pct": round(sp * 100, 1),
                "ask_price": round(ask_price, 0),
                "exp_dom_days": round(exp_dom, 1),
                "p_sell_30d_pct": round(p_sell_30d * 100, 1),
                "p_sell_60d_pct": round(p_sell_60d * 100, 1),
                "p_price_cut_pct": round(prob_cut * 100, 1),
                "carrying_cost": round(carrying_cost, 0),
                "expected_net": round(expected_net, 0),
            }
        )
    return pl.DataFrame(rows)


def demo_frontiers() -> None:
    """Compare optimal seller pricing in a hot suburb (Roswell 30075) vs. high-cut suburb (Kennesaw 30152)."""
    for zc, label, val in [
        ("30075", "Roswell / North Fulton (Hot Suburb)", 550_000.0),
        ("30152", "Kennesaw / Cobb (High Price-Cut Suburb)", 450_000.0),
    ]:
        res = simulate_ask_frontier(zc, val)
        best_idx = int(res["expected_net"].arg_max())
        tbl = Table(title=f"Seller Pricing Frontier — {label} (Fair Value ${val:,.0f})")
        for col in [
            "Ask vs FV",
            "Ask Price",
            "Exp DOM",
            "P(Sell 30d)",
            "P(Cut)",
            "Carry Cost",
            "Exp Net Proceeds",
        ]:
            tbl.add_column(col, justify="right")

        for i, row in enumerate(res.iter_rows(named=True)):
            style = "bold green" if i == best_idx else ""
            tbl.add_row(
                f"{row['ask_spread_pct']:+.1f}%",
                f"${row['ask_price']:,.0f}",
                f"{row['exp_dom_days']:.1f}d",
                f"{row['p_sell_30d_pct']:.0f}%",
                f"{row['p_price_cut_pct']:.0f}%",
                f"${row['carrying_cost']:,.0f}",
                f"${row['expected_net']:,.0f}",
                style=style,
            )
        console.print(tbl)


if __name__ == "__main__":
    demo_frontiers()
