"""Path C Survival & Hazard Analysis: Weibull AFT & Cox PH Models for Suburban Liquidity (DOM).

Models Metro Atlanta ZIP Days-on-Market as duration data using `lifelines`,
treating stale slowdown months (rising inventory + >45% price cuts) as partially
right-censored clearing events, and estimating interpretable hazard ratios for
buyer demand, supply tightness, 30Y mortgage rates, and lagged HMDA investor share.
"""

from __future__ import annotations

import duckdb
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter
from rich.console import Console
from rich.table import Table

from atl_housing.config import DUCKDB_PATH

console = Console()


def fit_dom_survival_models() -> dict[str, float]:
    """Fit Cox PH and Weibull AFT duration models on the dbt Gold Metro Atlanta mart."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df = con.sql(
        """
        SELECT
            zip_code,
            month_date,
            median_dom,
            price_cut_share,
            demand_score,
            supply_score,
            pending_ratio,
            mortgage_30y,
            hmda_investor_share_lag1y,
            hmda_denial_rate_lag1y
        FROM main.mart_atlanta_submarket_signals
        WHERE median_dom IS NOT NULL
          AND median_dom > 3
          AND demand_score IS NOT NULL
          AND supply_score IS NOT NULL
          AND mortgage_30y IS NOT NULL
          AND hmda_investor_share_lag1y IS NOT NULL
        """
    ).df()
    con.close()

    # Standardize covariates so hazard ratios represent 1-SD effect sizes
    covars = [
        "demand_score",
        "supply_score",
        "price_cut_share",
        "mortgage_30y",
        "hmda_investor_share_lag1y",
        "hmda_denial_rate_lag1y",
    ]
    model_df = df[["median_dom", *covars]].dropna().copy()
    for c in covars:
        std = model_df[c].std()
        model_df[c] = (model_df[c] - model_df[c].mean()) / (std if std > 1e-8 else 1.0)

    # Event observed = 1 when market is clearing normally; right-censored (0) during extreme stale overhangs
    model_df["cleared"] = (model_df["median_dom"] < 120.0).astype(int)
    model_df["duration"] = np.clip(model_df["median_dom"].astype(float), 4.0, 120.0)

    fit_cols = ["duration", "cleared", *covars]
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(model_df[fit_cols], duration_col="duration", event_col="cleared")

    waft = WeibullAFTFitter(penalizer=0.05)
    waft.fit(model_df[fit_cols], duration_col="duration", event_col="cleared")

    tbl = Table(
        title=f"Path C Liquidity Survival Analysis (N={len(model_df):,} ATL ZIP-Months · 1-SD Standardized)"
    )
    tbl.add_column("Covariate (1-SD shift)", style="cyan")
    tbl.add_column("Cox Hazard Ratio", justify="right")
    tbl.add_column("Weibull DOM Multiplier", justify="right")
    tbl.add_column("p-value", justify="right")
    tbl.add_column("Economic Interpretation")

    cph_sum = cph.summary
    aft_sum = waft.summary.loc["lambda_"]
    for c in covars:
        hr = float(cph_sum.loc[c, "exp(coef)"])
        pval = float(cph_sum.loc[c, "p"])
        dom_mult = float(np.exp(aft_sum.loc[c, "coef"]))
        direction = (
            f"Faster clearing ({(dom_mult - 1) * 100:+.1f}% DOM)"
            if dom_mult < 1.0
            else f"Slower clearing ({(dom_mult - 1) * 100:+.1f}% DOM)"
        )
        tbl.add_row(c, f"{hr:.3f}x", f"{dom_mult:.3f}x", f"{pval:.2e}", direction)

    console.print(tbl)
    console.print(
        f"[green]Concordance Index (C-index):[/green] Cox PH = {cph.concordance_index_:.3f} | "
        f"Weibull AFT = {waft.concordance_index_:.3f}"
    )
    return {
        "n_obs": float(len(model_df)),
        "cox_c_index": float(cph.concordance_index_),
        "weibull_c_index": float(waft.concordance_index_),
    }


if __name__ == "__main__":
    fit_dom_survival_models()
