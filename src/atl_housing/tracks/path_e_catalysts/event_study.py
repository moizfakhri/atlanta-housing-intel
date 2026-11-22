"""Path E Event Study: Johns Creek Rezoning Catalysts vs. Suburban Price & Demand Lead-Lag.

Tests whether spikes in Johns Creek zoning/rezoning filings lead 6M-12M home-price
appreciation and buyer-demand hotness relative to matched North Atlanta suburban controls.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr

from atl_housing.config import BRONZE_DIR, SILVER_DIR

console = Console()

# Treated ZIPs covering Johns Creek vs. matched affluent North Atlanta suburban controls
TREATED_ZIPS = ["30022", "30097", "30005"]
CONTROL_ZIPS = ["30075", "30076", "30024", "30062", "30068"]


def run_rezoning_event_study() -> dict[str, object]:
    """Inspect Johns Creek rezoning temporal profile and measure lead-lag vs. control suburbs."""
    jc_pq = BRONZE_DIR / "path_e" / "johns_creek_rezoning.parquet"
    silver_pq = SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet"

    gdf = gpd.read_parquet(jc_pq)
    date_candidates = [c for c in gdf.columns if any(k in c.lower() for k in ["date", "year", "hearing", "created"])]
    console.print(f"[cyan]Johns Creek Rezoning ({len(gdf):,} cases):[/cyan] date/time cols = {date_candidates}")

    # Parse best available date column (typically CaseID prefix RZ-YYYY or timestamp field)
    years: list[int | None] = []
    if "CaseID" in gdf.columns:
        for cid in gdf["CaseID"].astype(str):
            yr = None
            for token in cid.replace("_", "-").split("-"):
                if len(token) == 4 and token.isdigit() and 2000 <= int(token) <= 2026:
                    yr = int(token)
                    break
            years.append(yr)
    else:
        years = [None] * len(gdf)

    valid_yrs = [y for y in years if y is not None]
    if not valid_yrs:
        # Fallback: inspect epoch ms timestamp columns
        for col in date_candidates:
            vals = gdf[col].dropna()
            if len(vals) > 50 and vals.dtype.kind in ("i", "f"):
                dt = gpd.pd.to_datetime(vals, unit="ms", errors="coerce").dropna()
                if len(dt) > 50:
                    years = (
                        gpd.pd.to_datetime(gdf[col], unit="ms", errors="coerce").dt.year.tolist()
                    )
                    valid_yrs = [int(y) for y in years if gpd.pd.notna(y) and 2000 <= y <= 2026]
                    break

    rez_df = pl.DataFrame({"year": [int(y) for y in valid_yrs if y is not None]})
    annual_rez = rez_df.group_by("year").len().rename({"len": "rezoning_cases"}).sort("year")

    # Join to annual Johns Creek vs. Control Suburbs ZHVI appreciation & buyer hotness
    panel = pl.read_parquet(silver_pq).with_columns(pl.col("month_date").dt.year().alias("year"))
    treated_ann = (
        panel.filter(pl.col("zip_code").is_in(TREATED_ZIPS))
        .group_by("year")
        .agg(
            pl.col("zhvi").mean().alias("jc_zhvi"),
            pl.col("hotness_score").mean().alias("jc_hotness"),
            pl.col("median_dom").mean().alias("jc_dom"),
        )
        .sort("year")
        .with_columns((pl.col("jc_zhvi") / pl.col("jc_zhvi").shift(1) - 1.0).alias("jc_yoy_ret"))
    )
    ctrl_ann = (
        panel.filter(pl.col("zip_code").is_in(CONTROL_ZIPS))
        .group_by("year")
        .agg(
            pl.col("zhvi").mean().alias("ctrl_zhvi"),
            pl.col("hotness_score").mean().alias("ctrl_hotness"),
            pl.col("median_dom").mean().alias("ctrl_dom"),
        )
        .sort("year")
        .with_columns(
            (pl.col("ctrl_zhvi") / pl.col("ctrl_zhvi").shift(1) - 1.0).alias("ctrl_yoy_ret")
        )
    )

    merged = (
        treated_ann.join(ctrl_ann, on="year", how="inner")
        .join(annual_rez, on="year", how="left")
        .with_columns(
            pl.col("rezoning_cases").fill_null(0),
            (pl.col("jc_yoy_ret") - pl.col("ctrl_yoy_ret")).alias("excess_yoy_ret"),
            (pl.col("jc_hotness") - pl.col("ctrl_hotness")).alias("excess_hotness"),
        )
        .with_columns(
            pl.col("excess_yoy_ret").shift(-1).alias("lead_1y_excess_ret"),
            pl.col("excess_hotness").shift(-1).alias("lead_1y_excess_hotness"),
        )
        .filter((pl.col("year") >= 2017) & (pl.col("year") <= 2025))
    )

    tbl = Table(title="Path E Event Study: Johns Creek Rezoning Volume vs. Excess Return & Hotness")
    for col in ["Year", "Rezoning Cases", "JC YoY Ret", "Control YoY", "Excess Ret", "Next-Yr Excess"]:
        tbl.add_column(col, justify="right")
    for row in merged.iter_rows(named=True):
        tbl.add_row(
            str(row["year"]),
            str(row["rezoning_cases"]),
            f"{row['jc_yoy_ret'] * 100:+.1f}%" if row["jc_yoy_ret"] is not None else "-",
            f"{row['ctrl_yoy_ret'] * 100:+.1f}%" if row["ctrl_yoy_ret"] is not None else "-",
            f"{row['excess_yoy_ret'] * 100:+.2f}%" if row["excess_yoy_ret"] is not None else "-",
            f"{row['lead_1y_excess_ret'] * 100:+.2f}%"
            if row["lead_1y_excess_ret"] is not None
            else "-",
        )
    console.print(tbl)

    valid_corr = merged.filter(
        pl.col("rezoning_cases").is_not_null() & pl.col("lead_1y_excess_ret").is_not_null()
    )
    corr_val = float("nan")
    p_val = float("nan")
    if valid_corr.height >= 4 and np.std(valid_corr["rezoning_cases"].to_numpy()) > 0:
        r_stat, p_stat = pearsonr(
            valid_corr["rezoning_cases"].to_numpy(),
            valid_corr["lead_1y_excess_ret"].to_numpy(),
        )
        corr_val, p_val = float(r_stat), float(p_stat)
        console.print(
            f"[green]Lead-Lag Correlation (Rezoning Cases_t vs. Excess Return_{{t+1}}):[/green] "
            f"r = {corr_val:+.3f} (p = {p_val:.3f}, N={valid_corr.height} yrs)"
        )
    else:
        console.print("[yellow]Insufficient annual variation in rezoning date timestamps for lead-lag r.[/yellow]")

    return {
        "n_cases_total": len(gdf),
        "n_cases_dated": len(valid_yrs),
        "lead_1y_r": corr_val,
        "lead_1y_p": p_val,
    }


if __name__ == "__main__":
    run_rezoning_event_study()
