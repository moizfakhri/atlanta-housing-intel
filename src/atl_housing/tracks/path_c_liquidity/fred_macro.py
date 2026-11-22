"""Path C Macro Ingestion: FRED 30Y/15Y mortgage rates & Metro Atlanta labor/HPI series."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import polars as pl
from dotenv import load_dotenv
from rich.console import Console

from atl_housing.config import BRONZE_DIR

console = Console()
load_dotenv()

FRED_SERIES: dict[str, str] = {
    "mortgage_30y": "MORTGAGE30US",
    "mortgage_15y": "MORTGAGE15US",
    "fed_funds": "FEDFUNDS",
    "atl_unemp_rate": "ATLA013URN",
    "atl_nonfarm_emp": "ATLA013NA",
    "atl_case_shiller": "ATXRSA",
}


def ingest_fred_macro(start_date: str = "2015-01-01") -> Path:
    """Fetch FRED mortgage & Atlanta macro series and align to monthly frequency."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found in .env. Add FRED_API_KEY=... to local .env first."
        )

    out_dir = BRONZE_DIR / "path_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fred_atlanta_macro.parquet"

    monthly_frames: list[pl.DataFrame] = []
    with httpx.Client(timeout=25.0) as client:
        for col_name, series_id in FRED_SERIES.items():
            r = client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "observation_start": start_date,
                },
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            df = (
                pl.DataFrame(
                    {
                        "date": [o["date"] for o in obs],
                        "val": [o["value"] for o in obs],
                    }
                )
                .with_columns(
                    pl.col("date").str.to_date("%Y-%m-%d").dt.truncate("1mo").alias("month_date"),
                    pl.when(pl.col("val") == ".")
                    .then(None)
                    .otherwise(pl.col("val"))
                    .cast(pl.Float64, strict=False)
                    .alias(col_name),
                )
                .group_by("month_date")
                .agg(pl.col(col_name).mean())
                .sort("month_date")
            )
            monthly_frames.append(df)
            console.print(f"  FRED {series_id:<12s} ({col_name}): {df.height} months")

    merged = monthly_frames[0]
    for frame in monthly_frames[1:]:
        merged = merged.join(frame, on="month_date", how="full", coalesce=True)
    merged = merged.sort("month_date")
    merged.write_parquet(out_path)
    console.print(f"[green]Saved FRED Atlanta macro panel:[/green] {merged.height} months -> {out_path}")
    return out_path


if __name__ == "__main__":
    ingest_fred_macro()
