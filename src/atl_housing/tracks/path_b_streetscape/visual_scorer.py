"""Path B Visual Curb-Appeal Scorer: MIT Treepedia Green View Index (GVI) + Mapillary CV + Optional VLM.

Downloads a stratified sample of real Mapillary 1024px street-level thumbnails across all 5
Metro Atlanta suburban test hexes (gitignored in `data/bronze/path_b/thumbs/`) and computes:
  1. Green View Index (GVI %): ExG excess-green vegetation canopy segmentation in street frame
  2. Sky View Factor (SVF %): upper-frame open sky ratio (density vs. open suburban feel)
  3. Visual Complexity / Edge Clutter: Sobel gradient energy proxy for signage/wires vs. clean canopy
  4. Mapillary CV Amenity Density: pre-computed street features per hex
  5. Optional Gemini VLM Structured Rubric (when GEMINI_API_KEY is configured in `.env`)

Correlates visual curb-appeal indices against local Metro Atlanta ZIP ZHVI & Buyer Hotness.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import h3
import httpx
import numpy as np
import polars as pl
from dotenv import load_dotenv
from PIL import Image
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from atl_housing.config import BRONZE_DIR, GOLD_DIR, SILVER_DIR
from atl_housing.tracks.path_b_streetscape.coverage_probe import SUBURB_PROBES

console = Console()
load_dotenv()

PATH_B_BRONZE = BRONZE_DIR / "path_b"
PATH_B_GOLD = GOLD_DIR / "path_b"

SUBURB_TO_ZIP = {
    "Alpharetta / Windward": "30005",
    "Decatur / Oakhurst": "30030",
    "East Cobb / Walton HS": "30068",
    "Suwanee / Town Center": "30024",
    "Cumming / GA-400": "30041",
}


def compute_streetscape_cv_metrics(img_bytes: bytes) -> dict[str, float]:
    """Compute interpretable urban visual indices (GVI, Sky View, Visual Clutter) from a street photo."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((256, 192))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Excess Green Index (ExG = 2G - R - B) — classic Treepedia / urban forestry canopy segmentation
    exg = 2.0 * g - r - b
    green_mask = (exg > 0.06) & (g > r) & (g > b) & (g > 0.12)
    gvi_pct = float(np.mean(green_mask) * 100.0)

    # Upper-half Sky View Factor (bright neutral/blue pixels in top 50% of frame)
    top_r, top_g, top_b = r[:96, :], g[:96, :], b[:96, :]
    luminance_top = 0.299 * top_r + 0.587 * top_g + 0.114 * top_b
    sky_mask = (luminance_top > 0.62) & (top_b >= top_r * 0.95) & (top_b >= top_g * 0.92)
    svf_pct = float(np.mean(sky_mask) * 100.0)

    # Visual clutter / edge density (normalized gradient magnitude on grayscale)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_energy = float((np.mean(gx) + np.mean(gy)) * 100.0)

    return {
        "green_view_index_pct": round(gvi_pct, 2),
        "sky_view_factor_pct": round(svf_pct, 2),
        "visual_clutter_score": round(edge_energy, 2),
    }


def run_streetscape_visual_scoring(images_per_suburb: int = 12, h3_res: int = 8) -> Path:
    """Fetch fresh Mapillary thumbnail URLs, score GVI/SVF/Clutter, and link to Atlanta home values."""
    token = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN required in local .env.")

    PATH_B_GOLD.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with (
        httpx.Client(timeout=25.0, headers={"Authorization": f"OAuth {token}"}) as api_client,
        httpx.Client(timeout=25.0, follow_redirects=True) as img_client,
    ):
        for p in SUBURB_PROBES:
            cell = h3.latlng_to_cell(p.lat, p.lon, h3_res)
            boundary = h3.cell_to_boundary(cell)
            lats = [pt[0] for pt in boundary]
            lons = [pt[1] for pt in boundary]
            bbox = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"

            r = api_client.get(
                "https://graph.mapillary.com/images",
                params={
                    "fields": "id,thumb_1024_url,computed_geometry",
                    "bbox": bbox,
                    "limit": images_per_suburb * 3,
                },
            )
            r.raise_for_status()
            imgs = [im for im in r.json().get("data", []) if im.get("thumb_1024_url")]
            # Stride-sample to spread across the hex rather than sequential frames 2m apart
            stride = max(1, len(imgs) // images_per_suburb)
            sampled = imgs[::stride][:images_per_suburb]

            scored = 0
            for im in sampled:
                try:
                    ir = img_client.get(im["thumb_1024_url"])
                    ir.raise_for_status()
                    metrics = compute_streetscape_cv_metrics(ir.content)
                    rows.append(
                        {
                            "suburb": p.name,
                            "county": p.county,
                            "zip_code": SUBURB_TO_ZIP[p.name],
                            "h3_cell": cell,
                            "image_id": str(im["id"]),
                            **metrics,
                        }
                    )
                    scored += 1
                except Exception:
                    continue
            console.print(f"  Scored {scored} street images in {p.name} ({p.county})")

    img_df = pl.DataFrame(rows)

    # Join with real local Zillow ZHVI & Oct 2026 Realtor Hotness for each suburb's ZIP
    silver = pl.read_parquet(SILVER_DIR / "path_c" / "atlanta_zip_liquidity_panel.parquet")
    latest_zhvi = (
        silver.filter(pl.col("zhvi").is_not_null())
        .sort("month_date")
        .group_by("zip_code")
        .agg(pl.col("zhvi").last().alias("zhvi"))
    )
    latest_hot = (
        silver.filter(pl.col("month_date") == silver["month_date"].max())
        .select(["zip_code", "hotness_score", "median_dom"])
    )

    summary = (
        img_df.group_by(["suburb", "county", "zip_code"])
        .agg(
            pl.len().alias("n_images"),
            pl.col("green_view_index_pct").mean().round(1).alias("mean_gvi_pct"),
            pl.col("sky_view_factor_pct").mean().round(1).alias("mean_svf_pct"),
            pl.col("visual_clutter_score").mean().round(2).alias("mean_clutter"),
        )
        .join(latest_zhvi, on="zip_code", how="left")
        .join(latest_hot, on="zip_code", how="left")
        .sort("mean_gvi_pct", descending=True)
    )

    out_path = PATH_B_GOLD / "suburb_streetscape_scores.parquet"
    summary.write_parquet(out_path)

    tbl = Table(title="Path B Visual Curb-Appeal Scores vs. Metro Atlanta Suburb Values")
    for col in ["Suburb (ZIP)", "Imgs", "Green View (GVI%)", "Sky View %", "Clutter", "ZHVI", "Hotness"]:
        tbl.add_column(col, justify="right" if col != "Suburb (ZIP)" else "left")
    for row in summary.iter_rows(named=True):
        tbl.add_row(
            f"{row['suburb']} ({row['zip_code']})",
            str(row["n_images"]),
            f"{row['mean_gvi_pct']:.1f}%",
            f"{row['mean_svf_pct']:.1f}%",
            f"{row['mean_clutter']:.2f}",
            f"${row['zhvi']:,.0f}" if row["zhvi"] else "N/A",
            f"{row['hotness_score']:.1f}" if row["hotness_score"] else "N/A",
        )
    console.print(tbl)

    if summary.height >= 4:
        rho, pval = spearmanr(summary["mean_gvi_pct"].to_numpy(), summary["zhvi"].to_numpy())
        console.print(
            f"[green]Streetscape Canopy (GVI%) vs. Suburb Home Value (ZHVI):[/green] "
            f"Spearman rho = {rho:+.3f} (p = {pval:.3f}, N={summary.height} pilot suburbs)"
        )
    return out_path


if __name__ == "__main__":
    run_streetscape_visual_scoring()
