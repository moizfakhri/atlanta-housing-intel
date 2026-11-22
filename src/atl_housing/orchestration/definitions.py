"""Dagster Definitions: End-to-End Metro Atlanta Lakehouse, dbt Gold Marts & PyTorch GNN Assets."""

import subprocess
from pathlib import Path

from dagster import AssetExecutionContext, Definitions, MaterializeResult, asset

from atl_housing.config import PROJECT_ROOT
from atl_housing.tracks.path_a_valuation.ingest import (
    ingest_county_parcels_sample,
    ingest_overture_3d_buildings,
)
from atl_housing.tracks.path_a_valuation.spatial_gnn import train_and_benchmark_gnn
from atl_housing.tracks.path_c_liquidity.fred_macro import ingest_fred_macro
from atl_housing.tracks.path_c_liquidity.ingest import (
    ingest_hmda_atlanta,
    ingest_irs_migration,
    ingest_realtor_atlanta,
    ingest_zillow_atlanta,
)
from atl_housing.tracks.path_c_liquidity.silver_panel import run_all_path_c_silver
from atl_housing.tracks.path_e_catalysts.ingest import ingest_catalyst_layers


@asset(group_name="bronze_path_c")
def bronze_zillow_atlanta(context: AssetExecutionContext) -> MaterializeResult:
    out = ingest_zillow_atlanta()
    return MaterializeResult(metadata={"path": str(out)})


@asset(group_name="bronze_path_c", deps=[bronze_zillow_atlanta])
def bronze_realtor_atlanta(context: AssetExecutionContext) -> MaterializeResult:
    outs = ingest_realtor_atlanta()
    return MaterializeResult(metadata={k: str(v) for k, v in outs.items()})


@asset(group_name="bronze_path_c")
def bronze_hmda_atlanta(context: AssetExecutionContext) -> MaterializeResult:
    out = ingest_hmda_atlanta()
    return MaterializeResult(metadata={"path": str(out)})


@asset(group_name="bronze_path_c")
def bronze_fred_macro(context: AssetExecutionContext) -> MaterializeResult:
    out = ingest_fred_macro()
    return MaterializeResult(metadata={"path": str(out)})


@asset(group_name="bronze_path_c")
def bronze_irs_migration(context: AssetExecutionContext) -> MaterializeResult:
    out = ingest_irs_migration()
    return MaterializeResult(metadata={"path": str(out)})


@asset(
    group_name="silver_path_c",
    deps=[bronze_zillow_atlanta, bronze_realtor_atlanta, bronze_hmda_atlanta, bronze_fred_macro],
)
def silver_atlanta_liquidity_and_credit(context: AssetExecutionContext) -> MaterializeResult:
    run_all_path_c_silver()
    return MaterializeResult(metadata={"status": "materialized"})


@asset(
    group_name="gold_dbt",
    deps=[silver_atlanta_liquidity_and_credit],
)
def dbt_gold_submarket_mart(context: AssetExecutionContext) -> MaterializeResult:
    """Run dbt build (models + automated schema & anti-leakage tests) against DuckDB."""
    dbt_dir = PROJECT_ROOT / "dbt"
    res = subprocess.run(
        ["dbt", "build", "--profiles-dir", str(dbt_dir)],
        cwd=str(dbt_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    context.log.info(res.stdout[-1500:])
    return MaterializeResult(metadata={"dbt_dir": str(dbt_dir), "exit_code": res.returncode})


@asset(group_name="bronze_path_a")
def bronze_fulton_parcels(context: AssetExecutionContext) -> MaterializeResult:
    out = ingest_county_parcels_sample("fulton", max_records=4000)
    return MaterializeResult(metadata={"path": str(out)})


@asset(group_name="bronze_path_a")
def bronze_overture_3d_sandy_springs(context: AssetExecutionContext) -> MaterializeResult:
    out = ingest_overture_3d_buildings(bbox=(-84.45, 33.87, -84.34, 33.91), tag="sandy_springs")
    return MaterializeResult(metadata={"path": str(out)})


@asset(
    group_name="gold_ml_path_a",
    deps=[bronze_fulton_parcels, bronze_overture_3d_sandy_springs],
)
def gold_path_a_gnn_3d_predictions(context: AssetExecutionContext) -> MaterializeResult:
    out: Path = train_and_benchmark_gnn()
    return MaterializeResult(metadata={"gold_parquet": str(out)})


@asset(group_name="bronze_path_e")
def bronze_atlanta_catalysts(context: AssetExecutionContext) -> MaterializeResult:
    outs = ingest_catalyst_layers()
    return MaterializeResult(metadata={k: str(v) for k, v in outs.items()})


defs = Definitions(
    assets=[
        bronze_zillow_atlanta,
        bronze_realtor_atlanta,
        bronze_hmda_atlanta,
        bronze_fred_macro,
        bronze_irs_migration,
        silver_atlanta_liquidity_and_credit,
        dbt_gold_submarket_mart,
        bronze_fulton_parcels,
        bronze_overture_3d_sandy_springs,
        gold_path_a_gnn_3d_predictions,
        bronze_atlanta_catalysts,
    ]
)
