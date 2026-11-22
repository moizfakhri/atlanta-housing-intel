"""Integration tests for FastAPI 3D Twin endpoints and Dagster asset definitions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from atl_housing.api.app import app
from atl_housing.orchestration.definitions import defs

client = TestClient(app)


def test_dagster_definitions_load_all_assets() -> None:
    asset_keys = {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()}
    expected = {
        "bronze_zillow_atlanta",
        "bronze_realtor_atlanta",
        "bronze_hmda_atlanta",
        "bronze_fred_macro",
        "silver_atlanta_liquidity_and_credit",
        "dbt_gold_submarket_mart",
        "gold_path_a_gnn_3d_predictions",
    }
    assert expected.issubset(asset_keys)


def test_fastapi_serves_index_and_health() -> None:
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "Metro Atlanta 3D Housing Twin" in r_home.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"


def test_fastapi_3d_and_simulator_endpoints() -> None:
    r_parcels = client.get("/api/path-a/parcels-3d?limit=100")
    assert r_parcels.status_code == 200
    data_p = r_parcels.json()
    assert data_p["count"] == 100
    assert len(data_p["parcels"][0]["comps"]) == 5

    r_zips = client.get("/api/path-c/zip-liquidity")
    assert r_zips.status_code == 200
    assert r_zips.json()["count"] >= 150

    r_arcs = client.get("/api/path-c/migration-arcs")
    assert r_arcs.status_code == 200
    assert r_arcs.json()["count"] >= 10

    r_sim = client.get("/api/path-c/simulate-pricing?zip_code=30075&fair_value=550000")
    assert r_sim.status_code == 200
    assert len(r_sim.json()["frontier"]) == 13
