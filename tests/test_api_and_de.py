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
    assert "Metro Atlanta Housing Intelligence Atlas" in r_home.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_counties = client.get("/api/boundaries/counties")
    assert r_counties.status_code == 200
    assert len(r_counties.json()["features"]) == 5

    r_zips_geo = client.get("/api/path-c/zips-geojson")
    assert r_zips_geo.status_code == 200
    assert len(r_zips_geo.json()["features"]) >= 180


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

    r_b3d = client.get("/api/path-a/buildings-3d")
    assert r_b3d.status_code == 200
    assert r_b3d.json()["type"] == "FeatureCollection"
    assert len(r_b3d.json()["features"]) >= 100

    r_lodes = client.get("/api/path-c/commute-arcs")
    assert r_lodes.status_code == 200
    assert r_lodes.json()["count"] >= 10

    r_sql = client.get("/api/spatial-query?q=highest+price+cuts+buyer+leverage")
    assert r_sql.status_code == 200
    assert len(r_sql.json()["rows"]) >= 1
    assert "SELECT" in r_sql.json()["sql"].upper()

    r_ts = client.get("/api/path-c/metro-timeseries")
    assert r_ts.status_code == 200
    assert len(r_ts.json()["series"]) >= 100

    r_zh = client.get("/api/path-c/zip-history/30075")
    assert r_zh.status_code == 200
    assert len(r_zh.json()["history"]) >= 100

    r_an = client.get("/api/analytics/summary")
    assert r_an.status_code == 200
    assert len(r_an.json()["hmda_2024"]) == 5


def test_viewport_3d_streaming_and_zero_dangling_comps() -> None:
    # 1. Featured Sandy Springs home loads cleanly with clean 5-digit ZIP and 5 distinct comps
    r_feat = client.get("/api/path-a/featured-home")
    assert r_feat.status_code == 200
    f_props = r_feat.json()["properties"]
    assert f_props["has_gnn"] is True
    assert "." not in f_props["zip"]
    comps = f_props["comps"]
    assert len(comps) == 5
    comp_oids = [c["comp_overture_id"] for c in comps]
    assert len(set(comp_oids)) == 5, "Duplicate comp buildings detected!"
    assert f_props["overture_id"] not in comp_oids

    # 2. Viewport streaming in Sandy Springs ALWAYS includes selected home + all 5 comp buildings
    lon, lat = f_props["lon"], f_props["lat"]
    r_vp = client.get(
        f"/api/path-a/buildings-viewport?west={lon - 0.004}&south={lat - 0.003}"
        f"&east={lon + 0.004}&north={lat + 0.003}&zoom=16.0&selected_oid={f_props['overture_id']}"
    )
    assert r_vp.status_code == 200
    vp_data = r_vp.json()
    vp_oids = {feat["properties"]["overture_id"] for feat in vp_data["features"]}
    assert f_props["overture_id"] in vp_oids
    for coid in comp_oids:
        assert coid in vp_oids, f"Dangling comp {coid} missing from viewport response!"

    # 3. Viewport streaming outside Sandy Springs (e.g. Alpharetta & Decatur) returns real 3D buildings!
    r_alpha = client.get(
        "/api/path-a/buildings-viewport?west=-84.300&south=34.070&east=-84.288&north=34.080&zoom=15.5"
    )
    assert r_alpha.status_code == 200
    assert len(r_alpha.json()["features"]) >= 50

    # 4. Zoomed-out request (<13.0) returns instantly with 0 features (LOD protection)
    r_far = client.get(
        "/api/path-a/buildings-viewport?west=-84.6&south=33.7&east=-84.0&north=34.1&zoom=10.0"
    )
    assert r_far.status_code == 200
    assert r_far.json()["meta"]["too_far_out"] is True
    assert len(r_far.json()["features"]) == 0
