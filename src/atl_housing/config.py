"""Canonical configuration for Metro Atlanta suburban real estate intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DUCKDB_PATH = DATA_DIR / "atlanta_housing.duckdb"


@dataclass(frozen=True)
class CountySpec:
    name: str
    fips: str
    suburban_role: str


# Core 5-county Metro Atlanta suburban spine (plus option to expand to outer exurbs)
METRO_ATLANTA_COUNTIES: tuple[CountySpec, ...] = (
    CountySpec("Fulton", "13121", "Core + North Fulton affluent suburbs (Alpharetta, Milton, Roswell, Sandy Springs) & South Fulton"),
    CountySpec("DeKalb", "13089", "Inner-ring east suburbs (Decatur, Brookhaven, Dunwoody, Tucker, Chamblee)"),
    CountySpec("Cobb", "13067", "Northwest suburbs & school-district cliffs (East Cobb/Marietta, Smyrna, Kennesaw)"),
    CountySpec("Gwinnett", "13135", "High-growth diverse northeast suburbs (Suwanee, Duluth, Lawrenceville, Buford)"),
    CountySpec("Forsyth", "13117", "Fast-growing outer north exurb (Cumming, Lambert HS corridor, GA-400 growth)"),
)

COUNTY_FIPS_LIST: list[str] = [c.fips for c in METRO_ATLANTA_COUNTIES]
COUNTY_FIPS_CSV: str = ",".join(COUNTY_FIPS_LIST)

# Bounding box covering the 5-county Metro Atlanta suburban spine (lon_min, lat_min, lon_max, lat_max)
METRO_ATLANTA_BBOX: tuple[float, float, float, float] = (-84.85, 33.48, -83.80, 34.34)

# Focused test bbox around Alpharetta / Suwanee / Johns Creek corridor for rapid queries
TEST_SUBURB_BBOX: tuple[float, float, float, float] = (-84.32, 34.03, -84.15, 34.12)

# Verified Realtor.com Economic Research public S3 endpoints (monthly historical panels)
REALTOR_URLS = {
    "zip_core": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv",
    "zip_hotness": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Hotness/RDC_Inventory_Hotness_Metrics_Zip_History.csv",
    "county_core": "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_County_History.csv",
}

# Verified Metro Atlanta public ArcGIS Parcel FeatureServers (zero-auth REST APIs)
COUNTY_PARCEL_ENDPOINTS = {
    "fulton": "https://services5.arcgis.com/buITjRsK0rZsAXbQ/arcgis/rest/services/CurrentParcels/FeatureServer/0",
    "dekalb": "https://services2.arcgis.com/IxVN2oUE9EYLSnPE/arcgis/rest/services/Tax_Parcels_2025/FeatureServer/0",
    "gwinnett": "https://services3.arcgis.com/RfpmnkSAQleRbndX/arcgis/rest/services/Property_and_Tax/FeatureServer/0",
    "cobb": "https://services2.arcgis.com/hE9igMm8RoNeQRbh/arcgis/rest/services/Cobb_Parcels/FeatureServer/0",
}

# Redfin Data Center S3 endpoints (note: Redfin occasionally restricts direct S3 via WAF; UI export supported)
REDFIN_URLS = {
    "zip": "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz",
    "neighborhood": "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/neighborhood_market_tracker.tsv000.gz",
    "county": "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/county_market_tracker.tsv000.gz",
}

# CFPB FFIEC HMDA Data Browser API (free loan-level mortgage microdata)
HMDA_API_BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

# Default H3 resolution for neighborhood microstructure (res 8 ≈ 0.74 km² / ~180 acres; res 9 ≈ 0.11 km²)
DEFAULT_H3_RES = 8
