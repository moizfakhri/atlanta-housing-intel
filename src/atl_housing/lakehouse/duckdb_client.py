"""DuckDB lakehouse connection helper with Spatial, HTTPFS, and H3 extensions."""

from __future__ import annotations

from pathlib import Path

import duckdb

from atl_housing.config import DUCKDB_PATH


def get_duckdb(
    db_path: Path | str | None = DUCKDB_PATH,
    *,
    read_only: bool = False,
    load_spatial: bool = True,
    load_httpfs: bool = False,
    load_h3: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for GeoParquet and spatial analytics."""
    target = ":memory:" if db_path is None else str(db_path)
    if db_path is not None and not read_only:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(target, read_only=read_only)

    if load_spatial:
        con.execute("INSTALL spatial; LOAD spatial;")
    if load_httpfs:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    if load_h3:
        con.execute("INSTALL h3 FROM community; LOAD h3;")

    return con
