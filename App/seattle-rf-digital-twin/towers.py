"""Load real Seattle towers from Unity Catalog and turn them into Sionna transmitters.

Source table: ``cmegdemos_catalog.network_analytics_enablement.cell_towers``
Columns used: ``tower_id, carrier, tower_type, coverage_radius_m, latitude, longitude``.

Two responsibilities:
  1. ``load_towers(neighborhood, spark)`` — read the bbox subset, project lat/lon to the
     neighborhood's local-ENU meters (see ``neighborhoods.project_lonlat``), and emit one
     cell dict per tower in the schema the Sionna pipeline expects.
  2. ``randomize_config(tower, rng)`` — assign a plausible, **deterministic** radio config
     per tower (carrier frequency keyed off ``tower_type``, TX UPA size, power, antenna
     height). Determinism (seeded on ``tower_id``) keeps the ``config_hash`` stable so the
     Lakebase cache stays warm across re-runs.

The per-tower dict extends the original 7-cell schema with ``frequency_hz``,
``num_rows_tx`` and ``num_cols_tx`` because, unlike the homogeneous etoile demo, real
towers are heterogeneous (a 5G-NR small cell next to an LTE macro). The render pipeline
groups towers by array geometry so a single ``PlanarArray`` can serve each pass; the
curated "stories" instead hold the array constant and vary one knob (see ``defaults.py``).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

import neighborhoods as nb

SOURCE_TABLE = "cmegdemos_catalog.network_analytics_enablement.cell_towers"

# The source table carries a `location geometry(4326)` column whose Delta "geospatial"
# reader feature isn't supported by the Sionna-validated GPU runtime (DBR 16.4). A
# serverless SQL warehouse *does* support it, so we read the (non-geometry) columns we need
# via the SQL Statement Execution API instead of Spark.
SQL_WAREHOUSE_ENV = "SEATTLE_SQL_WAREHOUSE_ID"

# Frequency menus (Hz) per tower technology. ITU materials in Sionna are only defined
# ≥1 GHz, so even GSM is bumped to LTE band 3 (1.8 GHz) — same caveat the etoile demo
# documents for its frequency-ladder story.
_BAND_HZ: Dict[str, List[float]] = {
    "GSM": [1.8e9],
    "UMTS": [2.1e9],
    "LTE": [1.8e9, 2.6e9, 3.5e9],
    "NR": [3.5e9, 28e9],
}
_DEFAULT_BANDS = [1.8e9, 2.6e9, 3.5e9]

# Antenna height (m) range per technology — small cells sit lower than macros.
_HEIGHT_M: Dict[str, tuple] = {
    "GSM": (28.0, 38.0),
    "UMTS": (25.0, 35.0),
    "LTE": (22.0, 35.0),
    "NR": (10.0, 18.0),
}
_DEFAULT_HEIGHT = (20.0, 30.0)

# TX uniform planar array choices (rows, cols).
_TX_ARRAYS = [(2, 2), (4, 4), (8, 2), (8, 8), (16, 16)]


def _rng_for(tower_id: int, seed: int) -> np.random.Generator:
    """A deterministic RNG seeded per tower so configs are reproducible."""
    return np.random.default_rng((int(tower_id) * 1_000_003) ^ int(seed))


def randomize_config(tower: Dict[str, Any], seed: int = 1234) -> Dict[str, Any]:
    """Assign a deterministic random radio config to a tower row.

    ``tower`` must already carry ``tower_id``, ``tower_type`` and projected ``x``/``y``.
    Returns a new dict; does not mutate the input.
    """
    rng = _rng_for(tower["tower_id"], seed)
    ttype = str(tower.get("tower_type", "")).upper()

    freq = float(rng.choice(_BAND_HZ.get(ttype, _DEFAULT_BANDS)))
    rows, cols = _TX_ARRAYS[int(rng.integers(len(_TX_ARRAYS)))]
    z_lo, z_hi = _HEIGHT_M.get(ttype, _DEFAULT_HEIGHT)
    z = float(rng.uniform(z_lo, z_hi))
    power_dbm = float(rng.integers(38, 51))  # 38..50 dBm inclusive

    out = dict(tower)
    out.update(
        z=z,
        power_dbm=power_dbm,
        frequency_hz=freq,
        num_rows_tx=int(rows),
        num_cols_tx=int(cols),
    )
    return out


def _to_cells(rows: List[Dict[str, Any]], hood: nb.Neighborhood, seed: int) -> List[Dict[str, Any]]:
    """Project + randomize a list of raw tower rows into Sionna cell dicts."""
    cells: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        x, y = nb.project_lonlat(float(r["latitude"]), float(r["longitude"]), hood.origin)
        base = {
            "cell_id": i,
            "tower_id": int(r["tower_id"]),
            "name": f"tx{i}_{r.get('tower_type', 'NA')}",
            "x": float(x),
            "y": float(y),
            "tower_type": r.get("tower_type"),
            "coverage_radius_m": int(r.get("coverage_radius_m") or 0),
            # Antennas tilt toward the neighborhood center (origin → local 0,0).
            "look_at_x": 0.0,
            "look_at_y": 0.0,
            "look_at_z": 0.0,
        }
        cells.append(randomize_config(base, seed=seed))
    return cells


def _resolve_warehouse_id(w, warehouse_id: Optional[str]) -> str:
    """Pick a SQL warehouse: explicit arg → env → first serverless (prefer RUNNING)."""
    wid = warehouse_id or os.environ.get(SQL_WAREHOUSE_ENV)
    if wid:
        return wid
    serverless, running = [], []
    for wh in w.warehouses.list():
        if str(getattr(wh, "warehouse_type", "")).upper().endswith("PRO") or \
           getattr(wh, "enable_serverless_compute", False):
            serverless.append(wh)
            if str(getattr(wh, "state", "")).upper() == "RUNNING":
                running.append(wh)
    pick = (running or serverless)
    if not pick:
        raise RuntimeError("No SQL warehouse available to read the geospatial tower table; "
                           f"set {SQL_WAREHOUSE_ENV}.")
    return pick[0].id


def _query_via_warehouse(sql: str, warehouse_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run a read-only query on a SQL warehouse and return list-of-dict rows."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    wid = _resolve_warehouse_id(w, warehouse_id)
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wid, wait_timeout="50s",
    )
    # Poll if the statement is still running past the inline wait window.
    deadline = time.time() + 300
    while str(resp.status.state.value) in ("PENDING", "RUNNING") and time.time() < deadline:
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    state = str(resp.status.state.value)
    if state != "SUCCEEDED":
        err = getattr(resp.status, "error", None)
        raise RuntimeError(f"Tower query {state}: {err}")
    cols = [c.name for c in resp.manifest.schema.columns]
    data = (resp.result.data_array or []) if resp.result else []
    return [dict(zip(cols, row)) for row in data]


def load_towers(
    neighborhood: str,
    spark=None,
    seed: int = 1234,
    limit: Optional[int] = None,
    warehouse_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read towers inside ``neighborhood`` from UC and return Sionna cell dicts.

    Reads via a serverless SQL warehouse (the geometry column blocks Spark on DBR 16.4).
    ``spark`` is accepted for call-site compatibility but unused. ``limit`` caps the tower
    count (useful for a quick calibration render).
    """
    hood = nb.get(neighborhood)
    sql = (
        f"SELECT tower_id, carrier, tower_type, coverage_radius_m, latitude, longitude "
        f"FROM {SOURCE_TABLE} WHERE {hood.sql_bbox_filter()} ORDER BY tower_id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = _query_via_warehouse(sql, warehouse_id)
    # Warehouse returns all values as strings — coerce the numerics _to_cells relies on.
    for r in rows:
        r["tower_id"] = int(r["tower_id"])
        r["latitude"] = float(r["latitude"])
        r["longitude"] = float(r["longitude"])
        r["coverage_radius_m"] = int(r["coverage_radius_m"]) if r.get("coverage_radius_m") else 0
    return _to_cells(rows, hood, seed)
