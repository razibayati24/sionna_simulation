"""Lakehouse (Unity Catalog Delta + SQL warehouse) access layer.

Functionally equivalent to `lakebase_client.py` in the Lakebase variant
of the app — same `compute_config_hash`, same shape of read/write helpers,
but reads/writes Delta tables via the Databricks SQL Connector instead of
Postgres.

Storage layout, all in `cmegdemos_catalog.sionna_rf_data`:

  scene_configs   (config_hash STRING PK, scene-level params, is_preset)
  cell_configs    (config_hash STRING, cell_id INT, …)
  cached_renders  (config_hash STRING PK, PNG BINARYs, kpis_json STRING, …)
  compute_jobs    (config_hash STRING PK, status, run_id, …)

Tradeoffs vs Lakebase:
  + One platform (UC + Delta + warehouse). No Postgres ops.
  + Queryable by every downstream consumer (BI, notebooks, dashboards).
  + Time-travel + lineage built in.
  − DBSQL queries are ~100–300 ms vs Lakebase's ~10–30 ms — fine for an
    interactive app on cached blobs, slower than Postgres for OLTP.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from typing import Any, Iterable

from databricks import sql
from databricks.sdk.core import Config


CATALOG     = os.environ.get("LAKEHOUSE_CATALOG", "cmegdemos_catalog")
SCHEMA      = os.environ.get("LAKEHOUSE_SCHEMA",  "sionna_rf_data")
FQ          = f"{CATALOG}.{SCHEMA}"

T_SCENE     = f"{FQ}.scene_configs"
T_CELLS     = f"{FQ}.cell_configs"
T_RENDERS   = f"{FQ}.cached_renders"
T_JOBS      = f"{FQ}.compute_jobs"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _warehouse_http_path() -> str:
    """Resolve `/sql/1.0/warehouses/<id>` from the bound resource."""
    wid = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not wid:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is not set. Bind a SQL Warehouse resource "
            "to this Databricks App (see app.yaml)."
        )
    return f"/sql/1.0/warehouses/{wid}"


@contextmanager
def _connect():
    cfg = Config()  # picks up DATABRICKS_HOST / SP creds from the runtime
    host = cfg.host.replace("https://", "").rstrip("/")
    conn = sql.connect(
        server_hostname=host,
        http_path=_warehouse_http_path(),
        credentials_provider=lambda: cfg.authenticate,
    )
    try:
        yield conn
    finally:
        conn.close()


def _row_to_dict(cursor, row) -> dict:
    if row is None:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Hashing — identical to lakebase_client so cache keys stay portable.
# ---------------------------------------------------------------------------

_HASH_SCENE_FIELDS = (
    "num_rows_tx", "num_cols_tx", "num_rows_rx", "num_cols_rx",
    "frequency_hz", "bandwidth_hz", "max_depth", "samples_per_tx",
    "cell_size_x", "cell_size_y", "pattern", "polarization",
    "num_user_samples", "min_sinr_db", "min_user_dist_m", "max_user_dist_m",
)

_HASH_CELL_FIELDS = (
    "cell_id", "x", "y", "z",
    "look_at_x", "look_at_y", "look_at_z", "power_dbm",
)


def compute_config_hash(scene: dict, cells: Iterable[dict]) -> str:
    payload = {
        "scene": {k: scene[k] for k in _HASH_SCENE_FIELDS},
        "cells": [
            {k: c[k] for k in _HASH_CELL_FIELDS}
            for c in sorted(cells, key=lambda c: c["cell_id"])
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_DDL = [
    f"CREATE SCHEMA IF NOT EXISTS {FQ}",
    f"""CREATE TABLE IF NOT EXISTS {T_SCENE} (
        config_hash       STRING NOT NULL,
        name              STRING,
        num_rows_tx       INT,
        num_cols_tx       INT,
        num_rows_rx       INT,
        num_cols_rx       INT,
        frequency_hz      DOUBLE,
        bandwidth_hz      DOUBLE,
        max_depth         INT,
        samples_per_tx    BIGINT,
        cell_size_x       DOUBLE,
        cell_size_y       DOUBLE,
        pattern           STRING,
        polarization      STRING,
        num_user_samples  INT,
        min_sinr_db       DOUBLE,
        min_user_dist_m   DOUBLE,
        max_user_dist_m   DOUBLE,
        is_preset         BOOLEAN,
        created_at        TIMESTAMP
    ) USING DELTA""",
    f"""CREATE TABLE IF NOT EXISTS {T_CELLS} (
        config_hash  STRING NOT NULL,
        cell_id      INT NOT NULL,
        name         STRING,
        x            DOUBLE,
        y            DOUBLE,
        z            DOUBLE,
        look_at_x    DOUBLE,
        look_at_y    DOUBLE,
        look_at_z    DOUBLE,
        power_dbm    DOUBLE
    ) USING DELTA""",
    f"""CREATE TABLE IF NOT EXISTS {T_RENDERS} (
        config_hash       STRING NOT NULL,
        scene_render_png  BINARY,
        sinr_map_png      BINARY,
        association_png   BINARY,
        sinr_cdf_png      BINARY,
        rss_cdf_png       BINARY,
        kpis_json         STRING,
        compute_seconds   DOUBLE,
        created_at        TIMESTAMP
    ) USING DELTA""",
    f"""CREATE TABLE IF NOT EXISTS {T_JOBS} (
        config_hash    STRING NOT NULL,
        status         STRING,
        run_id         BIGINT,
        error_message  STRING,
        submitted_at   TIMESTAMP,
        updated_at     TIMESTAMP
    ) USING DELTA""",
]


def init_schema() -> None:
    with _connect() as conn, conn.cursor() as cur:
        for stmt in SCHEMA_DDL:
            cur.execute(stmt)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_render(config_hash: str) -> dict | None:
    """Return the cached_renders row for this hash, or None."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT config_hash, scene_render_png, sinr_map_png, association_png,
                   sinr_cdf_png, rss_cdf_png, kpis_json, compute_seconds, created_at
            FROM {T_RENDERS}
            WHERE config_hash = %(h)s
            LIMIT 1
            """,
            {"h": config_hash},
        )
        return _row_to_dict(cur, cur.fetchone())


def get_job(config_hash: str) -> dict | None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT * FROM {T_JOBS} WHERE config_hash = %(h)s LIMIT 1",
            {"h": config_hash},
        )
        return _row_to_dict(cur, cur.fetchone())


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def upsert_scene_config(
    scene: dict,
    cells: list[dict],
    is_preset: bool = False,
) -> tuple[str, str]:
    """Insert (or replace) a scene_config + its cells. Returns (config_hash, config_hash).

    The first element of the tuple is preserved for API parity with the
    lakebase_client.upsert_scene_config signature.
    """
    config_hash = compute_config_hash(scene, cells)

    with _connect() as conn, conn.cursor() as cur:
        # Delete + insert is simpler than MERGE for parametrized Delta writes
        # via the SQL connector.
        cur.execute(f"DELETE FROM {T_SCENE} WHERE config_hash = %(h)s", {"h": config_hash})
        cur.execute(
            f"""
            INSERT INTO {T_SCENE} (
                config_hash, name,
                num_rows_tx, num_cols_tx, num_rows_rx, num_cols_rx,
                frequency_hz, bandwidth_hz, max_depth, samples_per_tx,
                cell_size_x, cell_size_y, pattern, polarization,
                num_user_samples, min_sinr_db, min_user_dist_m, max_user_dist_m,
                is_preset, created_at
            ) VALUES (
                %(config_hash)s, %(name)s,
                %(num_rows_tx)s, %(num_cols_tx)s, %(num_rows_rx)s, %(num_cols_rx)s,
                %(frequency_hz)s, %(bandwidth_hz)s, %(max_depth)s, %(samples_per_tx)s,
                %(cell_size_x)s, %(cell_size_y)s, %(pattern)s, %(polarization)s,
                %(num_user_samples)s, %(min_sinr_db)s, %(min_user_dist_m)s, %(max_user_dist_m)s,
                %(is_preset)s, current_timestamp()
            )
            """,
            {**scene, "config_hash": config_hash, "is_preset": is_preset},
        )

        cur.execute(f"DELETE FROM {T_CELLS} WHERE config_hash = %(h)s", {"h": config_hash})
        for cell in cells:
            cur.execute(
                f"""
                INSERT INTO {T_CELLS} (
                    config_hash, cell_id, name, x, y, z,
                    look_at_x, look_at_y, look_at_z, power_dbm
                ) VALUES (
                    %(config_hash)s, %(cell_id)s, %(name)s, %(x)s, %(y)s, %(z)s,
                    %(look_at_x)s, %(look_at_y)s, %(look_at_z)s, %(power_dbm)s
                )
                """,
                {**cell, "config_hash": config_hash},
            )

    return config_hash, config_hash


def write_render(config_hash: str, results: dict) -> None:
    """Persist render bytes + KPIs for a config_hash."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(f"DELETE FROM {T_RENDERS} WHERE config_hash = %(h)s", {"h": config_hash})
        cur.execute(
            f"""
            INSERT INTO {T_RENDERS} (
                config_hash, scene_render_png, sinr_map_png,
                association_png, sinr_cdf_png, rss_cdf_png,
                kpis_json, compute_seconds, created_at
            ) VALUES (
                %(config_hash)s, %(scene_render_png)s, %(sinr_map_png)s,
                %(association_png)s, %(sinr_cdf_png)s, %(rss_cdf_png)s,
                %(kpis_json)s, %(compute_seconds)s, current_timestamp()
            )
            """,
            {"config_hash": config_hash, **results},
        )


def set_job_status(
    config_hash: str,
    status: str,
    run_id: int | None = None,
    error_message: str | None = None,
) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {T_JOBS} WHERE config_hash = %(h)s",
            {"h": config_hash},
        )
        cur.execute(
            f"""
            INSERT INTO {T_JOBS} (
                config_hash, status, run_id, error_message, submitted_at, updated_at
            ) VALUES (
                %(h)s, %(s)s, %(r)s, %(e)s, current_timestamp(), current_timestamp()
            )
            """,
            {"h": config_hash, "s": status, "r": run_id, "e": error_message},
        )


# ---------------------------------------------------------------------------
# Optional helpers (used by setup / migration)
# ---------------------------------------------------------------------------

def list_cached_hashes() -> list[str]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT config_hash FROM {T_RENDERS}")
        return [r[0] for r in cur.fetchall()]
