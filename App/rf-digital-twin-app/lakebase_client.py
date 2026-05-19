"""Lakebase (Postgres) access layer for the RF Digital Twin app.

Schema overview
---------------
- scene_configs        : one row per saved scene-level configuration
- cell_configs         : 7 rows per scene_config, the per-TX setup
- cached_renders       : cached PNG renders + KPI blobs, keyed by config_hash

A "config_hash" is a deterministic SHA-256 over the canonicalised scene+cells
payload. Two identical configs produce the same hash, so cache lookups work
across users and sessions.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

# Default Lakebase instance name — overridden via LAKEBASE_INSTANCE env var.
_DEFAULT_INSTANCE_NAME = "rf-digital-twin-pg"

# OAuth token cache. Lakebase tokens last ~1h; refresh proactively.
_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}
_TOKEN_TTL_SECONDS = 45 * 60


def _pick(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _generate_password() -> str:
    """Mint a fresh OAuth token for Lakebase via the Databricks SDK.

    Databricks Apps populate PGHOST/PGUSER/PGDATABASE from the bound Lakebase
    resource but do not provide PGPASSWORD — the app must generate its own
    credential using the SP's identity.
    """
    now = time.time()
    if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires_at"]:
        return _TOKEN_CACHE["token"]

    instance_name = _pick("LAKEBASE_INSTANCE", default=_DEFAULT_INSTANCE_NAME)
    # Local import so the module remains importable without the SDK installed
    # (e.g. from the precompute notebook on a vanilla cluster).
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[instance_name],
    )
    _TOKEN_CACHE["token"] = cred.token
    _TOKEN_CACHE["expires_at"] = now + _TOKEN_TTL_SECONDS
    return cred.token


def _conn_kwargs() -> dict:
    """Build psycopg connect kwargs from environment.

    Databricks Apps with a Lakebase resource binding exposes:
      PGHOST, PGPORT, PGDATABASE, PGUSER  (PGPASSWORD is NOT set).
    The password is minted on demand via the SDK. For local dev, set
    PGPASSWORD/LAKEBASE_PASSWORD manually to skip the SDK call.
    """
    host = _pick("PGHOST", "LAKEBASE_HOST")
    if not host:
        raise RuntimeError(
            "Lakebase connection not configured. Set PGHOST/PGUSER/PGDATABASE "
            "(or LAKEBASE_* equivalents) in the app environment."
        )
    password = _pick("PGPASSWORD", "LAKEBASE_PASSWORD") or _generate_password()
    return dict(
        host=host,
        port=int(_pick("PGPORT", "LAKEBASE_PORT", default="5432")),
        dbname=_pick("PGDATABASE", "LAKEBASE_DATABASE", default="rf_digital_twin"),
        user=_pick("PGUSER", "LAKEBASE_USER"),
        password=password,
        sslmode=_pick("PGSSLMODE", "LAKEBASE_SSLMODE", default="require"),
    )


@contextmanager
def connect():
    """Yield a psycopg connection. Caller is responsible for transaction commit."""
    conn = psycopg.connect(**_conn_kwargs(), row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS scene_configs (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    config_hash     TEXT NOT NULL UNIQUE,
    num_rows_tx     INTEGER NOT NULL,
    num_cols_tx     INTEGER NOT NULL,
    num_rows_rx     INTEGER NOT NULL,
    num_cols_rx     INTEGER NOT NULL,
    frequency_hz    DOUBLE PRECISION NOT NULL,
    bandwidth_hz    DOUBLE PRECISION NOT NULL,
    max_depth       INTEGER NOT NULL,
    samples_per_tx  BIGINT NOT NULL,
    cell_size_x     DOUBLE PRECISION NOT NULL,
    cell_size_y     DOUBLE PRECISION NOT NULL,
    pattern         TEXT NOT NULL,
    polarization    TEXT NOT NULL,
    num_user_samples INTEGER NOT NULL,
    min_sinr_db     DOUBLE PRECISION NOT NULL,
    min_user_dist_m DOUBLE PRECISION NOT NULL,
    max_user_dist_m DOUBLE PRECISION NOT NULL,
    is_preset       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cell_configs (
    id              BIGSERIAL PRIMARY KEY,
    scene_config_id BIGINT NOT NULL REFERENCES scene_configs(id) ON DELETE CASCADE,
    cell_id         INTEGER NOT NULL,
    name            TEXT NOT NULL,
    x               DOUBLE PRECISION NOT NULL,
    y               DOUBLE PRECISION NOT NULL,
    z               DOUBLE PRECISION NOT NULL,
    look_at_x       DOUBLE PRECISION NOT NULL,
    look_at_y       DOUBLE PRECISION NOT NULL,
    look_at_z       DOUBLE PRECISION NOT NULL,
    power_dbm       DOUBLE PRECISION NOT NULL,
    UNIQUE (scene_config_id, cell_id)
);

CREATE TABLE IF NOT EXISTS cached_renders (
    config_hash         TEXT PRIMARY KEY,
    scene_render_png    BYTEA,
    sinr_map_png        BYTEA,
    association_png     BYTEA,
    sinr_cdf_png        BYTEA,
    rss_cdf_png         BYTEA,
    kpis_json           JSONB,
    compute_seconds     DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS compute_jobs (
    config_hash     TEXT PRIMARY KEY,
    status          TEXT NOT NULL,         -- PENDING | RUNNING | SUCCEEDED | FAILED
    run_id          BIGINT,
    error_message   TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_schema() -> None:
    """Create all tables if they do not yet exist."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_DDL)
        conn.commit()


# ---------------------------------------------------------------------------
# Hashing
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
    """Deterministic hash over scene + ordered cell list."""
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
# Writes
# ---------------------------------------------------------------------------

def upsert_scene_config(
    scene: dict,
    cells: list[dict],
    is_preset: bool = False,
) -> tuple[int, str]:
    """Insert (or fetch) a scene_config + its cells. Returns (id, config_hash)."""
    config_hash = compute_config_hash(scene, cells)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scene_configs (
                    name, config_hash,
                    num_rows_tx, num_cols_tx, num_rows_rx, num_cols_rx,
                    frequency_hz, bandwidth_hz, max_depth, samples_per_tx,
                    cell_size_x, cell_size_y, pattern, polarization,
                    num_user_samples, min_sinr_db, min_user_dist_m, max_user_dist_m,
                    is_preset
                ) VALUES (
                    %(name)s, %(config_hash)s,
                    %(num_rows_tx)s, %(num_cols_tx)s, %(num_rows_rx)s, %(num_cols_rx)s,
                    %(frequency_hz)s, %(bandwidth_hz)s, %(max_depth)s, %(samples_per_tx)s,
                    %(cell_size_x)s, %(cell_size_y)s, %(pattern)s, %(polarization)s,
                    %(num_user_samples)s, %(min_sinr_db)s, %(min_user_dist_m)s, %(max_user_dist_m)s,
                    %(is_preset)s
                )
                ON CONFLICT (config_hash) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                {**scene, "config_hash": config_hash, "is_preset": is_preset},
            )
            scene_id = cur.fetchone()["id"]

            cur.execute("DELETE FROM cell_configs WHERE scene_config_id = %s", (scene_id,))
            cur.executemany(
                """
                INSERT INTO cell_configs (
                    scene_config_id, cell_id, name,
                    x, y, z, look_at_x, look_at_y, look_at_z, power_dbm
                ) VALUES (
                    %(scene_config_id)s, %(cell_id)s, %(name)s,
                    %(x)s, %(y)s, %(z)s, %(look_at_x)s, %(look_at_y)s, %(look_at_z)s, %(power_dbm)s
                )
                """,
                [{**c, "scene_config_id": scene_id} for c in cells],
            )
        conn.commit()

    return scene_id, config_hash


def write_render(config_hash: str, results: dict) -> None:
    """Persist render bytes + KPIs for a config_hash."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cached_renders (
                    config_hash, scene_render_png, sinr_map_png,
                    association_png, sinr_cdf_png, rss_cdf_png,
                    kpis_json, compute_seconds
                ) VALUES (
                    %(config_hash)s, %(scene_render_png)s, %(sinr_map_png)s,
                    %(association_png)s, %(sinr_cdf_png)s, %(rss_cdf_png)s,
                    %(kpis_json)s, %(compute_seconds)s
                )
                ON CONFLICT (config_hash) DO UPDATE SET
                    scene_render_png = EXCLUDED.scene_render_png,
                    sinr_map_png     = EXCLUDED.sinr_map_png,
                    association_png  = EXCLUDED.association_png,
                    sinr_cdf_png     = EXCLUDED.sinr_cdf_png,
                    rss_cdf_png      = EXCLUDED.rss_cdf_png,
                    kpis_json        = EXCLUDED.kpis_json,
                    compute_seconds  = EXCLUDED.compute_seconds,
                    created_at       = now()
                """,
                {"config_hash": config_hash, **results},
            )
        conn.commit()


def set_job_status(
    config_hash: str,
    status: str,
    run_id: int | None = None,
    error_message: str | None = None,
) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO compute_jobs (config_hash, status, run_id, error_message)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (config_hash) DO UPDATE SET
                    status        = EXCLUDED.status,
                    run_id        = COALESCE(EXCLUDED.run_id, compute_jobs.run_id),
                    error_message = EXCLUDED.error_message,
                    updated_at    = now()
                """,
                (config_hash, status, run_id, error_message),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_render(config_hash: str) -> dict | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM cached_renders WHERE config_hash = %s",
                (config_hash,),
            )
            return cur.fetchone()


def get_job(config_hash: str) -> dict | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM compute_jobs WHERE config_hash = %s",
                (config_hash,),
            )
            return cur.fetchone()


def list_presets() -> list[dict]:
    """Return preset scene_configs + their cells (one row per preset, with cells list)."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM scene_configs WHERE is_preset = TRUE ORDER BY id"
            )
            scenes = cur.fetchall()
            for s in scenes:
                cur.execute(
                    "SELECT * FROM cell_configs WHERE scene_config_id = %s ORDER BY cell_id",
                    (s["id"],),
                )
                s["cells"] = cur.fetchall()
    return scenes


def load_scene_by_hash(config_hash: str) -> dict | None:
    """Return scene + cells dict for a given hash, or None."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM scene_configs WHERE config_hash = %s",
                (config_hash,),
            )
            scene = cur.fetchone()
            if not scene:
                return None
            cur.execute(
                "SELECT * FROM cell_configs WHERE scene_config_id = %s ORDER BY cell_id",
                (scene["id"],),
            )
            scene["cells"] = cur.fetchall()
    return scene
