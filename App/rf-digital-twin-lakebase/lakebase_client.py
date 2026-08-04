"""Lakebase (Postgres) access layer for the Seattle RF Digital Twin app.

Shares the etoile demo's connection/credential pattern (OAuth token minted via the
Databricks SDK, cached ~45 min) and the same ``rf-digital-twin-pg`` instance, but the
schema is **neighborhood- and tile-aware**:

- ``scene_configs``  : one row per (neighborhood, tile, render config); keyed by config_hash.
- ``cell_configs``   : the towers (TXs) for each scene_config.
- ``cached_renders`` : cached PNGs + KPI JSON, keyed by config_hash (unchanged shape).
- ``neighborhoods``  : per-neighborhood render status — drives the app's dropdown. A neighborhood
                       is ``CACHED`` (renders ready), ``RENDERING`` (a job is in flight), or ``NONE``.

The app never recomputes a config_hash from tower data (it has no Spark): it selects cached
renders **by neighborhood + story name**. ``compute_config_hash`` is used setup/job-side only,
to mint stable storage keys, and folds in the per-tower frequency/array (towers are
heterogeneous) plus the neighborhood/tile identity.
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

_DEFAULT_INSTANCE_NAME = "rf-digital-twin-pg"
# This app's tables live in their own Postgres schema inside the shared rf_digital_twin
# database, so they never collide with the other demos on the same instance (override via
# PG_SCHEMA).
_PG_SCHEMA = os.environ.get("PG_SCHEMA", "lakebase_only")
_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}
_TOKEN_TTL_SECONDS = 45 * 60
_INSTANCE_CACHE: dict[str, Any] = {"host": None, "name": None}


def _pick(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _ws_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _instance_host() -> str:
    instance_name = _pick("LAKEBASE_INSTANCE", default=_DEFAULT_INSTANCE_NAME)
    if _INSTANCE_CACHE["name"] == instance_name and _INSTANCE_CACHE["host"]:
        return _INSTANCE_CACHE["host"]
    inst = _ws_client().database.get_database_instance(name=instance_name)
    _INSTANCE_CACHE["name"] = instance_name
    _INSTANCE_CACHE["host"] = inst.read_write_dns
    return inst.read_write_dns


def _current_user() -> str:
    return _ws_client().current_user.me().user_name


def _generate_password() -> str:
    now = time.time()
    if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires_at"]:
        return _TOKEN_CACHE["token"]
    instance_name = _pick("LAKEBASE_INSTANCE", default=_DEFAULT_INSTANCE_NAME)
    cred = _ws_client().database.generate_database_credential(
        request_id=str(uuid.uuid4()), instance_names=[instance_name],
    )
    _TOKEN_CACHE["token"] = cred.token
    _TOKEN_CACHE["expires_at"] = now + _TOKEN_TTL_SECONDS
    return cred.token


def _conn_kwargs() -> dict:
    host = _pick("PGHOST", "LAKEBASE_HOST") or _instance_host()
    user = _pick("PGUSER", "LAKEBASE_USER") or _current_user()
    password = _pick("PGPASSWORD", "LAKEBASE_PASSWORD") or _generate_password()
    return dict(
        host=host,
        port=int(_pick("PGPORT", "LAKEBASE_PORT", default="5432")),
        dbname=_pick("PGDATABASE", "LAKEBASE_DATABASE", default="rf_digital_twin"),
        user=user, password=password,
        sslmode=_pick("PGSSLMODE", "LAKEBASE_SSLMODE", default="require"),
        # Fail fast instead of hanging forever when the DB endpoint is unreachable.
        connect_timeout=int(_pick("PGCONNECT_TIMEOUT", default="10")),
        # Resolve all unqualified table names to this app's schema (created by init_schema).
        options=f"-c search_path={_PG_SCHEMA},public",
    )


@contextmanager
def connect():
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
    neighborhood    TEXT,
    tile_id         TEXT,
    story_key       TEXT,
    bbox_x_lo       DOUBLE PRECISION,
    bbox_x_hi       DOUBLE PRECISION,
    bbox_y_lo       DOUBLE PRECISION,
    bbox_y_hi       DOUBLE PRECISION,
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
CREATE INDEX IF NOT EXISTS idx_scene_configs_hood ON scene_configs (neighborhood);

CREATE TABLE IF NOT EXISTS cell_configs (
    id              BIGSERIAL PRIMARY KEY,
    scene_config_id BIGINT NOT NULL REFERENCES scene_configs(id) ON DELETE CASCADE,
    cell_id         INTEGER NOT NULL,
    tower_id        BIGINT,
    tower_type      TEXT,
    name            TEXT NOT NULL,
    x               DOUBLE PRECISION NOT NULL,
    y               DOUBLE PRECISION NOT NULL,
    z               DOUBLE PRECISION NOT NULL,
    look_at_x       DOUBLE PRECISION NOT NULL,
    look_at_y       DOUBLE PRECISION NOT NULL,
    look_at_z       DOUBLE PRECISION NOT NULL,
    power_dbm       DOUBLE PRECISION NOT NULL,
    frequency_hz    DOUBLE PRECISION,
    num_rows_tx     INTEGER,
    num_cols_tx     INTEGER,
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
    status          TEXT NOT NULL,
    run_id          BIGINT,
    error_message   TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS neighborhoods (
    name            TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'NONE',   -- NONE | RENDERING | CACHED | FAILED
    run_id          BIGINT,
    n_renders       INTEGER NOT NULL DEFAULT 0,
    n_towers        INTEGER,
    error_message   TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_schema() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            # Create the isolated schema first; search_path already points at it so the
            # CREATE TABLE statements below land in `seattle`, not the shared `public`.
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_PG_SCHEMA}")
            cur.execute(SCHEMA_DDL)
            # The app runs as its own service-principal Postgres role. A custom schema
            # doesn't grant USAGE to other roles by default, so open read access on the
            # Seattle objects to all roles (idempotent; safe for this demo cache).
            cur.execute(f"GRANT USAGE ON SCHEMA {_PG_SCHEMA} TO PUBLIC")
            cur.execute(f"GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA {_PG_SCHEMA} TO PUBLIC")
            cur.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {_PG_SCHEMA} TO PUBLIC")
            cur.execute(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA {_PG_SCHEMA} "
                f"GRANT SELECT, INSERT, UPDATE ON TABLES TO PUBLIC"
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Hashing (setup/job-side only)
# ---------------------------------------------------------------------------

_HASH_SCENE_FIELDS = (
    "num_rows_tx", "num_cols_tx", "num_rows_rx", "num_cols_rx",
    "frequency_hz", "bandwidth_hz", "max_depth", "samples_per_tx",
    "cell_size_x", "cell_size_y", "pattern", "polarization",
    "num_user_samples", "min_sinr_db", "min_user_dist_m", "max_user_dist_m",
)
_HASH_CELL_FIELDS = (
    "cell_id", "x", "y", "z", "look_at_x", "look_at_y", "look_at_z",
    "power_dbm", "frequency_hz", "num_rows_tx", "num_cols_tx",
)


# Decimal places floats are quantized to before hashing. Tower coordinates are metres, so 6 dp
# is sub-micron — far below anything the ray tracer can resolve — while making the hash
# immune to last-ULP float noise.
#
# This matters because the app and the render job run on different platforms (an app container
# on Linux x86_64 vs a dev laptop on macOS arm64), and libm's ``cos`` for the projection in
# ``neighborhoods.project_lonlat`` differs between them by 1 ULP. Hashing raw ``repr`` floats
# made every preset a cache miss depending on which machine did the hashing. ``round`` is
# correctly-rounded and platform-independent in CPython, so quantizing first fixes it.
_HASH_FLOAT_DP = 6


def _quantize(v: Any) -> Any:
    """Round floats to _HASH_FLOAT_DP so the hash can't depend on last-bit float noise."""
    if isinstance(v, float):
        # Normalise -0.0 to 0.0 too; they're equal but repr differently.
        return round(v, _HASH_FLOAT_DP) + 0.0
    return v


def compute_config_hash(scene: dict, cells: Iterable[dict]) -> str:
    """Deterministic hash over scene + ordered cells + neighborhood/tile identity.

    Stable across platforms: see ``_HASH_FLOAT_DP``.
    """
    payload = {
        "neighborhood": scene.get("neighborhood"),
        "tile_id": scene.get("tile_id"),
        "story_key": scene.get("story_key"),
        "scene": {k: _quantize(scene[k]) for k in _HASH_SCENE_FIELDS},
        "cells": [
            {k: _quantize(c.get(k)) for k in _HASH_CELL_FIELDS}
            for c in sorted(cells, key=lambda c: c["cell_id"])
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def upsert_scene_config(scene: dict, cells: list[dict], is_preset: bool = False) -> tuple[int, str]:
    config_hash = compute_config_hash(scene, cells)
    bbox = scene.get("render_bounds") or [None, None, None, None]
    params = {
        **scene, "config_hash": config_hash, "is_preset": is_preset,
        "bbox_x_lo": bbox[0], "bbox_x_hi": bbox[1], "bbox_y_lo": bbox[2], "bbox_y_hi": bbox[3],
        "neighborhood": scene.get("neighborhood"),
        "tile_id": scene.get("tile_id"),
        "story_key": scene.get("story_key"),
    }
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scene_configs (
                    name, config_hash, neighborhood, tile_id, story_key,
                    bbox_x_lo, bbox_x_hi, bbox_y_lo, bbox_y_hi,
                    num_rows_tx, num_cols_tx, num_rows_rx, num_cols_rx,
                    frequency_hz, bandwidth_hz, max_depth, samples_per_tx,
                    cell_size_x, cell_size_y, pattern, polarization,
                    num_user_samples, min_sinr_db, min_user_dist_m, max_user_dist_m, is_preset
                ) VALUES (
                    %(name)s, %(config_hash)s, %(neighborhood)s, %(tile_id)s, %(story_key)s,
                    %(bbox_x_lo)s, %(bbox_x_hi)s, %(bbox_y_lo)s, %(bbox_y_hi)s,
                    %(num_rows_tx)s, %(num_cols_tx)s, %(num_rows_rx)s, %(num_cols_rx)s,
                    %(frequency_hz)s, %(bandwidth_hz)s, %(max_depth)s, %(samples_per_tx)s,
                    %(cell_size_x)s, %(cell_size_y)s, %(pattern)s, %(polarization)s,
                    %(num_user_samples)s, %(min_sinr_db)s, %(min_user_dist_m)s, %(max_user_dist_m)s,
                    %(is_preset)s
                )
                ON CONFLICT (config_hash) DO UPDATE SET
                    name = EXCLUDED.name, neighborhood = EXCLUDED.neighborhood,
                    tile_id = EXCLUDED.tile_id, story_key = EXCLUDED.story_key
                RETURNING id
                """,
                params,
            )
            scene_id = cur.fetchone()["id"]
            cur.execute("DELETE FROM cell_configs WHERE scene_config_id = %s", (scene_id,))
            cur.executemany(
                """
                INSERT INTO cell_configs (
                    scene_config_id, cell_id, tower_id, tower_type, name,
                    x, y, z, look_at_x, look_at_y, look_at_z, power_dbm,
                    frequency_hz, num_rows_tx, num_cols_tx
                ) VALUES (
                    %(scene_config_id)s, %(cell_id)s, %(tower_id)s, %(tower_type)s, %(name)s,
                    %(x)s, %(y)s, %(z)s, %(look_at_x)s, %(look_at_y)s, %(look_at_z)s, %(power_dbm)s,
                    %(frequency_hz)s, %(num_rows_tx)s, %(num_cols_tx)s
                )
                """,
                [{
                    "scene_config_id": scene_id,
                    "tower_id": c.get("tower_id"), "tower_type": c.get("tower_type"),
                    "frequency_hz": c.get("frequency_hz"),
                    "num_rows_tx": c.get("num_rows_tx"), "num_cols_tx": c.get("num_cols_tx"),
                    **{k: c[k] for k in ("cell_id", "name", "x", "y", "z",
                                         "look_at_x", "look_at_y", "look_at_z", "power_dbm")},
                } for c in cells],
            )
        conn.commit()
    return scene_id, config_hash


def write_render(config_hash: str, results: dict) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cached_renders (
                    config_hash, scene_render_png, sinr_map_png, association_png,
                    sinr_cdf_png, rss_cdf_png, kpis_json, compute_seconds
                ) VALUES (
                    %(config_hash)s, %(scene_render_png)s, %(sinr_map_png)s, %(association_png)s,
                    %(sinr_cdf_png)s, %(rss_cdf_png)s, %(kpis_json)s, %(compute_seconds)s
                )
                ON CONFLICT (config_hash) DO UPDATE SET
                    scene_render_png = EXCLUDED.scene_render_png,
                    sinr_map_png = EXCLUDED.sinr_map_png,
                    association_png = EXCLUDED.association_png,
                    sinr_cdf_png = EXCLUDED.sinr_cdf_png,
                    rss_cdf_png = EXCLUDED.rss_cdf_png,
                    kpis_json = EXCLUDED.kpis_json,
                    compute_seconds = EXCLUDED.compute_seconds,
                    created_at = now()
                """,
                {"config_hash": config_hash, **results},
            )
        conn.commit()


def set_job_status(config_hash: str, status: str, run_id: int | None = None,
                   error_message: str | None = None) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO compute_jobs (config_hash, status, run_id, error_message)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (config_hash) DO UPDATE SET
                    status = EXCLUDED.status,
                    run_id = COALESCE(EXCLUDED.run_id, compute_jobs.run_id),
                    error_message = EXCLUDED.error_message, updated_at = now()
                """,
                (config_hash, status, run_id, error_message),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Neighborhood status (drives the app dropdown)
# ---------------------------------------------------------------------------

def upsert_neighborhood(name: str, status: str = "NONE", run_id: int | None = None,
                        n_towers: int | None = None, error_message: str | None = None) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO neighborhoods (name, status, run_id, n_towers, error_message)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    status = EXCLUDED.status,
                    run_id = COALESCE(EXCLUDED.run_id, neighborhoods.run_id),
                    n_towers = COALESCE(EXCLUDED.n_towers, neighborhoods.n_towers),
                    error_message = EXCLUDED.error_message,
                    n_renders = (SELECT COUNT(*) FROM scene_configs sc
                                 JOIN cached_renders r ON r.config_hash = sc.config_hash
                                 WHERE sc.neighborhood = neighborhoods.name),
                    updated_at = now()
                """,
                (name, status, run_id, n_towers, error_message),
            )
        conn.commit()


def get_neighborhood(name: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM neighborhoods WHERE name = %s", (name,))
        return cur.fetchone()


def list_neighborhoods() -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM neighborhoods ORDER BY name")
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Reads (app-facing)
# ---------------------------------------------------------------------------

def get_render(config_hash: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM cached_renders WHERE config_hash = %s", (config_hash,))
        return cur.fetchone()


def get_job(config_hash: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM compute_jobs WHERE config_hash = %s", (config_hash,))
        return cur.fetchone()


def list_neighborhood_renders(neighborhood: str) -> list[dict]:
    """Scene metadata + render bytes for every cached render in a neighborhood.

    The app uses this for both the Downtown story gallery and on-demand coverage tiles —
    no config_hash recomputation needed.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT sc.id, sc.name, sc.config_hash, sc.neighborhood, sc.tile_id, sc.story_key,
                   sc.is_preset, sc.frequency_hz, sc.num_rows_tx, sc.num_cols_tx,
                   r.scene_render_png, r.sinr_map_png, r.association_png,
                   r.sinr_cdf_png, r.rss_cdf_png, r.kpis_json, r.compute_seconds
            FROM scene_configs sc
            JOIN cached_renders r ON r.config_hash = sc.config_hash
            WHERE sc.neighborhood = %s
            ORDER BY sc.is_preset DESC, sc.story_key NULLS LAST, sc.tile_id, sc.id
            """,
            (neighborhood,),
        )
        return cur.fetchall()
