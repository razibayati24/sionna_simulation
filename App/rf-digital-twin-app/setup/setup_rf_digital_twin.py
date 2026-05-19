# Databricks notebook source
# MAGIC %md
# MAGIC # RF Digital Twin — Workspace Setup
# MAGIC
# MAGIC One-shot setup notebook that prepares everything the **RF Digital Twin**
# MAGIC Databricks App needs in this workspace:
# MAGIC
# MAGIC 1. Creates the `cmegdemos_catalog.sionna_rf_data` schema in Unity Catalog.
# MAGIC 2. Generates the default 7-cell network configuration and writes it as a
# MAGIC    Delta table (`cell_configs_default`) — this is the **source of truth**
# MAGIC    that engineers can edit later.
# MAGIC 3. Provisions a Lakebase Postgres instance + database (idempotent).
# MAGIC 4. Initialises the Lakebase schema used by the app for caching.
# MAGIC 5. Loads the default cells from UC into Lakebase.
# MAGIC 6. Runs Sionna RT for **Config 1 (8×2 TX)** and **Config 2 (16×16 TX)**
# MAGIC    and persists the renders (scene, SINR map, association, CDFs) + KPIs
# MAGIC    into Lakebase so the app loads them instantly.
# MAGIC 7. Prints the connection details to plug into the app's resource bindings.
# MAGIC
# MAGIC ## Cluster you need to provision
# MAGIC
# MAGIC Sionna RT calls NVIDIA OptiX for ray tracing — you **must** run this
# MAGIC notebook on a GPU cluster whose Databricks Runtime ships with OptiX.
# MAGIC DBR ML runtimes and plain DBR 16.x + GPU instance both include it; a
# MAGIC CPU cluster will fail with `libnvoptix.so.1 could not be loaded`.
# MAGIC
# MAGIC **Validated configuration** (the one used to seed the demo):
# MAGIC
# MAGIC | Setting | Value |
# MAGIC | --- | --- |
# MAGIC | Cluster name | `sionna` (any name) |
# MAGIC | Databricks Runtime | **16.4 LTS** (Scala 2.13) — plain, not ML |
# MAGIC | Runtime engine | Standard |
# MAGIC | Driver node | `g5.xlarge` (1× NVIDIA A10G GPU, 16 GB) |
# MAGIC | Worker node | `g5.xlarge` |
# MAGIC | Autoscaling | min 2, max 8 workers |
# MAGIC | Access mode | Single user (your identity) |
# MAGIC | Auto-termination | 120 minutes |
# MAGIC | AWS availability | Spot with fallback to on-demand |
# MAGIC | Init scripts | none |
# MAGIC | Custom Spark conf | none |
# MAGIC
# MAGIC Cheaper alternatives that also work: `g4dn.xlarge` (T4 GPU) — slower
# MAGIC ray tracing, but full OptiX. Avoid CPU-only or A1 (ARM) instances.
# MAGIC
# MAGIC Wall-clock you can expect: **~2–3 min per config** on g5.xlarge,
# MAGIC ~5–8 min on g4dn.xlarge. Total notebook run on g5: ~10 minutes.
# MAGIC
# MAGIC > **Permissions required:**
# MAGIC > - `USE CATALOG` + `CREATE SCHEMA` on `cmegdemos_catalog`
# MAGIC > - Workspace permission to create Lakebase Database Instances
# MAGIC >   (workspace quota is 10 — delete an unused one first if needed)
# MAGIC > - You'll be acting as the Postgres role for the data load.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. (Optional) Provision the GPU cluster from this notebook
# MAGIC
# MAGIC If you don't already have a Sionna-ready GPU cluster, run the cell
# MAGIC below from a tiny job/serverless context to create one matching the
# MAGIC validated config above. Then **attach this notebook to that new
# MAGIC cluster** and continue from section 1. Skip if you've already got
# MAGIC a GPU cluster attached.

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.compute import (
#     AutoScale, AwsAttributes, AwsAvailability, ClusterSpec, DataSecurityMode,
#     RuntimeEngine,
# )
#
# w = WorkspaceClient()
# new_cluster = w.clusters.create(
#     cluster_name="sionna-rf-digital-twin",
#     spark_version="16.4.x-scala2.13",
#     node_type_id="g5.xlarge",
#     driver_node_type_id="g5.xlarge",
#     autoscale=AutoScale(min_workers=2, max_workers=8),
#     autotermination_minutes=120,
#     data_security_mode=DataSecurityMode.SINGLE_USER,
#     runtime_engine=RuntimeEngine.STANDARD,
#     aws_attributes=AwsAttributes(
#         availability=AwsAvailability.SPOT_WITH_FALLBACK,
#         first_on_demand=1,
#         spot_bid_price_percent=100,
#         zone_id="auto",
#     ),
# )
# print(f"Created cluster {new_cluster.cluster_id} — attach this notebook to it.")

# COMMAND ----------

# MAGIC ## 0b. Install dependencies

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade \
# MAGIC   drjit mitsuba sionna-rt \
# MAGIC   "psycopg[binary]>=3.1.18" \
# MAGIC   "databricks-sdk>=0.30.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import base64
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("Authenticated as:", w.current_user.me().user_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration knobs
# MAGIC
# MAGIC Edit these if you want a different catalog/schema, instance name, or
# MAGIC Sionna sample count.

# COMMAND ----------

CATALOG = "cmegdemos_catalog"
SCHEMA  = "sionna_rf_data"
UC_FQN  = f"{CATALOG}.{SCHEMA}"

CELL_TABLE = f"{UC_FQN}.cell_configs_default"

LAKEBASE_INSTANCE = "rf-digital-twin-pg"
LAKEBASE_CAPACITY = "CU_1"
LAKEBASE_DB       = "rf_digital_twin"

# Light/heavy Sionna sample-per-tx setting. 10^7 matches the article; drop to
# 10^6 if you want a fast smoke test on CPU.
SAMPLES_PER_TX = 10 ** 7

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Default network configuration
# MAGIC
# MAGIC 7 macro cells around the Arc de Triomphe (Sionna `etoile` scene), height
# MAGIC 25 m, 44 dBm each. Edit these dicts to change the default layout.

# COMMAND ----------

DEFAULT_CELLS = [
    {"cell_id": 0, "name": "tx0", "x": -150.0, "y":    0.0, "z": 25.0,
     "look_at_x": -300.0, "look_at_y":    0.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 1, "name": "tx1", "x": -100.0, "y":  100.0, "z": 25.0,
     "look_at_x": -200.0, "look_at_y":  200.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 2, "name": "tx2", "x":    0.0, "y":  150.0, "z": 25.0,
     "look_at_x":    0.0, "look_at_y":  300.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 3, "name": "tx3", "x":  100.0, "y":  100.0, "z": 25.0,
     "look_at_x":  200.0, "look_at_y":  200.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 4, "name": "tx4", "x":  150.0, "y":    0.0, "z": 25.0,
     "look_at_x":  300.0, "look_at_y":    0.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 5, "name": "tx5", "x":  100.0, "y": -100.0, "z": 25.0,
     "look_at_x":  200.0, "look_at_y": -200.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 6, "name": "tx6", "x": -100.0, "y": -100.0, "z": 25.0,
     "look_at_x": -200.0, "look_at_y": -200.0, "look_at_z": 0.0, "power_dbm": 44.0},
]


@dataclass
class SceneConfig:
    name: str
    num_rows_tx: int
    num_cols_tx: int
    num_rows_rx: int
    num_cols_rx: int
    frequency_hz: float = 28e9
    bandwidth_hz: float = 1e8
    max_depth: int = 5
    samples_per_tx: int = SAMPLES_PER_TX
    cell_size_x: float = 1.0
    cell_size_y: float = 1.0
    pattern: str = "tr38901"
    polarization: str = "V"
    num_user_samples: int = 50
    min_sinr_db: float = 3.0
    min_user_dist_m: float = 10.0
    max_user_dist_m: float = 200.0
    # When set, every cell's power_dbm gets replaced with this value.
    cell_power_override_dbm: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def cells_for_preset(cfg: SceneConfig, base_cells: list[dict]) -> list[dict]:
    """Apply preset-level overrides (e.g. uniform TX power) to the cell list."""
    cells = [dict(c) for c in base_cells]
    if cfg.cell_power_override_dbm is not None:
        for c in cells:
            c["power_dbm"] = float(cfg.cell_power_override_dbm)
    return cells


# Story A — Antenna densification
CONFIG_DENS_2X2  = SceneConfig("A · Densification — 2×2 TX (baseline)",   2,  2, 2, 2)
CONFIG_DENS_4X4  = SceneConfig("A · Densification — 4×4 TX",              4,  4, 2, 2)
CONFIG_1         = SceneConfig("A · Densification — 8×2 TX (Config 1)",   8,  2, 2, 2)
CONFIG_DENS_8X8  = SceneConfig("A · Densification — 8×8 TX",              8,  8, 2, 2)
CONFIG_2         = SceneConfig("A · Densification — 16×16 TX (Config 2)", 16, 16, 2, 2)
CONFIG_DENS_32X8 = SceneConfig("A · Densification — 32×8 TX (elongated)", 32, 8, 2, 2)

# Story B — Frequency band ladder (8×2 TX held constant)
CONFIG_FREQ_700M = SceneConfig("B · Frequency — 8×2 @ 700 MHz",  8, 2, 2, 2, frequency_hz=7e8,    bandwidth_hz=2e7)
CONFIG_FREQ_2P6G = SceneConfig("B · Frequency — 8×2 @ 2.6 GHz",  8, 2, 2, 2, frequency_hz=2.6e9,  bandwidth_hz=2e7)
CONFIG_FREQ_3P5G = SceneConfig("B · Frequency — 8×2 @ 3.5 GHz",  8, 2, 2, 2, frequency_hz=3.5e9,  bandwidth_hz=1e8)
CONFIG_FREQ_39G  = SceneConfig("B · Frequency — 8×2 @ 39 GHz",   8, 2, 2, 2, frequency_hz=3.9e10, bandwidth_hz=4e8)

# Story C — Antenna pattern (16×16 TX held constant)
CONFIG_PAT_ISO    = SceneConfig("C · Pattern — 16×16 isotropic", 16, 16, 2, 2, pattern="iso")
CONFIG_PAT_DIPOLE = SceneConfig("C · Pattern — 16×16 dipole",    16, 16, 2, 2, pattern="dipole")

# Story D — Polarization (16×16 TX held constant)
CONFIG_POL_VH = SceneConfig("D · Polarization — 16×16 cross (VH)", 16, 16, 2, 2, polarization="VH")

# Story E — Power level (uniform across cells)
CONFIG_PWR_LOW  = SceneConfig("E · Power — 16×16 @ 38 dBm", 16, 16, 2, 2, cell_power_override_dbm=38.0)
CONFIG_PWR_HIGH = SceneConfig("E · Power — 16×16 @ 50 dBm", 16, 16, 2, 2, cell_power_override_dbm=50.0)

# Story F — Bandwidth (16×16 TX held constant)
CONFIG_BW_20M  = SceneConfig("F · Bandwidth — 16×16 @ 20 MHz",  16, 16, 2, 2, bandwidth_hz=2e7)
CONFIG_BW_400M = SceneConfig("F · Bandwidth — 16×16 @ 400 MHz", 16, 16, 2, 2, bandwidth_hz=4e8)

# Story G — Ray tracing fidelity (16×16 TX held constant)
CONFIG_DEPTH_3 = SceneConfig("G · Fidelity — 16×16 max_depth=3", 16, 16, 2, 2, max_depth=3)
CONFIG_DEPTH_8 = SceneConfig("G · Fidelity — 16×16 max_depth=8", 16, 16, 2, 2, max_depth=8)

PRESETS = [
    CONFIG_DENS_2X2, CONFIG_DENS_4X4, CONFIG_1, CONFIG_DENS_8X8, CONFIG_2, CONFIG_DENS_32X8,
    CONFIG_FREQ_700M, CONFIG_FREQ_2P6G, CONFIG_FREQ_3P5G, CONFIG_FREQ_39G,
    CONFIG_PAT_ISO, CONFIG_PAT_DIPOLE,
    CONFIG_POL_VH,
    CONFIG_PWR_LOW, CONFIG_PWR_HIGH,
    CONFIG_BW_20M, CONFIG_BW_400M,
    CONFIG_DEPTH_3, CONFIG_DEPTH_8,
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create UC schema + write default cell table

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_FQN}")
print(f"Schema ready: {UC_FQN}")

# COMMAND ----------

cells_pdf = pd.DataFrame(DEFAULT_CELLS)
cells_sdf = spark.createDataFrame(cells_pdf)

(
    cells_sdf.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CELL_TABLE)
)
print(f"Wrote default cells → {CELL_TABLE}")
display(spark.table(CELL_TABLE).orderBy("cell_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Provision Lakebase instance + database

# COMMAND ----------

from databricks.sdk.service.database import DatabaseInstance, DatabaseCatalog

try:
    instance = w.database.create_database_instance(
        DatabaseInstance(name=LAKEBASE_INSTANCE, capacity=LAKEBASE_CAPACITY)
    )
    print(f"Created instance: {instance.name}")
except Exception as e:
    print(f"Instance create skipped (likely exists): {e}")
    instance = w.database.get_database_instance(name=LAKEBASE_INSTANCE)
    print(f"Using existing instance: {instance.name}")

# Wait until the instance is AVAILABLE.
deadline = time.time() + 15 * 60
while True:
    inst = w.database.get_database_instance(name=LAKEBASE_INSTANCE)
    state = str(inst.state)
    print(f"  state = {state}")
    if "AVAILABLE" in state:
        break
    if time.time() > deadline:
        raise RuntimeError(f"Lakebase instance did not become AVAILABLE within 15min (last state={state})")
    time.sleep(20)

instance = w.database.get_database_instance(name=LAKEBASE_INSTANCE)
print(f"\nLakebase DNS: {instance.read_write_dns}")

# COMMAND ----------

# Create the actual Postgres database inside the instance.
try:
    catalog = w.database.create_database_catalog(
        DatabaseCatalog(
            name=LAKEBASE_DB,
            database_instance_name=LAKEBASE_INSTANCE,
            database_name=LAKEBASE_DB,
            create_database_if_not_exists=True,
        )
    )
    print(f"Created database: {catalog.name}")
except Exception as e:
    print(f"Database create skipped (likely exists): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Connect to Lakebase as the current user

# COMMAND ----------

cred = w.database.generate_database_credential(
    request_id="rf-digital-twin-setup",
    instance_names=[LAKEBASE_INSTANCE],
)

PG_HOST = instance.read_write_dns
PG_PORT = 5432
PG_USER = w.current_user.me().user_name
PG_PASS = cred.token
PG_DB   = LAKEBASE_DB

print(f"Connecting to {PG_HOST}:{PG_PORT} as {PG_USER} / db={PG_DB}")


def lb_connect():
    return psycopg.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASS,
        sslmode="require", row_factory=dict_row,
    )


with lb_connect() as conn, conn.cursor() as cur:
    cur.execute("SELECT version()")
    print(cur.fetchone())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Initialise Lakebase schema

# COMMAND ----------

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
    status          TEXT NOT NULL,
    run_id          BIGINT,
    error_message   TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

with lb_connect() as conn:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_DDL)
    conn.commit()

print("Lakebase schema initialised.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Hash helper + UC → Lakebase ingest helpers

# COMMAND ----------

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
        "cells": [{k: c[k] for k in _HASH_CELL_FIELDS}
                  for c in sorted(cells, key=lambda c: c["cell_id"])],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def upsert_scene(scene: dict, cells: list[dict], is_preset: bool) -> tuple[int, str]:
    config_hash = compute_config_hash(scene, cells)
    with lb_connect() as conn:
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
    with lb_connect() as conn:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sionna RT pipeline
# MAGIC
# MAGIC Builds the etoile scene, runs `RadioMapSolver`, and emits PNG bytes for
# MAGIC the scene render, SINR map, user-to-TX association plot, and SINR/RSS
# MAGIC CDFs, plus a KPI dict.

# COMMAND ----------

import sionna.rt
from sionna.rt import (
    load_scene, PlanarArray, Transmitter, Camera,
    RadioMapSolver,
)


def _fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _build_scene(scene_cfg: dict, cells: list[dict]):
    scene = load_scene(sionna.rt.scene.etoile)
    scene.frequency = float(scene_cfg["frequency_hz"])
    scene.bandwidth = float(scene_cfg["bandwidth_hz"])
    scene.tx_array = PlanarArray(
        num_rows=int(scene_cfg["num_rows_tx"]),
        num_cols=int(scene_cfg["num_cols_tx"]),
        pattern=scene_cfg["pattern"],
        polarization=scene_cfg["polarization"],
    )
    scene.rx_array = PlanarArray(
        num_rows=int(scene_cfg["num_rows_rx"]),
        num_cols=int(scene_cfg["num_cols_rx"]),
        pattern=scene_cfg["pattern"],
        polarization=scene_cfg["polarization"],
    )
    for c in sorted(cells, key=lambda r: r["cell_id"]):
        scene.add(Transmitter(
            name=c["name"],
            position=[float(c["x"]), float(c["y"]), float(c["z"])],
            look_at=[float(c["look_at_x"]), float(c["look_at_y"]), float(c["look_at_z"])],
            power_dbm=float(c["power_dbm"]),
        ))
    return scene


def _scene_render_png(scene, radio_map) -> bytes:
    cam = Camera(position=[0.0, 0.0, 1000.0],
                 orientation=np.array([0.0, np.pi/2, -np.pi/2]))
    fig = scene.render(
        camera=cam, radio_map=radio_map, rm_metric="sinr",
        rm_vmin=-10, rm_vmax=60, rm_show_color_bar=True,
    )
    if fig is None:                       # newer Sionna: render to current fig
        fig = plt.gcf()
    return _fig_png(fig)


def _association_png(radio_map, scene_cfg) -> tuple[bytes, dict]:
    pos, cell_ids = radio_map.sample_positions(
        num_pos=int(scene_cfg["num_user_samples"]),
        metric="sinr",
        min_val_db=float(scene_cfg["min_sinr_db"]),
        min_dist=float(scene_cfg["min_user_dist_m"]),
        max_dist=float(scene_cfg["max_user_dist_m"]),
        tx_association=True,
    )
    fig = radio_map.show(metric="sinr", vmin=-10, vmax=70)
    cell_ids_np = cell_ids.numpy() if hasattr(cell_ids, "numpy") else np.asarray(cell_ids)
    cmap = mpl.colormaps["Dark2"].colors
    for tx, ids in enumerate(cell_ids_np):
        fig.axes[0].plot(ids[:, 1], ids[:, 0],
                         marker="o", markersize=2, linestyle="",
                         color=cmap[tx % len(cmap)])
    users_per_tx = {int(tx): int((ids != 0).any(axis=1).sum())
                    for tx, ids in enumerate(cell_ids_np)}
    return _fig_png(fig), {"users_per_tx": users_per_tx}


def _cdf_png(radio_map, metric: str, xlim: tuple[float, float]) -> tuple[bytes, dict]:
    # Let Sionna's cdf() create its own figure, then grab it.
    plt.close("all")
    radio_map.cdf(metric=metric, bins=400)
    plt.xlim(*xlim)
    plt.title(f"CDF of {metric.upper()}")
    fig = plt.gcf()
    summary: dict[str, float] = {}
    for line in fig.gca().get_lines():
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs):
            for p in (10, 50, 90):
                idx = min(int(np.searchsorted(ys, p/100.0)), len(xs) - 1)
                summary[f"p{p}"] = float(xs[idx])
            break
    return _fig_png(fig), summary


def run_simulation(scene_cfg: dict, cells: list[dict]) -> dict[str, Any]:
    t0 = time.time()
    scene = _build_scene(scene_cfg, cells)
    rm_solver = RadioMapSolver()
    radio_map = rm_solver(
        scene,
        max_depth=int(scene_cfg["max_depth"]),
        samples_per_tx=int(scene_cfg["samples_per_tx"]),
        cell_size=(float(scene_cfg["cell_size_x"]), float(scene_cfg["cell_size_y"])),
    )
    scene_png = _scene_render_png(scene, radio_map)

    fig = radio_map.show_association("sinr")
    sinr_map_png = _fig_png(fig)

    association_png, assoc_kpis = _association_png(radio_map, scene_cfg)
    sinr_cdf_png, sinr_pct = _cdf_png(radio_map, "sinr", xlim=(-40.0, 75.0))
    rss_cdf_png,  rss_pct  = _cdf_png(radio_map, "rss",  xlim=(-150.0, 25.0))

    kpis = {
        "sinr_percentiles_db": sinr_pct,
        "rss_percentiles_dbm": rss_pct,
        **assoc_kpis,
        "num_tx": len(cells),
    }
    return {
        "scene_render_png": scene_png,
        "sinr_map_png":     sinr_map_png,
        "association_png":  association_png,
        "sinr_cdf_png":     sinr_cdf_png,
        "rss_cdf_png":      rss_cdf_png,
        "kpis_json":        json.dumps(kpis),
        "compute_seconds":  time.time() - t0,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Read cell configs from UC + run Sionna for every preset
# MAGIC
# MAGIC Loops through `PRESETS`, applies any preset-level cell overrides
# MAGIC (e.g. uniform TX power), and runs Sionna RT for each. Skips configs
# MAGIC whose hash is already cached in Lakebase — so re-running this cell
# MAGIC after adding new presets only renders the new ones.

# COMMAND ----------

# Pull the canonical cell layout out of UC so we can edit there later.
base_cells = (
    spark.table(CELL_TABLE)
    .orderBy("cell_id")
    .toPandas()
    .to_dict(orient="records")
)
print(f"Loaded {len(base_cells)} cells from {CELL_TABLE}")

# COMMAND ----------

def _is_already_cached(config_hash: str) -> bool:
    """True when a non-empty cached_renders row exists for this hash."""
    with lb_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT length(scene_render_png) AS n "
            "FROM cached_renders WHERE config_hash = %s",
            (config_hash,),
        )
        row = cur.fetchone()
    return bool(row and row["n"] and row["n"] > 10_000)


precompute_summary = []

for cfg in PRESETS:
    scene_cfg = cfg.to_dict()
    cells = cells_for_preset(cfg, base_cells)
    scene_id, config_hash = upsert_scene(scene_cfg, cells, is_preset=True)
    print(f"\n=== {cfg.name} ===")
    print(f"  scene_id     = {scene_id}")
    print(f"  config_hash  = {config_hash}")

    if _is_already_cached(config_hash):
        print("  Already cached — skipping Sionna run.")
        precompute_summary.append({
            "name": cfg.name,
            "config_hash": config_hash,
            "status": "skipped (already cached)",
        })
        continue

    print(f"  Running Sionna RT (samples_per_tx={SAMPLES_PER_TX:,})…")
    results = run_simulation(scene_cfg, cells)
    write_render(config_hash, results)

    print(f"  Done in {results['compute_seconds']:.1f}s, cached to Lakebase.")
    precompute_summary.append({
        "name": cfg.name,
        "config_hash": config_hash,
        "compute_seconds": results["compute_seconds"],
        "status": "rendered",
        "kpis": json.loads(results["kpis_json"]),
    })

print(f"\nFinished. {sum(1 for s in precompute_summary if s.get('status') == 'rendered')} new renders, "
      f"{sum(1 for s in precompute_summary if s.get('status') == 'skipped (already cached)')} skipped.")
display(pd.DataFrame(precompute_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9b. Demo cheat sheet — sidebar values per preset
# MAGIC
# MAGIC Run this cell to print the exact sidebar inputs to set in the app for
# MAGIC each cached preset. Bookmark / screenshot the resulting table; during
# MAGIC the demo, glance at it to know what to type so the hash matches a
# MAGIC cached render and you get an instant flip.

# COMMAND ----------

cheatsheet_rows = []
for cfg in PRESETS:
    scene_cfg = cfg.to_dict()
    cells = cells_for_preset(cfg, base_cells)
    config_hash = compute_config_hash(scene_cfg, cells)

    # Effective TX power across cells (the override, or the default 44 dBm).
    power_dbm = (
        cfg.cell_power_override_dbm
        if cfg.cell_power_override_dbm is not None
        else 44.0
    )

    cheatsheet_rows.append({
        "preset":      cfg.name,
        "hash":        config_hash[:12],
        "TX array":    f"{cfg.num_rows_tx} × {cfg.num_cols_tx}",
        "RX array":    f"{cfg.num_rows_rx} × {cfg.num_cols_rx}",
        "Pattern":     cfg.pattern,
        "Polariz.":    cfg.polarization,
        "TX pwr dBm":  power_dbm,
        "Freq GHz":    round(cfg.frequency_hz / 1e9, 3),
        "BW MHz":      round(cfg.bandwidth_hz / 1e6, 1),
        "max_depth":   cfg.max_depth,
        "samples 10^": int(round(__import__("math").log10(cfg.samples_per_tx))),
    })

cheatsheet_df = pd.DataFrame(cheatsheet_rows)

# Pretty-print the same thing as plain text so the cheat sheet is copy-pastable.
print("APP SIDEBAR CHEAT SHEET — type these values to match each cached preset")
print("=" * 110)
for row in cheatsheet_rows:
    print(
        f"  {row['preset']:<55}  hash={row['hash']}"
    )
    diffs = []
    if row['TX array']    != "8 × 2":     diffs.append(f"TX={row['TX array']}")
    if row['RX array']    != "2 × 2":     diffs.append(f"RX={row['RX array']}")
    if row['Pattern']     != "tr38901":   diffs.append(f"pattern={row['Pattern']}")
    if row['Polariz.']    != "V":         diffs.append(f"pol={row['Polariz.']}")
    if row['TX pwr dBm']  != 44.0:        diffs.append(f"power={row['TX pwr dBm']} dBm")
    if row['Freq GHz']    != 28.0:        diffs.append(f"freq={row['Freq GHz']} GHz")
    if row['BW MHz']      != 100.0:       diffs.append(f"BW={row['BW MHz']} MHz")
    if row['max_depth']   != 5:           diffs.append(f"max_depth={row['max_depth']}")
    print(f"      vs Config 1 default → {', '.join(diffs) if diffs else 'no changes (same as Config 1)'}")
print("=" * 110)

display(cheatsheet_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Quick verification — pull one preset back from Lakebase

# COMMAND ----------

with lb_connect() as conn, conn.cursor() as cur:
    cur.execute("""
        SELECT s.name, s.config_hash, r.compute_seconds, length(r.scene_render_png) AS png_bytes
        FROM scene_configs s
        JOIN cached_renders r ON r.config_hash = s.config_hash
        WHERE s.is_preset = TRUE
        ORDER BY s.id
    """)
    for row in cur.fetchall():
        print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Deployment checklist
# MAGIC
# MAGIC The Lakebase cache is populated. Now wire the app up:
# MAGIC
# MAGIC 1. **Upload the app source** in `App/rf-digital-twin-app/` to your workspace
# MAGIC    (e.g. `/Workspace/Users/<you>/rf-digital-twin-app`).
# MAGIC 2. In **Compute → Apps**, create a new app pointing at that directory.
# MAGIC 3. Add resources (Secrets) matching the names referenced in `app.yaml`:
# MAGIC    - `lakebase-host`      → printed below
# MAGIC    - `lakebase-port`      → `5432`
# MAGIC    - `lakebase-database`  → `rf_digital_twin`
# MAGIC    - `lakebase-user`      → app service principal's Postgres user
# MAGIC    - `lakebase-password`  → OAuth token from
# MAGIC      `WorkspaceClient().database.generate_database_credential(...)`
# MAGIC      (rotate periodically, or have the app generate its own at startup —
# MAGIC      easier path is to set `PGUSER`/`PGPASSWORD` to a service-principal
# MAGIC      Postgres role provisioned on the Lakebase instance).
# MAGIC    - `sionna-job-id`      → optional, only if you wire up the live job
# MAGIC 4. **Grant the app's service principal** `CAN USE` on the Lakebase
# MAGIC    instance so it can connect.
# MAGIC 5. Deploy the app. On first load it will pull Config 1 from the cache
# MAGIC    instantly; clicking "Load Config 2" + Render does the same for Config 2.

# COMMAND ----------

print("=" * 70)
print("Connection details for the Databricks Apps resource binding:")
print("=" * 70)
print(f"  PGHOST     = {PG_HOST}")
print(f"  PGPORT     = 5432")
print(f"  PGDATABASE = {LAKEBASE_DB}")
print(f"  PGSSLMODE  = require")
print(f"  PGUSER     = <app service principal application_id>")
print(f"  PGPASSWORD = <OAuth token from generate_database_credential>")
print()
print("UC source table for cell configs (edit here to change defaults):")
print(f"  {CELL_TABLE}")
print()
print("Lakebase tables created:")
print("  scene_configs, cell_configs, cached_renders, compute_jobs")
