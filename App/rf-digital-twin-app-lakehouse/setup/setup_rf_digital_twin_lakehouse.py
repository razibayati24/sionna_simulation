# Databricks notebook source
# MAGIC %md
# MAGIC # RF Digital Twin — Lakehouse setup
# MAGIC
# MAGIC Variant of `setup_rf_digital_twin.py` that uses Unity Catalog Delta
# MAGIC tables instead of Lakebase Postgres. Run once on a GPU cluster.
# MAGIC
# MAGIC What it does:
# MAGIC
# MAGIC 1. Creates the `cmegdemos_catalog.sionna_rf_data` schema if needed.
# MAGIC 2. Creates Delta tables: `scene_configs`, `cell_configs`,
# MAGIC    `cached_renders`, `compute_jobs` (UC-managed).
# MAGIC 3. (Optional) Migrates already-rendered presets from the Lakebase
# MAGIC    cache so you don't have to re-run Sionna. Skips if Lakebase
# MAGIC    isn't configured.
# MAGIC 4. Renders any preset that's still missing through Sionna RT and
# MAGIC    writes the results into the Delta `cached_renders` table.
# MAGIC 5. Prints the cheat sheet of hashes per preset.
# MAGIC
# MAGIC ## Cluster
# MAGIC
# MAGIC Same as the Lakebase variant: **DBR 16.4 LTS** + `g5.xlarge`
# MAGIC (1× NVIDIA A10G), Standard runtime. CPU clusters can't run Sionna
# MAGIC (missing OptiX). See the Lakebase setup notebook header for the full
# MAGIC cluster spec.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade \
# MAGIC   drjit mitsuba sionna-rt \
# MAGIC   "psycopg[binary]>=3.1.18" \
# MAGIC   "databricks-sdk>=0.55.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

CATALOG = "cmegdemos_catalog"
SCHEMA  = "sionna_rf_data"
FQ      = f"{CATALOG}.{SCHEMA}"

T_SCENE   = f"{FQ}.scene_configs"
T_CELLS   = f"{FQ}.cell_configs"
T_RENDERS = f"{FQ}.cached_renders"
T_JOBS    = f"{FQ}.compute_jobs"

SAMPLES_PER_TX = 10 ** 7

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Schema + tables

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T_SCENE} (
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
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T_CELLS} (
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
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T_RENDERS} (
    config_hash       STRING NOT NULL,
    scene_render_png  BINARY,
    sinr_map_png      BINARY,
    association_png   BINARY,
    sinr_cdf_png      BINARY,
    rss_cdf_png       BINARY,
    kpis_json         STRING,
    compute_seconds   DOUBLE,
    created_at        TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T_JOBS} (
    config_hash    STRING NOT NULL,
    status         STRING,
    run_id         BIGINT,
    error_message  STRING,
    submitted_at   TIMESTAMP,
    updated_at     TIMESTAMP
) USING DELTA
""")

print(f"Schema + 4 Delta tables ready under {FQ}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Default 7-cell layout + scene preset definitions
# MAGIC
# MAGIC Identical to the Lakebase variant. Edit `DEFAULT_CELLS` to change
# MAGIC the network. Edit `PRESETS` to add or remove cached configurations.

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
    cell_power_override_dbm: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def cells_for_preset(cfg, base_cells):
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

# Story B — Frequency band ladder
CONFIG_FREQ_1P8G = SceneConfig("B · Frequency — 8×2 @ 1.8 GHz",  8, 2, 2, 2, frequency_hz=1.8e9,  bandwidth_hz=2e7)
CONFIG_FREQ_2P6G = SceneConfig("B · Frequency — 8×2 @ 2.6 GHz",  8, 2, 2, 2, frequency_hz=2.6e9,  bandwidth_hz=2e7)
CONFIG_FREQ_3P5G = SceneConfig("B · Frequency — 8×2 @ 3.5 GHz",  8, 2, 2, 2, frequency_hz=3.5e9,  bandwidth_hz=1e8)
CONFIG_FREQ_39G  = SceneConfig("B · Frequency — 8×2 @ 39 GHz",   8, 2, 2, 2, frequency_hz=3.9e10, bandwidth_hz=4e8)

# Story C — Antenna pattern
CONFIG_PAT_ISO    = SceneConfig("C · Pattern — 16×16 isotropic", 16, 16, 2, 2, pattern="iso")
CONFIG_PAT_DIPOLE = SceneConfig("C · Pattern — 16×16 dipole",    16, 16, 2, 2, pattern="dipole")

# Story D — Polarization
CONFIG_POL_VH = SceneConfig("D · Polarization — 16×16 cross (VH)", 16, 16, 2, 2, polarization="VH")

# Story E — Power
CONFIG_PWR_LOW  = SceneConfig("E · Power — 16×16 @ 38 dBm", 16, 16, 2, 2, cell_power_override_dbm=38.0)
CONFIG_PWR_HIGH = SceneConfig("E · Power — 16×16 @ 50 dBm", 16, 16, 2, 2, cell_power_override_dbm=50.0)

# Story F — Bandwidth
CONFIG_BW_20M  = SceneConfig("F · Bandwidth — 16×16 @ 20 MHz",  16, 16, 2, 2, bandwidth_hz=2e7)
CONFIG_BW_400M = SceneConfig("F · Bandwidth — 16×16 @ 400 MHz", 16, 16, 2, 2, bandwidth_hz=4e8)

# Story G — Ray tracing fidelity
CONFIG_DEPTH_3 = SceneConfig("G · Fidelity — 16×16 max_depth=3", 16, 16, 2, 2, max_depth=3)
CONFIG_DEPTH_8 = SceneConfig("G · Fidelity — 16×16 max_depth=8", 16, 16, 2, 2, max_depth=8)

PRESETS = [
    CONFIG_DENS_2X2, CONFIG_DENS_4X4, CONFIG_1, CONFIG_DENS_8X8, CONFIG_2, CONFIG_DENS_32X8,
    CONFIG_FREQ_1P8G, CONFIG_FREQ_2P6G, CONFIG_FREQ_3P5G, CONFIG_FREQ_39G,
    CONFIG_PAT_ISO, CONFIG_PAT_DIPOLE,
    CONFIG_POL_VH,
    CONFIG_PWR_LOW, CONFIG_PWR_HIGH,
    CONFIG_BW_20M, CONFIG_BW_400M,
    CONFIG_DEPTH_3, CONFIG_DEPTH_8,
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Hashing helpers (identical to lakehouse_client)

# COMMAND ----------

_HASH_SCENE_FIELDS = (
    "num_rows_tx", "num_cols_tx", "num_rows_rx", "num_cols_rx",
    "frequency_hz", "bandwidth_hz", "max_depth", "samples_per_tx",
    "cell_size_x", "cell_size_y", "pattern", "polarization",
    "num_user_samples", "min_sinr_db", "min_user_dist_m", "max_user_dist_m",
)
_HASH_CELL_FIELDS = (
    "cell_id", "x", "y", "z", "look_at_x", "look_at_y", "look_at_z", "power_dbm",
)


def compute_config_hash(scene, cells):
    payload = {
        "scene": {k: scene[k] for k in _HASH_SCENE_FIELDS},
        "cells": [{k: c[k] for k in _HASH_CELL_FIELDS}
                  for c in sorted(cells, key=lambda c: c["cell_id"])],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delta write helpers (via Spark)

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, BooleanType, BinaryType, TimestampType,
)
from pyspark.sql.functions import current_timestamp


def upsert_scene_into_lakehouse(scene, cells, is_preset=True):
    config_hash = compute_config_hash(scene, cells)

    # scene_configs
    scene_row = {
        "config_hash": config_hash,
        "name": scene["name"],
        "num_rows_tx": int(scene["num_rows_tx"]),
        "num_cols_tx": int(scene["num_cols_tx"]),
        "num_rows_rx": int(scene["num_rows_rx"]),
        "num_cols_rx": int(scene["num_cols_rx"]),
        "frequency_hz": float(scene["frequency_hz"]),
        "bandwidth_hz": float(scene["bandwidth_hz"]),
        "max_depth": int(scene["max_depth"]),
        "samples_per_tx": int(scene["samples_per_tx"]),
        "cell_size_x": float(scene["cell_size_x"]),
        "cell_size_y": float(scene["cell_size_y"]),
        "pattern": scene["pattern"],
        "polarization": scene["polarization"],
        "num_user_samples": int(scene["num_user_samples"]),
        "min_sinr_db": float(scene["min_sinr_db"]),
        "min_user_dist_m": float(scene["min_user_dist_m"]),
        "max_user_dist_m": float(scene["max_user_dist_m"]),
        "is_preset": bool(is_preset),
    }
    spark.sql(f"DELETE FROM {T_SCENE} WHERE config_hash = '{config_hash}'")
    (spark.createDataFrame([scene_row])
        .withColumn("created_at", current_timestamp())
        .write.mode("append").saveAsTable(T_SCENE))

    spark.sql(f"DELETE FROM {T_CELLS} WHERE config_hash = '{config_hash}'")
    cell_rows = [{**c, "config_hash": config_hash} for c in cells]
    spark.createDataFrame(cell_rows).write.mode("append").saveAsTable(T_CELLS)

    return config_hash


def write_render_into_lakehouse(config_hash, results):
    schema = StructType([
        StructField("config_hash",      StringType(), False),
        StructField("scene_render_png", BinaryType(), True),
        StructField("sinr_map_png",     BinaryType(), True),
        StructField("association_png",  BinaryType(), True),
        StructField("sinr_cdf_png",     BinaryType(), True),
        StructField("rss_cdf_png",      BinaryType(), True),
        StructField("kpis_json",        StringType(), True),
        StructField("compute_seconds",  DoubleType(), True),
    ])
    row = (
        config_hash,
        bytes(results["scene_render_png"]) if results.get("scene_render_png") else None,
        bytes(results["sinr_map_png"])      if results.get("sinr_map_png")      else None,
        bytes(results["association_png"])   if results.get("association_png")   else None,
        bytes(results["sinr_cdf_png"])      if results.get("sinr_cdf_png")      else None,
        bytes(results["rss_cdf_png"])       if results.get("rss_cdf_png")       else None,
        results.get("kpis_json"),
        float(results.get("compute_seconds") or 0.0),
    )
    spark.sql(f"DELETE FROM {T_RENDERS} WHERE config_hash = '{config_hash}'")
    (spark.createDataFrame([row], schema=schema)
        .withColumn("created_at", current_timestamp())
        .write.mode("append").saveAsTable(T_RENDERS))


def is_already_cached(config_hash):
    n = spark.sql(
        f"SELECT length(scene_render_png) AS n FROM {T_RENDERS} "
        f"WHERE config_hash = '{config_hash}'"
    ).collect()
    return bool(n) and n[0]["n"] and n[0]["n"] > 10_000

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. (Optional) Migrate already-rendered presets from Lakebase
# MAGIC
# MAGIC If you ran the Lakebase variant first, every preset is already
# MAGIC rendered as Postgres bytea. This cell copies those rows into the
# MAGIC Delta tables — no re-render needed. Skip if you never set up Lakebase.

# COMMAND ----------

LAKEBASE_INSTANCE = "rf-digital-twin-pg"
LAKEBASE_DB       = "rf_digital_twin"
RUN_MIGRATION     = True   # set False to skip the migration cell

if RUN_MIGRATION:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from databricks.sdk import WorkspaceClient
        import uuid as _uuid

        w = WorkspaceClient()
        inst = w.database.get_database_instance(name=LAKEBASE_INSTANCE)
        cred = w.database.generate_database_credential(
            request_id=str(_uuid.uuid4()),
            instance_names=[LAKEBASE_INSTANCE],
        )

        with psycopg.connect(
            host=inst.read_write_dns, port=5432,
            dbname=LAKEBASE_DB,
            user=w.current_user.me().user_name,
            password=cred.token,
            sslmode="require",
            row_factory=dict_row,
        ) as conn, conn.cursor() as cur:

            cur.execute("""
                SELECT s.*, r.scene_render_png, r.sinr_map_png, r.association_png,
                       r.sinr_cdf_png, r.rss_cdf_png, r.kpis_json, r.compute_seconds
                FROM scene_configs s
                JOIN cached_renders r ON r.config_hash = s.config_hash
                WHERE s.is_preset = TRUE
            """)
            migrated = 0
            for row in cur:
                # Build scene_cfg + cells dicts in the same shape as the
                # PRESETS loop.
                cur.execute(
                    "SELECT * FROM cell_configs WHERE scene_config_id = %s ORDER BY cell_id",
                    (row["id"],),
                )
                cells = list(cur)
                scene = {k: row[k] for k in (
                    "name", "num_rows_tx", "num_cols_tx", "num_rows_rx", "num_cols_rx",
                    "frequency_hz", "bandwidth_hz", "max_depth", "samples_per_tx",
                    "cell_size_x", "cell_size_y", "pattern", "polarization",
                    "num_user_samples", "min_sinr_db", "min_user_dist_m", "max_user_dist_m",
                )}

                config_hash = upsert_scene_into_lakehouse(scene, cells, is_preset=True)
                # Sanity-check the hash actually matches what Lakebase had:
                if config_hash != row["config_hash"]:
                    print(f"  hash mismatch on {row['name']}: lakebase={row['config_hash'][:12]}, "
                          f"recomputed={config_hash[:12]} — using recomputed")
                results = {
                    "scene_render_png": bytes(row["scene_render_png"]) if row["scene_render_png"] else None,
                    "sinr_map_png":     bytes(row["sinr_map_png"])     if row["sinr_map_png"]     else None,
                    "association_png":  bytes(row["association_png"])  if row["association_png"]  else None,
                    "sinr_cdf_png":     bytes(row["sinr_cdf_png"])     if row["sinr_cdf_png"]     else None,
                    "rss_cdf_png":      bytes(row["rss_cdf_png"])      if row["rss_cdf_png"]      else None,
                    "kpis_json":        json.dumps(row["kpis_json"])   if isinstance(row["kpis_json"], dict) else row["kpis_json"],
                    "compute_seconds":  float(row["compute_seconds"] or 0.0),
                }
                write_render_into_lakehouse(config_hash, results)
                migrated += 1

            print(f"Migrated {migrated} preset(s) from Lakebase → Lakehouse.")
    except Exception as e:
        print(f"Migration skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Sionna RT pipeline
# MAGIC
# MAGIC Identical to the Lakebase variant. Renders the scene, SINR overlay,
# MAGIC user-to-TX association, and SINR/RSS CDFs.

# COMMAND ----------

import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Camera, RadioMapSolver


def _fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _build_scene(scene_cfg, cells):
    scene = load_scene(sionna.rt.scene.etoile)
    scene.frequency = float(scene_cfg["frequency_hz"])
    scene.bandwidth = float(scene_cfg["bandwidth_hz"])
    scene.tx_array = PlanarArray(
        num_rows=int(scene_cfg["num_rows_tx"]),
        num_cols=int(scene_cfg["num_cols_tx"]),
        pattern=scene_cfg["pattern"], polarization=scene_cfg["polarization"],
    )
    scene.rx_array = PlanarArray(
        num_rows=int(scene_cfg["num_rows_rx"]),
        num_cols=int(scene_cfg["num_cols_rx"]),
        pattern=scene_cfg["pattern"], polarization=scene_cfg["polarization"],
    )
    for c in sorted(cells, key=lambda r: r["cell_id"]):
        scene.add(Transmitter(
            name=c["name"],
            position=[float(c["x"]), float(c["y"]), float(c["z"])],
            look_at=[float(c["look_at_x"]), float(c["look_at_y"]), float(c["look_at_z"])],
            power_dbm=float(c["power_dbm"]),
        ))
    return scene


def _scene_render_png(scene, radio_map):
    cam = Camera(position=[0.0, 0.0, 1000.0],
                 orientation=np.array([0.0, np.pi/2, -np.pi/2]))
    fig = scene.render(camera=cam, radio_map=radio_map, rm_metric="sinr",
                       rm_vmin=-10, rm_vmax=60, rm_show_color_bar=True)
    if fig is None:
        fig = plt.gcf()
    return _fig_png(fig)


def _association_png(radio_map, scene_cfg):
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


def _cdf_png(radio_map, metric, xlim):
    plt.close("all")
    radio_map.cdf(metric=metric, bins=400)
    plt.xlim(*xlim)
    plt.title(f"CDF of {metric.upper()}")
    fig = plt.gcf()
    summary = {}
    for line in fig.gca().get_lines():
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs):
            for p in (10, 50, 90):
                idx = min(int(np.searchsorted(ys, p/100.0)), len(xs) - 1)
                summary[f"p{p}"] = float(xs[idx])
            break
    return _fig_png(fig), summary


def run_simulation(scene_cfg, cells):
    t0 = time.time()
    scene = _build_scene(scene_cfg, cells)
    rm_solver = RadioMapSolver()
    radio_map = rm_solver(scene,
                          max_depth=int(scene_cfg["max_depth"]),
                          samples_per_tx=int(scene_cfg["samples_per_tx"]),
                          cell_size=(float(scene_cfg["cell_size_x"]),
                                     float(scene_cfg["cell_size_y"])))
    scene_png = _scene_render_png(scene, radio_map)
    fig = radio_map.show_association("sinr")
    sinr_map_png = _fig_png(fig)
    association_png, assoc_kpis = _association_png(radio_map, scene_cfg)
    sinr_cdf_png, sinr_pct = _cdf_png(radio_map, "sinr", xlim=(-40.0, 75.0))
    rss_cdf_png,  rss_pct  = _cdf_png(radio_map, "rss",  xlim=(-150.0, 25.0))
    kpis = {"sinr_percentiles_db": sinr_pct,
            "rss_percentiles_dbm": rss_pct,
            **assoc_kpis,
            "num_tx": len(cells)}
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
# MAGIC ## 8. Render the gap — only the presets that aren't already cached

# COMMAND ----------

precompute_summary = []

for cfg in PRESETS:
    scene_cfg = cfg.to_dict()
    cells = cells_for_preset(cfg, DEFAULT_CELLS)
    config_hash = compute_config_hash(scene_cfg, cells)

    print(f"\n=== {cfg.name} ===")
    print(f"  config_hash = {config_hash}")

    if is_already_cached(config_hash):
        print("  Already cached — skipping Sionna run.")
        precompute_summary.append({"name": cfg.name, "hash": config_hash, "status": "skipped"})
        continue

    upsert_scene_into_lakehouse(scene_cfg, cells, is_preset=True)
    print(f"  Running Sionna RT (samples_per_tx={SAMPLES_PER_TX:,})…")
    try:
        results = run_simulation(scene_cfg, cells)
        write_render_into_lakehouse(config_hash, results)
        print(f"  Done in {results['compute_seconds']:.1f}s, cached to Lakehouse.")
        precompute_summary.append({
            "name": cfg.name, "hash": config_hash,
            "compute_seconds": results["compute_seconds"],
            "status": "rendered",
        })
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  FAILED — {msg}")
        precompute_summary.append({
            "name": cfg.name, "hash": config_hash, "status": "FAILED", "error": msg,
        })

display(pd.DataFrame(precompute_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verification + cheat sheet
# MAGIC
# MAGIC One row per preset showing the hash, byte sizes, and the sidebar
# MAGIC values an app user types to land on it. Hashes here must match the
# MAGIC Lakebase variant's hashes — they're computed from the same
# MAGIC `(scene, cells)` payload.

# COMMAND ----------

display(spark.sql(f"""
    SELECT s.name,
           substr(s.config_hash, 1, 12) AS hash_prefix,
           COALESCE(length(r.scene_render_png), 0) AS scene_bytes,
           r.compute_seconds,
           r.created_at
    FROM {T_SCENE} s
    LEFT JOIN {T_RENDERS} r ON r.config_hash = s.config_hash
    WHERE s.is_preset = TRUE
    ORDER BY s.name
"""))

# COMMAND ----------

print("APP SIDEBAR CHEAT SHEET — Lakehouse variant")
print("=" * 110)
for cfg in PRESETS:
    scene_cfg = cfg.to_dict()
    cells = cells_for_preset(cfg, DEFAULT_CELLS)
    h = compute_config_hash(scene_cfg, cells)
    pwr = cfg.cell_power_override_dbm if cfg.cell_power_override_dbm is not None else 44.0
    diffs = []
    if (cfg.num_rows_tx, cfg.num_cols_tx) != (8, 2): diffs.append(f"TX={cfg.num_rows_tx}×{cfg.num_cols_tx}")
    if cfg.pattern != "tr38901": diffs.append(f"pattern={cfg.pattern}")
    if cfg.polarization != "V":  diffs.append(f"pol={cfg.polarization}")
    if pwr != 44.0:              diffs.append(f"power={pwr} dBm")
    if cfg.frequency_hz != 28e9: diffs.append(f"freq={cfg.frequency_hz/1e9:g} GHz")
    if cfg.bandwidth_hz != 1e8:  diffs.append(f"BW={cfg.bandwidth_hz/1e6:g} MHz")
    if cfg.max_depth != 5:       diffs.append(f"max_depth={cfg.max_depth}")
    print(f"  {cfg.name:<55}  hash={h[:12]}")
    print(f"      vs Config 1 default → " + (", ".join(diffs) if diffs else "no changes (same as Config 1)"))
print("=" * 110)
