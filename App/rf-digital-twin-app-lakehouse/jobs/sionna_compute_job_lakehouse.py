# Databricks notebook source
# MAGIC %md
# MAGIC # Sionna compute job — Lakehouse variant
# MAGIC
# MAGIC Triggered by `rf-digital-twin-lh` on cache miss. Receives
# MAGIC `config_hash`, `scene_json`, `cells_json` as widgets, renders Sionna
# MAGIC RT, and writes the results into the Delta `cached_renders` table
# MAGIC (and `compute_jobs` for status tracking).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade drjit mitsuba sionna-rt "databricks-sdk>=0.55.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import hashlib
import io
import json
import os
import sys
import time
from typing import Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, BooleanType, BinaryType,
)
from pyspark.sql.functions import current_timestamp

dbutils.widgets.text("config_hash", "")
dbutils.widgets.text("scene_json", "")
dbutils.widgets.text("cells_json", "")

config_hash = dbutils.widgets.get("config_hash")
scene_cfg   = json.loads(dbutils.widgets.get("scene_json"))
cells       = json.loads(dbutils.widgets.get("cells_json"))

assert config_hash, "config_hash widget is empty"
print(f"config_hash = {config_hash}")
print(f"scene       = {scene_cfg}")
print(f"cells       = {len(cells)} TXs")

CATALOG   = "cmegdemos_catalog"
SCHEMA    = "sionna_rf_data"
FQ        = f"{CATALOG}.{SCHEMA}"
T_RENDERS = f"{FQ}.cached_renders"
T_JOBS    = f"{FQ}.compute_jobs"

# COMMAND ----------

# ---------------------------------------------------------------------------
# Sionna pipeline — same as the setup notebook.
# ---------------------------------------------------------------------------

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
    scene.tx_array = PlanarArray(num_rows=int(scene_cfg["num_rows_tx"]),
                                 num_cols=int(scene_cfg["num_cols_tx"]),
                                 pattern=scene_cfg["pattern"],
                                 polarization=scene_cfg["polarization"])
    scene.rx_array = PlanarArray(num_rows=int(scene_cfg["num_rows_rx"]),
                                 num_cols=int(scene_cfg["num_cols_rx"]),
                                 pattern=scene_cfg["pattern"],
                                 polarization=scene_cfg["polarization"])
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
    radio_map = rm_solver(
        scene,
        max_depth=int(scene_cfg["max_depth"]),
        samples_per_tx=int(scene_cfg["samples_per_tx"]),
        cell_size=(float(scene_cfg["cell_size_x"]),
                   float(scene_cfg["cell_size_y"])),
    )
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

# ---------------------------------------------------------------------------
# Lakehouse write helpers
# ---------------------------------------------------------------------------

def write_render_to_lakehouse(config_hash, results):
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


def set_job_status(config_hash, status, run_id=None, error_message=None):
    spark.sql(f"DELETE FROM {T_JOBS} WHERE config_hash = '{config_hash}'")
    row = (config_hash, status, run_id, error_message)
    schema = StructType([
        StructField("config_hash", StringType(), False),
        StructField("status",      StringType(), True),
        StructField("run_id",      LongType(),   True),
        StructField("error_message", StringType(), True),
    ])
    (spark.createDataFrame([row], schema=schema)
        .withColumn("submitted_at", current_timestamp())
        .withColumn("updated_at",   current_timestamp())
        .write.mode("append").saveAsTable(T_JOBS))

# COMMAND ----------

try:
    set_job_status(config_hash, "RUNNING")
    results = run_simulation(scene_cfg, cells)
    write_render_to_lakehouse(config_hash, results)
    set_job_status(config_hash, "SUCCEEDED")
    print(f"Done in {results['compute_seconds']:.1f}s")
except Exception as e:
    set_job_status(config_hash, "FAILED", error_message=str(e))
    raise
