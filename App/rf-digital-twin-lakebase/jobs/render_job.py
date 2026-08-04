# Databricks notebook source
# MAGIC %md
# MAGIC # RF Digital Twin — Sionna render job
# MAGIC
# MAGIC Renders Sionna RT radio maps into Lakebase, keyed by `config_hash`. Three modes:
# MAGIC
# MAGIC - **`custom`** — what the app triggers on a cache miss. Takes the knobs the user set
# MAGIC   plus the `config_hash` the app computed, rebuilds the identical tower list via
# MAGIC   `scene_spec.resolve`, and **asserts the hash matches** before rendering. That
# MAGIC   assertion is what stops the app and the job from silently drifting apart and caching
# MAGIC   rows nothing will ever read.
# MAGIC - **`stories`** — the curated S1–S7 gallery over a neighborhood's core tile. This is
# MAGIC   what seeds the instant-load presets.
# MAGIC - **`coverage`** — batch-renders a neighborhood's coverage tiles (backfill; no UI).
# MAGIC
# MAGIC Parameters (notebook widgets):
# MAGIC
# MAGIC | name             | default    | description                                        |
# MAGIC | ---------------- | ---------- | -------------------------------------------------- |
# MAGIC | `mode`           | `coverage` | `custom` \| `stories` \| `coverage`                |
# MAGIC | `config_hash`    | ``         | **custom**: hash the app is polling for            |
# MAGIC | `scene_json`     | ``         | **custom**: the story/knobs dict, JSON             |
# MAGIC | `neighborhood`   | `Downtown` | neighborhood name (see `neighborhoods.py`)         |
# MAGIC | `batch_index`    | `0`        | which batch of tiles to render (coverage mode)     |
# MAGIC | `n_batches`      | `1`        | total batches (coverage mode)                      |
# MAGIC | `tiles_per_batch`| ``         | optional explicit batch size (overrides n_batches) |
# MAGIC | `force`          | `false`    | re-render even if already cached                   |
# MAGIC
# MAGIC Must run on a GPU cluster with `drjit`, `mitsuba`, `sionna-rt`, plus `requests`,
# MAGIC `shapely`, `trimesh`, `mapbox_earcut` (OSM scenes) and `psycopg[binary]`. Needs
# MAGIC `SEATTLE_SQL_WAREHOUSE_ID` set (the tower table's geometry column blocks Spark) and
# MAGIC `PG_SCHEMA` matching the app's schema.

# COMMAND ----------

# MAGIC %pip install --quiet drjit mitsuba sionna-rt "psycopg[binary]>=3.1.18" \
# MAGIC   "databricks-sdk>=0.55.0" requests shapely trimesh mapbox_earcut "numpy<2"
# MAGIC dbutils.library.restartPython()
#
# Install these here, NOT as job/cluster libraries. Cluster libraries are installed before the
# Python kernel starts, and several of these resolve numpy 2, which breaks DBR 16.4's numpy-1
# ABI — the run then dies at "Failure starting repl" with "numpy.dtype size changed, Expected
# 96 from C header, got 88" before a single line of this notebook executes. Installed here,
# after the kernel is up, the pin holds and restartPython() picks it all up cleanly.

# COMMAND ----------

import os
import sys

dbutils.widgets.text("mode", "coverage")
dbutils.widgets.text("config_hash", "")
dbutils.widgets.text("scene_json", "")
dbutils.widgets.text("neighborhood", "Downtown")
dbutils.widgets.text("batch_index", "0")
dbutils.widgets.text("n_batches", "1")
dbutils.widgets.text("tiles_per_batch", "")
dbutils.widgets.text("force", "false")

mode = dbutils.widgets.get("mode").strip()
config_hash = dbutils.widgets.get("config_hash").strip()
scene_json = dbutils.widgets.get("scene_json")
neighborhood = dbutils.widgets.get("neighborhood")
batch_index = int(dbutils.widgets.get("batch_index") or "0")
n_batches = int(dbutils.widgets.get("n_batches") or "1")
_tpb = dbutils.widgets.get("tiles_per_batch")
tiles_per_batch = int(_tpb) if _tpb.strip() else None
force = dbutils.widgets.get("force").strip().lower() in ("true", "1", "yes")

print(f"mode={mode} neighborhood={neighborhood} batch={batch_index}/{n_batches} "
      f"tpb={tiles_per_batch} force={force} config_hash={config_hash[:12] or '—'}")

# COMMAND ----------

# App source is mounted next to this notebook (jobs/ is one level under the app dir).
APP_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import lakebase_client as lb
import render_pipeline as rp

lb.init_schema()

# COMMAND ----------

import json

if mode == "custom":
    # One off-menu config the app is waiting on. render_custom owns its own status
    # bookkeeping (including FAILED), so the app's poller sees the outcome either way.
    if not config_hash or not scene_json.strip():
        raise ValueError("mode=custom requires both config_hash and scene_json.")
    result = rp.render_custom(scene_json, config_hash)
    print("Done —", result)
    dbutils.notebook.exit(json.dumps({"mode": mode, **result}))

try:
    if mode == "stories":
        summary = rp.render_stories(spark, neighborhood, force=force)
    else:
        summary = rp.render_coverage(
            spark, neighborhood,
            batch_index=batch_index, n_batches=n_batches, tiles_per_batch=tiles_per_batch,
            force=force,
        )
except Exception as e:
    lb.upsert_neighborhood(neighborhood, status="FAILED", error_message=str(e))
    raise

print(f"Done — {len([s for s in summary if s.get('status') == 'rendered'])} rendered, "
      f"{len([s for s in summary if s.get('status') == 'skipped'])} skipped, "
      f"{len([s for s in summary if s.get('status') == 'FAILED'])} failed.")
for s in summary:
    print(" ", s)

# Authoritative verification: what's actually cached for this neighborhood now.
# Outside the try so a reporting glitch can't flip the neighborhood status to FAILED.
cached = lb.list_neighborhood_renders(neighborhood)
verify = {
    "neighborhood": neighborhood,
    "mode": mode,
    "rendered": len([s for s in summary if s.get("status") == "rendered"]),
    "skipped": len([s for s in summary if s.get("status") == "skipped"]),
    "failed": len([s for s in summary if s.get("status") == "FAILED"]),
    "cached_renders": [
        {"name": r["name"], "hash": r["config_hash"][:12],
         "scene_png_bytes": len(r["scene_render_png"]) if r.get("scene_render_png") else 0}
        for r in cached
    ],
    "fail_details": [s for s in summary if s.get("status") == "FAILED"],
}
dbutils.notebook.exit(json.dumps(verify))
