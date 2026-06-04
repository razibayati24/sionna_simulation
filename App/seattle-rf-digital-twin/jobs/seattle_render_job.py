# Databricks notebook source
# MAGIC %md
# MAGIC # Seattle RF — neighborhood render job
# MAGIC
# MAGIC Triggered by the Seattle RF Digital Twin app when a user picks a neighborhood that
# MAGIC isn't cached yet. Renders that neighborhood's coverage tiles into Lakebase and flips
# MAGIC its status to `CACHED`. Also used by the setup notebook for batch backfills.
# MAGIC
# MAGIC Parameters (notebook widgets):
# MAGIC
# MAGIC | name             | default    | description                                        |
# MAGIC | ---------------- | ---------- | -------------------------------------------------- |
# MAGIC | `neighborhood`   | `Downtown` | neighborhood name (see `neighborhoods.py`)         |
# MAGIC | `mode`           | `coverage` | `coverage` (tiles) or `stories` (curated gallery)  |
# MAGIC | `batch_index`    | `0`        | which batch of tiles to render (coverage mode)     |
# MAGIC | `n_batches`      | `1`        | total batches (coverage mode)                      |
# MAGIC | `tiles_per_batch`| ``         | optional explicit batch size (overrides n_batches) |
# MAGIC
# MAGIC Must run on a GPU cluster with `drjit`, `mitsuba`, `sionna-rt`, plus `requests`,
# MAGIC `shapely`, `trimesh` (OSM scenes) and `psycopg[binary]`, with Lakebase env vars set.

# COMMAND ----------

# MAGIC %pip install --quiet drjit mitsuba sionna-rt "psycopg[binary]>=3.1.18" \
# MAGIC   "databricks-sdk>=0.55.0" requests shapely trimesh mapbox_earcut
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

dbutils.widgets.text("neighborhood", "Downtown")
dbutils.widgets.text("mode", "coverage")
dbutils.widgets.text("batch_index", "0")
dbutils.widgets.text("n_batches", "1")
dbutils.widgets.text("tiles_per_batch", "")
dbutils.widgets.text("force", "false")

neighborhood = dbutils.widgets.get("neighborhood")
mode = dbutils.widgets.get("mode")
batch_index = int(dbutils.widgets.get("batch_index") or "0")
n_batches = int(dbutils.widgets.get("n_batches") or "1")
_tpb = dbutils.widgets.get("tiles_per_batch")
tiles_per_batch = int(_tpb) if _tpb.strip() else None
force = dbutils.widgets.get("force").strip().lower() in ("true", "1", "yes")

print(f"neighborhood={neighborhood} mode={mode} batch={batch_index}/{n_batches} "
      f"tpb={tiles_per_batch} force={force}")

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
