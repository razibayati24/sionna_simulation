# Databricks notebook source
# MAGIC %md
# MAGIC # Seattle RF Digital Twin — Workspace Setup
# MAGIC
# MAGIC One-shot setup for the **Seattle** RF Digital Twin app. Unlike the etoile demo (synthetic
# MAGIC 7-cell ring on the Paris scene), this drives the simulation from **real T-Mobile towers**
# MAGIC in `cmegdemos_catalog.network_analytics_enablement.cell_towers` (2,312 towers across the
# MAGIC Seattle metro) on **OpenStreetMap building geometry**.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Installs Sionna RT + OSM scene deps.
# MAGIC 2. Reuses the existing **`rf-digital-twin-pg`** Lakebase instance and initialises the
# MAGIC    neighborhood-aware schema (adds `neighborhoods` table + neighborhood/tile columns).
# MAGIC 3. Registers the metro neighborhoods (`neighborhoods.py`).
# MAGIC 4. **Calibrates** one Downtown tile to size batch jobs to ~20–30 min.
# MAGIC 5. Renders the **Downtown** curated story gallery + coverage tiles into Lakebase.
# MAGIC 6. Prints the deployment checklist for the new `seattle-rf-digital-twin` app.
# MAGIC
# MAGIC ## Cluster
# MAGIC Same GPU requirement as the etoile demo — Sionna RT needs NVIDIA OptiX. Validated:
# MAGIC **DBR 16.4 LTS (non-ML), driver+worker `g5.xlarge` (A10G), single-user**. A CPU cluster
# MAGIC fails with `libnvoptix.so.1 could not be loaded`. The cluster also needs **internet
# MAGIC egress** for the OSM Overpass API (a flat-ground fallback covers tiles where it fails).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade drjit mitsuba sionna-rt \
# MAGIC   "psycopg[binary]>=3.1.18" "databricks-sdk>=0.55.0" requests shapely trimesh
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Make the app modules importable (this notebook lives in setup/, one level under the app dir).
APP_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from databricks.sdk import WorkspaceClient

import neighborhoods as nb
import lakebase_client as lb
import render_pipeline as rp

w = WorkspaceClient()
print("Authenticated as:", w.current_user.me().user_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# The app mints OAuth tokens for this instance at runtime (shared with the etoile demo).
os.environ.setdefault("LAKEBASE_INSTANCE", "rf-digital-twin-pg")
os.environ.setdefault("PGDATABASE", "rf_digital_twin")

CALIBRATION_TARGET_MIN = 25.0     # aim each coverage batch at ~25 min
RENDER_COVERAGE = True            # also render Downtown's full coverage tiles (not just stories)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Lakebase schema + neighborhood registry

# COMMAND ----------

lb.init_schema()
print("Lakebase schema ready (scene_configs, cell_configs, cached_renders, compute_jobs, neighborhoods).")

for name in nb.names():
    lb.upsert_neighborhood(name, status="NONE")
print(f"Registered {len(nb.names())} neighborhoods: {', '.join(nb.names())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tower sanity check — confirm the Downtown subset

# COMMAND ----------

hood = nb.get("Downtown")
display(spark.sql(f"""
    SELECT tower_type, COUNT(*) AS n, ROUND(AVG(coverage_radius_m)) AS avg_radius_m
    FROM {rp.towers.SOURCE_TABLE}
    WHERE {hood.sql_bbox_filter()}
    GROUP BY tower_type ORDER BY n DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Calibrate — render one Downtown tile, size the batches

# COMMAND ----------

cal = rp.calibrate(spark, "Downtown", target_min=CALIBRATION_TARGET_MIN)
TILES_PER_BATCH = cal["tiles_per_batch"]
print(cal)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Render the Downtown curated story gallery (~7 instant-load renders)

# COMMAND ----------

import pandas as pd

story_summary = rp.render_stories(spark, "Downtown")
display(pd.DataFrame(story_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Render Downtown coverage tiles (batched ~20–30 min each)
# MAGIC
# MAGIC Loops over every calibrated batch. For a one-off setup we render all batches inline; in
# MAGIC production the app triggers `jobs/seattle_render_job.py` per batch on demand.

# COMMAND ----------

if RENDER_COVERAGE:
    n_batches = cal["n_batches"]
    for b in range(n_batches):
        print(f"\n=== Coverage batch {b + 1}/{n_batches} ===")
        rp.render_coverage(spark, "Downtown", batch_index=b, n_batches=n_batches,
                           tiles_per_batch=TILES_PER_BATCH)
    print("\nDowntown coverage complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify — what's cached for Downtown

# COMMAND ----------

rows = lb.list_neighborhood_renders("Downtown")
print(f"Downtown has {len(rows)} cached renders:")
for r in rows:
    png = r.get("scene_render_png")
    print(f"  {r['name']:<55} {r['config_hash'][:12]}  {len(png) if png else 0} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Deployment checklist
# MAGIC
# MAGIC 1. Upload `App/seattle-rf-digital-twin/` to the workspace.
# MAGIC 2. Create the GPU job from `jobs/seattle_render_job.py` (g5.xlarge, DBR 16.4 + the deps
# MAGIC    above). Put its `job_id` into `app.yaml` as `SEATTLE_RENDER_JOB_ID`.
# MAGIC 3. Create the Databricks App `seattle-rf-digital-twin` pointing at the app dir; bind the
# MAGIC    `rf-digital-twin-pg` Lakebase database (populates PGHOST/PGPORT/PGDATABASE/PGUSER).
# MAGIC 4. Grant the app's service principal `CAN USE` on the Lakebase instance and `CAN MANAGE
# MAGIC    RUN` on the render job.
# MAGIC 5. Deploy. The app loads Downtown from cache instantly; picking another neighborhood
# MAGIC    triggers the render job and the results auto-appear when the batches land.

# COMMAND ----------

inst = w.database.get_database_instance(name="rf-digital-twin-pg")
print("=" * 70)
print("App resource binding:")
print(f"  PGHOST     = {inst.read_write_dns}")
print(f"  PGPORT     = 5432")
print(f"  PGDATABASE = rf_digital_twin")
print(f"  PGSSLMODE  = require")
print(f"  PGUSER     = <app service principal application_id>")
print("Source towers:", rp.towers.SOURCE_TABLE)
