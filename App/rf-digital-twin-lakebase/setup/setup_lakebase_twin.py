# Databricks notebook source
# MAGIC %md
# MAGIC # RF Digital Twin — Workspace Setup
# MAGIC
# MAGIC One-shot setup for the RF Digital Twin app: seeds the Lakebase cache so the app's presets
# MAGIC load instantly. The simulation is driven from **real T-Mobile towers** in
# MAGIC `cmegdemos_catalog.network_analytics_enablement.cell_towers` (2,312 towers across the
# MAGIC Seattle metro) on **OpenStreetMap building geometry**.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Installs Sionna RT + OSM scene deps.
# MAGIC 2. Initialises the Lakebase schema on the **`rf-digital-twin-pg`** instance (set
# MAGIC    `PG_SCHEMA` to match the app's — default `lakebase_only`).
# MAGIC 3. Registers the metro neighborhoods (`neighborhoods.py`).
# MAGIC 4. **Calibrates** one Downtown tile to size batch jobs to ~20–30 min.
# MAGIC 5. Renders the **Downtown** preset gallery (S1–S7) + coverage tiles into Lakebase.
# MAGIC 6. Prints the deployment checklist.
# MAGIC
# MAGIC Afterwards, run `tools/check_hash_parity.py` to confirm the app will actually hit these
# MAGIC rows — that's the check that keeps the "instant" demo path honest.
# MAGIC
# MAGIC ## Cluster
# MAGIC Sionna RT needs NVIDIA OptiX. Validated: **DBR 16.4 LTS (non-ML), driver+worker
# MAGIC `g5.xlarge` (A10G), single-user**. A CPU cluster fails with
# MAGIC `libnvoptix.so.1 could not be loaded`. The cluster also needs **internet egress** for the
# MAGIC OSM Overpass API (a flat-ground fallback covers tiles where it fails), and
# MAGIC `SEATTLE_SQL_WAREHOUSE_ID` set so the geospatial tower table can be read.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade drjit mitsuba sionna-rt \
# MAGIC   "psycopg[binary]>=3.1.18" "databricks-sdk>=0.55.0" requests shapely trimesh \
# MAGIC   mapbox_earcut "numpy<2"
# MAGIC dbutils.library.restartPython()
#
# Keep numpy pinned below 2 and install here rather than as cluster libraries: DBR 16.4 ships
# numpy 1, and a cluster-library install of these deps pulls numpy 2 in before the Python
# kernel starts, killing the run at "Failure starting repl".

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

# The app mints OAuth tokens for this instance at runtime.
os.environ.setdefault("LAKEBASE_INSTANCE", "rf-digital-twin-pg")
os.environ.setdefault("PGDATABASE", "rf_digital_twin")
# Must match the app's PG_SCHEMA, or the app will look for these renders in the wrong schema.
os.environ.setdefault("PG_SCHEMA", "lakebase_only")
# The tower table's geometry column can't be read by Spark on DBR 16.4, so towers.py goes
# through a SQL warehouse. Leave unset to auto-pick a running serverless one.
# os.environ.setdefault("SEATTLE_SQL_WAREHOUSE_ID", "<warehouse id>")

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

# Read via the SQL warehouse — the table's geometry column blocks Spark on DBR 16.4.
import pandas as pd

hood = nb.get("Downtown")
_rows = rp.towers._query_via_warehouse(f"""
    SELECT tower_type, COUNT(*) AS n, ROUND(AVG(coverage_radius_m)) AS avg_radius_m
    FROM {rp.towers.SOURCE_TABLE}
    WHERE {hood.sql_bbox_filter()}
    GROUP BY tower_type ORDER BY n DESC
""")
display(pd.DataFrame(_rows))

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
# MAGIC Loops over every calibrated batch. For a one-off setup we render all batches inline; for
# MAGIC bigger backfills run `jobs/render_job.py` with `mode=coverage` per batch instead.

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
# MAGIC 1. Get `App/rf-digital-twin-lakebase/` into the workspace (a Git folder on this branch is
# MAGIC    easiest — the app and the job then stay in sync automatically).
# MAGIC 2. Create the GPU job from `jobs/render_job.py` (g5.xlarge, DBR 16.4 + the deps above,
# MAGIC    plus `SEATTLE_SQL_WAREHOUSE_ID` and `PG_SCHEMA` as spark env vars). Put its `job_id`
# MAGIC    into `app.yaml` as `SIONNA_RENDER_JOB_ID`.
# MAGIC 3. Create the Databricks App pointing at the app dir; bind the `rf-digital-twin-pg`
# MAGIC    Lakebase database (populates PGHOST/PGPORT/PGDATABASE/PGUSER).
# MAGIC 4. Grant the app's service principal: `CAN MANAGE RUN` on the render job, `CAN_USE` on the
# MAGIC    SQL warehouse, and `SELECT` on the tower table plus `USE_CATALOG`/`USE_SCHEMA` on its
# MAGIC    parents. Without that last one the tower read 403s and **no config can be hashed**.
# MAGIC 5. Run `tools/check_hash_parity.py`. Every preset must resolve to a cached render.
# MAGIC 6. Deploy. Presets load from cache instantly; any off-menu config triggers the render job
# MAGIC    and appears automatically when it lands.

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
