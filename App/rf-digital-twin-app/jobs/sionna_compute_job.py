# Databricks notebook source
# MAGIC %md
# MAGIC # Sionna RT compute job
# MAGIC
# MAGIC Triggered by the RF Digital Twin app when a user submits a custom
# MAGIC configuration that is not in the Lakebase cache.
# MAGIC
# MAGIC Parameters (notebook widgets):
# MAGIC
# MAGIC | name        | description                                  |
# MAGIC | ----------- | -------------------------------------------- |
# MAGIC | config_hash | sha256 hash to write back into the cache row |
# MAGIC | scene_json  | JSON of scene-level config                   |
# MAGIC | cells_json  | JSON of cell list                            |
# MAGIC
# MAGIC The job must run on a GPU cluster with `drjit`, `mitsuba`, and
# MAGIC `sionna-rt` available, and have Lakebase connection env vars set.

# COMMAND ----------

# MAGIC %pip install drjit mitsuba sionna-rt psycopg[binary]

# COMMAND ----------

import json
import os
import sys

dbutils.widgets.text("config_hash", "")
dbutils.widgets.text("scene_json", "")
dbutils.widgets.text("cells_json", "")

config_hash = dbutils.widgets.get("config_hash")
scene_cfg = json.loads(dbutils.widgets.get("scene_json"))
cells = json.loads(dbutils.widgets.get("cells_json"))

assert config_hash, "config_hash widget is empty"
print(f"config_hash = {config_hash}")
print(f"scene_cfg   = {scene_cfg}")
print(f"cells       = {len(cells)} TXs")

# COMMAND ----------

# App source is mounted next to this notebook; make it importable.
APP_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import lakebase_client as lb
from sionna_compute import run_simulation

# COMMAND ----------

try:
    lb.set_job_status(config_hash, "RUNNING")
    results = run_simulation(scene_cfg, cells)
    lb.write_render(config_hash, results)
    lb.set_job_status(config_hash, "SUCCEEDED")
    print(f"Done in {results['compute_seconds']:.1f}s")
except Exception as e:
    lb.set_job_status(config_hash, "FAILED", error_message=str(e))
    raise
