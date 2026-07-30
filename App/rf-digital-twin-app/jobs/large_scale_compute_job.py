# Databricks notebook source
# MAGIC %md
# MAGIC # Large-scale radio-map compute job
# MAGIC
# MAGIC Triggered by the RF Digital Twin app's "Large-scale map" tab when a
# MAGIC region has no cached render. Runs the NVIDIA
# MAGIC [`sionna-large-radio-maps`](https://github.com/NVlabs/sionna-large-radio-maps)
# MAGIC pipeline (tiling → OSM scene build → per-tile ray tracing), mosaics the
# MAGIC per-tile arrays, and writes the coverage/tiling/CDF PNGs + KPIs into the
# MAGIC Lakebase `large_scale_maps` cache.
# MAGIC
# MAGIC Parameters (notebook widgets):
# MAGIC
# MAGIC | name        | description                                     |
# MAGIC | ----------- | ----------------------------------------------- |
# MAGIC | region_hash | hash to write back into the cache row           |
# MAGIC | region_json | JSON of the region config (bbox + radio params) |
# MAGIC
# MAGIC Must run on a **GPU cluster** (RTX cores preferred, e.g. L40S) with
# MAGIC internet access to OpenStreetMap/Overpass and Lakebase env vars set.
# MAGIC
# MAGIC **Dependency isolation (important):** this notebook has NO `%pip` cell.
# MAGIC A `%pip install` restarts the Python REPL, and on DBR 16.4 that restart
# MAGIC intermittently fails while re-importing the runtime's pandas/numpy
# MAGIC ("numpy.dtype size changed" → "Failure starting repl"). So:
# MAGIC   * The two light packages the KERNEL imports (psycopg, databricks-sdk —
# MAGIC     neither pulls numpy) are attached as **cluster libraries** on the job
# MAGIC     (same pattern as the working etoile compute job). numpy/matplotlib
# MAGIC     ship with the runtime.
# MAGIC   * The heavy Sionna RT + geo stack
# MAGIC     (sionna-rt/mitsuba/drjit/geopandas/shapely/rasterio) is installed into
# MAGIC     an ISOLATED venv at runtime by
# MAGIC     `large_scale_compute._ensure_subproc_python` and only ever runs in
# MAGIC     subprocesses — it never touches the kernel's site-packages.

# COMMAND ----------

import json
import os
import subprocess
import sys

dbutils.widgets.text("region_hash", "")
dbutils.widgets.text("region_json", "")

region_hash = dbutils.widgets.get("region_hash")
region = json.loads(dbutils.widgets.get("region_json"))
assert region_hash, "region_hash widget is empty"
print(f"region_hash = {region_hash}")
print(f"region      = {region}")

# COMMAND ----------

# Clone the NVlabs pipeline if it isn't already present, and point the app's
# large_scale_compute at it via SIONNA_LRM_REPO.
REPO_DIR = "/tmp/sionna-large-radio-maps"
if not os.path.isdir(os.path.join(REPO_DIR, "scripts")):
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/NVlabs/sionna-large-radio-maps.git", REPO_DIR],
        check=True,
    )
os.environ["SIONNA_LRM_REPO"] = REPO_DIR

# App source is mounted next to this notebook; make it importable.
APP_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import lakebase_client as lb
from large_scale_compute import run_large_scale

# COMMAND ----------

try:
    # allow_demo=False: this job exists to produce *real* Sionna RT output.
    results = run_large_scale(region, allow_demo=False)
    lb.write_large_scale_map(region_hash, region, results)
    print(f"Done in {results['compute_seconds']:.1f}s (is_demo={results['is_demo']})")
except Exception as e:
    print(f"Large-scale job failed: {e}")
    raise
