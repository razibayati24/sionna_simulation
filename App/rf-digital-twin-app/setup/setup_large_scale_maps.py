# Databricks notebook source
# MAGIC %md
# MAGIC # Large-scale radio maps — setup / seed
# MAGIC
# MAGIC Prepares the **Large-scale map** tab of the RF Digital Twin app:
# MAGIC
# MAGIC 1. Creates the `large_scale_maps` cache table in Lakebase (idempotent).
# MAGIC 2. Seeds a **demo** coverage render for each region preset so the tab
# MAGIC    shows something the moment the app loads — even before a GPU job has
# MAGIC    ever run. Demo rows are clearly flagged (`is_demo = TRUE`) and are
# MAGIC    replaced by real Sionna RT output when a user computes that region
# MAGIC    with `LARGE_SCALE_JOB_ID` configured.
# MAGIC
# MAGIC The seed step is **CPU-only** (log-distance synthetic coverage), so this
# MAGIC notebook runs on any cluster — no GPU required. To precompute *real*
# MAGIC Sionna RT large-scale maps, run `jobs/large_scale_compute_job.py` on a
# MAGIC GPU cluster instead (or let the app submit it on a cache miss).

# COMMAND ----------

# MAGIC %pip install numpy matplotlib psycopg[binary] databricks-sdk

# COMMAND ----------

import os
import sys

# Make the app modules importable (this notebook lives in App/.../setup).
APP_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import lakebase_client as lb
from large_scale_defaults import REGION_PRESETS
from large_scale_compute import run_large_scale

# COMMAND ----------

# 1. Create the large_scale_maps table (plus any other app tables).
lb.init_schema()
print("Schema ready.")

# COMMAND ----------

# 2. Seed a demo render for every region preset.
for key, preset in REGION_PRESETS.items():
    region = preset.to_dict()
    region_hash = lb.compute_region_hash(region)
    existing = lb.get_large_scale_map(region_hash)
    if existing and not existing.get("is_demo"):
        print(f"[{key}] real render already cached — skipping.")
        continue
    print(f"[{key}] computing demo coverage for {preset.name} …")
    results = run_large_scale(region, allow_demo=True)  # forced demo on CPU
    lb.write_large_scale_map(region_hash, region, results)
    print(f"[{key}] seeded ({region_hash[:12]}, {results['compute_seconds']:.1f}s).")

print("Done — the Large-scale map tab will now load instantly for each preset.")
