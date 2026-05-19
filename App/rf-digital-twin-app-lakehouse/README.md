# RF Digital Twin — Lakehouse variant

Same Shiny UI and Sionna pipeline as the [Lakebase variant](../rf-digital-twin-app/README.md), but caches every render in **Unity Catalog Delta tables** and reads them through a **SQL warehouse** via `databricks-sql-connector`. No Postgres, no separate managed service — everything lives in UC.

For the cache-layer tradeoffs (~200–400 ms warm reads vs ~10–30 ms on Lakebase, scale-to-zero idle cost vs always-on, etc.), see the [top-level README's "Lakebase vs Lakehouse — performance at scale" section](../../README.md#lakebase-vs-lakehouse--performance-at-scale).

```
                ┌─────────────────────────────────────────────┐
                │ Shiny app (Databricks Apps)                 │
                │  ─ user edits sidebar / preset summary      │
                │  ─ Render → sha256(scene + cells)           │
                │                                             │
                │      hit ──► reads cached_renders ─► views  │
                │       │      (SQL connector → warehouse)    │
                │       miss                                  │
                │       ▼                                     │
                │  jobs API run_now ─► Sionna GPU job ────────┐
                │                       writes Delta via Spark│
                └─────────────────────────────────────────────┘
                                  ▲ ▼ databricks-sql-connector
                ┌─────────────────────────────────────────────┐
                │ Unity Catalog Delta tables                  │
                │   cmegdemos_catalog.sionna_rf_data.*        │
                │  scene_configs, cell_configs                │
                │  cached_renders (BINARY PNGs + STRING KPIs) │
                │  compute_jobs                               │
                └─────────────────────────────────────────────┘
```

## How it differs from the Lakebase variant

| | Lakebase | **Lakehouse** |
| --- | --- | --- |
| Cache store | Postgres `bytea` columns | UC Delta `BINARY` columns |
| App read driver | `psycopg` | `databricks-sql-connector` over a SQL warehouse |
| Job write driver | `psycopg` | Spark `DataFrame.write.saveAsTable()` |
| Auxiliary service | Lakebase instance | SQL warehouse (scale-to-zero) |
| Cache portability | — | **Optional Lakebase → Delta migration cell** copies every preset without re-running Sionna |
| `config_hash` algorithm | identical | identical (cache content is byte-for-byte portable) |

## Layout

```
App/rf-digital-twin-app-lakehouse/
├── app.py                                              Shiny UI (imports lakehouse_client as `lb`)
├── app.yaml                                            Binds SQL warehouse instead of Lakebase
├── requirements.txt                                    databricks-sql-connector replaces psycopg
├── defaults.py                                         Identical preset definitions
├── lakehouse_client.py                                 SQL connector + Delta read/write
├── sionna_compute.py                                   Identical Sionna pipeline
├── setup/
│   └── setup_rf_digital_twin_lakehouse.py              Creates UC tables + optional migration + Sionna renders
└── jobs/
    └── sionna_compute_job_lakehouse.py                 Cache-miss job — writes results to Delta via Spark
```

## Deploy in 10 minutes — quickstart

After the setup notebook has succeeded (UC schema + 4 Delta tables + 19 cached presets exist), these are the remaining steps in execution order.

```bash
# Prereqs: setup notebook done, databricks CLI authed, workspace Repo on the right branch.
WS_USER="<your.email@databricks.com>"
APP_SRC="/Workspace/Users/$WS_USER/sionna_simulation/App/rf-digital-twin-app-lakehouse"
WAREHOUSE_ID="<sql_warehouse_id>"   # databricks warehouses list
REPO_ID="<workspace-repo-id>"

# 1. Create the live-render Job (see § Create the Job for full JSON with $WS_USER substituted).
JOB_ID=$(databricks jobs create --json @job.json | jq -r .job_id)
echo "Job ID: $JOB_ID"

# 2. Wire $JOB_ID into app.yaml, commit, push, sync workspace Repo.
sed -i.bak "s|REPLACE_WITH_LAKEHOUSE_JOB_ID|$JOB_ID|" App/rf-digital-twin-app-lakehouse/app.yaml
git add App/rf-digital-twin-app-lakehouse/app.yaml && git commit -m "wire SIONNA_JOB_ID" && git push
databricks repos update $REPO_ID --branch main

# 3. Create the App with the SQL warehouse bound.
SP=$(databricks apps create --json @app.json --compute-size MEDIUM | jq -r .service_principal_client_id)
echo "App SP: $SP"
while [ "$(databricks apps get rf-digital-twin-lh --output json | jq -r .compute_status.state)" != "ACTIVE" ]; do sleep 12; done

# 4. Grant the SP CAN_MANAGE_RUN on the job.
databricks permissions update jobs $JOB_ID \
  --json "{\"access_control_list\":[{\"service_principal_name\":\"$SP\",\"permission_level\":\"CAN_MANAGE_RUN\"}]}"

# 5. Grant the SP UC permissions on the four Delta tables.
databricks grants update catalog cmegdemos_catalog \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_CATALOG\"]}]}"
databricks grants update schema cmegdemos_catalog.sionna_rf_data \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_SCHEMA\"]}]}"
for t in scene_configs cell_configs cached_renders compute_jobs; do
  databricks grants update table cmegdemos_catalog.sionna_rf_data.$t \
    --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"SELECT\",\"MODIFY\"]}]}"
done

# 6. Deploy the app source.
databricks apps deploy rf-digital-twin-lh --source-code-path $APP_SRC

# 7. Open the URL.
databricks apps get rf-digital-twin-lh --output json | jq -r .url
```

`job.json` and `app.json` referenced above are in [§ Create the Job](#3-create-the-live-render-job) and [§ Create the Databricks App](#4-create-the-databricks-app--bind-the-sql-warehouse) below.

## Common gotchas

- **SP doesn't exist until step 3** — UC `GRANT`s only work after the app is created.
- **`DELTA_FAILED_TO_MERGE_FIELDS num_rows_tx and num_rows_tx`** — `spark.createDataFrame` inferred `BIGINT` for Python `int` while the Delta table declared `INT`. The setup notebook uses explicit `StructType`s to avoid this; if you write directly, do the same.
- **First Render after >5 min of idle takes 30–60 s** — the SQL warehouse scaled to zero. Either keep it warm or accept the cold-start tax.
- **`PERMISSION_DENIED on cached_renders`** — the app SP needs `USE_CATALOG` + `USE_SCHEMA` + `SELECT`/`MODIFY` on each Delta table (step 5 above).
- **Account-level OAuth quota** — Databricks caps custom OAuth app integrations at 10 K per account. Each Databricks App creates one. If `databricks apps create` returns `QUOTA_EXCEEDED`, an admin must clean up unused integrations.
- **GPU cluster ≠ OptiX** — Sionna RT needs NVIDIA OptiX, shipped with `g4dn`/`g5`/`g6` instances but **not** CPU or ARM. Plain DBR 16.4 LTS on `g5.xlarge` works.

## One-time setup

### 1. Provision a GPU cluster

Same Sionna RT requirements as the Lakebase variant:

| Setting | Value |
| --- | --- |
| Databricks Runtime | **16.4 LTS** (Scala 2.13) — plain, not ML |
| Driver / worker | `g5.xlarge` (1× A10G, 24 GB VRAM) |
| Access mode | Single user |
| Auto-termination | 120 minutes |
| AWS availability | Spot with fallback to on-demand |

CPU clusters can't run Sionna (missing OptiX). See the [Lakebase variant README](../rf-digital-twin-app/README.md#1-provision-a-gpu-cluster) for the full cluster spec.

### 2. Run the setup notebook

Open `setup/setup_rf_digital_twin_lakehouse.py`, attach the GPU cluster, run all cells. It will:

1. `pip install` drjit / mitsuba / sionna-rt / databricks-sdk (no psycopg needed for the Lakehouse path).
2. Create the `cmegdemos_catalog.sionna_rf_data` UC schema (idempotent).
3. Create four Delta tables: `scene_configs`, `cell_configs`, `cached_renders` (with `BINARY` columns for PNGs), `compute_jobs`.
4. **Section 6 — optional migration:** if you've already populated Lakebase from the other variant, this cell copies every preset across in minutes — no Sionna re-runs. Set `RUN_MIGRATION = False` to skip.
5. Renders any preset whose hash isn't in `cached_renders` yet through Sionna RT (Section 8). Idempotent + per-preset try/except.
6. Prints the cheat sheet of sidebar values → config_hash for every cached preset.

Wall-clock:
- ~minutes if you migrate from Lakebase.
- ~30–50 min for a fresh run on `g5.xlarge`.

### 3. Create the live-render Job

```bash
databricks jobs create --json @- <<'JSON'
{
  "name": "rf-digital-twin-lh-sionna-compute",
  "tasks": [{
    "task_key": "sionna_compute_lh",
    "notebook_task": {
      "notebook_path": "/Workspace/Users/<you>/sionna_simulation/App/rf-digital-twin-app-lakehouse/jobs/sionna_compute_job_lakehouse",
      "source": "WORKSPACE",
      "base_parameters": {"config_hash": "", "scene_json": "", "cells_json": ""}
    },
    "job_cluster_key": "sionna_gpu",
    "libraries": [
      {"pypi": {"package": "drjit"}},
      {"pypi": {"package": "mitsuba"}},
      {"pypi": {"package": "sionna-rt"}},
      {"pypi": {"package": "databricks-sdk>=0.55.0"}}
    ],
    "timeout_seconds": 1800
  }],
  "job_clusters": [{
    "job_cluster_key": "sionna_gpu",
    "new_cluster": {
      "spark_version": "16.4.x-scala2.13",
      "node_type_id": "g5.xlarge", "driver_node_type_id": "g5.xlarge",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      "single_user_name": "<you>",
      "runtime_engine": "STANDARD",
      "spark_conf": {"spark.master": "local[*]", "spark.databricks.cluster.profile": "singleNode"},
      "custom_tags": {"ResourceClass": "SingleNode"},
      "aws_attributes": {"availability": "SPOT_WITH_FALLBACK", "first_on_demand": 1, "zone_id": "auto"}
    }
  }],
  "max_concurrent_runs": 3, "queue": {"enabled": true}
}
JSON
```

Note the returned `job_id`. Edit `app.yaml` so `SIONNA_JOB_ID` matches (replace the `REPLACE_WITH_LAKEHOUSE_JOB_ID` placeholder), commit, push, and sync the workspace Repo before continuing.

### 4. Create the Databricks App + bind the SQL warehouse

```bash
databricks apps create --json @- <<'JSON'
{
  "name": "rf-digital-twin-lh",
  "description": "Lakehouse variant of the RF Digital Twin app.",
  "resources": [{
    "name": "warehouse",
    "sql_warehouse": {
      "id": "<sql_warehouse_id>",
      "permission": "CAN_USE"
    }
  }]
}
JSON
```

A small serverless SQL warehouse is plenty for the cache reads. The binding auto-populates `DATABRICKS_WAREHOUSE_ID` in the app container. Wait until `compute_status.state` is `ACTIVE` and capture the `service_principal_client_id` — you'll need it next.

### 5. Grant the SP permissions

Two grants, in either order:

**a) `CAN_MANAGE_RUN` on the live-render Job:**

```bash
databricks permissions update jobs <job_id> --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_client_id>",
    "permission_level": "CAN_MANAGE_RUN"
  }]
}'
```

**b) Unity Catalog access on the four Delta tables:**

```bash
SP="<app_sp_client_id>"
databricks grants update catalog cmegdemos_catalog \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_CATALOG\"]}]}"
databricks grants update schema cmegdemos_catalog.sionna_rf_data \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_SCHEMA\"]}]}"
for t in scene_configs cell_configs cached_renders compute_jobs; do
  databricks grants update table cmegdemos_catalog.sionna_rf_data.$t \
    --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"SELECT\",\"MODIFY\"]}]}"
done
```

### 6. Deploy the app source

```bash
databricks apps deploy rf-digital-twin-lh \
  --source-code-path /Workspace/Users/<you>/sionna_simulation/App/rf-digital-twin-app-lakehouse
```

Open the URL — Config 1 auto-loads from the Delta cache.

## What's in UC

All under `cmegdemos_catalog.sionna_rf_data`:

| Table | Type | Purpose |
| --- | --- | --- |
| `scene_configs`  | Delta | One row per saved scene config + sha256 `config_hash` |
| `cell_configs`   | Delta | 7 rows per `config_hash` — per-TX positions / look-at / power |
| `cached_renders` | Delta | PNG `BINARY` for scene / SINR / association / CDFs + KPI JSON string |
| `compute_jobs`   | Delta | Tracking row per submitted Sionna job |
| `cell_configs_default` | Delta | Source-of-truth cell layout (engineers edit this; setup notebook re-renders affected configs) |

Because everything is in UC, the same `cached_renders` table is queryable from BI tools, dashboards, surrogate-training pipelines, etc. without an ETL step.

## Demo flow

Identical to the Lakebase variant — see [its demo flow](../rf-digital-twin-app/README.md#demo-flow). The only user-visible difference is the URL.

## Local dev

```bash
cd App/rf-digital-twin-app-lakehouse
pip install -r requirements.txt
export DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
export DATABRICKS_WAREHOUSE_ID=<wh_id>
# personal access token in DATABRICKS_TOKEN, OR use OAuth via `databricks auth login`
shiny run --reload app.py:app
```

The app uses `databricks.sdk.core.Config()` to pick up credentials from env. The SQL connector authenticates via the same OAuth/PAT flow.

## Runtime troubleshooting

Setup/deployment gotchas are in [Common gotchas](#common-gotchas) at the top of this file. The bullets here are things that surface **after** the app is running.

- **`PERMISSION_DENIED ... SELECT on TABLE cached_renders`** — the app SP needs `USE_CATALOG` + `USE_SCHEMA` + `SELECT`/`MODIFY` on each Delta table. See step 5 of the deployment flow.
- **`Properties of ITU material 'marble' are not defined for this frequency`** — the user typed a frequency below 1 GHz; ITU `marble` is only defined for 1–100 GHz. Keep frequencies ≥ 1.8 GHz on the etoile scene.
- **App shows Config 1 on load but every Render goes to the live job** — workspace Repo wasn't synced to the deploy branch before `databricks apps deploy`. Re-run `databricks repos update <repo_id> --branch <name>` and redeploy.

## See also

- [Lakebase variant](../rf-digital-twin-app/README.md) — the other version of this app.
- [Top-level README](../../README.md) — preset gallery, render-time × compute table, Lakebase vs Lakehouse tradeoffs, scaling strategies.
