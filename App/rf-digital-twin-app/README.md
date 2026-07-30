# RF Digital Twin — Lakebase variant

The default app. Caches every Sionna RT render in **Lakebase Postgres** and reads them back through a `psycopg` connection. Sub-30 ms cache hits, OLTP-grade throughput, instance is always warm.

For the cache-layer tradeoffs vs the [Lakehouse variant](../rf-digital-twin-app-lakehouse/README.md), see the [top-level README's "Lakebase vs Lakehouse — performance at scale" section](../../README.md#lakebase-vs-lakehouse--performance-at-scale).

```
                ┌─────────────────────────────────────────────┐
                │ Shiny app (Databricks Apps)                 │
                │  ─ user edits sidebar / preset summary      │
                │  ─ Render → sha256(scene + cells)           │
                │                                             │
                │      hit ──► reads cached_renders ─► views  │
                │       │                                     │
                │       miss                                  │
                │       ▼                                     │
                │  jobs API run_now ─► Sionna GPU job ────────┐
                │                       writes to Lakebase    │
                └─────────────────────────────────────────────┘
                                  ▲ ▼ psycopg
                ┌─────────────────────────────────────────────┐
                │ Lakebase Postgres (rf-digital-twin-pg)      │
                │  scene_configs, cell_configs                │
                │  cached_renders (bytea PNGs + JSONB KPIs)   │
                │  compute_jobs                               │
                └─────────────────────────────────────────────┘
```

## Layout

```
App/rf-digital-twin-app/
├── app.py                                Shiny UI + non-blocking server
├── app.yaml                              Databricks Apps deploy config
├── requirements.txt                      App runtime deps (no Sionna; that's job-side)
├── defaults.py                           19-preset gallery + 7-cell layout
├── large_scale_defaults.py               Region presets for the large-scale tab
├── large_scale_compute.py                NVlabs sionna_lrm pipeline + GPU-free demo
├── lakebase_client.py                    Postgres connection + Delta-equivalent helpers
├── sionna_compute.py                     Sionna RT pipeline (shared by setup + job)
├── setup/
│   ├── setup_rf_digital_twin.py          One-shot workspace setup notebook
│   └── setup_large_scale_maps.py         Init large_scale_maps table + seed demo regions
└── jobs/
    ├── sionna_compute_job.py             Notebook the app triggers on cache miss
    └── large_scale_compute_job.py        Large-scale (sionna_lrm) GPU job (optional)
```

## Large-scale radio maps (NVIDIA sionna-large-radio-maps)

The **Large-scale map** tab computes coverage across a *real geographic region*
— a WGS84 lat/lon bounding box with OpenStreetMap buildings and a base-station
layout — by wrapping NVIDIA's
[`sionna-large-radio-maps`](https://github.com/NVlabs/sionna-large-radio-maps)
pipeline (adaptive tiling → OSM scene build → per-tile ray tracing → mosaic).
This complements the single synthetic `etoile` scene the other tabs use.

Configure a region in the sidebar's **Large-scale map** section (preset +
editable bbox / frequency / TX power), then click **Compute large-scale map**.
Two execution paths, chosen automatically:

- **Demo (GPU-free, default).** When `LARGE_SCALE_JOB_ID` is not set, the app
  computes a synthetic log-distance coverage map inline (no cluster needed) and
  flags it with a yellow "Demo mode" banner. Great for a quick walkthrough.
- **Real Sionna RT.** Set `LARGE_SCALE_JOB_ID` in `app.yaml` to a job created
  from `jobs/large_scale_compute_job.py` (GPU cluster, RTX cores preferred).
  On a cache miss the app submits that job; it clones the NVlabs repo, runs the
  full pipeline, mosaics the per-tile path-gain arrays, and writes the result to
  the Lakebase `large_scale_maps` cache. A background poller auto-loads it when
  ready.

Region presets ship for **Seattle** (default), **San Francisco**, and **Paris
(étoile)**. Run `setup/setup_large_scale_maps.py` (CPU-only) once to create the
`large_scale_maps` table and seed a demo render for each preset so the tab loads
instantly.

## Deploy in 10 minutes — quickstart

After the setup notebook has succeeded (Lakebase + 19 cached presets exist), these are the remaining steps in execution order. Each one has a detailed section below; this block is for skimming and copy-paste.

```bash
# Prereqs: setup notebook done, databricks CLI authed, workspace Repo on the right branch.
WS_USER="<your.email@databricks.com>"
APP_SRC="/Workspace/Users/$WS_USER/sionna_simulation/App/rf-digital-twin-app"
REPO_ID="<workspace-repo-id>"     # databricks repos list

# 1. Create the live-render Job (see § Create the Job for the full JSON; substitute $WS_USER).
JOB_ID=$(databricks jobs create --json @job.json | jq -r .job_id)
echo "Job ID: $JOB_ID"

# 2. Wire $JOB_ID into app.yaml on the branch you'll deploy from, push, sync workspace Repo.
sed -i.bak "s|REPLACE_WITH_JOB_ID|$JOB_ID|" App/rf-digital-twin-app/app.yaml
git add App/rf-digital-twin-app/app.yaml && git commit -m "wire SIONNA_JOB_ID" && git push
databricks repos update $REPO_ID --branch main

# 3. Create the App. The Lakebase binding makes a Postgres role for the app SP automatically.
SP=$(databricks apps create --json @app.json --compute-size MEDIUM | jq -r .service_principal_client_id)
echo "App SP: $SP"
# Wait for compute ACTIVE
while [ "$(databricks apps get rf-digital-twin --output json | jq -r .compute_status.state)" != "ACTIVE" ]; do sleep 12; done

# 4. Grant the SP CAN_MANAGE_RUN on the job.
databricks permissions update jobs $JOB_ID \
  --json "{\"access_control_list\":[{\"service_principal_name\":\"$SP\",\"permission_level\":\"CAN_MANAGE_RUN\"}]}"

# 5. Grant the SP Postgres access — run this cell in your setup notebook (lb_connect() is already defined):
#
#   with lb_connect() as conn, conn.cursor() as cur:
#       cur.execute("""
#           GRANT USAGE ON SCHEMA public TO PUBLIC;
#           GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO PUBLIC;
#           GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;
#           ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO PUBLIC;
#           ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO PUBLIC;
#       """)
#       conn.commit()

# 6. Deploy the app source.
databricks apps deploy rf-digital-twin --source-code-path $APP_SRC

# 7. Open the URL.
databricks apps get rf-digital-twin --output json | jq -r .url
```

`job.json` and `app.json` referenced above are in [§ Create the Job](#3-create-the-live-render-job) and [§ Create the Databricks App](#4-create-the-databricks-app--bind-the-database) below.

## Common gotchas

- **SP doesn't exist until step 3** — Postgres `GRANT`s only work after the app is created. The Postgres role is auto-provisioned by the Lakebase binding, so the GRANT step is just permissioning the tables you created earlier as a different user.
- **`'WorkspaceClient' object has no attribute 'database'`** — `databricks-sdk` in the app base image is 0.33.0; the Lakebase API namespace needs `>=0.55.0`. `requirements.txt` already pins it; just make sure your deploy is from the latest commit.
- **Account-level OAuth quota** — Databricks accounts cap custom OAuth app integrations at 10 K. Each Databricks App creates one. If `databricks apps create` returns `QUOTA_EXCEEDED`, an admin needs to clean up unused integrations.
- **Lakebase 10-instance per workspace cap** — the setup notebook will fail if you're at the cap. Delete an unused instance first.
- **Workspace Repo branch drift** — `databricks apps deploy` snapshots the path at deploy time. If the workspace Repo is behind, you'll silently ship stale code. Always `databricks repos update <id> --branch <name>` before deploying.
- **GPU cluster ≠ OptiX** — Sionna RT needs NVIDIA OptiX, which ships with `g4dn`/`g5`/`g6` instances but **not** CPU or ARM. Plain DBR 16.4 LTS on `g5.xlarge` works.

## One-time setup

### 1. Provision a GPU cluster

Sionna RT calls NVIDIA OptiX. Required cluster spec (validated):

| Setting | Value |
| --- | --- |
| Databricks Runtime | **16.4 LTS** (Scala 2.13) — plain, not ML |
| Runtime engine | Standard |
| Driver / worker | `g5.xlarge` (1× NVIDIA A10G, 24 GB VRAM) |
| Autoscaling | min 2, max 8 workers (or single node) |
| Access mode | Single user |
| Auto-termination | 120 minutes |
| AWS availability | Spot with fallback to on-demand |
| Init scripts | none |
| Custom Spark conf | none |

Cheaper alternative: `g4dn.xlarge` (T4, slower but full OptiX). **Don't use CPU or ARM instances** — Sionna will fail with `libnvoptix.so.1 could not be loaded`.

### 2. Run the setup notebook

Open `setup/setup_rf_digital_twin.py` in the workspace, attach the GPU cluster, run all cells. It will:

1. `pip install` drjit / mitsuba / sionna-rt / `psycopg[binary]` / databricks-sdk.
2. Create the `cmegdemos_catalog.sionna_rf_data` UC schema.
3. Generate the default 7-cell network and persist it as `cmegdemos_catalog.sionna_rf_data.cell_configs_default` Delta table (engineers can edit this later).
4. Provision the `rf-digital-twin-pg` Lakebase instance (CU_1, ~$0.30/hr) + `rf_digital_twin` database. **Mind the account-level 10-instance Lakebase quota** — delete an unused instance first if needed.
5. Create Lakebase tables: `scene_configs`, `cell_configs`, `cached_renders`, `compute_jobs`.
6. Render every preset in `PRESETS` (19 by default) through Sionna RT and write the results into `cached_renders`. Idempotent: re-running the notebook skips configs whose hash is already cached.
7. Print the **cheat sheet** of sidebar values → config_hash for every cached preset.

Wall-clock: ~30–50 min for a fresh run on `g5.xlarge`; ~minutes on re-runs.

### 3. Create the live-render Job

The app submits this Job whenever a user picks an off-menu config.

```bash
databricks jobs create --json @- <<'JSON'
{
  "name": "rf-digital-twin-sionna-compute",
  "tasks": [{
    "task_key": "sionna_compute",
    "notebook_task": {
      "notebook_path": "/Workspace/Users/<you>/sionna_simulation/App/rf-digital-twin-app/jobs/sionna_compute_job",
      "source": "WORKSPACE",
      "base_parameters": {"config_hash": "", "scene_json": "", "cells_json": ""}
    },
    "job_cluster_key": "sionna_gpu",
    "libraries": [
      {"pypi": {"package": "drjit"}},
      {"pypi": {"package": "mitsuba"}},
      {"pypi": {"package": "sionna-rt"}},
      {"pypi": {"package": "psycopg[binary]>=3.1.18"}},
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

Note the `job_id` it returns.

Note the returned `job_id`. Edit `app.yaml` so `SIONNA_JOB_ID` matches, commit, push, and sync the workspace Repo before continuing.

### 4. Create the Databricks App + bind the database

```bash
databricks apps create --json @- <<'JSON'
{
  "name": "rf-digital-twin",
  "description": "Sionna RT digital twin (Lakebase variant)",
  "resources": [{
    "name": "lakebase",
    "database": {
      "instance_name": "rf-digital-twin-pg",
      "database_name": "rf_digital_twin",
      "permission": "CAN_CONNECT_AND_CREATE"
    }
  }]
}
JSON
```

Wait until `compute_status.state` is `ACTIVE`. Capture the `service_principal_client_id` — you'll need it for the next step. The Lakebase binding auto-populates `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER` in the app container and creates a Postgres role for the SP automatically; **`PGPASSWORD` is not populated** — `lakebase_client.py` mints a fresh OAuth token at runtime via `WorkspaceClient().database.generate_database_credential()` and caches it for 45 minutes.

### 5. Grant the SP permissions

Two things to grant, in either order:

**a) `CAN_MANAGE_RUN` on the live-render Job** (so the app can submit jobs on cache miss):

```bash
databricks permissions update jobs <job_id> --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_client_id>",
    "permission_level": "CAN_MANAGE_RUN"
  }]
}'
```

**b) Postgres access** on the tables your setup notebook created (the SP can connect but can't read tables owned by your personal user). Easiest: run this cell in the setup notebook where `lb_connect()` is already defined:

```python
with lb_connect() as conn, conn.cursor() as cur:
    cur.execute("""
        GRANT USAGE ON SCHEMA public TO PUBLIC;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO PUBLIC;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO PUBLIC;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO PUBLIC;
    """)
    conn.commit()
```

For production, swap `PUBLIC` for the app SP's UUID and grant least-privilege.

### 6. Deploy the app source

```bash
databricks apps deploy rf-digital-twin \
  --source-code-path /Workspace/Users/<you>/sionna_simulation/App/rf-digital-twin-app
```

Open the URL — Config 1 auto-loads from the cache.

## What's in Lakebase

| Table | Purpose |
| --- | --- |
| `scene_configs`  | One row per saved scene config + sha256 `config_hash` |
| `cell_configs`   | 7 rows per `scene_configs.id` — per-TX positions / look-at / power |
| `cached_renders` | PNG bytea for scene / SINR / association / CDFs + KPI JSON, keyed by config_hash |
| `compute_jobs`   | Tracking row per submitted Sionna job (PENDING / RUNNING / SUCCEEDED / FAILED) |

## Demo flow

1. Open the app — Config 1 (8×2 TX) auto-loads from the cache.
2. Walk through the visualisation tabs: **Scene render → SINR association → Users → CDFs → KPIs**.
3. Edit the sidebar to match a row in the [preset gallery](../../README.md#preset-gallery--whats-cached-in-lakebase) (e.g. flip to 16×16 for Config 2). Click **Render** — cached hit, instant.
4. Bonus: type an off-menu value. Status banner shows "Sionna job submitted (run_id=…)". Switch to any cached preset while it runs. When the job finishes, results auto-appear. **Cancel pending job** in the sidebar kills the cluster if you change your mind.

## Local dev

```bash
cd App/rf-digital-twin-app
pip install -r requirements.txt
export PGHOST=… PGUSER=… PGPASSWORD=… PGDATABASE=rf_digital_twin PGSSLMODE=require
shiny run --reload app.py:app
```

App starts on `http://localhost:8000`. Sionna isn't required locally — the app only reads `cached_renders` from Lakebase.

## Runtime troubleshooting

Setup/deployment gotchas are in [Common gotchas](#common-gotchas) at the top of this file. The bullets here are things that surface **after** the app is running.

- **`fe_sendauth: no password supplied`** — `lakebase_client._generate_password()` couldn't mint an OAuth token. Check that `LAKEBASE_INSTANCE` is set in `app.yaml` and the app SP has `CAN_USE` on the Lakebase instance.
- **`Properties of ITU material 'marble' are not defined for this frequency`** — the user typed a frequency below 1 GHz. The etoile scene's ITU `marble` material is only defined for 1–100 GHz. Keep frequencies ≥ 1.8 GHz.
- **App shows Config 1 on load but every Render goes to the live job** — workspace Repo wasn't synced to the deploy branch before `databricks apps deploy`. Re-run `databricks repos update <repo_id> --branch <name>` and redeploy.
