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
├── lakebase_client.py                    Postgres connection + Delta-equivalent helpers
├── sionna_compute.py                     Sionna RT pipeline (shared by setup + job)
├── setup/
│   └── setup_rf_digital_twin.py          One-shot workspace setup notebook
└── jobs/
    └── sionna_compute_job.py             Notebook the app triggers on cache miss
```

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

### 3. Grant the app's SP Postgres access

After the App is created (step 5), the runtime gives you a service principal. Lakebase auto-creates a Postgres role for it when the database resource is bound to the app, but it doesn't get permissions on tables created by another user.

Easiest grant (in a notebook cell using `lb_connect()`):

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

For production, replace `PUBLIC` with the app SP's UUID and grant least-privilege.

### 4. Create the live-render Job

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

### 5. Create the Databricks App + bind resources

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

Once the App is created, capture its `service_principal_client_id` and grant it `CAN_MANAGE_RUN` on the job:

```bash
databricks permissions update jobs <job_id> --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_client_id>",
    "permission_level": "CAN_MANAGE_RUN"
  }]
}'
```

Edit `app.yaml` to point `SIONNA_JOB_ID` at the job_id from step 4, then deploy:

```bash
databricks apps deploy rf-digital-twin \
  --source-code-path /Workspace/Users/<you>/sionna_simulation/App/rf-digital-twin-app
```

The Lakebase resource binding auto-populates `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` in the app container. `PGPASSWORD` is **not** populated — `lakebase_client.py` mints a fresh OAuth token at runtime via `WorkspaceClient().database.generate_database_credential()` and caches it for 45 minutes.

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

## Troubleshooting

- **`fe_sendauth: no password supplied`** — `lakebase_client._generate_password()` couldn't mint a token. Check that `LAKEBASE_INSTANCE` env var is set in `app.yaml` and the app SP has `CAN_USE` on the Lakebase instance.
- **`'WorkspaceClient' object has no attribute 'database'`** — `databricks-sdk` is too old. Pin `>=0.55.0` in `requirements.txt`.
- **`libnvoptix.so.1 could not be loaded`** — Sionna ran on a non-GPU or non-OptiX runtime. Switch the setup notebook to DBR 16.4 on `g5.xlarge`.
- **`Properties of ITU material 'marble' are not defined for this frequency`** — your scene uses a frequency below 1 GHz; ITU `marble` is only defined for 1–100 GHz. Keep frequencies ≥ 1.8 GHz on the etoile scene.
- **Render submitted but Status tab shows nothing** — confirm the workspace Repo is on the latest branch (`databricks repos update <id> --branch main`) and the App was redeployed after the code change.
