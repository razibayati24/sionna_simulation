# RF Digital Twin App

A Databricks App that lets RF planning engineers edit a cell-network configuration and instantly visualise the resulting Sionna RT simulation: scene render with SINR overlay, cell-to-TX association, sampled user map, and SINR/RSS CDFs.

Built around the workflow in `rf_planning_optimization/simulation_RT_light.ipynb` and the two-part Medium write-up ("NVIDIA's AI-Native Digital Twin on Databricks").

## How it works

```
                ┌─────────────────────────────────────────────┐
                │ Shiny app (Databricks Apps)                 │
                │                                             │
   user edits   │  Sidebar:                                   │
   ─────────►   │   • TX/RX array, freq, BW, depth, samples   │
                │   • Pattern, polarization, cell size        │
                │   • User sampling params                    │
                │  Cells tab: 7 editable TX rows              │
                │                                             │
                │  ┌──────────────┐    sha256 hash             │
                │  │ "Render"     │──────────┐                 │
                │  └──────────────┘          ▼                 │
                │                       ┌─────────┐            │
                │  Cached?  ──► YES ──► │Lakebase │ ─► views   │
                │   │                   └─────────┘            │
                │   │ NO                     ▲                 │
                │   ▼                        │                 │
                │ Jobs API run-now ─► Sionna RT GPU job ──────┘│
                │                     (writes back to Lakebase)│
                └─────────────────────────────────────────────┘
```

- **Preset switches are instant** — Config 1 (8×2 TX / 2×2 RX) and Config 2 (16×16 TX / 2×2 RX) are precomputed once and stored in Lakebase, keyed by SHA-256 of the full config.
- **Custom edits trigger a job** — the app submits a Databricks job that runs Sionna RT on a GPU cluster and writes results to the same Lakebase cache. The app then polls for the result.
- **Lakebase is the cache** — Postgres tables hold scene configs, per-cell configs, render PNGs (bytea), and KPI JSON.

## Layout

```
App/rf-digital-twin-app/
├── app.py                  Shiny UI + server
├── app.yaml                Databricks Apps deploy config
├── requirements.txt        App runtime deps (light; Sionna only on the job)
├── defaults.py             Config 1, Config 2, and the 7-cell default layout
├── lakebase_client.py      Postgres connection + schema + queries
├── sionna_compute.py       The Sionna RT pipeline (shared by setup + job)
├── setup/
│   └── setup_rf_digital_twin.py One-shot workspace setup notebook (see below)
└── jobs/
    └── sionna_compute_job.py    Notebook the app triggers for custom configs
```

## One-time setup

### 1. Run the workspace setup notebook

Open `setup/setup_rf_digital_twin.py` as a Databricks notebook on a **GPU cluster** (CPU works but Sionna is slow) and run all cells. It will:

1. Create the `cmegdemos_catalog.sionna_rf_data` schema in Unity Catalog.
2. Generate the default 7-cell network and persist it as `cmegdemos_catalog.sionna_rf_data.cell_configs_default` — engineers can edit this Delta table later to change the network.
3. Provision a `rf-digital-twin-pg` Lakebase Postgres instance + `rf_digital_twin` database.
4. Create the Lakebase tables the app expects (`scene_configs`, `cell_configs`, `cached_renders`, `compute_jobs`).
5. Run Sionna RT for **Config 1 (8×2 TX)** and **Config 2 (16×16 TX)** and cache both into Lakebase.
6. Print the connection details to plug into the app's resource bindings.

Wall-clock: ~5–10 min on a GPU cluster, ~30–40 min on CPU.

### 2. (Optional) Wire up the live-edit job

If you want users to be able to render custom configs (positions, power, antenna sizes other than the presets):

1. Upload `jobs/sionna_compute_job.py` to your workspace.
2. Create a Databricks Job pointing at it, configured to run on a GPU cluster with `drjit`/`mitsuba`/`sionna-rt`/`psycopg[binary]` available, and parameterised by the three notebook widgets (`config_hash`, `scene_json`, `cells_json`).
3. Note the job id.

### 3. Create the App + bind resources

In the Databricks Apps UI, create a new app from this directory. Add resources matching the `valueFrom` names in `app.yaml`:

| Resource name        | Type     | Value                                            |
| -------------------- | -------- | ------------------------------------------------ |
| `lakebase-host`      | secret   | `<instance>.database.<region>.databricks.com`    |
| `lakebase-port`      | secret   | `5432`                                           |
| `lakebase-database`  | secret   | `rf_digital_twin`                                |
| `lakebase-user`      | secret   | Postgres user (service principal recommended)    |
| `lakebase-password`  | secret   | Postgres password / OAuth token                  |
| `sionna-job-id`      | secret   | Job ID from step 3 (or leave unset)              |

The app's service principal needs:
- `CAN USE` on the Lakebase instance
- `CAN MANAGE RUN` on the Sionna compute job (only if live edits are enabled)

## Demo flow

1. App opens — Config 1 is already loaded from Lakebase (scene render, SINR map, association, CDFs all populated).
2. Walk through tabs: **Scene render → SINR association → Users → CDFs → KPIs**.
3. Click **"Load Config 2 (16x16)"** in the sidebar. All controls flip to Config 2 values.
4. Click **Render**. Cache hit — the new visuals appear within a second. Compare CDFs side-by-side with the prior screenshots / KPI table.
5. (Bonus) Edit a TX position or power, then **Render**. The app submits a GPU job and polls until results land in the cache, demonstrating the live-edit loop.

## What's in Lakebase

| Table             | Purpose                                                              |
| ----------------- | -------------------------------------------------------------------- |
| `scene_configs`   | One row per saved scene config + sha256 `config_hash`                |
| `cell_configs`    | 7 rows per `scene_configs.id` — per-TX positions, look-at, power     |
| `cached_renders`  | PNG bytes for scene/SINR/assoc/CDFs + KPI JSON, keyed by config_hash |
| `compute_jobs`    | Tracking row per submitted Sionna job (PENDING/RUNNING/SUCCEEDED/…)  |

Schema is created idempotently on first connection by `lakebase_client.init_schema()`.

## Local dev

```bash
cd App/rf-digital-twin-app
pip install -r requirements.txt
export PGHOST=... PGUSER=... PGPASSWORD=... PGDATABASE=rf_digital_twin PGSSLMODE=require
shiny run --reload app.py:app
```

The app starts on `http://localhost:8000`. Sionna itself is not needed locally — the app only reads cached results from Lakebase.

## References

- Notebook: `rf_planning_optimization/simulation_RT_light.ipynb`
- Article I: <https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-bdb81ef87b70>
- Article II: <https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-ii-065938ca112c>
- Sionna RT Radio Maps tutorial: <https://nvlabs.github.io/sionna/rt/tutorials.html>
