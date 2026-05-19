# RF Digital Twin — Lakehouse-only variant

Same UI and Sionna pipeline as `App/rf-digital-twin-app/`, but the
storage/cache layer is **Unity Catalog Delta tables** queried through a
**SQL warehouse** — no Lakebase Postgres.

## What's the same

- The 19 cached presets, the cheat sheet, the demo flow.
- The Sionna RT pipeline (`sionna_compute.py` unchanged).
- The Shiny sidebar / tabs / preset summary / Cancel button.
- The hash function — config_hash values are identical to the Lakebase
  variant, so the same `(scene, cells)` payload lands on the same row.

## What's different

| | Lakebase variant | **Lakehouse variant** |
| --- | --- | --- |
| Cache storage | Postgres `rf_digital_twin` database | Delta tables in `cmegdemos_catalog.sionna_rf_data` |
| App access | `psycopg` over Lakebase OAuth | `databricks-sql-connector` over a SQL warehouse |
| Job writes | Postgres bytea via psycopg | Spark DataFrame `.write.saveAsTable()` into Delta BINARY columns |
| Latency on cache hit | ~10–30 ms | ~100–300 ms (warm warehouse) |
| Auxiliary cost | Lakebase instance (~$0.30/h idle) | SQL warehouse + Delta storage (scale-to-zero) |
| Side benefit | OLTP-grade hot path | Cached renders are queryable from BI, notebooks, dashboards without an extra hop |

## Layout

```
App/rf-digital-twin-app-lakehouse/
├── app.py                       # Shiny app (imports lakehouse_client as `lb`)
├── app.yaml                     # binds DATABRICKS_WAREHOUSE_ID + job id
├── requirements.txt             # databricks-sql-connector instead of psycopg
├── defaults.py                  # identical to the Lakebase variant
├── lakehouse_client.py          # SQL-connector + Delta read/write helpers
├── sionna_compute.py            # identical
├── setup/
│   └── setup_rf_digital_twin_lakehouse.py
│       # creates schema + tables, optionally migrates from Lakebase,
│       # renders any missing presets, prints cheat sheet
└── jobs/
    └── sionna_compute_job_lakehouse.py
        # triggered by the app on cache miss; writes results into the
        # Delta `cached_renders` table
```

## Deployment

### 1. Run the setup notebook
Attach `setup/setup_rf_digital_twin_lakehouse.py` to a GPU cluster
(DBR 16.4 LTS + `g5.xlarge`, same as the Lakebase variant — see that
notebook's header for the validated spec). Run all cells.

If you've already run the Lakebase variant, the optional migration cell
copies every preset into Delta without re-running Sionna (~minutes
instead of ~30–50 min).

### 2. Create the Databricks Job
Upload `jobs/sionna_compute_job_lakehouse.py`. Create a Job with a
`g5.xlarge` job cluster, three text widgets (`config_hash`,
`scene_json`, `cells_json`), and a `sionna_compute` task pointing at the
notebook. Note the job id and paste it into `app.yaml` (replace
`REPLACE_WITH_LAKEHOUSE_JOB_ID`).

### 3. Create the Databricks App
Create a new app (`rf-digital-twin-lh` is the suggested name).
Resource binding:

| Resource name | Type | Value |
| --- | --- | --- |
| `warehouse`   | SQL warehouse | a serverless or PRO warehouse the app SP has `CAN_USE` on |

Permissions:
- App SP needs `SELECT`, `INSERT`, `UPDATE`, `DELETE` on the four Delta
  tables (`scene_configs`, `cell_configs`, `cached_renders`,
  `compute_jobs`).
- App SP needs `CAN MANAGE RUN` on the Lakehouse compute job.

### 4. Deploy
`databricks apps deploy rf-digital-twin-lh --source-code-path
 /Workspace/.../App/rf-digital-twin-app-lakehouse`

## When to use which variant

- **Lakebase variant** when you need OLTP-grade hot-path latency, are
  comfortable owning a Postgres instance, or are demoing the
  Apps + Lakebase product story end-to-end.
- **Lakehouse variant** when you'd rather keep everything in Delta + UC,
  want the cache itself to be a queryable data product (analysts can
  build dashboards directly on `cached_renders`), or are minimising
  managed-service surface area.
