# RF Digital Twin — Sionna RT on Databricks (Lakebase)

The app. A Shiny sidebar drives a Sionna RT radio-map solve over **real T-Mobile towers** from
`cmegdemos_catalog.network_analytics_enablement.cell_towers` (2,312 across the Seattle metro),
ray-traced against **OpenStreetMap building geometry**, with every render cached in Lakebase
Postgres and keyed by a hash of its config.

See the [top-level README](../../README.md) for the architecture, the performance/cost tables, and
the metro-scale story. This file is the operational detail.

## How it works

```
sidebar knobs ─▶ scene_spec.resolve() ─▶ compute_config_hash() ─▶ Lakebase
                                                                    ├─ hit  → PNGs + KPIs, 5–15 ms
                                                                    └─ miss → GPU job, then poll
```

The app has no Spark and never renders anything. It resolves the tower set for the chosen
neighborhood, hashes the config, and either loads the cached render or asks the GPU job for it.

**Hash parity is the contract.** The cache key folds in each tower's position/height/power/band/array
plus the neighborhood and tile, so the app must resolve the *identical* tower set the job rendered.
`scene_spec.py` is the single module both sides use, and on a cache miss `render_pipeline.render_custom`
re-derives the towers job-side and asserts the hash matches what the app is polling for. Run
`tools/check_hash_parity.py` after changing anything hash-relevant.

## How 2,312 towers become tractable

A radio-map solve is ~`O(num_tx × samples_per_tx × max_depth)` with an output tensor of
`area / cell_size²` cells per TX, so the whole 44×28 km metro in one scene is infeasible. Four levers:

1. **Tiling** — the metro is cut into named **neighborhoods** (`neighborhoods.py`); each is gridded
   into ~800 m **tiles** with a 150 m overlap margin so coverage isn't clipped at seams. A tile
   renders 10–30 towers, not 2,312.
2. **Tower cap** — a tile keeps at most 30 towers (`scene_spec.TOWER_CAP`); tallest/highest-power win.
3. **Coarse Sionna knobs** — `samples_per_tx=1e6`, `max_depth=3`, `cell_size=5 m` (vs 1e7 / 5 / 1 m),
   cutting each solve 10–100×. These are the app's sidebar defaults.
4. **Batched jobs** — coverage tiles are grouped so each job runs ~20–30 min; the setup notebook
   **calibrates** one tile to pick the batch size.

## The sidebar

| Group | Controls |
| --- | --- |
| Scene | `neighborhood`, `preset` (S1–S7 or Custom) |
| Towers | `tower_filter` (All/NR/LTE/UMTS/GSM), `power_override` (blank = per-tower random) |
| Antenna array | TX rows/cols, RX rows/cols, `pattern`, `polarization` |
| Radio | frequency (GHz), bandwidth (MHz) |
| Ray tracing | `max_depth`, samples per TX (10^x), cell X/Y (m) |
| User sampling | users/TX, min SINR, min/max distance |

Selecting a preset pushes its values into every knob, which is what makes it an exact cache hit.
Moving any knob off a preset flips the dropdown to **Custom** — a cache miss. Moving it back
restores the preset and its instant hit.

Presets are pre-rendered for **Downtown** only. Another neighborhood is a legitimate cache miss:
the app resolves its towers (a real SQL-warehouse read) and renders on the GPU job.

## Render modes (`jobs/render_job.py`)

| `mode` | Used by | What it does |
| --- | --- | --- |
| `custom` | the app, on cache miss | renders one config from `scene_json` + `config_hash`, asserting hash parity |
| `stories` | setup / re-seeding | renders the S1–S7 gallery over a neighborhood's core tile |
| `coverage` | backfill (no UI) | batch-renders a neighborhood's coverage tiles |

## Files

```
App/rf-digital-twin-lakebase/
├── app.py                  # Shiny UI + server: knobs → hash → cache-or-job → poll
├── app.yaml                # Apps config (Lakebase binding, job id, warehouse, PG_SCHEMA)
├── requirements.txt        # app runtime only — shiny, sdk, psycopg, numpy
├── neighborhoods.py        # metro bounding boxes + lat/lon → local-ENU projection
├── towers.py               # UC tower read via SQL warehouse + deterministic per-tower config
├── tiling.py               # neighborhood → tiles → time-boxed batches
├── scene_spec.py           # ★ shared scene resolution — the hash-parity guarantee
├── osm_scene.py            # headless OSM → Mitsuba scene (+ flat-ground fallback)
├── sionna_compute.py       # Sionna RT pipeline (GPU-side only; imports drjit/mitsuba)
├── defaults.py             # the S1–S7 stories + the knob dataclass
├── lakebase_client.py      # Postgres schema, OAuth token minting, hashing, reads/writes
├── render_pipeline.py      # render_custom / render_stories / render_coverage / calibrate
├── setup/setup_lakebase_twin.py   # one-shot: schema, calibrate, render the gallery
├── jobs/render_job.py             # the GPU job
└── tools/
    ├── check_hash_parity.py        # asserts presets resolve to cached renders
    └── seed_schema_from_seattle.py # copy a cache into a new schema, re-keyed
```

> The OSM + Sionna deps (`drjit`, `mitsuba`, `sionna-rt`, `requests`, `shapely`, `trimesh`,
> `mapbox_earcut`) belong to the **setup notebook and render job**, not the app — the app process
> only reads Lakebase and the tower table, so its `requirements.txt` stays light.
> `mapbox_earcut` is the triangulation backend `trimesh` needs to extrude building footprints;
> without it every building silently drops to the flat-ground fallback.

## Deploy

1. Run `setup/setup_lakebase_twin.py` on a **GPU cluster** (g5.xlarge, DBR 16.4 non-ML, internet
   egress for Overpass). It creates the schema, registers neighborhoods, calibrates, and renders
   the gallery into Lakebase. Idempotent.
2. Create the GPU job from `jobs/render_job.py` (spec in the top-level README) and put its id in
   `app.yaml` as `SIONNA_RENDER_JOB_ID`.
3. Create the Databricks App pointing at this directory, bind the Lakebase database, and grant the
   app service principal: `CAN MANAGE RUN` on the job, `CAN_USE` on the SQL warehouse, and
   `SELECT` on the tower table plus `USE_CATALOG`/`USE_SCHEMA` on its parents.
4. Run `tools/check_hash_parity.py`. All presets must resolve to a cached render before you demo it.

### Environment

| Variable | Purpose |
| --- | --- |
| `LAKEBASE_INSTANCE` | Lakebase instance name (default `rf-digital-twin-pg`) |
| `PG_SCHEMA` | Postgres schema for this app's cache (default `lakebase_only`) |
| `SIONNA_RENDER_JOB_ID` | GPU job to trigger on a cache miss |
| `SEATTLE_SQL_WAREHOUSE_ID` | warehouse for the tower read; unset = auto-pick a running serverless one |
| `DATABRICKS_WORKSPACE_URL` | builds clickable Jobs-UI links for in-flight runs |

`PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER` come from the Lakebase resource binding; `PGPASSWORD` is
never set — the app mints an OAuth token per session and caches it ~45 min.

## Caveats

- **Never put `--reload` in `app.yaml`.** It kills the Shiny session websocket in Databricks Apps,
  and every panel renders blank while the page still returns HTTP 200.
- **OSM dependency** — geometry comes from the public Overpass API. If it's unreachable or a tile
  has no buildings, that tile renders over a flat ground plane (propagation only). Generated scenes
  are cached on disk per tile bbox.
- **Monochromatic solve** — Sionna's `scene.frequency` and `scene.tx_array` are scene-level, so each
  render fixes one band and array. The per-tower random bands are stored and used to *filter* towers
  per story (S3 renders only NR towers at 28 GHz), not to mix bands in one solve.
- **The tower table needs a SQL warehouse** — its `geometry(4326)` column can't be read by Spark on
  DBR 16.4, which is why `towers.py` goes through the Statement Execution API on both sides.
- **GPU cost is real on a miss** — a fresh config is minutes of A10G time plus cluster start. The
  presets are pre-rendered so the demo path stays instant.
