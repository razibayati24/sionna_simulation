# Seattle Coverage Analysis — RF Digital Twin

A Seattle variant of the RF Digital Twin. Instead of a synthetic 7-cell ring on the Paris
`etoile` scene, this drives Sionna RT from **real T-Mobile towers** in
`cmegdemos_catalog.network_analytics_enablement.cell_towers` (2,312 towers across the metro)
ray-traced against **OpenStreetMap Seattle building geometry**, with results cached in the
same `rf-digital-twin-pg` Lakebase instance.

## What's different from the etoile demo

| | etoile demo (`App/rf-digital-twin-app/`) | this app (`App/seattle-rf-digital-twin/`) |
| --- | --- | --- |
| Transmitters | synthetic 7-cell ring | **real towers** from Unity Catalog, projected to local meters |
| Geometry | built-in Paris `etoile` mesh | **OSM Seattle buildings** per tile (flat-ground fallback) |
| Scale unit | one scene | **neighborhood → tiles** (2,312 towers can't share one scene) |
| Gallery | presets A–G | **Seattle stories S1–S7** over the Downtown core tile |
| Cache miss | custom config → job | **uncached neighborhood → job** (dropdown-driven) |
| Fidelity | full (1e7 samples, depth 5, 1 m) | **approximated** (1e6, depth 3, 5 m) — see below |

## How it scales 2,312 towers — the approximation

A radio-map solve is ~`O(num_tx × samples_per_tx × max_depth)` with an output tensor of
`area / cell_size²` cells per TX, so the whole 44×28 km metro in one scene is infeasible.
Four levers (all in `tiling.py` / `defaults.py` / `sionna_compute.py`) bring it down:

1. **Tiling** — the metro is cut into named **neighborhoods**; each neighborhood is gridded
   into ~800 m **tiles** with a 150 m overlap margin (so coverage isn't clipped at seams).
   A tile renders ~10–30 towers, not 2,312.
2. **Tower cap** — a tile keeps at most 30 towers (tallest/highest-power dominate).
3. **Coarse Sionna knobs** — `samples_per_tx=1e6`, `max_depth=3`, `cell_size=5 m`
   (vs 1e7 / 5 / 1 m), cutting each solve 10–100×.
4. **Batched jobs** — tiles are grouped so each render job runs ~20–30 min; the setup
   notebook **calibrates** one tile to pick the batch size.

## Neighborhood dropdown (render-on-demand)

- **Downtown** ships pre-rendered: the S1–S7 story gallery + coverage tiles load from cache instantly.
- Selecting any other neighborhood (Capitol Hill, Bellevue, Ballard, …) and clicking
  **Render this neighborhood** submits the `seattle-rf-render` GPU job, flips the
  neighborhood to `RENDERING`, and a background poller fills the tabs as batches land —
  the same cache-miss → Jobs → poll pattern as the etoile app, at neighborhood grain.

## Files

```
App/seattle-rf-digital-twin/
├── app.py                  # Shiny UI — neighborhood + story dropdowns, render trigger, tabs
├── app.yaml                # Apps deploy config (binds rf-digital-twin-pg, SEATTLE_RENDER_JOB_ID)
├── requirements.txt        # app runtime deps (shiny, sdk, psycopg) — light, read-only
├── neighborhoods.py        # metro bounding boxes + lat/lon → local-ENU projection
├── towers.py               # load towers from UC + deterministic per-tower random config
├── tiling.py               # neighborhood → tiles → time-boxed batches
├── osm_scene.py            # headless OSM → Mitsuba scene (+ flat-ground fallback)
├── sionna_compute.py       # Sionna RT pipeline (OSM scene + real-tower TXs + approximation)
├── defaults.py             # Seattle stories S1–S7
├── lakebase_client.py      # neighborhood/tile-aware Lakebase schema + queries
├── render_pipeline.py      # render_stories / render_coverage / calibrate (shared)
├── setup/
│   └── setup_seattle_rf_digital_twin.py   # one-shot: schema, calibrate, render Downtown
└── jobs/
    └── seattle_render_job.py              # on-demand neighborhood render (app triggers this)
```

> The OSM + Sionna deps (`drjit`, `mitsuba`, `sionna-rt`, `requests`, `shapely`, `trimesh`,
> `mapbox_earcut`) are installed by the **setup notebook and render job** via `%pip` — the app
> process itself only reads from Lakebase, so its `requirements.txt` stays light.
> `mapbox_earcut` is the triangulation backend `trimesh` needs to extrude building footprints;
> without it every building silently drops to the flat-ground fallback.

## Deploy

1. Run `setup/setup_seattle_rf_digital_twin.py` on a **GPU cluster** (g5.xlarge, DBR 16.4
   non-ML; needs internet egress for Overpass). It initialises the schema, registers
   neighborhoods, calibrates, and renders Downtown into Lakebase.
2. Create the GPU job from `jobs/seattle_render_job.py`; put its `job_id` in `app.yaml` as
   `SEATTLE_RENDER_JOB_ID`.
3. Create the Databricks App pointing at this directory; bind the `rf-digital-twin-pg`
   Lakebase database and grant the app SP `CAN USE` on the instance + `CAN MANAGE RUN` on the job.
4. Open the app — Downtown loads instantly; pick another neighborhood to render on demand.

## Caveats

- **OSM dependency** — building geometry comes from the public Overpass API. If it's
  unreachable or a tile has no buildings, that tile renders over a flat ground plane
  (propagation only). Generated scenes are cached on disk per tile bbox.
- **Monochromatic solve** — Sionna's `scene.frequency` / `scene.tx_array` are scene-level,
  so each render fixes the band + array; the per-tower random band/array are stored and used
  to *filter* towers per story (e.g. S3 renders only NR towers at 28 GHz), not to mix bands
  in a single solve.
- **GPU cost/time** — rendering a fresh neighborhood is real GPU time (a few 20–30 min
  batches). Downtown is pre-rendered so the demo path is instant.
