# Sionna RF Digital Twin on Databricks

A reference implementation of NVIDIA Sionna RT as an interactive Databricks App, backed by Lakebase Postgres for cached renders and a Databricks Job for on-demand custom configurations. Built around the Arc de Triomphe (etoile) scene with a 7-cell mmWave network — but the scaling characteristics matter long before you hit 100+ cells, so this README documents where the time goes and how to get to production scale.

For app-specific deployment instructions, see [`App/rf-digital-twin-app/README.md`](App/rf-digital-twin-app/README.md). For the original notebook that the app is built around, see [`rf_planning_optimization/simulation_RT_light.ipynb`](rf_planning_optimization/simulation_RT_light.ipynb).

---

## What this project is

A digital twin of an RF network running in Databricks. Edit antenna parameters, ray-tracing settings, or per-TX positions / power; render the scene with Sionna RT; visualise the coverage (scene render, SINR map, user-to-TX association, SINR/RSS CDFs); compare configurations side-by-side.

Three components:

1. **`App/rf-digital-twin-app/`** — Shiny app deployed via Databricks Apps. Reads cached renders from Lakebase. On cache miss, submits the Sionna compute job.
2. **`App/rf-digital-twin-app/setup/setup_rf_digital_twin.py`** — One-shot setup notebook. Provisions Lakebase, creates the UC schema (`cmegdemos_catalog.sionna_rf_data`), seeds the default 7-cell network, runs Sionna for the preset configs, caches everything.
3. **`App/rf-digital-twin-app/jobs/sionna_compute_job.py`** — Databricks Job notebook. Triggered by the app on cache miss. Runs Sionna RT on a GPU job cluster and writes results back to Lakebase.

```
┌─────────────────┐    cache miss    ┌────────────────────┐
│ Shiny app       │ ───── Jobs API ──▶│ Sionna compute job │
│ (Databricks App)│                  │ (g5.xlarge GPU)    │
└────────┬────────┘                  └─────────┬──────────┘
         │                                     │
         │       reads renders                 │ writes renders
         │       on cache hit                  ▼
         │                            ┌────────────────────┐
         └───────────────────────────▶│ Lakebase Postgres  │
                                      │  scene_configs     │
                                      │  cell_configs      │
                                      │  cached_renders    │
                                      └────────────────────┘
```

---

## Where the time goes

A single Sionna RT render breaks down roughly as:

| Stage | What it does | Share of total time |
| --- | --- | --- |
| **`RadioMapSolver`** | Ray tracing — fires `samples_per_tx` rays from each TX, traces up to `max_depth` bounces, computes per-cell SINR / RSS / path gain. **GPU-bound.** | **~85–95 %** |
| `scene.render` | Renders the 3D scene from a top-down camera with the radio map overlaid as a colour map. | ~3–5 % |
| `radio_map.show_association` | Plots SINR-best-TX assignment as a 2D image. | ~1–2 % |
| `radio_map.sample_positions` + user plot | Samples user positions per TX, plots them coloured by serving TX. | ~1–2 % |
| `radio_map.cdf` ×2 | Computes + plots SINR and RSS CDFs over the cell grid. | ~1–2 % |
| Lakebase write | PNG bytea + KPI JSON insert. | <1 % |

**The ray-tracing solver is what scales.** Its cost is roughly `O(num_tx × samples_per_tx × max_depth)` — linear in TX count, linear in samples, super-linear in depth.

The other variables that move the needle:
- **`cell_size`** — coverage area divided by cell_size² gives the radio map tensor size (per TX). At 1 m cells over a 600×600 m area you have ~360 k cells per TX. Drop to `cell_size=5` and the per-TX map shrinks 25×.
- **TX antenna array (`num_rows_tx × num_cols_tx`)** — bigger arrays slow down per-ray channel computation. 16×16 is ~5–10 % slower than 8×2 in practice.
- **`max_depth`** — each extra bounce roughly doubles ray traversal cost. Default 5 is moderate; pushing to 8 makes the simulation 2–4× slower with minimal accuracy gain in most urban scenes.

---

## Render time — compute × cell count

Estimates for the etoile scene with `samples_per_tx=10⁷`, `max_depth=5`, `cell_size=(1, 1)`, mixed 8×2 / 16×16 TX arrays. **Per single render.** Real workloads vary ±30 %; treat as planning estimates, not SLAs.

| Compute (driver) ↓ / Number of cells → |   **7**   |  **25**  |  **50**  |   **100**   |   **250**    |   **500**    |
| --- | --- | --- | --- | --- | --- | --- |
| **CPU only** (`c5.4xlarge`, 16 vCPU) | 30–45 min | 1.5–2 h | 3–4 h | 6–8 h | 16+ h | OOM† |
| **g4dn.xlarge** (1× T4, 16 GB) | 4–6 min | 15–20 min | 30–40 min | 60–90 min | 4–5 h | OOM† |
| **g5.xlarge** (1× A10G, 24 GB) ← *current setup* | 2–3 min | 7–10 min | 15–25 min | 30–45 min | 2–3 h | OOM† |
| **g5.4xlarge** (1× A10G, 24 GB, big CPU/RAM) | 2–3 min | 6–9 min | 12–20 min | 25–35 min | 1.5–2 h | 4–5 h |
| **g5.12xlarge** (4× A10G, 96 GB total) | 1.5–2 min | 3–5 min | 6–10 min | 10–15 min | 30–45 min | 1–1.5 h |
| **g6.xlarge** (1× L4, 24 GB) | 2–3 min | 6–9 min | 12–18 min | 22–32 min | 1–1.5 h | OOM† |
| **p4d.24xlarge** (8× A100, 320 GB total) | 1–1.5 min | 1.5–3 min | 3–5 min | 5–8 min | 12–18 min | 25–35 min |

† **OOM**: A10G/T4/L4 GPUs (~24 GB VRAM) hit memory limits when you combine many TXs with a fine `cell_size` and high `samples_per_tx`. Mitigations: increase `cell_size` to 2–5 m, drop `samples_per_tx` to 10⁶, or move to multi-GPU instances.

The numbers above are dominated by the ray tracing solver (see the table earlier). Post-processing (scene render, CDFs, association plot, Lakebase write) is **~30–60 s of fixed overhead** regardless of TX count, so for tiny scenes (7 cells, 1× A10G) that overhead is 25 % of total time; for 500 cells it's <1 %.

---

## How to scale

Two complementary strategies. Both are needed for production-scale deployments (full metro coverage, ~thousands of cells).

### Option A — Pre-compute by area, serve from cache

Treat the simulation as a **data product**. The output of Sionna is deterministic for a given (scene, cell config, tracing params) — exactly the kind of computation that benefits from offline materialization.

```
┌─────────────────────────┐      offline / nightly         ┌───────────────────────┐
│ City divided into tiles │ ── Sionna RT on each tile ───▶ │ Pre-rendered cache    │
│ (e.g. H3 hexes, 1–5 km) │    in parallel jobs            │ (Lakebase + Volumes)  │
└─────────────────────────┘                                └──────────┬────────────┘
                                                                      │
                                                          read-only   │
                                                                      ▼
                                                            ┌──────────────────┐
                                                            │ Inference app    │
                                                            │  (latency: ms)   │
                                                            └──────────────────┘
```

**How it works:**
- Decompose the geography into tiles (H3 hex resolution 7–9, or arbitrary rectangles). Each tile covers a few km² and a manageable number of cells (10–50).
- For each tile, render Sionna once with the **current** TX layout in that tile, cache the result. Re-render only when the layout in that tile changes.
- The app stitches per-tile results at query time.
- A new build / network change → render only the affected tiles, not the entire city.
- **Storage strategy:**
  - **Lakebase** for hot-path lookups (per-tile config_hash → rendered PNG + KPI JSON). Sub-100 ms reads.
  - **Lakehouse (Delta + UC Volumes)** for the raw radio map tensors, raster outputs, and historical comparisons. Better for analytics ("show me SINR p10 distribution across all tiles, by month").
  - Hybrid: indexes + KPIs in Lakebase, big blobs in Volumes addressed by URL.

**Cost / latency trade-off:** a one-time render of every tile is expensive (think: 1000 tiles × 5 min on g5.xlarge ≈ 80 hours of GPU time, ~$80–100). But you pay it once. Day-to-day, the app serves from cache in milliseconds.

**Cache invalidation:**
- TX configs are versioned (`scene_configs` table already supports this).
- Whenever a tower changes, mark the affected tile's hash stale and re-render. Use Lakeflow / Workflows to drive the orchestration.

### Option B — Approximate the simulation

When you don't need photorealistic ray tracing, lean on cheaper models. Each of these can cut render time 10–100× per tile.

| Knob | Default | Approximate setting | Speed-up | Accuracy hit |
| --- | --- | --- | --- | --- |
| `samples_per_tx` | 10⁷ | 10⁵ – 10⁶ | 10–100× | Noisier maps, especially shadows / sharp features |
| `max_depth` | 5 | 2 – 3 | 2–4× | Misses higher-order multi-path; LOS-dominated areas still accurate |
| `cell_size` | 1 m | 5 – 10 m | 25–100× | Coarser map; fine for coverage planning, not pixel-perfect |
| Antenna pattern | `tr38901` | `iso` | 10–20 % | Loses directional gain — only OK for sanity checks |
| Scene complexity | full OSM buildings | simplified mesh (5–10× fewer triangles) | 2–4× | Diffraction artifacts at building corners |
| ML surrogate | full RT | NN trained on RT outputs | 100–1000× | Learns the macro pattern; loses sharp interference fringes |

**Recommended pattern:** offer two modes in the app — *preview* (10⁶ samples, depth 3, cell_size 5 m → ~10× faster) for live iteration, *full* (10⁷ samples, depth 5, cell_size 1 m) for the committed cached result.

**ML surrogate** is the long-game move. Train a small CNN / U-Net on a few thousand (config → radio_map) pairs from real Sionna runs. Inference is ~1 second instead of minutes; you still validate against real Sionna periodically. Sionna RT was designed with this workflow in mind — the [RT tutorials](https://nvlabs.github.io/sionna/rt/tutorials.html) cover differentiable ray tracing end-to-end, and the broader [Sionna research](https://developer.nvidia.com/blog/tag/sionna/) corpus from NVIDIA has examples of learned propagation models. For production RAN, the surrogate pattern is what underpins NVIDIA's [Aerial platform](https://developer.nvidia.com/aerial).

### Putting it together

For a metro deployment:

1. **Tile the city** with H3 hex resolution 8 (~0.7 km² per tile).
2. **Nightly Lakeflow job** renders Sionna for tiles whose `scene_configs` hash changed in the last 24 h. Parallelism across tiles is embarrassingly parallel — use a Workflows for-each over a job cluster pool.
3. **Lakebase** caches the renders. Index by `(tile_id, config_hash)` so the app can mix-and-match across tiles.
4. **App** stitches tiles for the visible viewport at query time. Adds preview-mode rendering for in-session edits before committing to a full re-render.
5. **ML surrogate** lives alongside Sionna; the app shows surrogate output during interactive editing, then triggers a real Sionna render for the committed configuration. The surrogate is retrained on Lakeflow weekly from the latest cached pairs.

---

## Repo layout

```
.
├── README.md                                # this file
├── rf_planning_optimization/
│   └── simulation_RT_light.ipynb            # original notebook (single-shot demo)
└── App/
    └── rf-digital-twin-app/
        ├── README.md                        # app-specific deployment guide
        ├── app.py                           # Shiny UI + server
        ├── app.yaml                         # Databricks Apps deploy config
        ├── requirements.txt
        ├── defaults.py                      # Config 1, Config 2, default 7-cell layout
        ├── lakebase_client.py               # Postgres connection + schema + queries
        ├── sionna_compute.py                # Sionna RT pipeline (shared)
        ├── setup/
        │   └── setup_rf_digital_twin.py     # one-shot workspace setup
        └── jobs/
            └── sionna_compute_job.py        # notebook triggered by the app on cache miss
```

---

## Further reading

- [Sionna RT documentation](https://nvlabs.github.io/sionna/rt/index.html) — official ray-tracing docs and tutorials.
- [Lakebase docs](https://docs.databricks.com/aws/en/oltp/) — managed Postgres on Databricks (Lakebase OLTP).
- [Databricks Apps docs](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) — running Shiny / FastAPI / etc. apps.
- Medium series this project is built on:
  - [Part I — "NVIDIA's AI-Native Digital Twin on Databricks"](https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-bdb81ef87b70)
  - [Part II](https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-ii-065938ca112c)
