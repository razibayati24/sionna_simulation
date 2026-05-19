# Sionna RF Digital Twin on Databricks

A reference implementation of NVIDIA Sionna RT as an interactive Databricks App. Edit a cell-network configuration in a Shiny sidebar; instantly see the resulting scene render, SINR coverage map, user-to-TX association, and SINR/RSS CDFs. Built around the Arc de Triomphe (etoile) scene with a 7-cell mmWave network — but the scaling characteristics matter long before you hit 100+ cells, so this README documents where the time goes and how to get to production scale.

---

## Two app variants

Pick the variant whose deployment fits your environment. **Both share the same UI, the same Sionna pipeline, and the same 19-preset cache** (`config_hash` is computed identically).

| | [Lakebase variant](App/rf-digital-twin-app/README.md) | [Lakehouse variant](App/rf-digital-twin-app-lakehouse/README.md) |
| --- | --- | --- |
| Cache store | Lakebase Postgres (`bytea`) | Unity Catalog Delta (`BINARY`) via SQL warehouse |
| Warm-hit latency | ~5–30 ms | ~200–400 ms |
| Idle cost | ~$200/mo (CU_1 always-on) | $0 (warehouse scale-to-zero) |
| Best for | OLTP-grade hot path, sustained traffic | Bursty traffic, cache-as-data-product, UC-only stack |
| Deployment guide | [`App/rf-digital-twin-app/README.md`](App/rf-digital-twin-app/README.md) | [`App/rf-digital-twin-app-lakehouse/README.md`](App/rf-digital-twin-app-lakehouse/README.md) |

The detailed tradeoffs (concurrent-user scaling, cost shape, hybrid pattern at real scale) are in [Lakebase vs Lakehouse — performance at scale](#lakebase-vs-lakehouse--performance-at-scale).

---

## Architecture (both variants)

Same three-stage pattern; the only thing that changes between variants is the cache layer in the middle.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  ① ONE-TIME SETUP — populates the cache                                 │
   │                                                                         │
   │     setup notebook   (run once on a GPU cluster, ~30–50 min)            │
   │     ─ creates schema / tables / database                                │
   │     ─ renders the 19 presets through Sionna RT                          │
   │     ─ writes scene render + SINR map + association + CDFs + KPIs        │
   └────────────────────────────────────────┬────────────────────────────────┘
                                            ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  CACHE LAYER  (Lakebase Postgres  OR  UC Delta tables)                  │
   │  ─ scene_configs       one row per saved scene-level config + hash      │
   │  ─ cell_configs        7 TX rows per scene config                       │
   │  ─ cached_renders      PNGs + KPI JSON keyed by config_hash             │
   │  ─ compute_jobs        run-id and status for live re-renders            │
   └────────────┬─────────────────────────────────────────┬──────────────────┘
                ▲ reads (cache hit)                       ▲ writes (on done)
                │                                         │
   ┌────────────┴───────────────────┐  cache miss   ┌─────┴─────────────────┐
   │  ② RUNTIME APP                 │ ── Jobs API ─▶│  ③ LIVE RE-RENDER     │
   │  Shiny on Databricks Apps      │               │  Sionna RT on a       │
   │  ─ user types sidebar values   │               │  g5.xlarge GPU job    │
   │  ─ Render → sha256(config)     │               │  cluster              │
   │  ─ hit: load cache in <1 s     │               │  ~5–8 min cold,       │
   │  ─ miss: submit job, poll cache│               │  ~2–3 min warm        │
   │  ─ Cancel button kills the run │               │                       │
   └────────────────────────────────┘               └───────────────────────┘
```

- **Path 1 — Cache hit (the demo path):** sidebar values → `config_hash` → cache row → tabs render in under a second.
- **Path 2 — Cache miss (the live-edit path):** off-menu config → live job → results flow into the cache → app picks them up via a background poller.
- **Setup is idempotent** — re-running the setup notebook only renders presets whose hash isn't already cached, so adding new configs later is cheap.

For exact notebook paths, cluster specs, and resource-binding commands per variant, see the per-app READMEs linked above.

---

## Preset gallery — what's cached

These are the 19 sidebar combinations that resolve to a **cached render** (instant load) in **either variant**. Anything outside this list triggers the live Sionna job. The `config_hash` algorithm is identical across Lakebase and Lakehouse, so the same hash prefix lands the same render regardless of which variant you're running.

> **Reading the table:** every column is a sidebar input. To reach a row, type **all** of its values in the app sidebar — partial matches (e.g. "20 MHz BW" without also setting "TX 16×16") will not hash to a cached row and will fall to the live job. **All cells in the table that aren't called out keep their Story-A default values** (28 GHz, 100 MHz, tr38901, V, 44 dBm, max_depth 5, 8×2 RX 2×2, etc).

### Story A — Antenna densification
Same 28 GHz, 100 MHz, tr38901, V polarization, 44 dBm, max_depth 5. Only TX UPA changes.

| Hash prefix | TX array | RX array | Freq | BW | Pattern | Pol | TX power | max_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `4d99ce0ad66c` | **2 × 2** | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | 5 |
| `1947611f5ab1` | **4 × 4** | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | 5 |
| `07934c589015` | **8 × 2** *(= Config 1)* | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | 5 |
| `9dc696f16498` | **8 × 8** | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | 5 |
| `7d40e2f4cf67` | **16 × 16** *(= Config 2)* | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | 5 |
| `1f83af551835` | **32 × 8** | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | 5 |

### Story B — Frequency band ladder
TX held at 8 × 2 (Config 1 array). Only frequency + bandwidth change.

| Hash prefix | TX array | RX array | Freq | BW | Pattern | Pol | TX power | max_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c97b934ed651` | 8 × 2 | 2 × 2 | **1.8 GHz** | **20 MHz** | tr38901 | V | 44 dBm | 5 |
| `8e58d2b3aa3b` | 8 × 2 | 2 × 2 | **2.6 GHz** | **20 MHz** | tr38901 | V | 44 dBm | 5 |
| `2b4207dac650` | 8 × 2 | 2 × 2 | **3.5 GHz** | 100 MHz | tr38901 | V | 44 dBm | 5 |
| `07934c589015` | 8 × 2 | 2 × 2 | **28 GHz** *(= Config 1)* | 100 MHz | tr38901 | V | 44 dBm | 5 |
| `411bcd0ac9fc` | 8 × 2 | 2 × 2 | **39 GHz** | **400 MHz** | tr38901 | V | 44 dBm | 5 |

### Story C — Antenna pattern
TX held at 16 × 16. Only pattern changes.

| Hash prefix | TX array | RX array | Freq | BW | Pattern | Pol | TX power | max_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `7d40e2f4cf67` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | **tr38901** *(= Config 2)* | V | 44 dBm | 5 |
| `74315cd0c5a0` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | **iso** | V | 44 dBm | 5 |
| `4d5194498776` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | **dipole** | V | 44 dBm | 5 |

### Story D — Polarization
TX held at 16 × 16, tr38901 pattern. Only polarization changes.

| Hash prefix | TX array | RX array | Freq | BW | Pattern | Pol | TX power | max_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `7d40e2f4cf67` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | **V** *(= Config 2)* | 44 dBm | 5 |
| `1dcfea9eb314` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | **VH** | 44 dBm | 5 |

### Story E — TX power
TX held at 16 × 16. Power applied uniformly across all 7 cells.

| Hash prefix | TX array | RX array | Freq | BW | Pattern | Pol | TX power | max_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `4d3b2585ea77` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | **38 dBm** | 5 |
| `7d40e2f4cf67` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | **44 dBm** *(= Config 2)* | 5 |
| `c9960aa40101` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | **50 dBm** | 5 |

### Story F — Bandwidth
TX held at 16 × 16. Only bandwidth changes.

| Hash prefix | TX array | RX array | Freq | BW | Pattern | Pol | TX power | max_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1b0bae11e49c` | 16 × 16 | 2 × 2 | 28 GHz | **20 MHz** | tr38901 | V | 44 dBm | 5 |
| `7d40e2f4cf67` | 16 × 16 | 2 × 2 | 28 GHz | **100 MHz** *(= Config 2)* | tr38901 | V | 44 dBm | 5 |
| `d7d1abe8d6b0` | 16 × 16 | 2 × 2 | 28 GHz | **400 MHz** | tr38901 | V | 44 dBm | 5 |

### Story G — Ray tracing fidelity
TX held at 16 × 16. Only max_depth changes.

| Hash prefix | TX array | RX array | Freq | BW | Pattern | Pol | TX power | max_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1bfc66c11279` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | **3** |
| `7d40e2f4cf67` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | **5** *(= Config 2)* |
| `74c31dcc5ea6` | 16 × 16 | 2 × 2 | 28 GHz | 100 MHz | tr38901 | V | 44 dBm | **8** |

> The Status tab in the app shows the live `config_hash`. When you've typed values that match a row in the tables above, the first 12 chars of that hash should equal the table's hash prefix — that's your visual confirmation that you're about to hit the cache.

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

## Lakebase vs Lakehouse — performance at scale

This repo ships **two app variants** that share everything except the cache layer:

- **`App/rf-digital-twin-app/`** — caches renders in Lakebase Postgres (`bytea` columns).
- **`App/rf-digital-twin-app-lakehouse/`** — caches renders in Unity Catalog Delta tables (`BINARY` columns) read through a SQL warehouse.

`config_hash` is computed identically on both sides, so the cache content is portable — the Lakehouse setup notebook even includes an optional migration cell that copies every preset from Lakebase straight into Delta in minutes, no Sionna re-runs needed.

For a single demo user clicking through cached presets, both feel instant. The architectural choice matters once you stress the read path.

### Hot-path latency per cache lookup

| Workload | Lakebase (CU_1) | Lakehouse (Small serverless SQL warehouse) |
| --- | --- | --- |
| Cold path (warehouse idle ≥5 min, scaled to zero) | 10–30 ms | **30–60 s warehouse cold-start**, then 200–500 ms |
| Warm hit, single row by `config_hash` | **5–15 ms** | 200–400 ms |
| Single 1 MB PNG blob fetch | **20–50 ms** | 300–800 ms |
| Sustained QPS per instance/warehouse | **1 000+** | 50–200 |

Where the Lakehouse latency goes for a warm hit:

- HTTPS handshake to the SQL endpoint (~50 ms)
- Query planning + execution coordination (~50–100 ms)
- BINARY result transport over Thrift (~50–200 ms for 1 MB)
- Python connector deserialization (~10–50 ms)

Lakebase keeps an open `psycopg` connection on the app side, so the steady-state read is one round-trip + indexed PK lookup + `bytea` fetch — typical Postgres territory.

### Concurrent user scaling

| Concurrent app users | Lakebase | Lakehouse |
| --- | --- | --- |
| 1–5 | Instant, flat latency | First lookup cold-starts the warehouse; rest are warm |
| 10–50 | Flat latency on CU_1 | Small warehouse may queue at the upper end → bump to Medium |
| 50–200 | Bump to CU_2 for headroom | Need serverless + multi-cluster load balancing |
| 200+ | CU_4 or sharded instances | Serverless auto-scaling + a connection-pool layer in front |

SQL warehouses cap concurrent queries per cluster (Small ≈ 10, Medium ≈ 20, Large ≈ 40). Past that, queries queue. Lakebase Postgres sustains 1 000+ short queries/sec per `CU_1` and scales near-linearly up the capacity tiers.

### Cost shape (rough monthly, AWS list)

| Traffic pattern | Lakebase | Lakehouse (serverless SQL) |
| --- | --- | --- |
| Idle / no traffic | **~$200/mo** (CU_1 always-on) | **~$0** (warehouse scaled to zero) |
| Light, intermittent (~10 req/hr) | ~$200/mo | ~$30–80/mo (occasional wake-ups) |
| Steady (~100 req/hr) | ~$200/mo | ~$200–400/mo (warehouse stays warm) |
| Heavy (1 000+ req/hr) | ~$200–400/mo (size up if needed) | ~$500–1 500/mo (warm + scaled) |
| Render bytes storage | bytea in Postgres, ~included | Delta on S3, pennies for this dataset |

The crossover is roughly an order-of-magnitude apart: **Lakehouse wins on intermittent / bursty workloads** because of scale-to-zero; **Lakebase wins on sustained high-QPS** because Postgres has better throughput-per-dollar at saturation.

### When to pick which

Pick **Lakebase** when:
- The app is the primary read path and users expect sub-50 ms.
- Traffic is steady or bursty (warehouse cold-start is unacceptable).
- You'll also do session-state / user-action writes from the app at OLTP latencies.
- You're already running other Lakebase workloads, so the ops overhead is amortised.

Pick **Lakehouse** when:
- The cached renders are *also* a data product that analysts will query directly (BI dashboards, surrogate training, drift monitoring).
- Traffic is intermittent and idle cost matters more than tail latency.
- You'd rather minimise managed-service surface area (everything stays in UC).
- Time-travel, column-level governance, or built-in lineage are hard requirements.

### Hybrid — what production looks like at real scale

For a metro-scale deployment (thousands of cached configs, dozens of analysts, an interactive app that needs sub-50 ms reads):

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  Lakehouse (Delta + UC)  ←  Sionna RT writes here                  │
   │  ─ system of record + analytics surface                            │
   │  ─ BI, dashboards, surrogate training, drift monitoring read here  │
   └─────────────────────────────────┬──────────────────────────────────┘
                                     │  CDC / Lakeflow continuous task
                                     ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  Lakebase (Postgres)  ←  read-through hot cache                    │
   │  ─ mirrors the latest version of each cached_renders row           │
   │  ─ sub-30 ms reads, 1 000+ QPS per instance                        │
   └────────────────────────────────────────────────────────────────────┘
                                     ▲
                                     │  app reads only here
                                     │
                                ┌────┴─────┐
                                │  App     │
                                └──────────┘
```

Delta is the source of truth and analytics surface. Lakebase is a thin read-only cache populated by a Lakeflow pipeline that streams new `cached_renders` rows into Postgres. The app talks only to Lakebase. Analysts query Delta directly. You pay for both, but each is sized for its actual workload.

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

### Option B — Approximate the simulation, end-to-end on Databricks

When you don't need photorealistic ray tracing, lean on cheaper models. The Sionna-specific knobs each cut render time 10–100× per tile:

| Knob | Default | Approximate setting | Speed-up | Accuracy hit |
| --- | --- | --- | --- | --- |
| `samples_per_tx` | 10⁷ | 10⁵ – 10⁶ | 10–100× | Noisier maps, especially shadows / sharp features |
| `max_depth` | 5 | 2 – 3 | 2–4× | Misses higher-order multi-path; LOS-dominated areas still accurate |
| `cell_size` | 1 m | 5 – 10 m | 25–100× | Coarser map; fine for coverage planning, not pixel-perfect |
| Antenna pattern | `tr38901` | `iso` | 10–20 % | Loses directional gain — only OK for sanity checks |
| Scene complexity | full OSM buildings | simplified mesh (5–10× fewer triangles) | 2–4× | Diffraction artifacts at building corners |
| ML surrogate | full RT | NN trained on RT outputs | 100–1000× | Learns the macro pattern; loses sharp interference fringes |

**Recommended pattern:** offer two modes in the app — *preview* (10⁶ samples, depth 3, cell_size 5 m → ~10× faster) for live iteration, *full* (10⁷ samples, depth 5, cell_size 1 m) for the committed cached result.

The biggest unlock is the **ML surrogate** 

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                ML surrogate lifecycle — all on Databricks                   │
│                                                                             │
│   ① Lakeflow Jobs ──────────► Sionna RT renders (Ray / Spark GPU pool)      │
│      (parallel data gen)      │  thousands of (config → radio_map) pairs    │
│                               ▼                                             │
│   ② Delta Lake + UC Volumes ──► training corpus, governed by Unity Catalog  │
│                               ▼                                             │
│   ③ Mosaic AI Model Training / AutoML ──► small CNN / U-Net surrogate       │
│                               ▼                                             │
│   ④ MLflow Tracking + UC Model Registry ──► versioned, stage-promoted       │
│                               ▼                                             │
│   ⑤ Mosaic AI Model Serving (autoscale → 0) ──► <50 ms inference endpoint   │
│                               ▼                                             │
│   ⑥ Shiny app ──► Lakebase cache  +  surrogate endpoint  +  full Sionna job │
│                               ▲                                             │
│   ⑦ Lakehouse Monitoring ─────┴─── drift alert ──► retrain (back to ①)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each stage and the Databricks primitive that runs it:

1. **Training data generation** — embarrassingly parallel Sionna RT renders driven by **Lakeflow Jobs** with a for-each task over a job-cluster pool, or **Ray on Databricks** for finer-grained per-GPU parallelism. Generate 10 k labelled pairs over a weekend on a small GPU pool.
2. **Storage + governance** — config metadata + KPIs in **Delta Lake** for columnar analytics ("show me tiles where surrogate error > 5 dB"), raw radio-map tensors in **UC Volumes**, the whole lineage versioned by **Unity Catalog**. Same dataset is queryable from a notebook, BI dashboard, or model training job.
3. **Training** — **Mosaic AI Model Training** for distributed training on multi-GPU clusters, or **AutoML** for a one-click baseline that beats hand-rolled architectures in ~an hour. **Feature Store** registers the RF config schema as a versioned feature set so training and serving share the exact same definition.
4. **Experiment tracking + registry** — every run logged to **MLflow Tracking** with hyperparameters, training curves, RMSE-vs-Sionna on held-out tiles. **UC-backed Model Registry** holds the champion + challengers with dev/staging/prod stage transitions and per-model permissions.
5. **Serving** — **Mosaic AI Model Serving** hosts the surrogate as an autoscaling REST endpoint (scale-to-zero when idle). The app calls it for **interactive preview** during sidebar edits — sub-second feedback. When the user commits, the **same Lakebase cache** + the full Sionna job (Option A) handles the production render. Both surrogate and Sionna outputs flow back into the same cache, so a future query lands on whichever is fresher.
6. **Drift monitoring + retraining** — **Lakehouse Monitoring** computes daily distributional metrics between surrogate predictions and the periodic ground-truth Sionna renders. Alerts trigger a **Lakeflow workflow** that pulls the latest pairs, retrains, and auto-promotes the new model via **MLflow webhooks** if it beats the current champion on a held-out evaluation set.
7. **Analyst access** — RF engineers query the simulation results in natural language via **Genie**, build dashboards in **AI/BI**, or do similarity search ("find the cached config closest to my edits") with **Mosaic AI Vector Search**. All without writing a single line of model-serving code.

**Why Databricks is the right home for this — not a generic ML stack:**

- **One platform, one identity, one bill.** GPU training, Postgres serving, Delta storage, model registry, vector search, dashboards — all under the same workspace, the same Unity Catalog permissions, the same cost-attribution tags. No SageMaker ↔ RDS ↔ Snowflake ↔ Weights-and-Biases shuttle.
- **The training data and the production data are the same table.** The Delta table that holds 100 k Sionna renders is exactly what the Lakeflow pipeline updates nightly and what Mosaic AI Model Training reads from. No ETL between research and production.
- **Governance is built in.** Unity Catalog handles row/column-level security on the training corpus, lineage tracking from raw OSM data → cached render → trained model → serving endpoint, and audit logs for every change. For a telco shipping this internationally, that's table stakes that AWS-native equivalents need a separate compliance lift to match.
- **Cost flexibility.** Serverless GPU for ad-hoc training, job clusters for nightly batches, scale-to-zero serving for the surrogate endpoint, Lakebase for hot reads. You don't pay for idle infrastructure the way a self-managed Triton + EKS setup would force you to.
- **Iteration speed.** A data scientist can prototype a new surrogate in a notebook attached to the same Delta table the production pipeline uses, log to the same MLflow experiment, and promote a winning model to production with a single CLI call. The "deploy a Python model" friction that motivates most "let's use a managed ML platform" decisions just isn't there.

For a telco RF team, the practical decision tree looks like:

| If you want to… | The Databricks path | The DIY equivalent |
| --- | --- | --- |
| Render a few configs, share results | Notebook + plot | … same |
| Render thousands of configs nightly | Lakeflow Jobs + Spark/Ray on GPU pool | Custom Airflow + EKS + spot reaper |
| Cache renders for a Shiny app | Lakebase + Databricks Apps | RDS + ECS + ALB + IAM |
| Train an ML surrogate | Mosaic AI Model Training + MLflow + UC | SageMaker / Vertex + own registry + own permissions |
| Serve the surrogate at <50 ms | Mosaic AI Model Serving (scale-to-zero) | Triton + EKS + autoscaler + auth proxy |
| Detect surrogate drift, retrain | Lakehouse Monitoring + Lakeflow trigger | Custom metrics + Airflow + alerting glue |
| Let RF engineers query results | Genie + AI/BI Dashboards | Separate BI tool + secondary data warehouse |

Same goal, **one platform vs. seven**.

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
├── README.md                                       # this file
├── rf_planning_optimization/
│   └── simulation_RT_light.ipynb                   # original notebook (single-shot demo)
└── App/
    ├── rf-digital-twin-app/                        # Lakebase variant (Postgres cache)
    │   ├── README.md                               # app-specific deployment guide
    │   ├── app.py                                  # Shiny UI + server
    │   ├── app.yaml                                # Databricks Apps deploy config
    │   ├── requirements.txt
    │   ├── defaults.py                             # Config 1, Config 2, 7-cell layout
    │   ├── lakebase_client.py                      # Postgres connection + queries
    │   ├── sionna_compute.py                       # Sionna RT pipeline (shared)
    │   ├── setup/
    │   │   └── setup_rf_digital_twin.py            # one-shot Lakebase setup
    │   └── jobs/
    │       └── sionna_compute_job.py               # cache-miss job (Postgres write)
    │
    └── rf-digital-twin-app-lakehouse/              # Lakehouse variant (UC Delta cache)
        ├── README.md                               # comparison + deployment guide
        ├── app.py                                  # identical UI; imports lakehouse_client
        ├── app.yaml                                # binds SQL warehouse instead of Lakebase
        ├── requirements.txt                        # databricks-sql-connector + sdk
        ├── defaults.py                             # identical preset definitions
        ├── lakehouse_client.py                     # SQL-connector + Delta read/write
        ├── sionna_compute.py                       # identical pipeline
        ├── setup/
        │   └── setup_rf_digital_twin_lakehouse.py  # incl. optional Lakebase→Delta migration
        └── jobs/
            └── sionna_compute_job_lakehouse.py     # cache-miss job (Spark Delta write)
```

---

## Further reading

- [Sionna RT documentation](https://nvlabs.github.io/sionna/rt/index.html) — official ray-tracing docs and tutorials.
- [Lakebase docs](https://docs.databricks.com/aws/en/oltp/) — managed Postgres on Databricks (Lakebase OLTP).
- [Databricks Apps docs](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) — running Shiny / FastAPI / etc. apps.
- Medium series this project is built on:
  - [Part I — "NVIDIA's AI-Native Digital Twin on Databricks"](https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-bdb81ef87b70)
  - [Part II](https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-ii-065938ca112c)
