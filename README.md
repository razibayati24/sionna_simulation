# Sionna RF Digital Twin on Databricks — Lakebase variant

A reference implementation of NVIDIA Sionna RT as an interactive Databricks App. Edit a cell-network configuration in a Shiny sidebar; instantly see the resulting scene render, SINR coverage map, user-to-TX association, and SINR/RSS CDFs. Built around the Arc de Triomphe (etoile) scene with a 7-cell mmWave network.

This branch is trimmed to **one deployment path**: the Lakebase Postgres variant, deployed as a Databricks App. If you want to stand the app up and nothing else, everything you need is here.

---

## What you get

Two apps, deployed independently:

| App | What it is | Guide |
| --- | --- | --- |
| [`App/rf-digital-twin-app/`](App/rf-digital-twin-app/README.md) | **The digital twin.** Shiny app over a Lakebase Postgres render cache. Sub-30 ms cache hits; off-menu configs ray-trace live on a GPU job. | [deployment guide](App/rf-digital-twin-app/README.md) |
| [`App/chapter-0-rf-agent/`](App/chapter-0-rf-agent/README.md) | **Chapter 0.** A self-running, cinematic showcase that opens the story — "the network that fixes itself." Pure static front-end, no cluster or database needed. | [guide](App/chapter-0-rf-agent/README.md) |

Deploy Chapter 0 first if you're presenting: it sets up *why* the twin matters, then the twin lets you drive it by hand.

---

## Deploy the digital twin

Full step-by-step (GPU cluster spec, setup notebook, Lakebase provisioning, resource bindings, service-principal grants) lives in **[`App/rf-digital-twin-app/README.md`](App/rf-digital-twin-app/README.md)**. The short version, once setup has run:

```bash
export APP_NAME=rf-digital-twin-v2
export APP_SRC=/Workspace/Users/<you>/sionna_simulation/App/rf-digital-twin-app

# 1. Create the app + bind the Lakebase database
databricks apps create --json '{
  "name": "'"$APP_NAME"'",
  "description": "Sionna RT digital twin (Lakebase variant)",
  "resources": [{
    "name": "lakebase",
    "description": "Lakebase Postgres cache for Sionna renders",
    "database": {
      "instance_name": "rf-digital-twin-pg",
      "database_name": "rf_digital_twin",
      "permission": "CAN_CONNECT_AND_CREATE"
    }
  }]
}'

# 2. Grant the app service principal CAN_MANAGE_RUN on the live-render job
#    (required for cache misses; get the SP id from step 1's output)
databricks permissions update jobs <job_id> --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_client_id>",
    "permission_level": "CAN_MANAGE_RUN"
  }]
}'

# 3. Deploy the source
databricks apps deploy $APP_NAME --source-code-path $APP_SRC

# 4. Open it
databricks apps get $APP_NAME --output json | jq -r .url
```

> **Don't add `--reload` to the `shiny run` command in `app.yaml`.** It kills the Shiny
> session websocket on Databricks Apps and the panels come up blank.

---

## Architecture

Three stages: a one-time setup pass populates the cache, the app reads it on the hot path, and off-menu configs fall through to a GPU job that writes back.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  ① ONE-TIME SETUP — populates the cache                                 │
   │                                                                         │
   │     setup/setup_rf_digital_twin.py  (once, GPU cluster, ~30–50 min)     │
   │     ─ creates the UC schema, Lakebase instance + database, tables        │
   │     ─ renders the 19 presets through Sionna RT                          │
   │     ─ writes scene render + SINR map + association + CDFs + KPIs        │
   └────────────────────────────────────────┬────────────────────────────────┘
                                            ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  LAKEBASE POSTGRES  (instance: rf-digital-twin-pg / db: rf_digital_twin)│
   │  ─ scene_configs       one row per saved scene-level config + hash      │
   │  ─ cell_configs        7 TX rows per scene config                       │
   │  ─ cached_renders      PNG bytea + KPI JSONB keyed by config_hash       │
   │  ─ compute_jobs        run-id and status for live re-renders            │
   └────────────┬─────────────────────────────────────────┬──────────────────┘
                ▲ reads (cache hit, psycopg)              ▲ writes (on done)
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

- **Path 1 — Cache hit (the demo path):** sidebar values → `config_hash` → Postgres row → tabs render in under a second.
- **Path 2 — Cache miss (the live-edit path):** off-menu config → live GPU job → results written to Lakebase → app picks them up via a background poller.
- **Setup is idempotent** — re-running it only renders presets whose hash isn't already cached, so adding configs later is cheap.
- **No `PGPASSWORD`** — the Lakebase binding populates `PGHOST` / `PGPORT` / `PGDATABASE` / `PGUSER`; `lakebase_client.py` mints a fresh OAuth token at runtime via the Databricks SDK and caches it for 45 minutes.

---

## Preset gallery — what's cached

These are the 19 sidebar combinations that resolve to a **cached render** (instant load). Anything outside this list triggers the live Sionna job.

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

---

## Repo layout

```
.
├── README.md                                       # this file
├── rf_planning_optimization/
│   └── simulation_RT_light.ipynb                   # original notebook (single-shot demo)
└── App/
    ├── rf-digital-twin-app/                        # the digital twin (Lakebase cache)
    │   ├── README.md                               # full deployment guide
    │   ├── app.py                                  # Shiny UI + server
    │   ├── app.yaml                                # Databricks Apps deploy config
    │   ├── requirements.txt
    │   ├── defaults.py                             # Config 1, Config 2, 7-cell layout
    │   ├── lakebase_client.py                      # Postgres connection + queries
    │   ├── sionna_compute.py                       # Sionna RT pipeline
    │   ├── setup/
    │   │   └── setup_rf_digital_twin.py            # one-shot Lakebase setup
    │   └── jobs/
    │       └── sionna_compute_job.py               # cache-miss job (Postgres write)
    │
    └── chapter-0-rf-agent/                         # Chapter 0 — static showcase
        ├── README.md
        ├── index.html / app.js / style.css
        ├── towers.json                             # baked real tower locations
        └── app.yaml                                # static file server
```

---

## Further reading

- [Sionna RT documentation](https://nvlabs.github.io/sionna/rt/index.html) — official ray-tracing docs and tutorials.
- [Lakebase docs](https://docs.databricks.com/aws/en/oltp/) — managed Postgres on Databricks (Lakebase OLTP).
- [Databricks Apps docs](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) — running Shiny / FastAPI / etc. apps.
- Medium series this project is built on:
  - [Part I — "NVIDIA's AI-Native Digital Twin on Databricks"](https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-bdb81ef87b70)
  - [Part II](https://medium.com/@razibayati20/nvidias-ai-native-digital-twin-on-databricks-true-ai-democratization-for-telecom-ii-065938ca112c)
