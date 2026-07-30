"""RF Digital Twin app — Sionna RT on Databricks.

Edit the scene + per-TX configuration, hit "Render", and the app either pulls
a cached result from Lakebase (preset configs) or submits a Databricks job to
run Sionna RT for a custom configuration. Results: scene render, SINR map,
user-to-TX association, and CDFs of SINR/RSS.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import traceback
from typing import Any

from shiny import App, ui, render, reactive

import lakebase_client as lb
from defaults import CONFIG_1, preset_cells
from large_scale_defaults import REGION_PRESETS, DEFAULT_REGION, RegionConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIONNA_JOB_ID = os.environ.get("SIONNA_JOB_ID")
DATABRICKS_WORKSPACE_URL = os.environ.get(
    "DATABRICKS_WORKSPACE_URL",
    f"https://{os.environ.get('DATABRICKS_HOST', '').rstrip('/')}",
).rstrip("/")

# Estimated wall-clock for a cold job cluster run (used for the wait ETA).
JOB_ETA_SECONDS = 6 * 60

# Databricks Job that runs the large-scale (sionna_lrm) pipeline on a GPU
# cluster. Optional — when unset, the tab renders the GPU-free demo instead.
LARGE_SCALE_JOB_ID = os.environ.get("LARGE_SCALE_JOB_ID")
# A large-region ray-trace across many tiles takes far longer than one scene.
LARGE_SCALE_JOB_ETA_SECONDS = 25 * 60

PATTERN_CHOICES = ["tr38901", "iso", "dipole", "hw_dipole"]
POLARIZATION_CHOICES = ["V", "H", "VH", "cross"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_img(data: bytes | memoryview | None, alt: str) -> ui.Tag:
    """Wrap a PNG byte string into an <img>."""
    if not data:
        return ui.tags.div(
            f"No {alt} available — render hasn't completed yet.",
            style="padding: 20px; color: #888;",
        )
    if isinstance(data, memoryview):
        data = bytes(data)
    b64 = base64.b64encode(data).decode("ascii")
    return ui.tags.img(
        src=f"data:image/png;base64,{b64}",
        style="max-width: 100%; height: auto; border-radius: 4px;",
        alt=alt,
    )


def _as_b64(data: bytes | memoryview | None) -> str | None:
    """Base64-encode PNG bytes for a data URI, or None."""
    if not data:
        return None
    if isinstance(data, memoryview):
        data = bytes(data)
    return base64.b64encode(data).decode("ascii")


def _as_obj(value: Any) -> Any:
    """Coerce a JSONB column (already parsed) or a JSON string to a Python obj."""
    if value is None:
        return None
    if isinstance(value, (bytes, str)):
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return None
    return value


def _leaflet_map_html(overlay_b64: str | None, bounds: list | None,
                      base_stations: list | None, legend_b64: str | None,
                      title: str) -> str:
    """Build a self-contained Leaflet HTML doc for an <iframe srcdoc>.

    Renders a pannable/zoomable OSM basemap with the coverage raster as a
    georeferenced image overlay on `bounds` ([south, west, north, east]),
    base-station markers, an opacity slider, and a colour-bar legend — the
    same overlay-on-slippy-map idea as NVlabs sionna-large-radio-maps.
    """
    if not overlay_b64 or not bounds:
        return (
            "<!DOCTYPE html><html><body style='font-family:sans-serif;"
            "color:#888;padding:24px'>No coverage computed yet — click "
            "“Compute large-scale map” in the sidebar.</body></html>"
        )

    south, west, north, east = bounds
    bs_json = json.dumps(base_stations or [])
    legend_html = (
        f"<img src='data:image/png;base64,{legend_b64}' "
        f"style='width:220px;display:block'/>" if legend_b64 else ""
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map {{ height:100%; margin:0; }}
  .info-box {{ background:rgba(255,255,255,0.9); padding:6px 8px;
              border-radius:4px; font:12px/1.3 sans-serif; box-shadow:0 1px 4px rgba(0,0,0,0.3); }}
  .info-box input {{ width:120px; vertical-align:middle; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  var south={south}, west={west}, north={north}, east={east};
  var bounds = [[south, west], [north, east]];
  var map = L.map('map');
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20
  }}).addTo(map);

  var overlay = L.imageOverlay(
    'data:image/png;base64,{overlay_b64}', bounds, {{ opacity: 0.7, interactive: false }}
  ).addTo(map);
  map.fitBounds(bounds);

  // Base-station markers.
  var stations = {bs_json};
  stations.forEach(function(s, i) {{
    L.circleMarker([s[0], s[1]], {{
      radius: 5, color: '#d62728', weight: 1.5, fillColor: '#ff4136', fillOpacity: 0.9
    }}).addTo(map).bindPopup('Base station ' + (i+1) + '<br/>' +
        s[0].toFixed(5) + ', ' + s[1].toFixed(5));
  }});

  // Opacity control.
  var opacityCtl = L.control({{ position: 'topright' }});
  opacityCtl.onAdd = function() {{
    var d = L.DomUtil.create('div', 'info-box');
    d.innerHTML = '<b>{title}</b><br/>Coverage opacity ' +
      '<input type="range" min="0" max="100" value="70" id="op"/>';
    L.DomEvent.disableClickPropagation(d);
    return d;
  }};
  opacityCtl.addTo(map);
  document.getElementById('op').addEventListener('input', function(e) {{
    overlay.setOpacity(e.target.value / 100);
  }});

  // Legend.
  var legendCtl = L.control({{ position: 'bottomright' }});
  legendCtl.onAdd = function() {{
    var d = L.DomUtil.create('div', 'info-box');
    d.innerHTML = "{legend_html}";
    return d;
  }};
  legendCtl.addTo(map);
</script>
</body>
</html>"""


def _collect_scene(input) -> dict:
    return {
        "name": "Custom",
        "num_rows_tx": int(input.num_rows_tx()),
        "num_cols_tx": int(input.num_cols_tx()),
        "num_rows_rx": int(input.num_rows_rx()),
        "num_cols_rx": int(input.num_cols_rx()),
        "frequency_hz": float(input.frequency_ghz()) * 1e9,
        "bandwidth_hz": float(input.bandwidth_mhz()) * 1e6,
        "max_depth": int(input.max_depth()),
        "samples_per_tx": 10 ** int(input.samples_log10()),
        "cell_size_x": float(input.cell_size_x()),
        "cell_size_y": float(input.cell_size_y()),
        "pattern": input.pattern(),
        "polarization": input.polarization(),
        "num_user_samples": int(input.num_user_samples()),
        "min_sinr_db": float(input.min_sinr_db()),
        "min_user_dist_m": float(input.min_user_dist_m()),
        "max_user_dist_m": float(input.max_user_dist_m()),
    }


def _collect_cells(input) -> list[dict]:
    """7-cell layout with the sidebar's TX-power applied uniformly.

    The 7 positions/look-at points match what's seeded in Lakebase; only
    power_dbm flexes via the sidebar so power-variant presets remain
    reachable by hash.
    """
    cells = preset_cells()
    power = float(input.tx_power_dbm())
    for c in cells:
        c["power_dbm"] = power
    return cells


def _submit_databricks_job(config_hash: str, scene_cfg: dict, cells: list[dict]) -> int:
    """Trigger the Sionna compute job. Returns the Databricks run_id."""
    if not SIONNA_JOB_ID:
        raise RuntimeError(
            "Live compute disabled: SIONNA_JOB_ID env var is not set. "
            "Either select a preset (Config 1/2) or configure the job."
        )
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    run = w.jobs.run_now(
        job_id=int(SIONNA_JOB_ID),
        notebook_params={
            "config_hash": config_hash,
            "scene_json": json.dumps(scene_cfg),
            "cells_json": json.dumps(cells),
        },
    )
    return int(run.run_id)


def _cancel_databricks_run(run_id: int) -> None:
    """Cancel an in-flight Sionna run. Best-effort — errors are swallowed."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    w.jobs.cancel_run(run_id=int(run_id))


def _collect_region(input) -> dict:
    """Build a region config dict from the large-scale sidebar controls.

    Non-editable fields (tiling cell sizes, ray samples) are inherited from the
    selected preset so the region hash stays stable and matches any seeded row.
    """
    preset = REGION_PRESETS.get(input.ls_region(), DEFAULT_REGION)
    region = preset.to_dict()
    region.update({
        "name":              preset.name,
        "south":             float(input.ls_south()),
        "west":              float(input.ls_west()),
        "north":             float(input.ls_north()),
        "east":              float(input.ls_east()),
        "frequency_hz":      float(input.ls_frequency_ghz()) * 1e9,
        "tx_power_dbm":      float(input.ls_tx_power_dbm()),
        "num_base_stations": int(input.ls_num_bs()),
    })
    return region


def _submit_large_scale_job(region_hash: str, region: dict) -> int:
    """Trigger the large-scale (sionna_lrm) compute job. Returns run_id."""
    if not LARGE_SCALE_JOB_ID:
        raise RuntimeError("LARGE_SCALE_JOB_ID env var is not set.")
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    run = w.jobs.run_now(
        job_id=int(LARGE_SCALE_JOB_ID),
        notebook_params={
            "region_hash": region_hash,
            "region_json": json.dumps(region),
        },
    )
    return int(run.run_id)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_initial_scene = CONFIG_1


def _sidebar() -> ui.Tag:
    return ui.sidebar(
        ui.h5("Antenna array"),
        ui.row(
            ui.column(6, ui.input_numeric("num_rows_tx", "TX rows",
                                          value=_initial_scene.num_rows_tx, min=1, max=64)),
            ui.column(6, ui.input_numeric("num_cols_tx", "TX cols",
                                          value=_initial_scene.num_cols_tx, min=1, max=64)),
        ),
        ui.row(
            ui.column(6, ui.input_numeric("num_rows_rx", "RX rows",
                                          value=_initial_scene.num_rows_rx, min=1, max=16)),
            ui.column(6, ui.input_numeric("num_cols_rx", "RX cols",
                                          value=_initial_scene.num_cols_rx, min=1, max=16)),
        ),
        ui.input_select("pattern", "Pattern",
                        choices=PATTERN_CHOICES, selected=_initial_scene.pattern),
        ui.input_select("polarization", "Polarization",
                        choices=POLARIZATION_CHOICES, selected=_initial_scene.polarization),
        ui.h5("Network"),
        ui.input_numeric(
            "tx_power_dbm", "TX power per cell (dBm)",
            value=44.0, min=20.0, max=60.0, step=1.0,
        ),
        ui.h5("Radio"),
        ui.input_numeric("frequency_ghz", "Frequency (GHz)",
                         value=_initial_scene.frequency_hz / 1e9, min=0.1, max=100, step=0.1),
        ui.input_numeric("bandwidth_mhz", "Bandwidth (MHz)",
                         value=_initial_scene.bandwidth_hz / 1e6, min=1, max=10000, step=10),
        ui.h5("Ray tracing"),
        ui.input_numeric("max_depth", "Max depth", value=_initial_scene.max_depth, min=1, max=10),
        ui.input_select("samples_log10", "Samples per TX (10^x)",
                        choices=["5", "6", "7", "8"], selected="7"),
        ui.row(
            ui.column(6, ui.input_numeric("cell_size_x", "Cell X (m)",
                                          value=_initial_scene.cell_size_x, min=0.1, step=0.5)),
            ui.column(6, ui.input_numeric("cell_size_y", "Cell Y (m)",
                                          value=_initial_scene.cell_size_y, min=0.1, step=0.5)),
        ),
        ui.h5("User sampling"),
        ui.input_numeric("num_user_samples", "Users / TX",
                         value=_initial_scene.num_user_samples, min=1, max=1000),
        ui.input_numeric("min_sinr_db", "Min SINR (dB)", value=_initial_scene.min_sinr_db),
        ui.row(
            ui.column(6, ui.input_numeric("min_user_dist_m", "Min dist (m)",
                                          value=_initial_scene.min_user_dist_m)),
            ui.column(6, ui.input_numeric("max_user_dist_m", "Max dist (m)",
                                          value=_initial_scene.max_user_dist_m)),
        ),
        ui.hr(),
        ui.input_action_button(
            "render_btn", "Render scene", class_="btn-success",
            style="width: 100%;",
        ),
        ui.input_action_button(
            "cancel_btn", "Cancel pending job", class_="btn-outline-danger",
            style="width: 100%; margin-top: 6px;",
            disabled=True,
        ),
        ui.tags.div(
            "Active only while a Sionna job is running for an off-menu config. "
            "Cached presets always load instantly.",
            style="font-size: 11px; color: #888; margin-top: 4px;",
        ),
        ui.hr(),
        ui.h5("Large-scale map"),
        ui.tags.div(
            "Coverage across a real geographic region (OSM buildings + "
            "base stations) via NVIDIA sionna-large-radio-maps. Configure "
            "here, then open the “Large-scale map” tab.",
            style="font-size: 11px; color: #888; margin-bottom: 6px;",
        ),
        ui.input_select(
            "ls_region", "Region preset",
            choices={k: v.name for k, v in REGION_PRESETS.items()},
            selected="seattle",
        ),
        ui.row(
            ui.column(6, ui.input_numeric("ls_south", "South lat",
                                          value=DEFAULT_REGION.south, step=0.001)),
            ui.column(6, ui.input_numeric("ls_north", "North lat",
                                          value=DEFAULT_REGION.north, step=0.001)),
        ),
        ui.row(
            ui.column(6, ui.input_numeric("ls_west", "West lon",
                                          value=DEFAULT_REGION.west, step=0.001)),
            ui.column(6, ui.input_numeric("ls_east", "East lon",
                                          value=DEFAULT_REGION.east, step=0.001)),
        ),
        ui.input_numeric("ls_frequency_ghz", "Frequency (GHz)",
                         value=DEFAULT_REGION.frequency_hz / 1e9, min=0.1, max=100, step=0.1),
        ui.input_numeric("ls_tx_power_dbm", "TX power / cell (dBm)",
                         value=DEFAULT_REGION.tx_power_dbm, min=20.0, max=60.0, step=1.0),
        ui.input_numeric("ls_num_bs", "Base stations (demo)",
                         value=DEFAULT_REGION.num_base_stations, min=1, max=500),
        ui.input_action_button(
            "ls_compute_btn", "Compute large-scale map", class_="btn-primary",
            style="width: 100%; margin-top: 4px;",
        ),
        width=340,
    )


app_ui = ui.page_sidebar(
    _sidebar(),
    ui.navset_card_tab(
        ui.nav_panel("Scene render",     ui.output_ui("scene_render_view")),
        ui.nav_panel("SINR association", ui.output_ui("sinr_map_view")),
        ui.nav_panel("Users → TX",       ui.output_ui("association_view")),
        ui.nav_panel("CDFs",             ui.output_ui("cdf_view")),
        ui.nav_panel("KPIs",             ui.output_ui("kpis_view")),
        ui.nav_panel("Large-scale map",  ui.output_ui("large_scale_view")),
        ui.nav_panel("Status",           ui.output_ui("status_view")),
        id="main_tabs",
    ),
    title="RF Digital Twin — Sionna RT on Databricks",
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):

    render_state = reactive.Value({
        # What's currently shown in the tabs.
        "config_hash":      None,    # hash of the render in `data`, or hash of an in-flight job
        "scene":            None,    # scene_cfg dict last submitted
        "data":             None,    # cached_renders row to display
        "kpis":             None,
        # Pending-job tracker — when a cache miss has triggered a Sionna run.
        "pending_hash":     None,    # hash being computed; None when idle
        "pending_run_id":   None,    # Databricks Jobs run_id
        "pending_started":  None,    # epoch seconds when submitted
        # User-visible.
        "status":           "Ready. Edit the config in the sidebar, then click Render.",
        "error":            None,
    })

    # Independent state for the large-scale (sionna_lrm) tab.
    ls_state = reactive.Value({
        "region_hash":  None,
        "region":       None,
        "data":         None,     # large_scale_maps row to display
        "kpis":         None,
        "pending_hash": None,     # region_hash of an in-flight GPU job
        "pending_run_id": None,
        "pending_started": None,
        "status": ("Configure a region in the sidebar, then click "
                   "“Compute large-scale map”."),
        "error": None,
    })

    async def _try_load_cached(config_hash: str) -> dict | None:
        return await asyncio.to_thread(lb.get_render, config_hash)

    def _apply_cached(state, config_hash, scene_cfg, cached, status_msg):
        """Populate render_state with a cache hit."""
        kpis = cached.get("kpis_json")
        if isinstance(kpis, (bytes, str)):
            kpis = json.loads(kpis)
        render_state.set({
            **state,
            "config_hash":     config_hash,
            "scene":           scene_cfg,
            "data":            cached,
            "kpis":            kpis,
            # Clear any pending tracker if the cache hit matches it.
            "pending_hash":    None if state.get("pending_hash") == config_hash else state.get("pending_hash"),
            "pending_run_id":  None if state.get("pending_hash") == config_hash else state.get("pending_run_id"),
            "pending_started": None if state.get("pending_hash") == config_hash else state.get("pending_started"),
            "status":          status_msg,
            "error":           None,
        })

    async def _do_render() -> None:
        """Non-blocking. Cache hit -> load and return. Cache miss -> submit
        the Databricks job, record the pending run, and return immediately.
        A background reactive timer polls Lakebase until results land."""
        try:
            scene_cfg = _collect_scene(input)
            cells = _collect_cells(input)
            config_hash = lb.compute_config_hash(scene_cfg, cells)

            # 1) Cache hit — instant load, regardless of any in-flight job.
            cached = await _try_load_cached(config_hash)
            if cached:
                _apply_cached(
                    render_state(), config_hash, scene_cfg, cached,
                    f"Loaded cached render ({config_hash[:12]}) computed in "
                    f"{cached.get('compute_seconds', 0):.1f}s.",
                )
                ui.update_navs("main_tabs", selected="Scene render")
                return

            # 2) Cache miss — cancel any previous pending job first so we're
            # only ever holding one cluster at a time.
            prev_run = render_state().get("pending_run_id")
            if prev_run:
                try:
                    await asyncio.to_thread(_cancel_databricks_run, prev_run)
                except Exception as e:
                    print(f"Failed to cancel previous run {prev_run}: {e}")

            if not SIONNA_JOB_ID:
                render_state.set({
                    **render_state(),
                    "config_hash":     config_hash,
                    "scene":           scene_cfg,
                    "pending_hash":    None,
                    "pending_run_id":  None,
                    "pending_started": None,
                    "status": "Cache miss and live compute disabled.",
                    "error": (
                        "No cached render for this configuration and SIONNA_JOB_ID "
                        "is not set. Pick a preset from the cheat sheet."
                    ),
                })
                return

            await asyncio.to_thread(lb.upsert_scene_config, scene_cfg, cells, False)
            run_id = await asyncio.to_thread(
                _submit_databricks_job, config_hash, scene_cfg, cells,
            )
            await asyncio.to_thread(
                lb.set_job_status, config_hash, "RUNNING", run_id, None,
            )
            render_state.set({
                **render_state(),
                "config_hash":     config_hash,
                "scene":           scene_cfg,
                "pending_hash":    config_hash,
                "pending_run_id":  run_id,
                "pending_started": time.time(),
                "status": (
                    f"Sionna job submitted (run_id={run_id}). Spinning up GPU "
                    f"cluster. You can keep clicking cached presets while it runs; "
                    f"results will auto-appear here when ready."
                ),
                "error": None,
            })
            # Return immediately — the background poller (defined below)
            # picks up the pending hash and updates state when the job lands.

        except Exception as e:
            render_state.set({
                **render_state(),
                "error":  str(e),
                "status": "Render failed.",
            })
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Background poller — runs every ~10 s. Checks Lakebase for any
    # pending hash and pulls results when they land. Updates the status
    # text in between so the user sees elapsed time.
    # ------------------------------------------------------------------
    @reactive.effect
    async def _background_poll():
        reactive.invalidate_later(10)
        st = render_state()
        pending_hash = st.get("pending_hash")
        if not pending_hash:
            return

        try:
            cached = await asyncio.to_thread(lb.get_render, pending_hash)
        except Exception as e:
            print(f"Background poll: get_render failed: {e}")
            return

        if cached:
            elapsed = time.time() - (st.get("pending_started") or time.time())
            _apply_cached(
                st, pending_hash, st.get("scene"), cached,
                f"Pending job complete (run_id={st.get('pending_run_id')}) — "
                f"render cached in {cached.get('compute_seconds', 0):.1f}s "
                f"(end-to-end {elapsed:.0f}s).",
            )
            # Only nudge the user to the render tab if they're still on Status.
            return

        # Still running — refresh the elapsed time in the status banner.
        elapsed = int(time.time() - (st.get("pending_started") or time.time()))
        remaining = max(JOB_ETA_SECONDS - elapsed, 30)
        render_state.set({
            **st,
            "status": (
                f"Sionna job running (run_id={st.get('pending_run_id')}). "
                f"Elapsed {elapsed//60}m{elapsed%60:02d}s, "
                f"~{remaining//60}m{remaining%60:02d}s remaining. "
                f"Switch to any cached preset to keep exploring."
            ),
        })

    # ------------------------------------------------------------------
    # Cancel button — only emitted when a job is pending.
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.cancel_btn, ignore_init=True, ignore_none=True)
    async def _on_cancel():
        run_id = render_state().get("pending_run_id")
        if not run_id:
            return
        try:
            await asyncio.to_thread(_cancel_databricks_run, run_id)
            msg = f"Cancelled run {run_id}."
        except Exception as e:
            msg = f"Cancel requested for run {run_id} but failed: {e}"
            traceback.print_exc()
        render_state.set({
            **render_state(),
            "pending_hash":    None,
            "pending_run_id":  None,
            "pending_started": None,
            "status": msg,
        })

    # ------------------------------------------------------------------
    # Auto-load: on first server start, pull Config 1 from cache.
    # ------------------------------------------------------------------
    initial_loaded = reactive.Value(False)

    @reactive.effect
    async def _initial_load():
        if initial_loaded():
            return
        initial_loaded.set(True)
        try:
            scene_cfg = CONFIG_1.to_dict()
            cells = preset_cells()
            config_hash = lb.compute_config_hash(scene_cfg, cells)
            cached = await _try_load_cached(config_hash)
            if cached:
                kpis = cached.get("kpis_json")
                if isinstance(kpis, (bytes, str)):
                    kpis = json.loads(kpis)
                render_state.set({
                    "config_hash": config_hash,
                    "scene": scene_cfg,
                    "cells": cells,
                    "data": cached,
                    "kpis": kpis,
                    "status": (
                        f"Loaded Config 1 from cache ({config_hash[:12]}) "
                        f"computed in {cached.get('compute_seconds', 0):.1f}s."
                    ),
                    "error": None,
                })
            else:
                render_state.set({
                    **render_state(),
                    "status": (
                        "Config 1 is not cached in Lakebase yet. Run the setup "
                        "notebook to precompute it (see README)."
                    ),
                })
        except Exception as e:
            render_state.set({
                **render_state(),
                "error": str(e),
                "status": "Could not connect to Lakebase.",
            })
            traceback.print_exc()

    @reactive.effect
    @reactive.event(input.render_btn)
    async def _render():
        await _do_render()

    # ------------------------------------------------------------------
    # Cancel button — always in the sidebar; enabled only when a job is pending.
    # ------------------------------------------------------------------
    @reactive.effect
    def _toggle_cancel_enabled():
        has_pending = bool(render_state().get("pending_run_id"))
        ui.update_action_button(
            "cancel_btn",
            label="Cancel pending job" if has_pending else "No job to cancel",
            disabled=not has_pending,
        )

    # ------------------------------------------------------------------
    # Large-scale map — region presets, compute, background poll.
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.ls_region)
    def _sync_region_preset():
        """When the region preset changes, load its bbox/radio defaults into
        the editable numeric inputs."""
        preset = REGION_PRESETS.get(input.ls_region())
        if not preset:
            return
        ui.update_numeric("ls_south", value=preset.south)
        ui.update_numeric("ls_north", value=preset.north)
        ui.update_numeric("ls_west",  value=preset.west)
        ui.update_numeric("ls_east",  value=preset.east)
        ui.update_numeric("ls_frequency_ghz", value=preset.frequency_hz / 1e9)
        ui.update_numeric("ls_tx_power_dbm",   value=preset.tx_power_dbm)
        ui.update_numeric("ls_num_bs",         value=preset.num_base_stations)

    def _apply_ls_cached(state, region_hash, region, cached, status_msg):
        kpis = cached.get("kpis_json")
        if isinstance(kpis, (bytes, str)):
            kpis = json.loads(kpis)
        ls_state.set({
            **state,
            "region_hash":     region_hash,
            "region":          region,
            "data":            cached,
            "kpis":            kpis,
            "pending_hash":    None if state.get("pending_hash") == region_hash else state.get("pending_hash"),
            "pending_run_id":  None if state.get("pending_hash") == region_hash else state.get("pending_run_id"),
            "pending_started": None if state.get("pending_hash") == region_hash else state.get("pending_started"),
            "status":          status_msg,
            "error":           None,
        })

    @reactive.effect
    @reactive.event(input.ls_compute_btn)
    async def _on_large_scale_compute():
        try:
            region = _collect_region(input)
            region_hash = lb.compute_region_hash(region)

            # 1) Cache hit — instant load.
            cached = None
            try:
                cached = await asyncio.to_thread(lb.get_large_scale_map, region_hash)
            except Exception as e:  # noqa: BLE001 — cache lookup may fail (e.g., Lakebase unavailable)
                print(f"Cache lookup failed (proceeding to demo/job path): {e}")

            if cached:
                tag = "demo" if cached.get("is_demo") else "Sionna RT"
                _apply_ls_cached(
                    ls_state(), region_hash, region, cached,
                    f"Loaded cached large-scale map ({tag}, {region_hash[:12]}) "
                    f"computed in {cached.get('compute_seconds', 0):.1f}s.",
                )
                ui.update_navs("main_tabs", selected="Large-scale map")
                return

            # 2) Cache miss with a real GPU job configured — submit it.
            if LARGE_SCALE_JOB_ID:
                prev = ls_state().get("pending_run_id")
                if prev:
                    try:
                        await asyncio.to_thread(_cancel_databricks_run, prev)
                    except Exception as e:  # noqa: BLE001
                        print(f"Failed to cancel previous LS run {prev}: {e}")
                run_id = await asyncio.to_thread(
                    _submit_large_scale_job, region_hash, region,
                )
                ls_state.set({
                    **ls_state(),
                    "region_hash":     region_hash,
                    "region":          region,
                    "pending_hash":    region_hash,
                    "pending_run_id":  run_id,
                    "pending_started": time.time(),
                    "status": (
                        f"Large-scale Sionna job submitted (run_id={run_id}). "
                        f"Tiling → OSM scenes → per-tile ray tracing on a GPU "
                        f"cluster; results auto-appear here when ready."
                    ),
                    "error": None,
                })
                ui.update_navs("main_tabs", selected="Large-scale map")
                return

            # 3) No GPU job configured — run the GPU-free demo inline and cache it.
            ls_state.set({
                **ls_state(),
                "region_hash": region_hash,
                "region": region,
                "status": "No GPU job configured — computing demo coverage map…",
                "error": None,
            })
            ui.update_navs("main_tabs", selected="Large-scale map")

            from large_scale_compute import run_large_scale
            results = await asyncio.to_thread(run_large_scale, region, True)
            try:
                await asyncio.to_thread(
                    lb.write_large_scale_map, region_hash, region, results,
                )
            except Exception as e:  # noqa: BLE001 — demo still displays without cache
                print(f"Could not cache large-scale demo: {e}")

            kpis = json.loads(results["kpis_json"]) if results.get("kpis_json") else None
            ls_state.set({
                **ls_state(),
                "data": results,
                "kpis": kpis,
                "status": (
                    f"Demo large-scale map computed in "
                    f"{results.get('compute_seconds', 0):.1f}s (synthetic — set "
                    f"LARGE_SCALE_JOB_ID for real Sionna RT ray tracing)."
                ),
                "error": None,
            })
        except Exception as e:  # noqa: BLE001
            ls_state.set({
                **ls_state(),
                "error": str(e),
                "status": "Large-scale compute failed.",
            })
            traceback.print_exc()

    @reactive.effect
    async def _ls_background_poll():
        reactive.invalidate_later(15)
        st = ls_state()
        pending_hash = st.get("pending_hash")
        if not pending_hash:
            return
        try:
            cached = await asyncio.to_thread(lb.get_large_scale_map, pending_hash)
        except Exception as e:  # noqa: BLE001
            print(f"LS background poll failed: {e}")
            return
        if cached:
            elapsed = time.time() - (st.get("pending_started") or time.time())
            _apply_ls_cached(
                st, pending_hash, st.get("region"), cached,
                f"Large-scale job complete (run_id={st.get('pending_run_id')}) — "
                f"cached in {cached.get('compute_seconds', 0):.1f}s "
                f"(end-to-end {elapsed:.0f}s).",
            )
            return
        elapsed = int(time.time() - (st.get("pending_started") or time.time()))
        remaining = max(LARGE_SCALE_JOB_ETA_SECONDS - elapsed, 30)
        ls_state.set({
            **st,
            "status": (
                f"Large-scale Sionna job running "
                f"(run_id={st.get('pending_run_id')}). "
                f"Elapsed {elapsed//60}m{elapsed%60:02d}s, "
                f"~{remaining//60}m{remaining%60:02d}s remaining."
            ),
        })

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @render.ui
    def scene_render_view():
        data = render_state().get("data") or {}
        return ui.div(
            ui.h4("Scene render with SINR overlay"),
            _png_img(data.get("scene_render_png"), "scene render"),
        )

    @render.ui
    def sinr_map_view():
        data = render_state().get("data") or {}
        return ui.div(
            ui.h4("Cell-to-TX association (SINR)"),
            _png_img(data.get("sinr_map_png"), "SINR association"),
        )

    @render.ui
    def association_view():
        data = render_state().get("data") or {}
        return ui.div(
            ui.h4("Sampled users coloured by serving TX"),
            _png_img(data.get("association_png"), "user-to-TX association"),
        )

    @render.ui
    def cdf_view():
        data = render_state().get("data") or {}
        return ui.div(
            ui.h4("CDF of SINR"),
            _png_img(data.get("sinr_cdf_png"), "SINR CDF"),
            ui.h4("CDF of RSS", style="margin-top: 20px;"),
            _png_img(data.get("rss_cdf_png"), "RSS CDF"),
        )

    @render.ui
    def kpis_view():
        kpis = render_state().get("kpis")
        if not kpis:
            return ui.tags.div("No KPIs yet — render a scene first.",
                               style="padding: 20px; color: #888;")
        sinr = kpis.get("sinr_percentiles_db", {})
        rss  = kpis.get("rss_percentiles_dbm", {})
        users = kpis.get("users_per_tx", {})

        def _row(label: str, val: Any) -> ui.Tag:
            return ui.tags.tr(ui.tags.td(label), ui.tags.td(str(val)))

        rows = [
            _row("SINR p10 (dB)", f"{sinr.get('p10', 'n/a')}"),
            _row("SINR p50 (dB)", f"{sinr.get('p50', 'n/a')}"),
            _row("SINR p90 (dB)", f"{sinr.get('p90', 'n/a')}"),
            _row("RSS  p10 (dBm)", f"{rss.get('p10', 'n/a')}"),
            _row("RSS  p50 (dBm)", f"{rss.get('p50', 'n/a')}"),
            _row("RSS  p90 (dBm)", f"{rss.get('p90', 'n/a')}"),
            _row("Number of TXs", kpis.get("num_tx", "n/a")),
        ]
        for tx, n in sorted(users.items()):
            rows.append(_row(f"Users assigned to tx{tx}", n))

        return ui.div(
            ui.h4("KPI summary"),
            ui.tags.table(
                ui.tags.tbody(*rows),
                class_="table table-striped",
                style="max-width: 480px;",
            ),
        )

    @render.ui
    def large_scale_view():
        st = ls_state() or {}
        data = st.get("data") or {}
        kpis = st.get("kpis") or {}
        status_text = st.get("status") or ""
        error_text = st.get("error")
        is_demo = bool(data.get("is_demo"))

        banner = None
        if data and is_demo:
            banner = ui.tags.div(
                ui.tags.strong("Demo mode — "),
                "synthetic coverage (log-distance path loss), not Sionna RT ray "
                "tracing. Set LARGE_SCALE_JOB_ID to run the real "
                "sionna-large-radio-maps pipeline on a GPU cluster.",
                style="background:#fff3cd; border:1px solid #ffe69c; color:#664d03; "
                      "padding:8px 10px; border-radius:4px; margin-bottom:10px; "
                      "font-size:13px;",
            )

        kpi_rows = []
        for label, key, fmt in [
            ("Region",            "region",       str),
            ("Area (km²)",        "area_km2",     str),
            ("Base stations",     "num_base_stations", str),
            ("Coverage (%)",      "coverage_pct", str),
            ("Frequency (GHz)",   "frequency_ghz", str),
            ("TX power (dBm)",    "tx_power_dbm", str),
        ]:
            if key in kpis:
                kpi_rows.append(ui.tags.tr(ui.tags.td(label),
                                           ui.tags.td(fmt(kpis[key]))))
        pct = kpis.get("rss_percentiles_dbm") or kpis.get("path_gain_percentiles_db") or {}
        unit = "dBm" if "rss_percentiles_dbm" in kpis else "dB"
        for p in ("p10", "p50", "p90"):
            if p in pct:
                kpi_rows.append(ui.tags.tr(
                    ui.tags.td(f"{p} ({unit})"), ui.tags.td(str(pct[p]))))

        kpi_table = (
            ui.tags.table(ui.tags.tbody(*kpi_rows),
                          class_="table table-striped",
                          style="max-width:420px; margin-top:12px;")
            if kpi_rows else ""
        )

        # Interactive, zoomable Leaflet map: OSM basemap + georeferenced
        # coverage overlay + base-station markers (like NVlabs sionna_lrm).
        map_html = _leaflet_map_html(
            overlay_b64=_as_b64(data.get("overlay_png")),
            bounds=_as_obj(data.get("bounds_json")),
            base_stations=_as_obj(data.get("base_stations_json")),
            legend_b64=_as_b64(data.get("legend_png")),
            title=str(kpis.get("region") or "Coverage"),
        )
        map_frame = ui.tags.iframe(
            srcdoc=map_html,
            style="width:100%; height:560px; border:1px solid #ddd; "
                  "border-radius:6px;",
        )

        return ui.div(
            ui.h4("Large-scale radio map — NVIDIA sionna-large-radio-maps"),
            ui.tags.p(status_text, style="color:#555; font-size:13px;"),
            banner or "",
            ui.tags.div(
                ui.tags.strong("Error: "), str(error_text),
                style="color:#c00; padding:8px; border:1px solid #c00; "
                      "border-radius:4px; margin:8px 0;",
            ) if error_text else "",
            ui.tags.p(
                "Pan and zoom the map; drag the opacity slider (top-right) to "
                "fade the coverage layer against the streets underneath.",
                style="color:#777; font-size:12px; margin-bottom:6px;",
            ),
            map_frame,
            ui.row(
                ui.column(6,
                    ui.h5("Adaptive tiling", style="margin-top:16px;"),
                    _png_img(data.get("tiling_png"), "tiling preview"),
                ),
                ui.column(6,
                    ui.h5("Coverage CDF", style="margin-top:16px;"),
                    _png_img(data.get("cdf_png"), "coverage CDF"),
                ),
            ),
            kpi_table,
        )

    @render.ui
    def status_view():
        try:
            st = render_state() or {}
            status_text  = st.get("status") or "(no status)"
            config_hash  = st.get("config_hash") or "—"
            run_id       = st.get("pending_run_id")
            pending_hash = st.get("pending_hash")
            started      = st.get("pending_started")
            error_text   = st.get("error")
            scene        = st.get("scene")

            run_link = ""
            if run_id and SIONNA_JOB_ID and DATABRICKS_WORKSPACE_URL:
                run_url = f"{DATABRICKS_WORKSPACE_URL}/jobs/{SIONNA_JOB_ID}/runs/{run_id}"
                run_link = ui.tags.a(
                    f"open run_id={run_id}", href=run_url, target="_blank", rel="noopener",
                )

            scene_block = "(no scene yet — click Render)"
            if scene:
                try:
                    scene_block = json.dumps(scene, indent=2, default=str)
                except Exception as e:
                    scene_block = f"(could not serialise scene: {e})"

            return ui.div(
                ui.h4("Status"),
                ui.tags.p(ui.tags.strong("Message: "), status_text),
                ui.tags.p(
                    ui.tags.strong("Pending job: "),
                    "yes" if pending_hash else "no",
                    f"  (run_id={run_id})" if run_id else "",
                    "  ", run_link,
                ),
                ui.tags.p(
                    ui.tags.strong("Config hash: "), ui.tags.code(str(config_hash)),
                ),
                ui.tags.p(
                    ui.tags.strong("Job submitted at: "),
                    str(started) if started else "—",
                ),
                ui.tags.div(
                    ui.tags.strong("Error: "), str(error_text),
                    style="color:#c00; padding:8px; border:1px solid #c00; "
                          "border-radius:4px; margin:8px 0;",
                ) if error_text else "",
                ui.h5("Scene config"),
                ui.tags.pre(
                    scene_block,
                    style="background:#f8f9fa; padding:10px; border-radius:4px; "
                          "font-size:12px; max-height:400px; overflow:auto;",
                ),
                style="padding:12px;",
            )
        except Exception as e:
            import traceback as _tb
            return ui.tags.pre(
                f"status_view error: {e}\n\n{_tb.format_exc()}",
                style="color:#c00; padding:10px;",
            )


app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
