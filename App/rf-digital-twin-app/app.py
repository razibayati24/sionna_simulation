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
        "config_hash": None,
        "scene": None,
        "cells": None,
        "data": None,           # cached_renders row
        "kpis": None,
        "run_id": None,         # job run currently in flight (or last one)
        "started_at": None,     # epoch s when current job submitted; None if idle
        "status": "Ready. Edit the config in the sidebar, then click Render.",
        "error": None,
    })

    async def _try_load_cached(config_hash: str) -> dict | None:
        return await asyncio.to_thread(lb.get_render, config_hash)

    async def _do_render() -> None:
        try:
            scene_cfg = _collect_scene(input)
            cells = _collect_cells(input)
            config_hash = lb.compute_config_hash(scene_cfg, cells)

            render_state.set({
                **render_state(),
                "config_hash": config_hash,
                "scene": scene_cfg,
                "cells": cells,
                "status": f"Looking up cache for {config_hash[:12]}…",
                "error": None,
            })

            cached = await _try_load_cached(config_hash)
            if cached:
                kpis = cached.get("kpis_json")
                if isinstance(kpis, (bytes, str)):
                    kpis = json.loads(kpis)
                render_state.set({
                    **render_state(),
                    "data": cached,
                    "kpis": kpis,
                    "status": (
                        f"Loaded cached render ({config_hash[:12]}) "
                        f"computed in {cached.get('compute_seconds', 0):.1f}s."
                    ),
                })
                ui.update_navs("main_tabs", selected="Scene render")
                return

            if not SIONNA_JOB_ID:
                render_state.set({
                    **render_state(),
                    "status": "Cache miss and live compute disabled.",
                    "error": (
                        "No cached render for this configuration and SIONNA_JOB_ID "
                        "is not set. Pick Config 1 or Config 2, or wire up the "
                        "Sionna compute job (see README)."
                    ),
                })
                return

            await asyncio.to_thread(
                lb.upsert_scene_config, scene_cfg, cells, False,
            )
            run_id = await asyncio.to_thread(
                _submit_databricks_job, config_hash, scene_cfg, cells,
            )
            await asyncio.to_thread(
                lb.set_job_status, config_hash, "RUNNING", run_id, None,
            )
            started_at = time.time()
            render_state.set({
                **render_state(),
                "run_id": run_id,
                "started_at": started_at,
                "status": (
                    f"Sionna compute job submitted "
                    f"(run_id={run_id}). Spinning up GPU cluster…"
                ),
            })

            # Poll until the cache row appears, up to ~15 minutes.
            for tick in range(90):
                await asyncio.sleep(10)
                cached = await _try_load_cached(config_hash)
                if cached:
                    kpis = cached.get("kpis_json")
                    if isinstance(kpis, (bytes, str)):
                        kpis = json.loads(kpis)
                    elapsed = time.time() - started_at
                    render_state.set({
                        **render_state(),
                        "data": cached,
                        "kpis": kpis,
                        "run_id": run_id,
                        "started_at": None,
                        "status": (
                            f"Job complete (run_id={run_id}) — render cached "
                            f"in {cached.get('compute_seconds', 0):.1f}s "
                            f"(end-to-end {elapsed:.0f}s)."
                        ),
                    })
                    ui.update_navs("main_tabs", selected="Scene render")
                    return

                # Progress message every ~30s
                elapsed = int(time.time() - started_at)
                remaining = max(JOB_ETA_SECONDS - elapsed, 30)
                render_state.set({
                    **render_state(),
                    "run_id": run_id,
                    "started_at": started_at,
                    "status": (
                        f"Sionna job running (run_id={run_id}). "
                        f"Elapsed {elapsed//60}m{elapsed%60:02d}s, "
                        f"estimated ~{remaining//60}m{remaining%60:02d}s remaining."
                    ),
                })

            render_state.set({
                **render_state(),
                "status": (
                    f"Polling timed out after 15 minutes. The job may still be "
                    f"running — refresh the page or check run {run_id} in Databricks."
                ),
            })

        except Exception as e:
            render_state.set({
                **render_state(),
                "error": str(e),
                "status": "Render failed.",
            })
            traceback.print_exc()

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
    def status_view():
        try:
            st = render_state()
            is_running = bool(st.get("started_at"))
            run_id = st.get("run_id")
            status_text = st.get("status") or "(no status)"
            config_hash = st.get("config_hash") or "—"
            error_text = st.get("error")
            scene = st.get("scene")

            blocks: list = []

            # 1. Status banner (with spinner if a job is running).
            if is_running:
                blocks.append(ui.tags.style(
                    "@keyframes rfspin{from{transform:rotate(0deg)}"
                    "to{transform:rotate(360deg)}}"
                    ".rf-spinner{border:3px solid #eee;border-top:3px solid #2c7be5;"
                    "border-radius:50%;width:18px;height:18px;"
                    "animation:rfspin 1s linear infinite;"
                    "display:inline-block;vertical-align:middle;margin-right:8px}"
                ))
                blocks.append(ui.tags.div(
                    ui.tags.div(class_="rf-spinner"),
                    ui.tags.span(status_text),
                    style=(
                        "padding:10px;background:#f4f8ff;border:1px solid #2c7be5;"
                        "border-radius:4px;margin-bottom:12px;"
                    ),
                ))
            else:
                blocks.append(ui.tags.div(
                    ui.tags.strong("Status: "),
                    status_text,
                    style="padding:10px;background:#f8f9fa;border-radius:4px;margin-bottom:12px;",
                ))

            # 2. Job run link.
            if run_id and SIONNA_JOB_ID and DATABRICKS_WORKSPACE_URL:
                run_url = f"{DATABRICKS_WORKSPACE_URL}/jobs/{SIONNA_JOB_ID}/runs/{run_id}"
                blocks.append(ui.tags.p(
                    ui.tags.strong("Job run: "),
                    ui.tags.a(f"run_id={run_id}", href=run_url, target="_blank",
                              rel="noopener"),
                ))

            # 3. Config hash.
            blocks.append(ui.tags.p(
                ui.tags.strong("Config hash: "),
                ui.tags.code(config_hash),
            ))

            # 4. Error.
            if error_text:
                blocks.append(ui.tags.div(
                    ui.tags.strong("Error: "), error_text,
                    style=(
                        "color:#c00;padding:10px;border:1px solid #c00;"
                        "border-radius:4px;margin:12px 0;"
                    ),
                ))

            # 5. Scene dump.
            if scene:
                try:
                    scene_json = json.dumps(scene, indent=2, default=str)
                except Exception as e:
                    scene_json = f"(could not serialize scene: {e})"
                blocks.append(ui.h5("Current scene config"))
                blocks.append(ui.tags.pre(
                    scene_json,
                    style=(
                        "background:#f8f9fa;padding:10px;border-radius:4px;"
                        "font-size:12px;max-height:400px;overflow:auto;"
                    ),
                ))

            return ui.div(*blocks, style="padding:8px;")
        except Exception as e:
            import traceback as _tb
            return ui.tags.pre(
                f"status_view error: {e}\n\n{_tb.format_exc()}",
                style="color:#c00;padding:10px;",
            )


app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
