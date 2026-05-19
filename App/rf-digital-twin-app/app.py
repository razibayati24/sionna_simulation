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
import traceback
from typing import Any

from shiny import App, ui, render, reactive

import lakebase_client as lb
from defaults import CONFIG_1, CONFIG_2, PRESETS, preset_cells


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_CELLS = 7
SIONNA_JOB_ID = os.environ.get("SIONNA_JOB_ID")  # set if live re-renders enabled


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
    cells = []
    for i in range(NUM_CELLS):
        cells.append({
            "cell_id": i,
            "name": f"tx{i}",
            "x":         float(input[f"cell{i}_x"]()),
            "y":         float(input[f"cell{i}_y"]()),
            "z":         float(input[f"cell{i}_z"]()),
            "look_at_x": float(input[f"cell{i}_lax"]()),
            "look_at_y": float(input[f"cell{i}_lay"]()),
            "look_at_z": float(input[f"cell{i}_laz"]()),
            "power_dbm": float(input[f"cell{i}_pow"]()),
        })
    return cells


def _apply_preset(preset_key: str) -> None:
    """Push a preset into the input controls."""
    cfg = PRESETS[preset_key]
    cells = preset_cells()

    ui.update_numeric("num_rows_tx", value=cfg.num_rows_tx)
    ui.update_numeric("num_cols_tx", value=cfg.num_cols_tx)
    ui.update_numeric("num_rows_rx", value=cfg.num_rows_rx)
    ui.update_numeric("num_cols_rx", value=cfg.num_cols_rx)
    ui.update_select("pattern",      selected=cfg.pattern)
    ui.update_select("polarization", selected=cfg.polarization)
    ui.update_numeric("frequency_ghz", value=cfg.frequency_hz / 1e9)
    ui.update_numeric("bandwidth_mhz", value=cfg.bandwidth_hz / 1e6)
    ui.update_numeric("max_depth", value=cfg.max_depth)
    ui.update_select("samples_log10",
                     selected=str(int(round(__import__("math").log10(cfg.samples_per_tx)))))
    ui.update_numeric("cell_size_x", value=cfg.cell_size_x)
    ui.update_numeric("cell_size_y", value=cfg.cell_size_y)
    ui.update_numeric("num_user_samples", value=cfg.num_user_samples)
    ui.update_numeric("min_sinr_db", value=cfg.min_sinr_db)
    ui.update_numeric("min_user_dist_m", value=cfg.min_user_dist_m)
    ui.update_numeric("max_user_dist_m", value=cfg.max_user_dist_m)

    for i, c in enumerate(cells):
        ui.update_numeric(f"cell{i}_x",   value=c["x"])
        ui.update_numeric(f"cell{i}_y",   value=c["y"])
        ui.update_numeric(f"cell{i}_z",   value=c["z"])
        ui.update_numeric(f"cell{i}_lax", value=c["look_at_x"])
        ui.update_numeric(f"cell{i}_lay", value=c["look_at_y"])
        ui.update_numeric(f"cell{i}_laz", value=c["look_at_z"])
        ui.update_numeric(f"cell{i}_pow", value=c["power_dbm"])


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

def _cell_row_inputs(i: int, c: dict) -> ui.Tag:
    """One row of inputs for cell i."""
    return ui.row(
        ui.column(1, ui.tags.div(f"tx{i}", style="padding-top: 32px; font-weight: bold;")),
        ui.column(1, ui.input_numeric(f"cell{i}_x",   "x",   value=c["x"],   step=1)),
        ui.column(1, ui.input_numeric(f"cell{i}_y",   "y",   value=c["y"],   step=1)),
        ui.column(1, ui.input_numeric(f"cell{i}_z",   "z",   value=c["z"],   step=1)),
        ui.column(1, ui.input_numeric(f"cell{i}_lax", "look x", value=c["look_at_x"], step=1)),
        ui.column(1, ui.input_numeric(f"cell{i}_lay", "look y", value=c["look_at_y"], step=1)),
        ui.column(1, ui.input_numeric(f"cell{i}_laz", "look z", value=c["look_at_z"], step=1)),
        ui.column(2, ui.input_numeric(f"cell{i}_pow", "power (dBm)", value=c["power_dbm"], step=1)),
    )


_initial_cells = preset_cells()
_initial_scene = CONFIG_1


def _sidebar() -> ui.Tag:
    return ui.sidebar(
        ui.h4("Presets"),
        ui.input_action_button(
            "preset_1", "Load Config 1 (8x2)", class_="btn-primary",
            style="width: 100%; margin-bottom: 6px;",
        ),
        ui.input_action_button(
            "preset_2", "Load Config 2 (16x16)", class_="btn-warning",
            style="width: 100%;",
        ),
        ui.hr(),
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
        ui.nav_panel(
            "Cells",
            ui.h4("Per-TX configuration (7 cells around the Arc de Triomphe)"),
            ui.p("Edit positions, look-at points, and TX power. Coordinates are in scene meters."),
            *[_cell_row_inputs(i, _initial_cells[i]) for i in range(NUM_CELLS)],
        ),
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
        "status": "Ready. Edit a config or pick a preset, then click Render.",
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
            render_state.set({
                **render_state(),
                "status": (
                    f"Submitted Sionna compute job (run_id={run_id}). "
                    f"Polling cache every 10s…"
                ),
            })

            # Poll until the cache row appears, up to ~10 minutes.
            for _ in range(60):
                await asyncio.sleep(10)
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
                            f"Job complete (run_id={run_id}) — render cached "
                            f"in {cached.get('compute_seconds', 0):.1f}s."
                        ),
                    })
                    ui.update_navs("main_tabs", selected="Scene render")
                    return

            render_state.set({
                **render_state(),
                "status": "Job still running. Refresh to check again.",
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

    # ------------------------------------------------------------------
    # Preset buttons
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.preset_1)
    def _load_preset_1():
        _apply_preset("config_1")
        ui.notification_show("Loaded Config 1 (8x2 TX / 2x2 RX).", type="message", duration=3)

    @reactive.effect
    @reactive.event(input.preset_2)
    def _load_preset_2():
        _apply_preset("config_2")
        ui.notification_show("Loaded Config 2 (16x16 TX / 2x2 RX).", type="message", duration=3)

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
        st = render_state()
        items = [
            ui.tags.p(ui.tags.strong("Status: "), st.get("status", "")),
            ui.tags.p(ui.tags.strong("Config hash: "), st.get("config_hash") or "—"),
        ]
        if st.get("error"):
            items.append(ui.tags.div(
                ui.tags.strong("Error: "), st["error"],
                style="color: #c00; padding: 8px; border: 1px solid #c00; border-radius: 4px;",
            ))
        if st.get("scene"):
            items.append(ui.h5("Current scene config"))
            items.append(ui.tags.pre(json.dumps(st["scene"], indent=2)))
        return ui.div(*items, style="padding: 8px;")


app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
