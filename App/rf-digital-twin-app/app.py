"""RF Digital Twin app — Sionna RT on Databricks.

Two presets (Config 1: 8×2 TX / Config 2: 16×16 TX) backed by precomputed
Lakebase renders. Click a preset, then Render — the cached scene render,
SINR map, user-to-TX association, and CDFs appear instantly.
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


SIONNA_JOB_ID = os.environ.get("SIONNA_JOB_ID")  # set if live re-renders enabled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_img(data: bytes | memoryview | None, alt: str) -> ui.Tag:
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


def _config_summary_card(cfg) -> ui.Tag:
    """Read-only summary of the active preset."""
    def row(label: str, value: str) -> ui.Tag:
        return ui.tags.tr(
            ui.tags.td(label, style="color:#888; padding-right:10px;"),
            ui.tags.td(value, style="font-weight:600; font-family:monospace;"),
        )

    return ui.tags.table(
        ui.tags.tbody(
            row("TX array",  f"{cfg.num_rows_tx} × {cfg.num_cols_tx} UPA"),
            row("RX array",  f"{cfg.num_rows_rx} × {cfg.num_cols_rx} UPA"),
            row("Frequency", f"{cfg.frequency_hz / 1e9:g} GHz"),
            row("Bandwidth", f"{cfg.bandwidth_hz / 1e6:g} MHz"),
            row("Pattern",   cfg.pattern),
            row("Polarization", cfg.polarization),
            row("Cells",     "7 around Arc de Triomphe"),
        ),
        style="font-size:13px; margin-top:8px;",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Network configuration"),
        ui.input_action_button(
            "preset_1", "Load Config 1 (8×2)", class_="btn-primary",
            style="width:100%; margin-bottom:6px;",
        ),
        ui.input_action_button(
            "preset_2", "Load Config 2 (16×16)", class_="btn-warning",
            style="width:100%;",
        ),
        ui.hr(),
        ui.output_ui("current_config"),
        ui.hr(),
        ui.input_action_button(
            "render_btn", "Render scene", class_="btn-success",
            style="width:100%;",
        ),
        width=320,
    ),
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

    active_preset = reactive.Value("config_1")   # "config_1" | "config_2"

    render_state = reactive.Value({
        "config_hash": None,
        "scene": None,
        "data": None,
        "kpis": None,
        "status": "Ready. Pick Config 1 or Config 2, then click Render.",
        "error": None,
    })

    async def _try_load_cached(config_hash: str) -> dict | None:
        return await asyncio.to_thread(lb.get_render, config_hash)

    async def _do_render() -> None:
        try:
            cfg = PRESETS[active_preset()]
            scene_cfg = cfg.to_dict()
            cells = preset_cells()
            config_hash = lb.compute_config_hash(scene_cfg, cells)

            render_state.set({
                **render_state(),
                "config_hash": config_hash,
                "scene": scene_cfg,
                "status": f"Looking up cache for {cfg.name}…",
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
                        f"{cfg.name} loaded from cache "
                        f"(computed in {cached.get('compute_seconds', 0):.1f}s)."
                    ),
                })
                ui.update_navs("main_tabs", selected="Scene render")
                return

            render_state.set({
                **render_state(),
                "status": f"No cached render for {cfg.name}.",
                "error": (
                    f"No cached render for {cfg.name}. Run setup_rf_digital_twin.py "
                    f"in your workspace to populate the cache."
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
    # On first start, auto-load Config 1.
    # ------------------------------------------------------------------
    initial_loaded = reactive.Value(False)

    @reactive.effect
    async def _initial_load():
        if initial_loaded():
            return
        initial_loaded.set(True)
        await _do_render()

    # ------------------------------------------------------------------
    # Preset buttons
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.preset_1)
    def _load_preset_1():
        active_preset.set("config_1")
        ui.notification_show(
            "Loaded Config 1 (8×2 TX / 2×2 RX). Click Render.",
            type="message", duration=3,
        )

    @reactive.effect
    @reactive.event(input.preset_2)
    def _load_preset_2():
        active_preset.set("config_2")
        ui.notification_show(
            "Loaded Config 2 (16×16 TX / 2×2 RX). Click Render.",
            type="message", duration=3,
        )

    @reactive.effect
    @reactive.event(input.render_btn)
    async def _render():
        await _do_render()

    # ------------------------------------------------------------------
    # Sidebar summary
    # ------------------------------------------------------------------
    @render.ui
    def current_config():
        cfg = PRESETS[active_preset()]
        return ui.div(
            ui.h6(cfg.name, style="margin-bottom:0;"),
            _config_summary_card(cfg),
        )

    # ------------------------------------------------------------------
    # Tab views
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
            ui.h4("CDF of RSS", style="margin-top:20px;"),
            _png_img(data.get("rss_cdf_png"), "RSS CDF"),
        )

    @render.ui
    def kpis_view():
        kpis = render_state().get("kpis")
        if not kpis:
            return ui.tags.div("No KPIs yet — render a scene first.",
                               style="padding:20px; color:#888;")
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
                style="max-width:480px;",
            ),
        )

    @render.ui
    def status_view():
        st = render_state()
        items = [
            ui.tags.p(ui.tags.strong("Status: "), st.get("status", "")),
            ui.tags.p(ui.tags.strong("Config hash: "), st.get("config_hash") or "—"),
            ui.tags.p(ui.tags.strong("Active preset: "), active_preset()),
        ]
        if st.get("error"):
            items.append(ui.tags.div(
                ui.tags.strong("Error: "), st["error"],
                style="color:#c00; padding:8px; border:1px solid #c00; border-radius:4px;",
            ))
        if st.get("scene"):
            items.append(ui.h5("Scene config used for the current render"))
            items.append(ui.tags.pre(json.dumps(st["scene"], indent=2)))
        return ui.div(*items, style="padding:8px;")


app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
