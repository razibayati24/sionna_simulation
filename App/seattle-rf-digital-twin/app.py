"""Seattle Coverage Analysis — RF Digital Twin (Sionna RT on Databricks).

Pick a **neighborhood** and a **story/tile** from the sidebar; the app shows the cached
Sionna RT render (scene + SINR map + user→TX association + SINR/RSS CDFs + KPIs) for that
slice of Seattle's real T-Mobile network.

Cache model (Lakebase, shared ``rf-digital-twin-pg``):
  - **Downtown** is pre-rendered: ~7 curated "stories" + coverage tiles load instantly.
  - Any other neighborhood is **render-on-demand**: selecting it (or clicking "Render this
    neighborhood") triggers the ``seattle-rf-render`` GPU job, flips the neighborhood to
    ``RENDERING``, and a background poller surfaces the renders as the batches land — the
    same cache-miss → Jobs → poll pattern the etoile demo uses, but at neighborhood grain.

The app never touches Spark or the tower table: it selects cached renders by neighborhood +
story name straight from Lakebase.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import traceback
from typing import Any, Optional

from shiny import App, ui, render, reactive

import lakebase_client as lb
import neighborhoods as nb

SEATTLE_RENDER_JOB_ID = os.environ.get("SEATTLE_RENDER_JOB_ID")
DATABRICKS_WORKSPACE_URL = os.environ.get(
    "DATABRICKS_WORKSPACE_URL",
    f"https://{os.environ.get('DATABRICKS_HOST', '').rstrip('/')}",
).rstrip("/")

DEFAULT_NB = nb.DEFAULT_NEIGHBORHOOD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_img(data: bytes | memoryview | None, alt: str) -> ui.Tag:
    if not data:
        return ui.tags.div(f"No {alt} available yet.", style="padding:20px;color:#888;")
    if isinstance(data, memoryview):
        data = bytes(data)
    b64 = base64.b64encode(data).decode("ascii")
    return ui.tags.img(src=f"data:image/png;base64,{b64}",
                       style="max-width:100%;height:auto;border-radius:4px;", alt=alt)


def _trigger_render_job(neighborhood: str, mode: str = "coverage") -> int:
    if not SEATTLE_RENDER_JOB_ID:
        raise RuntimeError("SEATTLE_RENDER_JOB_ID is not set — cannot render on demand.")
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    run = w.jobs.run_now(
        job_id=int(SEATTLE_RENDER_JOB_ID),
        notebook_params={"neighborhood": neighborhood, "mode": mode},
    )
    return int(run.run_id)


def _render_label(row: dict) -> str:
    """Human label for a cached render row (story name, else tile id)."""
    return row.get("name") or row.get("story_key") or row.get("tile_id") or row["config_hash"][:12]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _sidebar() -> ui.Tag:
    return ui.sidebar(
        ui.h5("Neighborhood"),
        ui.input_select("neighborhood", None, choices=nb.names(), selected=DEFAULT_NB),
        ui.output_ui("neighborhood_status"),
        ui.input_action_button("render_btn", "Render this neighborhood",
                               class_="btn-success", style="width:100%;margin-top:6px;"),
        ui.hr(),
        ui.h5("Story / tile"),
        ui.output_ui("story_selector"),
        ui.tags.div(
            "Downtown ships pre-rendered (curated stories + coverage tiles). Other "
            "neighborhoods render on demand on a GPU job and cache into Lakebase.",
            style="font-size:11px;color:#888;margin-top:8px;",
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
    title="Seattle Coverage Analysis — RF Digital Twin",
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):

    state = reactive.Value({
        "neighborhood": DEFAULT_NB,
        "renders": [],          # list of cached render rows for the neighborhood
        "selected": None,       # the currently displayed render row
        "status": "Loading Downtown from cache…",
        "error": None,
        "pending": False,       # a render job is in flight for this neighborhood
        "run_id": None,
        "started": None,
    })

    async def _load_neighborhood(name: str, status_prefix: str = "") -> None:
        try:
            rows = await asyncio.to_thread(lb.list_neighborhood_renders, name)
            meta = await asyncio.to_thread(lb.get_neighborhood, name)
            nb_status = (meta or {}).get("status", "NONE")
            if rows:
                cur = state()
                # Keep current selection if it survives; else pick the first.
                sel = next((r for r in rows
                            if cur.get("selected") and r["config_hash"] == cur["selected"]["config_hash"]),
                           rows[0])
                state.set({
                    **cur, "neighborhood": name, "renders": rows, "selected": sel,
                    "pending": nb_status == "RENDERING",
                    "status": f"{status_prefix}{name}: {len(rows)} cached renders.",
                    "error": None,
                })
            else:
                state.set({
                    **state(), "neighborhood": name, "renders": [], "selected": None,
                    "pending": nb_status == "RENDERING",
                    "status": (f"{name} is rendering…" if nb_status == "RENDERING"
                               else f"{name} isn't rendered yet — click 'Render this neighborhood'."),
                    "error": None,
                })
        except Exception as e:
            state.set({**state(), "error": str(e), "status": "Could not connect to Lakebase."})
            traceback.print_exc()

    @reactive.effect
    async def _on_neighborhood_change():
        name = input.neighborhood()          # the only reactive dependency
        with reactive.isolate():
            cur = state()
        if not name:
            return
        # Reload when the dropdown changed, or on first load before anything is cached.
        if name != cur.get("neighborhood") or not cur.get("renders"):
            await _load_neighborhood(name)

    @reactive.effect
    @reactive.event(input.render_btn, ignore_init=True)
    async def _on_render():
        name = input.neighborhood()
        meta = await asyncio.to_thread(lb.get_neighborhood, name)
        if meta and meta.get("status") == "CACHED" and state().get("renders"):
            await _load_neighborhood(name, status_prefix="Already cached — ")
            return
        try:
            mode = "stories" if name == DEFAULT_NB else "coverage"
            run_id = await asyncio.to_thread(_trigger_render_job, name, mode)
            await asyncio.to_thread(lb.upsert_neighborhood, name, "RENDERING", run_id)
            state.set({
                **state(), "neighborhood": name, "pending": True, "run_id": run_id,
                "started": time.time(),
                "status": (f"Render job submitted for {name} (run_id={run_id}). Spinning up a "
                           f"GPU cluster; tiles will appear here as batches finish."),
                "error": None,
            })
        except Exception as e:
            state.set({**state(), "error": str(e), "status": "Could not submit render job."})
            traceback.print_exc()

    @reactive.effect
    async def _poll():
        reactive.invalidate_later(12)
        st = state()
        if not st.get("pending"):
            return
        name = st["neighborhood"]
        try:
            rows = await asyncio.to_thread(lb.list_neighborhood_renders, name)
            meta = await asyncio.to_thread(lb.get_neighborhood, name)
        except Exception as e:
            print(f"poll failed: {e}")
            return
        nb_status = (meta or {}).get("status", "RENDERING")
        if rows and nb_status == "CACHED":
            elapsed = int(time.time() - (st.get("started") or time.time()))
            state.set({**st, "renders": rows, "selected": rows[0], "pending": False,
                       "status": f"{name} render complete — {len(rows)} tiles ({elapsed}s)."})
        elif rows:
            state.set({**st, "renders": rows,
                       "selected": st.get("selected") or rows[0],
                       "status": f"{name} rendering… {len(rows)} tiles done so far."})

    # -- Selection --------------------------------------------------------
    @reactive.effect
    @reactive.event(input.story, ignore_init=True)
    def _on_story():
        key = input.story()
        row = next((r for r in state().get("renders", []) if _render_label(r) == key), None)
        if row:
            state.set({**state(), "selected": row})

    # -- Sidebar widgets --------------------------------------------------
    @render.ui
    def story_selector():
        rows = state().get("renders", [])
        if not rows:
            return ui.tags.div("No renders yet.", style="color:#888;font-size:12px;")
        labels = [_render_label(r) for r in rows]
        sel = state().get("selected")
        selected = _render_label(sel) if sel else labels[0]
        return ui.input_select("story", None, choices=labels, selected=selected)

    @render.ui
    def neighborhood_status():
        st = state()
        hood = nb.NEIGHBORHOODS.get(st["neighborhood"])
        blurb = hood.blurb if hood else ""
        badge = "rendering…" if st.get("pending") else (
            f"{len(st.get('renders', []))} renders" if st.get("renders") else "not rendered")
        return ui.tags.div(
            ui.tags.div(blurb, style="font-size:11px;color:#888;"),
            ui.tags.div(badge, style="font-size:11px;color:#0a7;margin-top:2px;"),
        )

    # -- Views ------------------------------------------------------------
    def _sel() -> dict:
        return state().get("selected") or {}

    @render.ui
    def scene_render_view():
        return ui.div(ui.h4("Scene render with SINR overlay"),
                      _png_img(_sel().get("scene_render_png"), "scene render"))

    @render.ui
    def sinr_map_view():
        return ui.div(ui.h4("Cell-to-TX association (SINR)"),
                      _png_img(_sel().get("sinr_map_png"), "SINR association"))

    @render.ui
    def association_view():
        return ui.div(ui.h4("Sampled users coloured by serving TX"),
                      _png_img(_sel().get("association_png"), "user-to-TX association"))

    @render.ui
    def cdf_view():
        s = _sel()
        return ui.div(
            ui.h4("CDF of SINR"), _png_img(s.get("sinr_cdf_png"), "SINR CDF"),
            ui.h4("CDF of RSS", style="margin-top:20px;"), _png_img(s.get("rss_cdf_png"), "RSS CDF"),
        )

    @render.ui
    def kpis_view():
        kpis = _sel().get("kpis_json")
        if isinstance(kpis, (bytes, str)):
            kpis = json.loads(kpis)
        if not kpis:
            return ui.tags.div("No KPIs yet — select a render.", style="padding:20px;color:#888;")
        sinr = kpis.get("sinr_percentiles_db", {})
        rss = kpis.get("rss_percentiles_dbm", {})
        mix = kpis.get("tower_type_mix", {})

        def _row(label: str, val: Any) -> ui.Tag:
            return ui.tags.tr(ui.tags.td(label), ui.tags.td(str(val)))

        rows = [
            _row("Neighborhood", kpis.get("neighborhood", "n/a")),
            _row("Tile", kpis.get("tile_id", "n/a")),
            _row("Towers (TX)", kpis.get("num_tx", "n/a")),
            _row("Tower mix", ", ".join(f"{k}:{v}" for k, v in mix.items()) or "n/a"),
            _row("SINR p10/p50/p90 (dB)",
                 f"{sinr.get('p10','?')} / {sinr.get('p50','?')} / {sinr.get('p90','?')}"),
            _row("RSS p10/p50/p90 (dBm)",
                 f"{rss.get('p10','?')} / {rss.get('p50','?')} / {rss.get('p90','?')}"),
        ]
        return ui.div(ui.h4("KPI summary"),
                      ui.tags.table(ui.tags.tbody(*rows), class_="table table-striped",
                                    style="max-width:520px;"))

    @render.ui
    def status_view():
        st = state()
        run_link = ""
        if st.get("run_id") and SEATTLE_RENDER_JOB_ID and DATABRICKS_WORKSPACE_URL:
            url = f"{DATABRICKS_WORKSPACE_URL}/jobs/{SEATTLE_RENDER_JOB_ID}/runs/{st['run_id']}"
            run_link = ui.tags.a(f"open run_id={st['run_id']}", href=url, target="_blank", rel="noopener")
        sel = st.get("selected") or {}
        return ui.div(
            ui.h4("Status"),
            ui.tags.p(ui.tags.strong("Message: "), st.get("status", "—")),
            ui.tags.p(ui.tags.strong("Neighborhood: "), st.get("neighborhood", "—"),
                      "  ", "(rendering)" if st.get("pending") else "", "  ", run_link),
            ui.tags.p(ui.tags.strong("Selected config hash: "),
                      ui.tags.code(str(sel.get("config_hash", "—")))),
            ui.tags.div(ui.tags.strong("Error: "), str(st.get("error")),
                        style="color:#c00;padding:8px;border:1px solid #c00;border-radius:4px;margin:8px 0;")
            if st.get("error") else "",
            style="padding:12px;",
        )


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
