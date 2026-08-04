"""RF Digital Twin — Sionna RT on Databricks, served from Lakebase.

Real network, real geometry, interactive: the transmitters are actual T-Mobile towers read
from Unity Catalog, the buildings are OpenStreetMap footprints for that city block, and the
propagation is a NVIDIA Sionna RT radio-map solve on a GPU cluster.

The interaction loop is the whole point of the demo:

  sidebar knobs ─→ config_hash ─→ Lakebase lookup
                                    ├─ hit  → PNGs + KPIs stream back in milliseconds
                                    └─ miss → submit a GPU job, poll, results appear here

That only works if the app and the render job agree on the hash for a given config. Both
resolve scenes through ``scene_spec``, and both hash through
``lakebase_client.compute_config_hash`` — see ``tools/check_hash_parity.py``, which asserts it.

Presets S1–S7 are pre-rendered for Downtown Seattle, so picking one is an instant cache hit.
Any knob moved off a preset is a cache miss that fires the GPU job — which is the honest
version of the story: cached is instant, uncached costs a cluster.
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

import defaults
import lakebase_client as lb
import neighborhoods as nb
import scene_spec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIONNA_RENDER_JOB_ID = os.environ.get("SIONNA_RENDER_JOB_ID") or os.environ.get("SIONNA_JOB_ID")
DATABRICKS_WORKSPACE_URL = os.environ.get(
    "DATABRICKS_WORKSPACE_URL",
    f"https://{os.environ.get('DATABRICKS_HOST', '').rstrip('/')}",
).rstrip("/")

# Estimated wall-clock for a cold GPU job cluster + one ~30-tower tile solve (wait ETA only).
JOB_ETA_SECONDS = 12 * 60

PATTERN_CHOICES = ["tr38901", "iso", "dipole", "hw_dipole"]
POLARIZATION_CHOICES = ["V", "H", "VH", "cross"]
# Tower types present in the UC table; "All" means don't filter.
TOWER_FILTER_CHOICES = ["All", "NR", "LTE", "UMTS", "GSM"]

CUSTOM_PRESET = "custom"
_PRESET_CHOICES = {CUSTOM_PRESET: "— Custom (off-menu) —",
                   **{k: s.name for k, s in defaults.STORIES.items()}}
_DEFAULT_PRESET = "s1_baseline"
_DEFAULT_NB = nb.DEFAULT_NEIGHBORHOOD

# The knob inputs a preset drives, so selecting one restores an exact cache hit.
_SCENE_INPUTS = (
    "num_rows_tx", "num_cols_tx", "num_rows_rx", "num_cols_rx", "pattern", "polarization",
    "frequency_ghz", "bandwidth_mhz", "max_depth", "samples_log10", "cell_size_x",
    "cell_size_y", "num_user_samples", "min_sinr_db", "min_user_dist_m", "max_user_dist_m",
    "tower_filter", "power_override",
)


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


def _story_from_inputs(input) -> defaults.SeattleStory:
    """Assemble a story (the hash-relevant knobs) from the sidebar."""
    tf = input.tower_filter()
    power = input.power_override()
    return defaults.SeattleStory(
        name="Custom",
        story_key=defaults.CUSTOM_STORY_KEY,
        num_rows_tx=int(input.num_rows_tx()),
        num_cols_tx=int(input.num_cols_tx()),
        num_rows_rx=int(input.num_rows_rx()),
        num_cols_rx=int(input.num_cols_rx()),
        frequency_hz=float(input.frequency_ghz()) * 1e9,
        bandwidth_hz=float(input.bandwidth_mhz()) * 1e6,
        max_depth=int(input.max_depth()),
        samples_per_tx=10 ** int(input.samples_log10()),
        cell_size_x=float(input.cell_size_x()),
        cell_size_y=float(input.cell_size_y()),
        pattern=input.pattern(),
        polarization=input.polarization(),
        num_user_samples=int(input.num_user_samples()),
        min_sinr_db=float(input.min_sinr_db()),
        min_user_dist_m=float(input.min_user_dist_m()),
        max_user_dist_m=float(input.max_user_dist_m()),
        neighborhood=input.neighborhood(),
        tower_filter=None if tf == "All" else tf,
        cell_power_override_dbm=None if power in (None, "") else float(power),
    )


def _matching_preset(story: defaults.SeattleStory) -> Optional[str]:
    """The preset key whose knobs equal this story, ignoring name/story_key. Else None."""
    ignore = {"name", "story_key"}
    mine = {k: v for k, v in story.to_dict().items() if k not in ignore}
    for key, preset in defaults.STORIES.items():
        theirs = {k: v for k, v in preset.to_dict().items() if k not in ignore}
        if mine == theirs:
            return key
    return None


def _identify(story: defaults.SeattleStory) -> defaults.SeattleStory:
    """Stamp name/story_key so a config that *is* a preset hashes as that preset.

    This is what makes the sidebar land on the pre-rendered cache rows: story_key is part of
    the hash, so nudging a knob back to a preset's values has to also restore its key.
    """
    key = _matching_preset(story)
    if key:
        preset = defaults.STORIES[key]
        story.name, story.story_key = preset.name, preset.story_key
    else:
        story.name = f"Custom · {story.neighborhood}"
        story.story_key = defaults.CUSTOM_STORY_KEY
    return story


def _submit_render_job(config_hash: str, story: defaults.SeattleStory) -> int:
    """Trigger the render job for an uncached config. Returns the Databricks run_id.

    Only the knobs travel; the job rebuilds the tower list itself via ``scene_spec.resolve``
    and asserts it arrives at the same hash.
    """
    if not SIONNA_RENDER_JOB_ID:
        raise RuntimeError(
            "Live compute disabled: SIONNA_RENDER_JOB_ID is not set. Pick one of the "
            "pre-rendered presets, or configure the render job (see README)."
        )
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    run = w.jobs.run_now(
        job_id=int(SIONNA_RENDER_JOB_ID),
        notebook_params={
            "mode": "custom",
            "config_hash": config_hash,
            "scene_json": json.dumps(story.to_dict()),
        },
    )
    return int(run.run_id)


def _cancel_databricks_run(run_id: int) -> None:
    from databricks.sdk import WorkspaceClient

    WorkspaceClient().jobs.cancel_run(run_id=int(run_id))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_initial = defaults.STORIES[_DEFAULT_PRESET]


def _sidebar() -> ui.Tag:
    return ui.sidebar(
        ui.h5("Scene"),
        ui.input_select("neighborhood", "Neighborhood",
                        choices=nb.names(), selected=_DEFAULT_NB),
        ui.input_select("preset", "Preset",
                        choices=_PRESET_CHOICES, selected=_DEFAULT_PRESET),
        ui.tags.div(
            "Presets are pre-rendered for Downtown — they load from Lakebase instantly. "
            "Move any knob and it becomes a custom config: a cache miss that renders on a "
            "GPU job.",
            style="font-size:11px;color:#888;margin:-6px 0 4px;",
        ),
        ui.h5("Towers"),
        ui.input_select("tower_filter", "Tower type",
                        choices=TOWER_FILTER_CHOICES, selected="All"),
        ui.input_numeric("power_override", "TX power override (dBm, blank = per-tower)",
                         value=None, min=20.0, max=60.0, step=1.0),
        ui.h5("Antenna array"),
        ui.row(
            ui.column(6, ui.input_numeric("num_rows_tx", "TX rows",
                                          value=_initial.num_rows_tx, min=1, max=64)),
            ui.column(6, ui.input_numeric("num_cols_tx", "TX cols",
                                          value=_initial.num_cols_tx, min=1, max=64)),
        ),
        ui.row(
            ui.column(6, ui.input_numeric("num_rows_rx", "RX rows",
                                          value=_initial.num_rows_rx, min=1, max=16)),
            ui.column(6, ui.input_numeric("num_cols_rx", "RX cols",
                                          value=_initial.num_cols_rx, min=1, max=16)),
        ),
        ui.input_select("pattern", "Pattern",
                        choices=PATTERN_CHOICES, selected=_initial.pattern),
        ui.input_select("polarization", "Polarization",
                        choices=POLARIZATION_CHOICES, selected=_initial.polarization),
        ui.h5("Radio"),
        ui.input_numeric("frequency_ghz", "Frequency (GHz)",
                         value=_initial.frequency_hz / 1e9, min=0.1, max=100, step=0.1),
        ui.input_numeric("bandwidth_mhz", "Bandwidth (MHz)",
                         value=_initial.bandwidth_hz / 1e6, min=1, max=10000, step=10),
        ui.h5("Ray tracing"),
        ui.input_numeric("max_depth", "Max depth", value=_initial.max_depth, min=1, max=10),
        ui.input_select("samples_log10", "Samples per TX (10^x)",
                        choices=["5", "6", "7", "8"],
                        selected=str(len(str(_initial.samples_per_tx)) - 1)),
        ui.row(
            ui.column(6, ui.input_numeric("cell_size_x", "Cell X (m)",
                                          value=_initial.cell_size_x, min=0.1, step=0.5)),
            ui.column(6, ui.input_numeric("cell_size_y", "Cell Y (m)",
                                          value=_initial.cell_size_y, min=0.1, step=0.5)),
        ),
        ui.tags.div(
            "Defaults are the approximation settings that keep a ~30-tower tile tractable "
            "(10⁶ samples, depth 3, 5 m cells). Raising them multiplies render time.",
            style="font-size:11px;color:#888;margin-top:-4px;",
        ),
        ui.h5("User sampling"),
        ui.input_numeric("num_user_samples", "Users / TX",
                         value=_initial.num_user_samples, min=1, max=1000),
        ui.input_numeric("min_sinr_db", "Min SINR (dB)", value=_initial.min_sinr_db),
        ui.row(
            ui.column(6, ui.input_numeric("min_user_dist_m", "Min dist (m)",
                                          value=_initial.min_user_dist_m)),
            ui.column(6, ui.input_numeric("max_user_dist_m", "Max dist (m)",
                                          value=_initial.max_user_dist_m)),
        ),
        ui.hr(),
        ui.input_action_button("render_btn", "Render scene", class_="btn-success",
                               style="width: 100%;"),
        ui.input_action_button("cancel_btn", "No job to cancel",
                               class_="btn-outline-danger",
                               style="width: 100%; margin-top: 6px;", disabled=True),
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
        "config_hash":     None,   # hash of the render in `data`, or of an in-flight job
        "scene":           None,   # scene_cfg dict last submitted
        "story":           None,   # the knobs behind it
        "data":            None,   # cached_renders row on display
        "kpis":            None,
        "n_towers":        None,
        "tile_id":         None,
        "pending_hash":    None,   # hash being computed; None when idle
        "pending_run_id":  None,
        "pending_started": None,
        "status":          "Loading the baseline preset from Lakebase…",
        "error":           None,
    })

    # Suppresses the "knob moved ⇒ Custom" watcher for the pass where a preset is being
    # applied. Deliberately a plain dict, not a reactive.Value: the watcher both reads and
    # clears it, and a reactive value would invalidate the watcher on its own write — an
    # endless re-run loop.
    applying_preset = {"on": False}

    def _apply_cached(state, config_hash, scene_cfg, story, cached, status_msg,
                      n_towers=None, tile_id=None):
        kpis = cached.get("kpis_json")
        if isinstance(kpis, (bytes, str)):
            kpis = json.loads(kpis)
        matched = state.get("pending_hash") == config_hash
        render_state.set({
            **state,
            "config_hash":     config_hash,
            "scene":           scene_cfg,
            "story":           story,
            "data":            cached,
            "kpis":            kpis,
            "n_towers":        n_towers if n_towers is not None else state.get("n_towers"),
            "tile_id":         tile_id if tile_id is not None else state.get("tile_id"),
            "pending_hash":    None if matched else state.get("pending_hash"),
            "pending_run_id":  None if matched else state.get("pending_run_id"),
            "pending_started": None if matched else state.get("pending_started"),
            "status":          status_msg,
            "error":           None,
        })

    async def _resolve(story: defaults.SeattleStory):
        """Tower load + tile pick off the event loop (it's a warehouse round trip)."""
        return await asyncio.to_thread(scene_spec.resolve, story.neighborhood, story)

    async def _do_render() -> None:
        """Cache hit → load and return. Miss → submit the GPU job and start polling."""
        try:
            story = _identify(_story_from_inputs(input))
            # The first render for a neighborhood reads its towers from the SQL warehouse, which
            # can take a few seconds (longer if the warehouse is cold) — say so rather than
            # looking hung. Cached thereafter for the life of the process.
            render_state.set({
                **render_state(),
                "status": f"Resolving {story.neighborhood} towers from Unity Catalog…",
                "error": None,
            })
            scene_cfg, cells, tile = await _resolve(story)
            config_hash = lb.compute_config_hash(scene_cfg, cells)

            cached = await asyncio.to_thread(lb.get_render, config_hash)
            if cached:
                _apply_cached(
                    render_state(), config_hash, scene_cfg, story, cached,
                    f"Cache hit from Lakebase ({config_hash[:12]}) — "
                    f"{len(cells)} towers on tile {tile.tile_id}, originally computed in "
                    f"{cached.get('compute_seconds', 0):.1f}s.",
                    n_towers=len(cells), tile_id=tile.tile_id,
                )
                ui.update_navs("main_tabs", selected="Scene render")
                return

            # Cache miss — only ever hold one cluster, so cancel any earlier run first.
            prev_run = render_state().get("pending_run_id")
            if prev_run:
                try:
                    await asyncio.to_thread(_cancel_databricks_run, prev_run)
                except Exception as e:
                    print(f"Failed to cancel previous run {prev_run}: {e}")

            if not SIONNA_RENDER_JOB_ID:
                render_state.set({
                    **render_state(),
                    "config_hash": config_hash, "scene": scene_cfg, "story": story,
                    "pending_hash": None, "pending_run_id": None, "pending_started": None,
                    "status": "Cache miss, and live compute is disabled.",
                    "error": ("No cached render for this config and SIONNA_RENDER_JOB_ID is "
                              "not set. Pick a preset, or configure the render job."),
                })
                return

            run_id = await asyncio.to_thread(_submit_render_job, config_hash, story)
            await asyncio.to_thread(lb.set_job_status, config_hash, "RUNNING", run_id, None)
            render_state.set({
                **render_state(),
                "config_hash": config_hash, "scene": scene_cfg, "story": story,
                "n_towers": len(cells), "tile_id": tile.tile_id,
                "pending_hash": config_hash,
                "pending_run_id": run_id,
                "pending_started": time.time(),
                "status": (
                    f"Cache miss — Sionna render job submitted (run_id={run_id}) for "
                    f"{len(cells)} towers on tile {tile.tile_id}. Spinning up the GPU "
                    f"cluster; results appear here automatically. Cached presets stay "
                    f"clickable meanwhile."
                ),
                "error": None,
            })

        except Exception as e:
            render_state.set({
                **render_state(),
                "error": f"{type(e).__name__}: {e}",
                "status": "Render failed.",
            })
            traceback.print_exc()

    @reactive.effect
    @reactive.event(input.render_btn)
    async def _render():
        await _do_render()

    # ------------------------------------------------------------------
    # Preset dropdown — push a preset's knob values into the sidebar.
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.preset)
    async def _on_preset():
        key = input.preset()
        if key == CUSTOM_PRESET:
            return
        story = defaults.STORIES[key]
        applying_preset["on"] = True
        ui.update_numeric("num_rows_tx", value=story.num_rows_tx)
        ui.update_numeric("num_cols_tx", value=story.num_cols_tx)
        ui.update_numeric("num_rows_rx", value=story.num_rows_rx)
        ui.update_numeric("num_cols_rx", value=story.num_cols_rx)
        ui.update_select("pattern", selected=story.pattern)
        ui.update_select("polarization", selected=story.polarization)
        ui.update_numeric("frequency_ghz", value=story.frequency_hz / 1e9)
        ui.update_numeric("bandwidth_mhz", value=story.bandwidth_hz / 1e6)
        ui.update_numeric("max_depth", value=story.max_depth)
        ui.update_select("samples_log10", selected=str(len(str(story.samples_per_tx)) - 1))
        ui.update_numeric("cell_size_x", value=story.cell_size_x)
        ui.update_numeric("cell_size_y", value=story.cell_size_y)
        ui.update_numeric("num_user_samples", value=story.num_user_samples)
        ui.update_numeric("min_sinr_db", value=story.min_sinr_db)
        ui.update_numeric("min_user_dist_m", value=story.min_user_dist_m)
        ui.update_numeric("max_user_dist_m", value=story.max_user_dist_m)
        ui.update_select("tower_filter", selected=story.tower_filter or "All")
        ui.update_numeric("power_override", value=story.cell_power_override_dbm)

    # ------------------------------------------------------------------
    # Knob watcher — moving anything off a preset flips the label to Custom, so the
    # dropdown never claims to show a preset the knobs no longer match.
    # ------------------------------------------------------------------
    @reactive.effect
    def _sync_preset_label():
        for name in _SCENE_INPUTS:
            input[name]()          # register as a reactive dependency
        if applying_preset["on"]:
            # This pass is the preset being applied; consume the flag and stop.
            applying_preset["on"] = False
            return
        with reactive.isolate():
            current = input.preset()
        try:
            key = _matching_preset(_story_from_inputs(input)) or CUSTOM_PRESET
        except Exception:
            return                 # mid-edit blank/invalid input; leave the label alone
        if key != current:
            ui.update_select("preset", selected=key)

    # ------------------------------------------------------------------
    # Background poller — pulls results when a pending render lands.
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
                st, pending_hash, st.get("scene"), st.get("story"), cached,
                f"Render complete (run_id={st.get('pending_run_id')}) — Sionna took "
                f"{cached.get('compute_seconds', 0):.1f}s, end-to-end {elapsed:.0f}s "
                f"including cluster start. It's cached now: this config is instant from here on.",
            )
            return

        # Still running — surface a job failure rather than polling forever.
        run_id = st.get("pending_run_id")
        job = None
        try:
            job = await asyncio.to_thread(lb.get_job, pending_hash)
        except Exception as e:
            print(f"Background poll: get_job failed: {e}")
        if job and job.get("status") == "FAILED":
            render_state.set({
                **st,
                "pending_hash": None, "pending_run_id": None, "pending_started": None,
                "status": f"Render job failed (run_id={run_id}).",
                "error": job.get("error_message") or "The render job reported FAILED.",
            })
            return

        elapsed = int(time.time() - (st.get("pending_started") or time.time()))
        remaining = max(JOB_ETA_SECONDS - elapsed, 30)
        render_state.set({
            **st,
            "status": (
                f"Sionna render running (run_id={run_id}). Elapsed "
                f"{elapsed // 60}m{elapsed % 60:02d}s, ~{remaining // 60}m"
                f"{remaining % 60:02d}s remaining. Switch to a cached preset to keep exploring."
            ),
        })

    # ------------------------------------------------------------------
    # Cancel button.
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
            msg = f"Cancel requested for run {run_id} but it failed: {e}"
            traceback.print_exc()
        render_state.set({
            **render_state(),
            "pending_hash": None, "pending_run_id": None, "pending_started": None,
            "status": msg,
        })

    @reactive.effect
    def _toggle_cancel_enabled():
        has_pending = bool(render_state().get("pending_run_id"))
        ui.update_action_button(
            "cancel_btn",
            label="Cancel pending job" if has_pending else "No job to cancel",
            disabled=not has_pending,
        )

    # ------------------------------------------------------------------
    # Auto-load the baseline preset on first paint.
    # ------------------------------------------------------------------
    initial_loaded = reactive.Value(False)

    @reactive.effect
    async def _initial_load():
        if initial_loaded():
            return
        initial_loaded.set(True)
        try:
            story = defaults.STORIES[_DEFAULT_PRESET]
            scene_cfg, cells, tile = await _resolve(story)
            config_hash = lb.compute_config_hash(scene_cfg, cells)
            cached = await asyncio.to_thread(lb.get_render, config_hash)
            if cached:
                _apply_cached(
                    render_state(), config_hash, scene_cfg, story, cached,
                    f"Loaded {story.name} from Lakebase ({config_hash[:12]}) — "
                    f"{len(cells)} real towers on tile {tile.tile_id}.",
                    n_towers=len(cells), tile_id=tile.tile_id,
                )
            else:
                render_state.set({
                    **render_state(),
                    "config_hash": config_hash, "scene": scene_cfg, "story": story,
                    "status": (
                        f"The baseline preset isn't cached in this Lakebase schema yet "
                        f"({config_hash[:12]}). Seed it with tools/seed_schema_from_seattle.py, "
                        f"or click Render to compute it on a GPU job."
                    ),
                })
        except Exception as e:
            render_state.set({
                **render_state(),
                "error": f"{type(e).__name__}: {e}",
                "status": "Could not load the baseline preset.",
            })
            traceback.print_exc()

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
        st = render_state()
        kpis = st.get("kpis")
        if not kpis:
            return ui.tags.div("No KPIs yet — render a scene first.",
                               style="padding: 20px; color: #888;")
        sinr = kpis.get("sinr_percentiles_db", {})
        rss = kpis.get("rss_percentiles_dbm", {})
        users = kpis.get("users_per_tx", {})

        def _row(label: str, val: Any) -> ui.Tag:
            return ui.tags.tr(ui.tags.td(label), ui.tags.td(str(val)))

        story = st.get("story")
        rows = [
            _row("Neighborhood", story.neighborhood if story else "n/a"),
            _row("Tile", st.get("tile_id") or "n/a"),
            _row("SINR p10 (dB)", sinr.get("p10", "n/a")),
            _row("SINR p50 (dB)", sinr.get("p50", "n/a")),
            _row("SINR p90 (dB)", sinr.get("p90", "n/a")),
            _row("RSS  p10 (dBm)", rss.get("p10", "n/a")),
            _row("RSS  p50 (dBm)", rss.get("p50", "n/a")),
            _row("RSS  p90 (dBm)", rss.get("p90", "n/a")),
            _row("Number of TXs (towers)", kpis.get("num_tx", "n/a")),
        ]
        mix = kpis.get("tower_type_mix")
        if mix:
            rows.append(_row("Tower type mix",
                             ", ".join(f"{k}: {v}" for k, v in sorted(mix.items()))))
        for tx, n in sorted(users.items()):
            rows.append(_row(f"Users assigned to tx{tx}", n))

        return ui.div(
            ui.h4("KPI summary"),
            ui.tags.table(ui.tags.tbody(*rows), class_="table table-striped",
                          style="max-width: 520px;"),
        )

    @render.ui
    def status_view():
        try:
            st = render_state() or {}
            run_id = st.get("pending_run_id")
            error_text = st.get("error")
            story = st.get("story")

            run_link = ""
            if run_id and SIONNA_RENDER_JOB_ID and DATABRICKS_WORKSPACE_URL:
                run_url = (f"{DATABRICKS_WORKSPACE_URL}/jobs/{SIONNA_RENDER_JOB_ID}"
                           f"/runs/{run_id}")
                run_link = ui.tags.a(f"open run_id={run_id}", href=run_url,
                                     target="_blank", rel="noopener")

            knobs = "(nothing rendered yet — click Render)"
            if story is not None:
                try:
                    knobs = json.dumps(story.to_dict(), indent=2, default=str)
                except Exception as e:
                    knobs = f"(could not serialise config: {e})"

            return ui.div(
                ui.h4("Status"),
                ui.tags.p(ui.tags.strong("Message: "), st.get("status") or "(no status)"),
                ui.tags.p(
                    ui.tags.strong("Pending job: "),
                    "yes" if st.get("pending_hash") else "no",
                    f"  (run_id={run_id})" if run_id else "", "  ", run_link,
                ),
                ui.tags.p(ui.tags.strong("Config hash: "),
                          ui.tags.code(str(st.get("config_hash") or "—"))),
                ui.tags.p(ui.tags.strong("Towers in this render: "),
                          str(st.get("n_towers") or "—"),
                          ui.tags.strong("   Tile: "), str(st.get("tile_id") or "—")),
                ui.tags.p(ui.tags.strong("Lakebase: "),
                          ui.tags.code(f"{lb._DEFAULT_INSTANCE_NAME} / {lb._PG_SCHEMA}")),
                ui.tags.div(
                    ui.tags.strong("Error: "), str(error_text),
                    style="color:#c00; padding:8px; border:1px solid #c00; "
                          "border-radius:4px; margin:8px 0;",
                ) if error_text else "",
                ui.h5("Config (the knobs that make the hash)"),
                ui.tags.pre(
                    knobs,
                    style="background:#f8f9fa; padding:10px; border-radius:4px; "
                          "font-size:12px; max-height:400px; overflow:auto;",
                ),
                style="padding:12px;",
            )
        except Exception as e:
            import traceback as _tb
            return ui.tags.pre(f"status_view error: {e}\n\n{_tb.format_exc()}",
                               style="color:#c00; padding:10px;")


app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
