"""Sionna RT pipeline for the Seattle RF Digital Twin (OSM scenes + real towers).

Adapted from the etoile demo's ``sionna_compute.py``. Two differences:

  1. **Scene geometry** is a per-tile OpenStreetMap scene (``osm_scene.load_tile_scene``)
     instead of the built-in Paris ``etoile`` mesh — so the ray tracing happens against
     real Seattle buildings (flat-ground fallback if OSM is unavailable).
  2. **Transmitters** are the real towers projected to local meters, with per-tower height
     and power from ``towers.randomize_config``. Sionna's ``scene.frequency`` and
     ``scene.tx_array`` are *scene-level* (one carrier + one array per radio-map solve), so
     each render config fixes those at the scene level and the curated stories vary one of
     them — exactly the etoile A–G pattern. Per-tower band/array are carried in the config
     (and drive band-specific story filtering) but a single solve is monochromatic.

The **approximation knobs** that make ~100-tower neighborhoods tractable live in the scene
config: ``samples_per_tx`` (default 1e6, vs 1e7), ``max_depth`` (3), ``cell_size`` (5 m).
These cut a solve 10–100× versus the full-fidelity etoile settings (see the repo README's
"Approximate the simulation" table).

Returned artefacts match the etoile pipeline (scene PNG, SINR map, association, SINR/RSS
CDFs, KPI JSON, compute_seconds) so the Lakebase cache + app are unchanged downstream.
"""
from __future__ import annotations

import io
import json
import time
from typing import Any, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _build_scene(scene_cfg: dict, cells: list[dict]):
    """Construct a Sionna RT scene: OSM (or flat-ground) geometry + real-tower TXs.

    ``scene_cfg`` extends the etoile config with optional geo keys:
      ``use_osm`` (bool), ``origin_lat``/``origin_lon`` (neighborhood ENU origin),
      ``render_bounds`` = [x_lo, x_hi, y_lo, y_hi] (tile bounds in local meters).
    Without ``use_osm`` it falls back to the etoile scene (keeps the module runnable for
    the homogeneous demo / smoke tests).
    """
    from sionna.rt import PlanarArray, Transmitter

    if scene_cfg.get("use_osm") and scene_cfg.get("render_bounds"):
        import osm_scene  # noqa: PLC0415

        origin = (float(scene_cfg["origin_lat"]), float(scene_cfg["origin_lon"]))
        scene = osm_scene.load_tile_scene(
            tuple(scene_cfg["render_bounds"]), origin,
            out_dir=scene_cfg.get("osm_scene_dir", "/tmp/seattle_osm_scenes"),
        )
    else:
        import sionna.rt  # noqa: PLC0415
        from sionna.rt import load_scene  # noqa: PLC0415
        scene = load_scene(sionna.rt.scene.etoile)

    scene.frequency = float(scene_cfg["frequency_hz"])
    scene.bandwidth = float(scene_cfg["bandwidth_hz"])
    scene.tx_array = PlanarArray(
        num_rows=int(scene_cfg["num_rows_tx"]),
        num_cols=int(scene_cfg["num_cols_tx"]),
        pattern=scene_cfg["pattern"],
        polarization=scene_cfg["polarization"],
    )
    scene.rx_array = PlanarArray(
        num_rows=int(scene_cfg["num_rows_rx"]),
        num_cols=int(scene_cfg["num_cols_rx"]),
        pattern=scene_cfg["pattern"],
        polarization=scene_cfg["polarization"],
    )

    for c in sorted(cells, key=lambda r: r["cell_id"]):
        scene.add(Transmitter(
            name=c["name"],
            position=[float(c["x"]), float(c["y"]), float(c["z"])],
            look_at=[float(c["look_at_x"]), float(c["look_at_y"]), float(c["look_at_z"])],
            power_dbm=float(c["power_dbm"]),
        ))
    return scene


def _camera_for(scene_cfg: dict):
    """Top-down camera framed over the tile center (etoile camera if no bounds)."""
    from sionna.rt import Camera

    rb = scene_cfg.get("render_bounds")
    if rb:
        x_lo, x_hi, y_lo, y_hi = rb
        cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
        # Altitude ~ the larger span so the whole tile is in frame.
        alt = max(x_hi - x_lo, y_hi - y_lo) * 1.3 + 300.0
        return Camera(position=[cx, cy, alt],
                      orientation=np.array([0.0, np.pi / 2, -np.pi / 2]))
    return Camera(position=[0.0, 0.0, 1000.0],
                  orientation=np.array([0.0, np.pi / 2, -np.pi / 2]))


def _scene_render_png(scene, radio_map, scene_cfg) -> bytes:
    fig = scene.render(
        camera=_camera_for(scene_cfg), radio_map=radio_map, rm_metric="sinr",
        rm_vmin=-10, rm_vmax=60, rm_show_color_bar=True,
    )
    if fig is None:
        fig = plt.gcf()
    return _fig_to_png(fig)


def _association_png(radio_map, num_user_samples, min_sinr_db, min_dist, max_dist):
    pos, cell_ids = radio_map.sample_positions(
        num_pos=num_user_samples, metric="sinr", min_val_db=min_sinr_db,
        min_dist=min_dist, max_dist=max_dist, tx_association=True,
    )
    fig = radio_map.show(metric="sinr", vmin=-10, vmax=70)
    cell_ids_np = cell_ids.numpy() if hasattr(cell_ids, "numpy") else np.asarray(cell_ids)
    cmap = mpl.colormaps["Dark2"].colors
    for tx, ids in enumerate(cell_ids_np):
        fig.axes[0].plot(ids[:, 1], ids[:, 0], marker="o", markersize=2,
                         linestyle="", color=cmap[tx % len(cmap)])
    users_per_tx = {int(tx): int((ids != 0).any(axis=1).sum())
                    for tx, ids in enumerate(cell_ids_np)}
    return _fig_to_png(fig), {"users_per_tx": users_per_tx}


def _cdf_png(radio_map, metric: str, xlim) -> tuple[bytes, dict]:
    plt.close("all")
    radio_map.cdf(metric=metric, bins=400)
    plt.xlim(*xlim)
    plt.title(f"CDF of {metric.upper()}")
    fig = plt.gcf()
    summary: dict[str, float] = {}
    for line in fig.gca().get_lines():
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs):
            for p in (10, 50, 90):
                idx = min(int(np.searchsorted(ys, p / 100.0)), len(xs) - 1)
                summary[f"p{p}"] = float(xs[idx])
            break
    return _fig_to_png(fig), summary


def run_simulation(scene_cfg: dict, cells: list[dict]) -> dict[str, Any]:
    """Run the full Sionna RT pipeline and return cacheable artefacts."""
    from sionna.rt import RadioMapSolver

    t0 = time.time()
    scene = _build_scene(scene_cfg, cells)

    rm_solver = RadioMapSolver()
    radio_map = rm_solver(
        scene,
        max_depth=int(scene_cfg["max_depth"]),
        samples_per_tx=int(scene_cfg["samples_per_tx"]),
        cell_size=(float(scene_cfg["cell_size_x"]), float(scene_cfg["cell_size_y"])),
    )

    scene_png = _scene_render_png(scene, radio_map, scene_cfg)
    sinr_map_png = _fig_to_png(radio_map.show_association("sinr"))
    association_png, assoc_kpis = _association_png(
        radio_map,
        num_user_samples=int(scene_cfg["num_user_samples"]),
        min_sinr_db=float(scene_cfg["min_sinr_db"]),
        min_dist=float(scene_cfg["min_user_dist_m"]),
        max_dist=float(scene_cfg["max_user_dist_m"]),
    )
    sinr_cdf_png, sinr_pct = _cdf_png(radio_map, "sinr", xlim=(-40.0, 75.0))
    rss_cdf_png, rss_pct = _cdf_png(radio_map, "rss", xlim=(-150.0, 25.0))

    type_mix: dict[str, int] = {}
    for c in cells:
        type_mix[str(c.get("tower_type", "NA"))] = type_mix.get(str(c.get("tower_type", "NA")), 0) + 1

    kpis = {
        "sinr_percentiles_db": sinr_pct,
        "rss_percentiles_dbm": rss_pct,
        **assoc_kpis,
        "num_tx": len(cells),
        "tower_type_mix": type_mix,
        "neighborhood": scene_cfg.get("neighborhood"),
        "tile_id": scene_cfg.get("tile_id"),
    }
    return {
        "scene_render_png": scene_png,
        "sinr_map_png": sinr_map_png,
        "association_png": association_png,
        "sinr_cdf_png": sinr_cdf_png,
        "rss_cdf_png": rss_cdf_png,
        "kpis_json": json.dumps(kpis),
        "compute_seconds": time.time() - t0,
    }
