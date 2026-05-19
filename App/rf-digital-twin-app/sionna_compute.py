"""Sionna RT pipeline shared by the setup script and the live-render job.

Inputs:
  scene (dict)         — scene-level config (see defaults.SceneConfig.to_dict())
  cells (list[dict])   — per-TX rows (cell_id, name, x, y, z, look_at_x/y/z, power_dbm)

Outputs (returned as a dict):
  scene_render_png   bytes
  sinr_map_png       bytes
  association_png    bytes
  sinr_cdf_png       bytes
  rss_cdf_png        bytes
  kpis_json          dict        — summary statistics
  compute_seconds    float

This module isolates everything that imports drjit/mitsuba/sionna so the app
process can stay light. Only run this on a Databricks job/cluster with the
heavy dependencies installed (and ideally a GPU).
"""
from __future__ import annotations

import io
import json
import time
from typing import Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _build_scene(scene_cfg: dict, cells: list[dict]):
    """Construct and configure a Sionna RT scene from the supplied config."""
    import sionna.rt
    from sionna.rt import load_scene, PlanarArray, Transmitter

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
        scene.add(
            Transmitter(
                name=c["name"],
                position=[float(c["x"]), float(c["y"]), float(c["z"])],
                look_at=[float(c["look_at_x"]), float(c["look_at_y"]), float(c["look_at_z"])],
                power_dbm=float(c["power_dbm"]),
            )
        )

    return scene


def _scene_render_png(scene, radio_map) -> bytes:
    """Render the scene from a top-down camera with the SINR radio map overlay."""
    from sionna.rt import Camera

    cam = Camera(
        position=[0.0, 0.0, 1000.0],
        orientation=np.array([0.0, np.pi / 2, -np.pi / 2]),
    )
    # Sionna's scene.render returns a matplotlib figure.
    fig = scene.render(
        camera=cam,
        radio_map=radio_map,
        rm_metric="sinr",
        rm_vmin=-10,
        rm_vmax=60,
        rm_show_color_bar=True,
    )
    return _fig_to_png(fig)


def _association_png(radio_map, num_user_samples: int, min_sinr_db: float,
                     min_dist: float, max_dist: float) -> tuple[bytes, dict]:
    """Sample user positions and render the cell-to-TX association plot."""
    pos, cell_ids = radio_map.sample_positions(
        num_pos=num_user_samples,
        metric="sinr",
        min_val_db=min_sinr_db,
        min_dist=min_dist,
        max_dist=max_dist,
        tx_association=True,
    )

    fig = radio_map.show(metric="sinr", vmin=-10, vmax=70)
    cell_ids_np = cell_ids.numpy() if hasattr(cell_ids, "numpy") else np.asarray(cell_ids)
    cmap = mpl.colormaps["Dark2"].colors

    for tx, ids in enumerate(cell_ids_np):
        fig.axes[0].plot(
            ids[:, 1], ids[:, 0],
            marker="o", markersize=2, linestyle="",
            color=cmap[tx % len(cmap)],
        )

    users_per_tx = {int(tx): int((ids != 0).any(axis=1).sum()) for tx, ids in enumerate(cell_ids_np)}
    return _fig_to_png(fig), {"users_per_tx": users_per_tx}


def _cdf_png(radio_map, metric: str, xlim: tuple[float, float]) -> tuple[bytes, dict]:
    """Render the CDF for `metric` ('sinr' or 'rss') and return percentile summary.

    Sionna's `radio_map.cdf()` opens its own matplotlib figure, so we let it
    do that and then grab whatever figure is current — pre-creating a figure
    here would just produce an empty plot.
    """
    plt.close("all")
    radio_map.cdf(metric=metric, bins=400)
    plt.xlim(*xlim)
    plt.title(f"CDF of {metric.upper()}")
    fig = plt.gcf()

    summary: dict[str, float] = {}
    for line in fig.gca().get_lines():
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs) > 0:
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

    scene_png = _scene_render_png(scene, radio_map)

    fig = radio_map.show_association("sinr")
    sinr_map_png = _fig_to_png(fig)

    association_png, assoc_kpis = _association_png(
        radio_map,
        num_user_samples=int(scene_cfg["num_user_samples"]),
        min_sinr_db=float(scene_cfg["min_sinr_db"]),
        min_dist=float(scene_cfg["min_user_dist_m"]),
        max_dist=float(scene_cfg["max_user_dist_m"]),
    )

    sinr_cdf_png, sinr_pct = _cdf_png(radio_map, "sinr", xlim=(-40.0, 75.0))
    rss_cdf_png,  rss_pct  = _cdf_png(radio_map, "rss",  xlim=(-150.0, 25.0))

    kpis = {
        "sinr_percentiles_db": sinr_pct,
        "rss_percentiles_dbm": rss_pct,
        **assoc_kpis,
        "num_tx": len(cells),
    }

    return {
        "scene_render_png": scene_png,
        "sinr_map_png":     sinr_map_png,
        "association_png":  association_png,
        "sinr_cdf_png":     sinr_cdf_png,
        "rss_cdf_png":      rss_cdf_png,
        "kpis_json":        json.dumps(kpis),
        "compute_seconds":  time.time() - t0,
    }
