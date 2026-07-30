"""Large-scale radio-map pipeline — NVIDIA `sionna-large-radio-maps` on Databricks.

This complements `sionna_compute.py` (single synthetic `etoile` scene) by
computing coverage across a *large real geographic region*: real lat/lon
bounding box, OpenStreetMap buildings, and a base-station layout. It wraps the
NVlabs `sionna_lrm` package pipeline:

    tiling  ->  scene build (OSM)  ->  radio-map ray tracing  ->  mosaic

Repo: https://github.com/NVlabs/sionna-large-radio-maps

Two execution paths, chosen automatically:

  * Real path (`_run_real`)  — used on a GPU job cluster where the NVlabs
    scripts + Sionna RT are installed. Drives the documented CLI
    (`generate_tiling.py`, `scene_builder.py`, `compute_radio_maps.py`),
    then mosaics the per-tile path-gain arrays into one coverage map.

  * Demo path (`_demo`)      — a lightweight, GPU-free synthetic coverage map
    over the same bounding box. Lets the app render the tab (and the setup
    notebook seed a cache row) without a cluster. Clearly labelled as a demo.

Outputs (returned as a dict, mirroring sionna_compute.run_simulation):

  coverage_png     bytes   — path-gain / RSS heatmap over the region
  tiling_png       bytes   — adaptive tile grid + base-station scatter
  cdf_png          bytes   — CDF of path gain across covered cells
  kpis_json        str     — summary statistics (JSON)
  compute_seconds  float
  is_demo          bool    — True when produced by the synthetic fallback
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# NVlabs repo defaults (see sionna_lrm/constants.py).
DEFAULT_RM_DB_VMIN = -120.0
DEFAULT_RM_DB_VMAX = -45.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _bbox(region_cfg: dict) -> tuple[float, float, float, float]:
    """Return (south, west, north, east)."""
    return (
        float(region_cfg["south"]),
        float(region_cfg["west"]),
        float(region_cfg["north"]),
        float(region_cfg["east"]),
    )


def _area_km2(south: float, west: float, north: float, east: float) -> float:
    """Rough planar area of the bbox in km² (fine at city scale)."""
    mean_lat = np.radians((south + north) / 2.0)
    dy_km = (north - south) * 111.32
    dx_km = (east - west) * 111.32 * np.cos(mean_lat)
    return abs(dx_km * dy_km)


def _synth_base_stations(
    south: float, west: float, north: float, east: float, n: int, seed: int,
) -> np.ndarray:
    """Deterministic pseudo-random base-station lat/lon layout for the demo."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(south, north, size=n)
    lon = rng.uniform(west, east, size=n)
    return np.column_stack([lat, lon])


# ---------------------------------------------------------------------------
# Demo (GPU-free) path
# ---------------------------------------------------------------------------

def _demo(region_cfg: dict) -> dict[str, Any]:
    """Synthesize a plausible large-scale coverage map without ray tracing.

    Builds a grid over the bbox, drops synthetic base stations, and computes a
    log-distance path-gain surface (best-server per cell). Good enough to
    exercise the UI, the Lakebase cache path, and the mosaic/CDF plots.
    """
    t0 = time.time()
    south, west, north, east = _bbox(region_cfg)

    grid_n = int(region_cfg.get("demo_grid", 320))
    n_bs = int(region_cfg.get("num_base_stations", 24))
    freq_hz = float(region_cfg["frequency_hz"])
    tx_power_dbm = float(region_cfg["tx_power_dbm"])

    lats = np.linspace(south, north, grid_n)
    lons = np.linspace(west, east, grid_n)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    bs = _synth_base_stations(south, west, north, east, n_bs, seed=grid_n + n_bs)

    mean_lat = np.radians((south + north) / 2.0)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(mean_lat)

    # Free-space-ish path loss reference at 1 m, plus log-distance decay and a
    # smooth shadowing field so the map reads like a real coverage plot.
    fspl_1m = 20.0 * np.log10(freq_hz) - 147.55  # dB at 1 m
    path_exp = 2.9  # urban-ish path-loss exponent

    shadow = np.zeros_like(lat_grid)
    rng = np.random.default_rng(1234)
    for _ in range(6):  # sum of a few low-freq sinusoids = correlated shadowing
        fx, fy = rng.uniform(0.5, 3.0, size=2)
        px, py = rng.uniform(0, 2 * np.pi, size=2)
        amp = rng.uniform(2.0, 6.0)
        u = (lon_grid - west) / (east - west)
        v = (lat_grid - south) / (north - south)
        shadow += amp * np.sin(2 * np.pi * fx * u + px) * np.sin(2 * np.pi * fy * v + py)

    best_rss = np.full_like(lat_grid, -np.inf)
    for b_lat, b_lon in bs:
        dlat_m = (lat_grid - b_lat) * m_per_deg_lat
        dlon_m = (lon_grid - b_lon) * m_per_deg_lon
        dist_m = np.sqrt(dlat_m ** 2 + dlon_m ** 2)
        dist_m = np.maximum(dist_m, 1.0)
        path_loss = fspl_1m + 10.0 * path_exp * np.log10(dist_m)
        rss = tx_power_dbm - path_loss + shadow
        best_rss = np.maximum(best_rss, rss)

    # RSS demo tends to sit well above the path-gain range; use a band that
    # keeps the colour ramp meaningful for RSS in dBm.
    vmin, vmax = -110.0, -45.0

    # --- Coverage heatmap (static PNG, kept for reference/thumbnails) ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        best_rss, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
        extent=[west, east, south, north], aspect="auto",
    )
    ax.scatter(bs[:, 1], bs[:, 0], c="red", s=18, marker="^",
               edgecolors="white", linewidths=0.4, label="Base stations")
    ax.set_title(f"{region_cfg.get('name', 'Region')} — best-server RSS (DEMO)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="RSS (dBm)")
    coverage_png = _fig_to_png(fig)

    # --- Georeferenced overlay + legend for the zoomable Leaflet map ---
    overlay_png = _overlay_raster_png(best_rss, vmin, vmax, alpha_floor=0.0)
    legend_png = _legend_png(vmin, vmax, "RSS (dBm)")

    # --- Tiling preview (adaptive-ish grid + BS scatter) ---
    tiling_png = _tiling_preview(
        south, west, north, east, bs,
        float(region_cfg["min_cell_size_m"]),
        float(region_cfg["max_cell_size_m"]),
        title=f"{region_cfg.get('name', 'Region')} — tiling preview (DEMO)",
    )

    # --- CDF of RSS over covered cells ---
    cdf_png, rss_pct = _cdf_png(best_rss.ravel(), "RSS (dBm)", (vmin, vmax))

    covered = best_rss >= float(region_cfg.get("coverage_threshold_dbm", -100.0))
    kpis = {
        "region":               region_cfg.get("name"),
        "bbox":                 [south, west, north, east],
        "area_km2":             round(_area_km2(south, west, north, east), 3),
        "num_base_stations":    int(n_bs),
        "grid_cells":           int(grid_n * grid_n),
        "coverage_pct":         round(100.0 * covered.mean(), 2),
        "rss_percentiles_dbm":  rss_pct,
        "frequency_ghz":        round(freq_hz / 1e9, 3),
        "tx_power_dbm":         tx_power_dbm,
    }

    return {
        "coverage_png":     coverage_png,
        "overlay_png":      overlay_png,
        "legend_png":       legend_png,
        "tiling_png":       tiling_png,
        "cdf_png":          cdf_png,
        "bounds_json":      json.dumps([south, west, north, east]),
        "base_stations_json": json.dumps([[float(la), float(lo)] for la, lo in bs]),
        "kpis_json":        json.dumps(kpis),
        "compute_seconds":  time.time() - t0,
        "is_demo":          True,
    }


def _tiling_preview(south, west, north, east, bs, min_cell_m, max_cell_m, title) -> bytes:
    """Draw an adaptive tile grid: denser tiles where base stations cluster."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(west, east); ax.set_ylim(south, north)
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    mean_lat = np.radians((south + north) / 2.0)
    span_m = (north - south) * 111_320.0
    # Coarse tiles, subdivided once near base stations (mimics adaptive tiling).
    n_coarse = max(2, int(span_m / max(max_cell_m, 1.0) / 8))
    lat_edges = np.linspace(south, north, n_coarse + 1)
    lon_edges = np.linspace(west, east, n_coarse + 1)

    for i in range(n_coarse):
        for j in range(n_coarse):
            y0, y1 = lat_edges[i], lat_edges[i + 1]
            x0, x1 = lon_edges[j], lon_edges[j + 1]
            in_tile = ((bs[:, 0] >= y0) & (bs[:, 0] < y1) &
                       (bs[:, 1] >= x0) & (bs[:, 1] < x1)).sum()
            if in_tile > 0:  # subdivide 2x2
                for di in range(2):
                    for dj in range(2):
                        ax.add_patch(Rectangle(
                            (x0 + dj * (x1 - x0) / 2, y0 + di * (y1 - y0) / 2),
                            (x1 - x0) / 2, (y1 - y0) / 2,
                            fill=False, edgecolor="#4c78a8", linewidth=0.6,
                        ))
            else:
                ax.add_patch(Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    fill=False, edgecolor="#bbbbbb", linewidth=0.5,
                ))
    ax.scatter(bs[:, 1], bs[:, 0], c="red", s=18, marker="^",
               edgecolors="white", linewidths=0.4, label="Base stations")
    ax.legend(loc="upper right", fontsize=8)
    return _fig_to_png(fig)


def _overlay_raster_png(values: np.ndarray, vmin: float, vmax: float,
                        cmap_name: str = "viridis",
                        alpha_floor: float = 0.0) -> bytes:
    """Render `values` as a bare, north-up RGBA PNG for a Leaflet image overlay.

    No axes, ticks, or colour bar — just the coloured raster, sized 1 px per
    grid cell. Cells below `vmin` fade toward transparent so the basemap shows
    through where there is no coverage. Row 0 of the returned image is the
    NORTH edge (Leaflet places the image's top row at the north bound).
    """
    vals = np.asarray(values, dtype=float)
    norm = np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap_name](norm)  # (H, W, 4), floats 0..1
    # Alpha ramps with signal so weak/no coverage is see-through.
    rgba[..., 3] = alpha_floor + (1.0 - alpha_floor) * norm
    rgba = np.flipud(rgba)  # north-up for Leaflet imageOverlay

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return buf.getvalue()


def _legend_png(vmin: float, vmax: float, label: str,
                cmap_name: str = "viridis") -> bytes:
    """Standalone horizontal colour-bar legend for the map overlay."""
    fig, ax = plt.subplots(figsize=(4.2, 0.7))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=matplotlib.colormaps[cmap_name], norm=norm, orientation="horizontal",
    )
    cb.set_label(label, fontsize=9)
    ax.tick_params(labelsize=8)
    return _fig_to_png(fig)


def _cdf_png(values: np.ndarray, label: str, xlim: tuple[float, float]) -> tuple[bytes, dict]:
    """Empirical CDF of `values` with p10/p50/p90 summary."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(7, 5))
    summary: dict[str, float] = {}
    if vals.size:
        xs = np.sort(vals)
        ys = np.linspace(0, 1, xs.size)
        ax.plot(xs, ys, color="#4c78a8")
        for p in (10, 50, 90):
            summary[f"p{p}"] = round(float(np.percentile(vals, p)), 2)
    ax.set_xlim(*xlim); ax.set_ylim(0, 1)
    ax.set_xlabel(label); ax.set_ylabel("CDF")
    ax.set_title(f"CDF of {label}")
    ax.grid(True, alpha=0.3)
    return _fig_to_png(fig), summary


# ---------------------------------------------------------------------------
# Real path — drives the NVlabs sionna_lrm CLI on a GPU cluster
# ---------------------------------------------------------------------------

# The NVlabs repo's runtime dependencies, mirrored from its pyproject.toml
# [project].dependencies. We install THESE into an isolated prefix (not the repo
# itself: its pyproject uses the SPDX `license = "Apache-2.0"` form that the
# cluster's older setuptools rejects at wheel-build time, and we don't need the
# repo as an installed package — `sionna_lrm` imports directly from repo_dir on
# PYTHONPATH). Kept OUT of the notebook kernel: this stack drags in a numpy build
# ABI-incompatible with the runtime's precompiled numpy/pyarrow, which crashes
# the kernel REPL. numpy is pinned <2 because the prefix's pandas transitively
# imports the runtime's pyarrow (compiled against numpy 1.x); numpy 2.x there
# raises "module compiled using NumPy 1.x cannot be run in NumPy 2.x".
_SUBPROC_PACKAGES = [
    "numpy<2",
    "sionna-rt==1.2.1",
    "basemap", "boto3", "geopandas", "matplotlib", "pandas", "Pillow",
    "pyproj", "scipy", "shapely", "tqdm", "open3d", "triangle", "osmnx",
]


def _run_checked(cmd: list[str], env: dict | None = None) -> None:
    """Run `cmd`, capturing output; on failure raise with stdout+stderr attached."""
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {r.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{r.stdout[-4000:]}\n"
            f"--- stderr ---\n{r.stderr[-4000:]}"
        )


def _ensure_subproc_env(repo_dir: str) -> tuple[str, str]:
    """Install the NVlabs deps into an isolated prefix; return (python, PYTHONPATH).

    Deliberately avoids `venv`: the Databricks runtime ships without ensurepip
    (no python3.x-venv apt package), so `python -m venv` can't seed pip. Instead
    we `pip install --target <prefix>` into a plain directory and run the scripts
    with that prefix (plus the repo, so `sionna_lrm` imports) prepended to
    PYTHONPATH. The scripts run under sys.executable but with the isolated numpy/
    geo stack shadowing the runtime's — safe because it's a subprocess that never
    imports into the notebook kernel. Cached under the repo dir for warm reuse.
    """
    prefix = os.path.join(repo_dir, ".slrm_deps")
    marker = os.path.join(prefix, ".deps_ok")
    pythonpath = os.pathsep.join([prefix, repo_dir])
    if os.path.exists(marker):
        return sys.executable, pythonpath

    os.makedirs(prefix, exist_ok=True)
    print(f"Installing NVlabs deps into isolated prefix {prefix} …", flush=True)
    # Install the declared dependency set into the prefix. --target keeps it out
    # of the kernel; the repo code itself imports from repo_dir on PYTHONPATH.
    _run_checked([sys.executable, "-m", "pip", "install",
                  "--target", prefix, *_SUBPROC_PACKAGES])
    open(marker, "w").write("ok")
    return sys.executable, pythonpath


def _run_real(region_cfg: dict, repo_dir: str) -> dict[str, Any]:
    """Run the documented NVlabs pipeline end-to-end for the bbox.

    Requires `repo_dir` to be a checkout of NVlabs/sionna-large-radio-maps on a
    GPU node. The heavy deps (Sionna RT / Mitsuba / drjit / geo stack) run in an
    isolated venv (see `_ensure_subproc_python`) so they never destabilise the
    notebook kernel. Writes into a temp data dir (SLRM_DATA_DIR) and mosaics the
    per-tile outputs. All Sionna/geo code executes in subprocesses — this module
    itself only needs numpy + matplotlib (already in the runtime).
    """
    t0 = time.time()
    south, west, north, east = _bbox(region_cfg)
    scripts = os.path.join(repo_dir, "scripts")

    subproc_py, pythonpath = _ensure_subproc_env(repo_dir)

    data_dir = tempfile.mkdtemp(prefix="slrm_")
    env = {**os.environ, "SLRM_DATA_DIR": data_dir}
    # Prepend the isolated deps prefix + repo so the scripts see the isolated
    # numpy/geo/sionna stack (and can import sionna_lrm) without touching the
    # notebook kernel.
    env["PYTHONPATH"] = os.pathsep.join(
        [pythonpath] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    )
    # sionna_lrm/__init__.py hard-requires remote/scenes to exist at import
    # time (and references these other dirs), so create the full skeleton.
    outputs_dir = os.path.join(data_dir, "remote", "outputs")
    for sub in ("remote/scenes", "remote/outputs", "remote/transmitters",
                "local/scenes", "local/optix_cache"):
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)

    def _run(cmd: list[str]) -> None:
        print("+ " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=scripts, env=env, capture_output=True, text=True)
        if r.stdout:
            print(r.stdout[-4000:], flush=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"NVlabs script failed (exit {r.returncode}): {' '.join(cmd)}\n"
                f"--- stderr ---\n{r.stderr[-4000:]}"
            )

    area = "region"
    tiling_npz = os.path.join(outputs_dir, "tiling.npz")

    # 1) Adaptive tiling for the bbox.
    _run([subproc_py, "generate_tiling.py",
          "--bbox", str(south), str(west), str(north), str(east),
          tiling_npz])

    # 2) Build Sionna RT scenes (pulls OSM buildings).
    _run([subproc_py, "scene_builder.py", "file", tiling_npz,
          "--subdir", area])

    # 3) Compute per-tile radio maps.
    scenes_dir = os.path.join(data_dir, "local", "scenes", area)
    rm_out = os.path.join(data_dir, "remote", "outputs", "radio_maps")
    os.makedirs(rm_out, exist_ok=True)
    _run([subproc_py, "compute_radio_maps.py",
          "-s", scenes_dir, "-o", rm_out,
          "--n-samples", str(int(region_cfg.get("samples", 20_000_000)))])

    # 4) Mosaic per-tile arrays (best-effort — output layout is loaded
    #    defensively; any deviation raises and we fall back to demo upstream).
    mosaic, bs = _load_and_mosaic(rm_out, tiling_npz)

    vmin, vmax = DEFAULT_RM_DB_VMIN, DEFAULT_RM_DB_VMAX
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mosaic, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[west, east, south, north], aspect="auto")
    ax.set_title(f"{region_cfg.get('name', 'Region')} — path gain (Sionna RT)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, label="Path gain (dB)")
    coverage_png = _fig_to_png(fig)

    overlay_png = _overlay_raster_png(mosaic, vmin, vmax, alpha_floor=0.0)
    legend_png = _legend_png(vmin, vmax, "Path gain (dB)")

    tiling_png = _tiling_preview(
        south, west, north, east,
        bs if bs is not None else np.empty((0, 2)),
        float(region_cfg["min_cell_size_m"]), float(region_cfg["max_cell_size_m"]),
        title=f"{region_cfg.get('name', 'Region')} — tiling",
    )
    cdf_png, pct = _cdf_png(mosaic.ravel(), "Path gain (dB)", (vmin, vmax))

    covered = mosaic >= float(region_cfg.get("coverage_threshold_db", -110.0))
    kpis = {
        "region":              region_cfg.get("name"),
        "bbox":                [south, west, north, east],
        "area_km2":            round(_area_km2(south, west, north, east), 3),
        "coverage_pct":        round(100.0 * float(np.mean(covered)), 2),
        "path_gain_percentiles_db": pct,
        "frequency_ghz":       round(float(region_cfg["frequency_hz"]) / 1e9, 3),
    }
    bs_list = ([[float(la), float(lo)] for la, lo in bs]
               if bs is not None else [])
    return {
        "coverage_png":    coverage_png,
        "overlay_png":     overlay_png,
        "legend_png":      legend_png,
        "tiling_png":      tiling_png,
        "cdf_png":         cdf_png,
        "bounds_json":     json.dumps([south, west, north, east]),
        "base_stations_json": json.dumps(bs_list),
        "kpis_json":       json.dumps(kpis),
        "compute_seconds": time.time() - t0,
        "is_demo":         False,
    }


def _load_and_mosaic(rm_out: str, tiling_npz: str):
    """Load per-tile radio-map arrays from `rm_out` and stitch into one grid.

    The NVlabs output layout is loaded defensively: any .npz/.npy under the
    output dir that exposes a 2-D array is accepted. Raises if nothing loads,
    so the caller can fall back to the demo path.
    """
    import glob

    tiles = []
    for path in sorted(glob.glob(os.path.join(rm_out, "**", "*.np[yz]"), recursive=True)):
        try:
            obj = np.load(path, allow_pickle=True)
            if isinstance(obj, np.lib.npyio.NpzFile):
                arr = next((obj[k] for k in obj.files
                            if getattr(obj[k], "ndim", 0) == 2), None)
            else:
                arr = obj if getattr(obj, "ndim", 0) == 2 else None
            if arr is not None:
                tiles.append(np.asarray(arr, dtype=float))
        except Exception as e:  # noqa: BLE001
            print(f"skip {path}: {e}")

    if not tiles:
        raise RuntimeError(f"No radio-map tiles loaded from {rm_out}")

    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    stack = np.full((len(tiles), h, w), np.nan)
    for i, t in enumerate(tiles):
        stack[i, : t.shape[0], : t.shape[1]] = t
    mosaic = np.nanmax(stack, axis=0)
    mosaic = np.where(np.isfinite(mosaic), mosaic, DEFAULT_RM_DB_VMIN)

    bs = None
    try:
        tz = np.load(tiling_npz, allow_pickle=True)
        for k in tz.files:
            v = tz[k]
            if getattr(v, "ndim", 0) == 2 and v.shape[1] >= 2:
                bs = v[:, :2]
                break
    except Exception:  # noqa: BLE001
        pass
    return mosaic, bs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_large_scale(region_cfg: dict, allow_demo: bool = True) -> dict[str, Any]:
    """Compute a large-scale radio map for `region_cfg`.

    Attempts the real NVlabs pipeline when its repo is available (env var
    `SIONNA_LRM_REPO`, default `/tmp/sionna-large-radio-maps`). Falls back to
    the synthetic demo when the repo/deps are missing or the run fails and
    `allow_demo` is True.
    """
    repo_dir = os.environ.get("SIONNA_LRM_REPO", "/tmp/sionna-large-radio-maps")
    can_run_real = os.path.isdir(os.path.join(repo_dir, "scripts"))

    if can_run_real:
        try:
            return _run_real(region_cfg, repo_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Real sionna_lrm pipeline failed: {e}")
            traceback.print_exc()
            if not allow_demo:
                raise

    if not allow_demo:
        raise RuntimeError(
            f"sionna_lrm repo not found at {repo_dir} and demo fallback disabled."
        )
    return _demo(region_cfg)
