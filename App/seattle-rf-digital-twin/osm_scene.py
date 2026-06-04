"""Headless OpenStreetMap → Sionna RT scene builder (no Blender).

Sionna RT ray-traces a Mitsuba 3 scene. To make the Seattle renders a *faithful* digital
twin we build that scene from real OSM building footprints instead of reusing the built-in
Paris ``etoile`` mesh:

  1. Convert a tile's local-meter render bounds back to a lat/lon bbox.
  2. Query the OSM **Overpass API** for ``building`` ways in that bbox.
  3. Project each footprint to the neighborhood's local-ENU meters (same projection as the
     towers), extrude to a prism using the building height (``height`` tag, else
     ``building:levels`` × 3 m, else a 10 m default).
  4. Emit a ground plane + the extruded buildings as PLY meshes and a Mitsuba scene XML
     that tags them with **ITU radio materials** (buildings → ``itu_concrete``,
     ground → ``itu_medium_dry_ground``). Sionna maps a BSDF whose id is ``mat-<name>``
     onto the ``RadioMaterial`` ``<name>``.

Everything degrades gracefully: if Overpass is unreachable, returns nothing, or any of the
optional geometry deps are missing, we still emit a **flat ground-plane** scene so the
render never hard-fails — the tile just loses its buildings. Generated scenes are cached on
disk keyed by the bbox so re-renders skip the network round-trip.

Optional deps (installed by the setup notebook / render job): ``requests``, ``shapely``,
``trimesh`` and a triangulation backend (``mapbox_earcut``) — without the latter,
``trimesh.creation.extrude_polygon`` raises "No available triangulation engine!" and every
building silently drops to the flat-ground fallback. Sionna's ``load_scene`` consumes the
XML this module writes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import List, Optional, Tuple

import neighborhoods as nb

OVERPASS_URL = os.environ.get("OVERPASS_URL", "https://overpass-api.de/api/interpreter")

# ITU material → diffuse reflectance RGB (visual only; RF behaviour comes from the named
# RadioMaterial Sionna attaches to the ``mat-<name>`` BSDF).
_BUILDING_MAT = "itu_concrete"          # valid 1–100 GHz
_GROUND_MAT = "itu_medium_dry_ground"   # ITU ground model only valid ~1–10 GHz
# At mmWave the ITU ground model is undefined, so use concrete (1–100 GHz) as a pragmatic
# ground stand-in above this cutoff.
_GROUND_MMWAVE_CUTOFF_HZ = 10e9
_MAT_RGB = {
    "itu_concrete": (0.55, 0.55, 0.57),
    "itu_medium_dry_ground": (0.30, 0.32, 0.28),
}


def _ground_material(frequency_hz: Optional[float]) -> str:
    """Ground RadioMaterial valid at the render frequency (ITU ground model caps ~10 GHz)."""
    if frequency_hz and float(frequency_hz) > _GROUND_MMWAVE_CUTOFF_HZ:
        return _BUILDING_MAT
    return _GROUND_MAT

_DEFAULT_BUILDING_H = 10.0


# ---------------------------------------------------------------------------
# Geographic <-> local helpers
# ---------------------------------------------------------------------------

def _local_to_lonlat(x: float, y: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    """Inverse of neighborhoods.project_lonlat."""
    lat0, lon0 = origin
    lon = lon0 + x / (111_320.0 * math.cos(math.radians(lat0)))
    lat = lat0 + y / 110_540.0
    return (lat, lon)


def _bbox_latlon(render_bounds: Tuple[float, float, float, float],
                 origin: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """(x_lo,x_hi,y_lo,y_hi) local → (south, west, north, east) lat/lon for Overpass."""
    x_lo, x_hi, y_lo, y_hi = render_bounds
    south, west = _local_to_lonlat(x_lo, y_lo, origin)
    north, east = _local_to_lonlat(x_hi, y_hi, origin)
    return (south, west, north, east)


# ---------------------------------------------------------------------------
# Overpass fetch + parse
# ---------------------------------------------------------------------------

def _building_height(tags: dict) -> float:
    h = tags.get("height")
    if h:
        try:
            return float(str(h).split()[0].replace("m", ""))
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return max(3.0, float(levels) * 3.0)
        except ValueError:
            pass
    return _DEFAULT_BUILDING_H


def fetch_buildings(render_bounds, origin, timeout: float = 60.0
                    ) -> List[Tuple[list, float]]:
    """Return [(polygon_local_xy, height_m)] for OSM buildings in the tile bbox.

    Empty list on any failure (caller falls back to ground-only).
    """
    try:
        import requests  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - dep missing
        print(f"[osm_scene] requests unavailable ({e}); skipping buildings.")
        return []

    south, west, north, east = _bbox_latlon(render_bounds, origin)
    query = (
        f"[out:json][timeout:{int(timeout)}];"
        f'(way["building"]({south},{west},{north},{east}););'
        f"out body geom;"
    )
    # Overpass returns 406 Not Acceptable without a User-Agent. A couple of mirrors are
    # tried with simple backoff so a single slow/rate-limited endpoint doesn't drop the
    # tile's buildings.
    headers = {"User-Agent": "seattle-rf-digital-twin/1.0 (Databricks Sionna demo)"}
    endpoints = [OVERPASS_URL]
    for extra in ("https://overpass-api.de/api/interpreter",
                  "https://overpass.kumi.systems/api/interpreter"):
        if extra not in endpoints:
            endpoints.append(extra)
    elements = None
    for attempt, url in enumerate(endpoints):
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=timeout + 10)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break
        except Exception as e:
            print(f"[osm_scene] Overpass fetch via {url} failed ({e}); trying next endpoint.")
            time.sleep(1.5 * (attempt + 1))
    if elements is None:
        print("[osm_scene] all Overpass endpoints failed; falling back to ground-only.")
        return []

    polys: List[Tuple[list, float]] = []
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 4:
            continue
        ring = [nb.project_lonlat(p["lat"], p["lon"], origin) for p in geom]
        polys.append((ring, _building_height(el.get("tags", {}))))
    print(f"[osm_scene] fetched {len(polys)} OSM buildings.")
    return polys


# ---------------------------------------------------------------------------
# Mesh + Mitsuba XML emission
# ---------------------------------------------------------------------------

def _extrude_buildings(polys: List[Tuple[list, float]]):
    """Concatenate extruded footprints into a single trimesh mesh (or None)."""
    if not polys:
        return None
    try:
        import numpy as np  # noqa: PLC0415
        import trimesh  # noqa: PLC0415
        from shapely.geometry import Polygon  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"[osm_scene] geometry deps unavailable ({e}); skipping buildings.")
        return None

    meshes = []
    for ring, height in polys:
        try:
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1.0:  # drop slivers < 1 m²
                continue
            m = trimesh.creation.extrude_polygon(poly, height=float(height))
            meshes.append(m)
        except Exception:
            continue
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _ground_mesh(render_bounds, pad: float = 50.0):
    """A flat ground quad (two triangles) covering the render bounds + pad."""
    import numpy as np  # noqa: PLC0415
    import trimesh  # noqa: PLC0415

    x_lo, x_hi, y_lo, y_hi = render_bounds
    x_lo, y_lo, x_hi, y_hi = x_lo - pad, y_lo - pad, x_hi + pad, y_hi + pad
    verts = np.array([[x_lo, y_lo, 0.0], [x_hi, y_lo, 0.0],
                      [x_hi, y_hi, 0.0], [x_lo, y_hi, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _xml(shapes: List[Tuple[str, str]]) -> str:
    """Build a Mitsuba 3 scene XML. ``shapes`` = [(ply_filename, material_name)]."""
    bsdfs = "\n".join(
        f'  <bsdf type="twosided" id="mat-{name}">\n'
        f'    <bsdf type="diffuse"><rgb value="{r} {g} {b}" name="reflectance"/></bsdf>\n'
        f'  </bsdf>'
        for name, (r, g, b) in _MAT_RGB.items()
    )
    shape_xml = "\n".join(
        f'  <shape type="ply" id="{os.path.splitext(os.path.basename(fn))[0]}">\n'
        f'    <string name="filename" value="meshes/{os.path.basename(fn)}"/>\n'
        f'    <ref id="mat-{mat}"/>\n'
        f'  </shape>'
        for fn, mat in shapes
    )
    return (
        '<scene version="3.0.0">\n'
        '  <integrator type="path"/>\n'
        f'{bsdfs}\n'
        f'{shape_xml}\n'
        '</scene>\n'
    )


def build_tile_scene_xml(render_bounds, origin, out_dir: str,
                         frequency_hz: Optional[float] = None, cache: bool = True) -> str:
    """Build (or reuse cached) Mitsuba XML for a tile; return the XML path.

    Always succeeds: on any building-fetch/geometry failure it still writes a ground-only
    scene. The output dir layout is ``<out_dir>/<bbox_hash>/{scene.xml, meshes/*.ply}``.
    The cache key folds in the ground-material band so a tile reused across sub-10 GHz and
    mmWave stories gets the right (frequency-valid) ground material.
    """
    import trimesh  # noqa: PLC0415  (always needed for ground + ply export)

    ground_mat = _ground_material(frequency_hz)
    key = hashlib.sha1(
        json.dumps([render_bounds, origin, ground_mat], sort_keys=True).encode()
    ).hexdigest()[:16]
    scene_dir = os.path.join(out_dir, key)
    mesh_dir = os.path.join(scene_dir, "meshes")
    xml_path = os.path.join(scene_dir, "scene.xml")
    if cache and os.path.exists(xml_path):
        print(f"[osm_scene] reusing cached scene {xml_path}")
        return xml_path
    os.makedirs(mesh_dir, exist_ok=True)

    shapes: List[Tuple[str, str]] = []
    # Ground (always present); material chosen valid for the render frequency.
    ground = _ground_mesh(render_bounds)
    ground.export(os.path.join(mesh_dir, "ground.ply"))
    shapes.append(("ground.ply", ground_mat))

    # Buildings (best-effort).
    polys = fetch_buildings(render_bounds, origin)
    buildings = _extrude_buildings(polys)
    if buildings is not None:
        buildings.export(os.path.join(mesh_dir, "buildings.ply"))
        shapes.append(("buildings.ply", _BUILDING_MAT))
        print(f"[osm_scene] scene with {len(polys)} buildings → {xml_path}")
    else:
        print(f"[osm_scene] FLAT-GROUND fallback (no buildings) → {xml_path}")

    with open(xml_path, "w") as f:
        f.write(_xml(shapes))
    return xml_path


def load_tile_scene(render_bounds, origin, out_dir: str = "/tmp/seattle_osm_scenes",
                    frequency_hz: Optional[float] = None):
    """Build the tile scene and return a loaded Sionna RT ``Scene`` with ITU materials set."""
    from sionna.rt import load_scene  # noqa: PLC0415

    xml_path = build_tile_scene_xml(render_bounds, origin, out_dir, frequency_hz=frequency_hz)
    return load_scene(xml_path)
