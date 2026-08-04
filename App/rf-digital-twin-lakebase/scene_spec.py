"""Scene resolution — the one code path the app and the GPU job both hash through.

A ``config_hash`` folds in the per-tower geometry and radio config
(``lakebase_client._HASH_CELL_FIELDS``) plus the neighborhood/tile identity. So for the app
to look up a render the job already wrote, both sides must resolve the **same tower list on
the same tile**. That's what this module guarantees: it owns tile geometry and tower
resolution, and both callers import it rather than each rolling their own.

It deliberately imports nothing from ``sionna_compute`` — that pulls in drjit/mitsuba, which
don't exist in an app container. Deps here are numpy-only, so ``resolve()`` runs equally well
in the Shiny app and on the GPU cluster.

Determinism, which is the whole ballgame:
  - ``towers.load_towers`` reads the tower table ordered by ``tower_id`` and derives per-tower
    frequency/height/power from a fixed seed (default 1234).
  - ``tiling.make_tiles`` / ``assign_towers`` are pure functions of the neighborhood box.
Same neighborhood + same seed ⇒ same cells ⇒ same hash, on either side.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import defaults
import neighborhoods as nb
import tiling
import towers

# Per-tile tower cap (keeps a single solve within A10G memory) and tile geometry.
# Changing any of these three re-keys every cached render.
TOWER_CAP = 30
TILE_SIZE_M = 800.0
TILE_MARGIN_M = 150.0

DEFAULT_SEED = 1234

# Towers for a (neighborhood, seed) are stable for the life of the process, and each load is
# a SQL-warehouse round trip — so cache them. Bounded by the neighborhood count (8).
_TOWER_CACHE: Dict[Tuple[str, int], List[dict]] = {}


def scene_cfg(story: defaults.SeattleStory, hood: nb.Neighborhood,
              tile: "tiling.Tile") -> Dict[str, Any]:
    """Assemble the scene config dict run_simulation + lakebase expect for one render."""
    cfg = story.to_dict()
    cfg.update(
        use_osm=True,
        origin_lat=hood.center_lat,
        origin_lon=hood.center_lon,
        render_bounds=list(tile.render_bounds),
        tile_id=tile.tile_id,
        osm_scene_dir="/tmp/seattle_osm_scenes",
    )
    return cfg


def core_tile(neighborhood: str, cells: List[dict]) -> Tuple["tiling.Tile", List[dict]]:
    """The tile owning the most towers — the gallery renders all stories here."""
    tiles = tiling.make_tiles(neighborhood, TILE_SIZE_M, TILE_MARGIN_M)
    assigned = tiling.assign_towers(tiles, cells, TOWER_CAP)
    if not assigned:
        raise RuntimeError(f"No towers found for neighborhood {neighborhood!r}")
    return max(assigned, key=lambda tc: len(tc[1]))


def load_neighborhood_towers(neighborhood: str, seed: int = DEFAULT_SEED,
                             warehouse_id: Optional[str] = None) -> List[dict]:
    """All towers in a neighborhood, memoised per (neighborhood, seed)."""
    key = (neighborhood, seed)
    if key not in _TOWER_CACHE:
        _TOWER_CACHE[key] = towers.load_towers(
            neighborhood, seed=seed, warehouse_id=warehouse_id
        )
    return _TOWER_CACHE[key]


def resolve(neighborhood: str, story: defaults.SeattleStory, seed: int = DEFAULT_SEED,
            warehouse_id: Optional[str] = None) -> Tuple[Dict[str, Any], List[dict], "tiling.Tile"]:
    """Resolve a story over a neighborhood's core tile.

    Returns ``(scene_cfg, cells, tile)`` — everything ``compute_config_hash`` and
    ``run_simulation`` need. The app calls this to hash a sidebar config; the job calls it to
    rebuild the identical cells before rendering.
    """
    hood = nb.get(neighborhood)
    cells = load_neighborhood_towers(neighborhood, seed=seed, warehouse_id=warehouse_id)
    tile, tile_cells = core_tile(neighborhood, cells)
    story_cells = defaults.apply_story_to_cells(story, tile_cells)
    if not story_cells:
        raise RuntimeError(
            f"No towers left in {neighborhood} core tile after filtering to "
            f"tower_type={story.tower_filter!r}."
        )
    return scene_cfg(story, hood, tile), story_cells, tile
