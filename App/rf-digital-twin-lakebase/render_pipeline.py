"""Neighborhood render orchestration — shared by the setup notebook and the on-demand job.

Two render modes, both writing PNG+KPI rows into Lakebase keyed by config_hash:

- ``render_stories(spark, neighborhood)`` — the curated **gallery**. Renders the ~7
  ``defaults.STORIES`` over the neighborhood's single densest "core" tile, each varying one
  knob. One render per story → ~7 instant-load cache rows.

- ``render_coverage(spark, neighborhood, batch_index, n_batches)`` — full neighborhood
  **coverage**. Tiles the neighborhood, assigns towers, and renders the requested batch of
  tiles with the baseline story. This is what the app's dropdown triggers for an uncached
  neighborhood; batches are sized (via ``calibrate``) so each job runs ~20–30 min.

``calibrate(spark, neighborhood)`` renders one tile, measures wall-clock, and recommends a
``TILES_PER_BATCH`` that lands a batch in the target window.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import neighborhoods as nb
import tiling
import towers
import defaults
import lakebase_client as lb
from sionna_compute import run_simulation

# Per-tile tower cap (keeps a single solve within A10G memory) and tile geometry.
TOWER_CAP = 30
TILE_SIZE_M = 800.0
TILE_MARGIN_M = 150.0


def _scene_cfg(story: defaults.SeattleStory, hood: nb.Neighborhood,
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


def _core_tile(neighborhood: str, cells: List[dict]) -> Tuple["tiling.Tile", List[dict]]:
    """The tile owning the most towers — the gallery renders all stories here."""
    tiles = tiling.make_tiles(neighborhood, TILE_SIZE_M, TILE_MARGIN_M)
    assigned = tiling.assign_towers(tiles, cells, TOWER_CAP)
    if not assigned:
        raise RuntimeError(f"No towers found for neighborhood {neighborhood!r}")
    return max(assigned, key=lambda tc: len(tc[1]))


def render_stories(spark, neighborhood: str = "Downtown", seed: int = 1234,
                   force: bool = False) -> List[dict]:
    """Render the curated story gallery for a neighborhood's core tile.

    ``force=True`` re-renders even if a row is already cached (e.g. to refresh renders
    after a scene-geometry fix that doesn't change the config_hash).
    """
    hood = nb.get(neighborhood)
    cells = towers.load_towers(neighborhood, spark, seed=seed)
    lb.upsert_neighborhood(neighborhood, status="RENDERING", n_towers=len(cells))
    tile, tile_cells = _core_tile(neighborhood, cells)
    print(f"[stories] {neighborhood}: {len(cells)} towers, core tile {tile.tile_id} "
          f"holds {len(tile_cells)} towers.")

    summary = []
    for story in defaults.STORIES.values():
        story_cells = defaults.apply_story_to_cells(story, tile_cells)
        if not story_cells:
            print(f"  {story.name}: no towers after filter — skipping.")
            continue
        cfg = _scene_cfg(story, hood, tile)
        _, config_hash = lb.upsert_scene_config(cfg, story_cells, is_preset=True)
        if not force and _is_cached(config_hash):
            print(f"  {story.name}: already cached ({config_hash[:12]}).")
            summary.append({"story": story.name, "hash": config_hash, "status": "skipped"})
            continue
        print(f"  {story.name}: rendering {len(story_cells)} towers …")
        try:
            results = run_simulation(cfg, story_cells)
            lb.write_render(config_hash, results)
            print(f"    done in {results['compute_seconds']:.1f}s.")
            summary.append({"story": story.name, "hash": config_hash,
                            "seconds": results["compute_seconds"], "status": "rendered"})
        except Exception as e:
            # One band/material failure shouldn't kill the whole gallery — log and continue.
            print(f"    FAILED — {type(e).__name__}: {e}")
            summary.append({"story": story.name, "hash": config_hash,
                            "status": "FAILED", "error": f"{type(e).__name__}: {e}"})

    lb.upsert_neighborhood(neighborhood, status="CACHED", n_towers=len(cells))
    return summary


def render_coverage(spark, neighborhood: str, batch_index: int = 0, n_batches: int = 1,
                    tiles_per_batch: Optional[int] = None, seed: int = 1234,
                    force: bool = False) -> List[dict]:
    """Render one batch of a neighborhood's coverage tiles (baseline story per tile)."""
    hood = nb.get(neighborhood)
    cells = towers.load_towers(neighborhood, spark, seed=seed)
    lb.upsert_neighborhood(neighborhood, status="RENDERING", n_towers=len(cells))

    tiles = tiling.make_tiles(neighborhood, TILE_SIZE_M, TILE_MARGIN_M)
    assigned = tiling.assign_towers(tiles, cells, TOWER_CAP)
    if tiles_per_batch is None:
        # Even split across n_batches when batch sizing isn't pre-calibrated.
        tiles_per_batch = max(1, -(-len(assigned) // max(1, n_batches)))
    batches = tiling.make_batches(assigned, tiles_per_batch)
    if batch_index >= len(batches):
        print(f"[coverage] batch {batch_index} ≥ {len(batches)} batches — nothing to do.")
        lb.upsert_neighborhood(neighborhood, status="CACHED", n_towers=len(cells))
        return []

    story = defaults.NEIGHBORHOOD_DEFAULT_STORY
    summary = []
    for tile, tile_cells in batches[batch_index]:
        story_cells = defaults.apply_story_to_cells(story, tile_cells)
        if not story_cells:
            continue
        cfg = _scene_cfg(story, hood, tile)
        cfg["story_key"] = f"coverage_{tile.tile_id}"
        cfg["name"] = f"Coverage · {neighborhood} · tile {tile.tile_id}"
        _, config_hash = lb.upsert_scene_config(cfg, story_cells, is_preset=False)
        if not force and _is_cached(config_hash):
            summary.append({"tile": tile.tile_id, "hash": config_hash, "status": "skipped"})
            continue
        print(f"[coverage] tile {tile.tile_id}: {len(story_cells)} towers …")
        try:
            results = run_simulation(cfg, story_cells)
            lb.write_render(config_hash, results)
            summary.append({"tile": tile.tile_id, "hash": config_hash,
                            "seconds": results["compute_seconds"], "status": "rendered"})
        except Exception as e:
            print(f"[coverage] tile {tile.tile_id} FAILED — {type(e).__name__}: {e}")
            summary.append({"tile": tile.tile_id, "hash": config_hash,
                            "status": "FAILED", "error": f"{type(e).__name__}: {e}"})

    # Mark CACHED only once the final batch lands.
    final = batch_index >= len(batches) - 1
    lb.upsert_neighborhood(neighborhood, status="CACHED" if final else "RENDERING",
                           n_towers=len(cells))
    return summary


def calibrate(spark, neighborhood: str = "Downtown", target_min: float = 25.0) -> dict:
    """Render the core tile once, measure wall-clock, recommend a batch size."""
    hood = nb.get(neighborhood)
    cells = towers.load_towers(neighborhood, spark)
    tile, tile_cells = _core_tile(neighborhood, cells)
    story = defaults.NEIGHBORHOOD_DEFAULT_STORY
    story_cells = defaults.apply_story_to_cells(story, tile_cells)
    cfg = _scene_cfg(story, hood, tile)
    cfg["story_key"] = "calibration"
    print(f"[calibrate] rendering core tile {tile.tile_id} with {len(story_cells)} towers …")
    results = run_simulation(cfg, story_cells)
    secs = results["compute_seconds"]
    tpb = tiling.recommend_tiles_per_batch(secs, target_min)
    n_tiles = len(tiling.assign_towers(tiling.make_tiles(neighborhood, TILE_SIZE_M, TILE_MARGIN_M),
                                       cells, TOWER_CAP))
    out = {"seconds_per_tile": secs, "tiles_per_batch": tpb, "n_tiles": n_tiles,
           "n_batches": max(1, -(-n_tiles // tpb))}
    print(f"[calibrate] {secs:.1f}s/tile → TILES_PER_BATCH={tpb} "
          f"({n_tiles} tiles → ~{out['n_batches']} batches of ~{target_min:.0f} min).")
    return out


def _is_cached(config_hash: str) -> bool:
    row = lb.get_render(config_hash)
    if not row:
        return False
    png = row.get("scene_render_png")
    return bool(png and len(png) > 10_000)
