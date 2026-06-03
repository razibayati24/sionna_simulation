"""Tile a neighborhood into render-sized squares and group tiles into time-boxed batches.

Why tile? A radio-map solve is roughly ``O(num_tx × samples_per_tx × max_depth)`` and the
output tensor is ``area / cell_size²`` cells **per TX**. Downtown's ~125 towers over a
2.0 × 1.0 km box at ``cell_size=5`` would be ~80 k cells × 125 TX — borderline OOM on a
24 GB A10G and slow. Splitting into ~700–1000 m tiles keeps each solve to ~10–30 towers
over a small map, which is the unit we render and cache.

Tiles carry an **overlap margin** so a tower just outside a tile still illuminates its edge
(coverage doesn't get clipped at tile seams). A tower is *assigned* to the tile whose core
(un-margined) cell contains it, but each tile's render *includes* every tower within
``margin_m`` of its bounds.

Batches group tiles so one job runs ~20–30 min. ``TILES_PER_BATCH`` is calibrated by the
setup notebook: it times one representative tile, then picks the batch size that lands in
the target window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import neighborhoods as nb


@dataclass(frozen=True)
class Tile:
    tile_id: str
    # Local-meter core bounds (tower assignment) — render adds the margin.
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    margin_m: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_lo + self.x_hi) / 2.0, (self.y_lo + self.y_hi) / 2.0)

    @property
    def render_bounds(self) -> Tuple[float, float, float, float]:
        """Margin-expanded bounds used to (a) include edge towers, (b) frame the camera."""
        return (
            self.x_lo - self.margin_m, self.x_hi + self.margin_m,
            self.y_lo - self.margin_m, self.y_hi + self.margin_m,
        )

    def core_contains(self, x: float, y: float) -> bool:
        return self.x_lo <= x < self.x_hi and self.y_lo <= y < self.y_hi

    def render_contains(self, x: float, y: float) -> bool:
        bx_lo, bx_hi, by_lo, by_hi = self.render_bounds
        return bx_lo <= x <= bx_hi and by_lo <= y <= by_hi


def make_tiles(neighborhood: str, tile_size_m: float = 800.0, margin_m: float = 150.0) -> List[Tile]:
    """Cut a neighborhood's local-ENU extent into a grid of ``tile_size_m`` squares.

    The grid is centered on the neighborhood origin (0, 0), matching ``towers.load_towers``.
    """
    hood = nb.get(neighborhood)
    ew, ns = hood.extent_m
    nx = max(1, math.ceil(ew / tile_size_m))
    ny = max(1, math.ceil(ns / tile_size_m))
    # Center the grid about (0,0).
    x0 = -nx * tile_size_m / 2.0
    y0 = -ny * tile_size_m / 2.0
    tiles: List[Tile] = []
    for j in range(ny):
        for i in range(nx):
            tiles.append(Tile(
                tile_id=f"{neighborhood.replace(' ', '_')}_{i}_{j}",
                x_lo=x0 + i * tile_size_m,
                x_hi=x0 + (i + 1) * tile_size_m,
                y_lo=y0 + j * tile_size_m,
                y_hi=y0 + (j + 1) * tile_size_m,
                margin_m=margin_m,
            ))
    return tiles


def assign_towers(tiles: List[Tile], cells: List[Dict[str, Any]], tower_cap: int = 30
                  ) -> List[Tuple[Tile, List[Dict[str, Any]]]]:
    """Return (tile, towers-to-render) for tiles that actually contain ≥1 core tower.

    A tile renders every tower within its margin, but only tiles that *own* (core-contain)
    at least one tower are emitted — empty grid squares are skipped. If a tile's render set
    exceeds ``tower_cap``, keep the tallest / highest-power ones (the dominant emitters).
    """
    out: List[Tuple[Tile, List[Dict[str, Any]]]] = []
    for t in tiles:
        owns = [c for c in cells if t.core_contains(c["x"], c["y"])]
        if not owns:
            continue
        render_set = [c for c in cells if t.render_contains(c["x"], c["y"])]
        if len(render_set) > tower_cap:
            render_set = sorted(
                render_set, key=lambda c: (c.get("z", 0.0), c.get("power_dbm", 0.0)), reverse=True
            )[:tower_cap]
        # Re-index cell_id within the tile so Sionna TX names/ids are contiguous.
        render_set = [dict(c, cell_id=k) for k, c in enumerate(
            sorted(render_set, key=lambda c: c.get("tower_id", 0)))]
        out.append((t, render_set))
    return out


def make_batches(tiled: List[Tuple[Tile, List[Dict[str, Any]]]], tiles_per_batch: int
                 ) -> List[List[Tuple[Tile, List[Dict[str, Any]]]]]:
    """Chunk assigned tiles into batches of ``tiles_per_batch`` (≈ one 20–30 min job each)."""
    tpb = max(1, int(tiles_per_batch))
    return [tiled[i:i + tpb] for i in range(0, len(tiled), tpb)]


def recommend_tiles_per_batch(seconds_per_tile: float, target_min: float = 25.0) -> int:
    """Given a calibrated per-tile wall-clock, size a batch toward ``target_min`` minutes."""
    if seconds_per_tile <= 0:
        return 1
    return max(1, round(target_min * 60.0 / seconds_per_tile))
