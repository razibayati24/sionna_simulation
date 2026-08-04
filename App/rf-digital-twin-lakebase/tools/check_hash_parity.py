"""Hash-parity gate — run this before deploying, and after touching anything hash-relevant.

The app looks up renders by ``config_hash``. The GPU job writes them under the same hash. If
the two sides ever disagree, every preset silently becomes a cache miss and the demo goes from
"instant" to "wait 20 minutes". This script asserts they agree, by resolving each preset the
way the app does and checking a cached render actually comes back.

Hash-relevant surface (change any of these and re-run this):
  - ``lakebase_client.compute_config_hash`` / ``_HASH_*_FIELDS`` / ``_HASH_FLOAT_DP``
  - ``scene_spec`` tile geometry (TOWER_CAP / TILE_SIZE_M / TILE_MARGIN_M) or DEFAULT_SEED
  - ``towers.randomize_config`` band/height/power tables
  - ``neighborhoods`` boxes or the projection
  - ``defaults`` story knobs

Usage:
    pip install "psycopg[binary]>=3.1.18" "databricks-sdk>=0.55.0" numpy
    DATABRICKS_CONFIG_PROFILE=fevm-cmegdemos python tools/check_hash_parity.py

Exits non-zero on any mismatch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import defaults           # noqa: E402
import lakebase_client as lb  # noqa: E402
import scene_spec         # noqa: E402

NEIGHBORHOOD = os.environ.get("PARITY_NEIGHBORHOOD", "Downtown")


def main() -> int:
    print(f"Resolving {NEIGHBORHOOD} towers (SQL warehouse) …")
    cells_all = scene_spec.load_neighborhood_towers(NEIGHBORHOOD)
    tile, tile_cells = scene_spec.core_tile(NEIGHBORHOOD, cells_all)
    print(f"  {len(cells_all)} towers; core tile {tile.tile_id} holds {len(tile_cells)}.")
    print(f"  schema={lb._PG_SCHEMA}  float_dp={lb._HASH_FLOAT_DP}\n")

    missing = []
    for key, story in defaults.STORIES.items():
        cfg, cells, _ = scene_spec.resolve(NEIGHBORHOOD, story)
        config_hash = lb.compute_config_hash(cfg, cells)
        row = lb.get_render(config_hash)
        png = len(row["scene_render_png"]) if row and row.get("scene_render_png") else 0
        if not png:
            missing.append((key, config_hash))
        print(f"{'OK  ' if png else 'MISS'} {key:16s} {config_hash[:12]}  "
              f"{len(cells):2d} towers  {png // 1024:4d} KB cached")

    print()
    if missing:
        print(f"PARITY BROKEN — {len(missing)}/{len(defaults.STORIES)} presets have no cached "
              f"render at the hash the app computes:")
        for key, h in missing:
            print(f"   {key}: {h}")
        print("\nEither the cache was never seeded for this schema, or something "
              "hash-relevant changed and the cached rows need re-keying.")
        return 1

    print(f"PARITY OK — all {len(defaults.STORIES)} presets resolve to a cached render.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
