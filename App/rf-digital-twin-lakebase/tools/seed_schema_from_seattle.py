"""Seed the lakebase-only app's own Postgres schema from the live `seattle` schema.

Copies the 13 already-rendered Downtown rows (7 stories + 6 coverage tiles) into a fresh
schema, re-keyed onto the platform-stable config_hash. The live seattle-rf-digital-twin app's
`seattle` schema is only ever READ here, never modified.

Dry-run by default; pass --apply to write.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SRC = "seattle"
DST = os.environ.get("PG_SCHEMA", "lakebase_only")

# Import with the destination schema so init_schema()/writes land in DST.
os.environ["PG_SCHEMA"] = DST
import lakebase_client as lb

APPLY = "--apply" in sys.argv

RENDER_COLS = ("scene_render_png", "sinr_map_png", "association_png",
               "sinr_cdf_png", "rss_cdf_png", "kpis_json", "compute_seconds")

# --- read source (untouched) ------------------------------------------------
with lb.connect() as conn, conn.cursor() as cur:
    cur.execute(f"SELECT * FROM {SRC}.scene_configs ORDER BY id")
    scenes = [dict(r) for r in cur.fetchall()]
    src = []
    for sc in scenes:
        cur.execute(
            f"SELECT * FROM {SRC}.cell_configs WHERE scene_config_id=%s ORDER BY cell_id",
            (sc["id"],))
        cells = [dict(r) for r in cur.fetchall()]
        cur.execute(
            f"SELECT * FROM {SRC}.cached_renders WHERE config_hash=%s", (sc["config_hash"],))
        render = cur.fetchone()
        src.append((sc, cells, dict(render) if render else None))

print(f"source schema {SRC!r}: {len(src)} scene_configs")
plan = []
for sc, cells, render in src:
    if not cells or not render:
        print(f"  SKIP id={sc['id']} ({len(cells)} cells, render={bool(render)}) {sc['name'][:40]}")
        continue
    new_hash = lb.compute_config_hash(sc, cells)
    plan.append((sc, cells, render, new_hash))

seen = {}
for sc, _, _, h in plan:
    seen.setdefault(h, []).append(sc["id"])
if any(len(v) > 1 for v in seen.values()):
    print("!! hash collision — aborting"); sys.exit(1)

print(f"\nwill copy {len(plan)} rows into schema {DST!r}:")
for sc, cells, render, h in plan:
    png = len(render["scene_render_png"]) if render.get("scene_render_png") else 0
    print(f"  {sc['config_hash'][:12]} -> {h[:12]}  ({len(cells):2d} cells, {png//1024:4d} KB)  "
          f"{sc['name'][:44]}")

if not APPLY:
    print("\nDRY RUN — pass --apply to write.")
    sys.exit(0)

# --- write destination ------------------------------------------------------
print(f"\nCreating schema {DST!r} + DDL ...")
lb.init_schema()

import neighborhoods as nb

with lb.connect() as conn:
    with conn.cursor() as cur:
        for name in nb.names():
            cur.execute(
                "INSERT INTO neighborhoods (name, status) VALUES (%s, 'NONE') "
                "ON CONFLICT (name) DO NOTHING", (name,))
    conn.commit()

for sc, cells, render, new_hash in plan:
    scene = dict(sc)
    # render_bounds is what upsert_scene_config unpacks into the bbox columns.
    scene["render_bounds"] = [scene["bbox_x_lo"], scene["bbox_x_hi"],
                              scene["bbox_y_lo"], scene["bbox_y_hi"]]
    scene_id, got = lb.upsert_scene_config(scene, cells, is_preset=sc["is_preset"])
    assert got == new_hash, f"expected {new_hash} got {got}"
    payload = {k: render.get(k) for k in RENDER_COLS}
    # kpis_json comes back from JSONB as a dict; write_render expects the JSON string
    # run_simulation produces.
    if isinstance(payload.get("kpis_json"), (dict, list)):
        payload["kpis_json"] = json.dumps(payload["kpis_json"])
    lb.write_render(new_hash, payload)
    print(f"  wrote {new_hash[:12]}  {sc['name'][:50]}")

for name in {sc["neighborhood"] for sc, _, _, _ in plan if sc["neighborhood"]}:
    lb.upsert_neighborhood(name, status="CACHED")

# --- verify -----------------------------------------------------------------
rows = lb.list_neighborhood_renders("Downtown")
print(f"\nverify: list_neighborhood_renders('Downtown') -> {len(rows)} rows")
for r in rows:
    png = len(r["scene_render_png"]) if r.get("scene_render_png") else 0
    print(f"   {r['config_hash'][:12]}  {png//1024:4d} KB  {r['name'][:50]}")

with lb.connect() as conn, conn.cursor() as cur:
    cur.execute(f"SELECT count(*) c FROM {SRC}.scene_configs")
    print(f"\nsource {SRC}.scene_configs still: {cur.fetchone()['c']} rows (untouched)")
