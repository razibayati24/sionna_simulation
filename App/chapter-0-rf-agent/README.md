# Chapter 0 — Sionna RF Agent (Seattle)

A cinematic, self-running showcase that opens the Sionna RF story: **"the network
that fixes itself."** It's the prequel to the [`rf-digital-twin-app`](../rf-digital-twin-app/README.md) —
where that app lets you drive the digital twin by hand, Chapter 0 shows *why* it matters.

## The narrative (auto-plays)

1. **The network at rest** — a bird's-eye of Seattle with every real T-Mobile site.
2. **Alarm** — a new 14-story building goes up at 5th & Pike; UEs on Tower #978 (5G NR)
   start reporting poor data experience.
3. **Zoom** — the camera flies down to the affected tower; its coverage sector and the
   new building (casting an NLOS shadow) are drawn on the map.
4. **Agent on the case** — the agent console streams its reasoning: it correlates UE
   telemetry with the 3D scene, reads the current radio config + KPIs, then ray-traces
   candidate configs with **Sionna RT on Databricks** and scores them.
5. **Recommendation** — a card animates in with the config diff (tilt / azimuth / power)
   and the *why*, with before→after SINR p10, RSS p50, and edge-user KPIs.

## What's real vs. illustrative

- **Real:** every tower location, type (LTE / NR / GSM / UMTS), and count come from
  `cmegdemos_catalog.network_analytics_enablement.cell_towers` (filtered to central
  Seattle), baked into `towers.json`. Hero site = tower #978, a downtown 5G NR cell.
- **Illustrative:** the alarm, the radio configs, the KPIs, the candidate scores, and
  the building geometry are made up to tell the story.

## Run locally

```bash
cd App/chapter-0-rf-agent
python -m http.server 8000
# open http://localhost:8000
```

Needs internet for the CARTO dark basemap tiles and the MapLibre / font CDNs.

## Deploy as a Databricks App

It's a pure static front-end, so `app.yaml` just runs a static file server. Sync the
folder to a workspace path and create an app pointing at it — no warehouse or resources
required.

## Refresh the tower data

`towers.json` was generated from the UC table:

```sql
SELECT tower_id, carrier, tower_type, coverage_radius_m,
       round(latitude,5) lat, round(longitude,5) lon
FROM cmegdemos_catalog.network_analytics_enablement.cell_towers
WHERE latitude  BETWEEN 47.49 AND 47.74
  AND longitude BETWEEN -122.46 AND -122.22;
```

Reshape into `{ hero_id, count, towers:[{id,type,r,lat,lon}] }` and overwrite the file.
