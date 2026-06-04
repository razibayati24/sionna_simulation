/* Chapter 0 — Sionna RF Agent. A scripted, cinematic showcase:
   alarm → bird's-eye Seattle → zoom to one tower → agent ray-traces
   candidate configs → recommends the fix. Tower coordinates are real
   (UC cell_towers); KPIs and configs are illustrative. */

const SEATTLE = { lng: -122.3321, lat: 47.6062 };

// ---- scripted scenario data (illustrative) -------------------------------
const SCN = {
  band: "n78 · 3.5 GHz",
  array: "64T64R massive MIMO",
  cur:  { power: 44, tilt: 2,  az: 120 },
  rec:  { power: 46, tilt: 6,  az: 138 },
  kpi: {
    sinr: { before: -1.2, after: 3.1,  unit: "dB",  lo: -5,   hi: 15,  better: "up"   },
    rss:  { before: -98.0, after: -94.9, unit: "dBm", lo: -110, hi: -80, better: "up"   },
    edge: { before: 22,  after: 9,    unit: "%",   lo: 0,    hi: 30,  better: "down" },
  },
  candidates: [
    { nm: "C1 down-tilt 2→6°",      sc: 71 },
    { nm: "C2 azimuth 120→138°",    sc: 78 },
    { nm: "C3 power 44→46 dBm",     sc: 64 },
    { nm: "C4 combined (tilt+az+P)", sc: 94, win: true },
  ],
};

let map, towers = [], hero = null, heroMarker = null;
let running = false, aborter = 0;

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => { const id = setTimeout(r, ms); window.__t = window.__t || []; window.__t.push(id); });

// --------------------------------------------------------------------------
// Geo helpers (spherical, meters)
// --------------------------------------------------------------------------
const R = 6378137;
function dest(lat, lon, bearingDeg, distM) {
  const br = (bearingDeg * Math.PI) / 180, dr = distM / R;
  const la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180;
  const la2 = Math.asin(Math.sin(la) * Math.cos(dr) + Math.cos(la) * Math.sin(dr) * Math.cos(br));
  const lo2 = lo + Math.atan2(Math.sin(br) * Math.sin(dr) * Math.cos(la), Math.cos(dr) - Math.sin(la) * Math.sin(la2));
  return [(lo2 * 180) / Math.PI, (la2 * 180) / Math.PI];
}
function circle(lat, lon, radM, steps = 64) {
  const ring = [];
  for (let i = 0; i <= steps; i++) ring.push(dest(lat, lon, (i / steps) * 360, radM));
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [ring] } };
}
function sector(lat, lon, bearing, beam, radM, steps = 28) {
  const ring = [[lon, lat]];
  for (let i = 0; i <= steps; i++) ring.push(dest(lat, lon, bearing - beam / 2 + (i / steps) * beam, radM));
  ring.push([lon, lat]);
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [ring] } };
}
function rect(lat, lon, bearing, distM, w, d) {
  const c = dest(lat, lon, bearing, distM); // building centroid
  const [clng, clat] = c;
  const corners = [
    dest(clat, clng, bearing - 90, w / 2),
    dest(clat, clng, bearing + 90, w / 2),
  ];
  const p1 = dest(corners[0][1], corners[0][0], bearing, d / 2);
  const p2 = dest(corners[1][1], corners[1][0], bearing, d / 2);
  const p3 = dest(corners[1][1], corners[1][0], bearing + 180, d / 2);
  const p4 = dest(corners[0][1], corners[0][0], bearing + 180, d / 2);
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [[p1, p2, p3, p4, p1]] } };
}

// --------------------------------------------------------------------------
// Map init
// --------------------------------------------------------------------------
function initMap() {
  map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      sources: {
        carto: {
          type: "raster",
          tiles: [
            "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
            "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
            "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
          ],
          tileSize: 256,
          attribution: "© CARTO © OpenStreetMap contributors",
        },
      },
      layers: [{ id: "carto", type: "raster", source: "carto" }],
    },
    center: [SEATTLE.lng, SEATTLE.lat],
    zoom: 10.4,
    pitch: 0,
    bearing: 0,
    attributionControl: true,
  });
  map.on("load", onMapLoad);
}

async function onMapLoad() {
  const data = await fetch("towers.json").then((r) => r.json());
  towers = data.towers;
  hero = towers.find((t) => t.id === data.hero_id) || towers[0];
  $("#stat-towers").textContent = data.count.toLocaleString();

  // tower point source
  map.addSource("towers", {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: towers.map((t) => ({
        type: "Feature",
        properties: { type: t.type, hero: t.id === hero.id ? 1 : 0 },
        geometry: { type: "Point", coordinates: [t.lon, t.lat] },
      })),
    },
  });
  // glow + core
  map.addLayer({
    id: "tower-glow", type: "circle", source: "towers",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 3.5, 14, 9],
      "circle-blur": 1, "circle-opacity": 0.5,
      "circle-color": ["match", ["get", "type"], "NR", "#e20074", "LTE", "#22d3ee", "#6b7a99"],
    },
  });
  map.addLayer({
    id: "tower-core", type: "circle", source: "towers",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 1.4, 14, 3.2],
      "circle-color": ["match", ["get", "type"], "NR", "#ff7ec1", "LTE", "#9ff3ff", "#aab6cc"],
      "circle-opacity": 0.9,
    },
  });

  // empty geometry layers we animate later (coverage, sector, building)
  map.addSource("cover", { type: "geojson", data: empty() });
  map.addLayer({ id: "cover-fill", type: "fill", source: "cover",
    paint: { "fill-color": "#22d3ee", "fill-opacity": 0 } });
  map.addLayer({ id: "cover-line", type: "line", source: "cover",
    paint: { "line-color": "#22d3ee", "line-width": 1.4, "line-opacity": 0, "line-dasharray": [2, 2] } });

  map.addSource("sector", { type: "geojson", data: empty() });
  map.addLayer({ id: "sector-fill", type: "fill", source: "sector",
    paint: { "fill-color": "#e20074", "fill-opacity": 0 } });

  map.addSource("bldg", { type: "geojson", data: empty() });
  map.addLayer({ id: "bldg-fill", type: "fill", source: "bldg",
    paint: { "fill-color": "#ffb020", "fill-opacity": 0 } });
  map.addLayer({ id: "bldg-line", type: "line", source: "bldg",
    paint: { "line-color": "#ffd27a", "line-width": 1.2, "line-opacity": 0 } });
}
const empty = () => ({ type: "FeatureCollection", features: [] });
const setData = (id, f) => map.getSource(id) && map.getSource(id).setData(f.type ? (f.features ? f : { type: "FeatureCollection", features: [f] }) : f);
const paint = (layer, prop, val, ms = 600) => { try { map.setPaintProperty(layer, prop, val); } catch (e) {} };

// --------------------------------------------------------------------------
// Console logging
// --------------------------------------------------------------------------
const logEl = () => $("#con-log");
function line(html, hold = 280) {
  const d = document.createElement("div");
  d.className = "line";
  d.innerHTML = html;
  logEl().appendChild(d);
  logEl().scrollTop = logEl().scrollHeight;
  return sleep(hold);
}
function ts() { const n = 7400 + Math.floor(performance.now() % 600); return `<span class="t">[t+${(n/1000).toFixed(2)}s]</span> `; }
async function type(prefix, text, hold = 420) {
  const d = document.createElement("div");
  d.className = "line cur"; d.innerHTML = prefix;
  logEl().appendChild(d);
  for (let i = 0; i < text.length; i++) {
    d.innerHTML = prefix + text.slice(0, i + 1);
    logEl().scrollTop = logEl().scrollHeight;
    await sleep(12);
  }
  d.classList.remove("cur");
  return sleep(hold);
}
const phase = (p) => { $("#con-phase").textContent = p; };

// --------------------------------------------------------------------------
// Scenario
// --------------------------------------------------------------------------
async function runScenario() {
  if (running) return;
  running = true; aborter++;
  reset();

  // ----- 1. bird's-eye establishing shot --------------------------------
  $("#chapter-name").textContent = "The network at rest";
  map.flyTo({ center: [SEATTLE.lng, SEATTLE.lat], zoom: 11, pitch: 25, bearing: -12, duration: 3200, curve: 1.3 });
  await sleep(2600);

  // ----- 2. ALARM --------------------------------------------------------
  $("#chapter-name").textContent = "Alarm";
  $("#alarm-text").innerHTML =
    "New 14-story construction at <b>5th Ave &amp; Pike St</b> — UE-reported RSRP " +
    "dropped on Tower&nbsp;#978 (5G&nbsp;NR). 340+ subscribers reporting poor data experience.";
  $("#alarm").classList.remove("hidden");
  requestAnimationFrame(() => $("#alarm").classList.add("show"));
  pulseHero();
  await sleep(3200);

  // ----- 3. ZOOM to the tower -------------------------------------------
  $("#chapter-name").textContent = "Zoom to Tower #978";
  map.flyTo({ center: [hero.lon, hero.lat], zoom: 15.6, pitch: 55, bearing: -22, duration: 4200, curve: 1.5 });
  await sleep(2600);

  // coverage ring fades in
  setData("cover", circle(hero.lat, hero.lon, 520));
  paint("cover-fill", "fill-opacity", 0.06);
  paint("cover-line", "line-opacity", 0.7);
  // current serving sector (azimuth 120)
  setData("sector", sector(hero.lat, hero.lon, SCN.cur.az, 65, 480));
  paint("sector-fill", "fill-opacity", 0.18);
  await sleep(1400);

  // the new building drops in, casting a shadow on the sector
  setData("bldg", rect(hero.lat, hero.lon, 125, 165, 70, 70));
  paint("bldg-fill", "fill-opacity", 0.7);
  paint("bldg-line", "line-opacity", 0.9);
  await sleep(1600);

  // ----- 4. AGENT CONSOLE ----------------------------------------------
  $("#chapter-name").textContent = "Agent on the case";
  $("#console").classList.remove("hidden");
  requestAnimationFrame(() => $("#console").classList.add("show"));

  phase("ingest");
  await line(ts() + '<span class="crit">⚠ ALARM</span> coverage_anomaly · cell <span class="key">NR-978</span> · sector A');
  await line(ts() + 'trigger: <span class="warn">obstruction_delta</span> + ue_experience &lt; SLA');
  await type(ts(), "correlating UE telemetry with 3D scene geometry…", 500);
  await line(ts() + 'new obstacle: <span class="warn">building</span> h≈48&nbsp;m, d≈165&nbsp;m, az≈125° → <span class="crit">NLOS shadow on sector A</span>');

  phase("read state");
  await line(ts() + 'current config → power <span class="key">' + SCN.cur.power + ' dBm</span> · tilt <span class="key">' + SCN.cur.tilt + '°</span> · az <span class="key">' + SCN.cur.az + '°</span> · ' + SCN.band);
  await line(ts() + 'live KPI → SINR p10 <span class="crit">' + SCN.kpi.sinr.before + ' dB</span> · RSS p50 <span class="crit">' + SCN.kpi.rss.before + ' dBm</span> · edge users <span class="crit">' + SCN.kpi.edge.before + '%</span>');

  phase("simulate");
  await type(ts(), "launching Sionna RT ray-tracing on Databricks (GPU)…", 600);
  await runCandidates();
  await line(ts() + '<span class="ok">✓</span> winner: <span class="mag">C4</span> — combined tilt + azimuth + power');

  phase("recommend");
  await line(ts() + 'rendering recommendation…', 350);

  // re-point sector to azimuth 138 (the fix), animate
  setData("sector", sector(hero.lat, hero.lon, SCN.rec.az, 65, 520));
  paint("sector-fill", "fill-color", "#34e89e");
  paint("sector-fill", "fill-opacity", 0.22);

  showReco();
  $("#replay").classList.remove("hidden");
  running = false;
}

async function runCandidates() {
  const host = document.createElement("div");
  logEl().appendChild(host);
  const rows = SCN.candidates.map((c) => {
    const el = document.createElement("div");
    el.className = "cand" + (c.win ? " win" : "");
    el.innerHTML = `<span class="nm">${c.nm}</span><span class="track"><i></i></span><span class="sc">·</span>`;
    host.appendChild(el);
    return { c, el };
  });
  logEl().scrollTop = logEl().scrollHeight;
  for (const { c, el } of rows) {
    await sleep(550);
    el.querySelector("i").style.width = c.sc + "%";
    let n = 0;
    const sc = el.querySelector(".sc");
    const iv = setInterval(() => { n += Math.ceil(c.sc / 14); if (n >= c.sc) { n = c.sc; clearInterval(iv); } sc.textContent = n; }, 55);
    await sleep(650);
  }
  await sleep(400);
}

// --------------------------------------------------------------------------
// Recommendation card
// --------------------------------------------------------------------------
function diffLine(label, oldv, newv, unit) {
  return `<div class="dline"><span class="dk">${label}</span>
    <span class="dv old">${oldv}${unit}</span><span class="ar">→</span>
    <span class="dv new">${newv}${unit}</span></div>`;
}
function barPct(k) {
  const m = SCN.kpi[k];
  const p = (v) => Math.max(4, Math.min(100, ((v - m.lo) / (m.hi - m.lo)) * 100));
  return m.better === "down" ? { before: 100 - p(m.before), after: 100 - p(m.after) } : { before: p(m.before), after: p(m.after) };
}
function showReco() {
  $("#reco-site").textContent = "NR-978 · Downtown Seattle";
  $("#reco-diff").innerHTML =
    diffLine("Electrical tilt", SCN.cur.tilt, SCN.rec.tilt, "°") +
    diffLine("Azimuth", SCN.cur.az, SCN.rec.az, "°") +
    diffLine("Tx power", SCN.cur.power, SCN.rec.power, " dBm");

  const fmt = (v, u) => (v > 0 && u !== "dBm" ? "+" : "") + v + u;
  setKpi("sinr", "SINR p10", SCN.kpi.sinr, "+4.3 dB");
  setKpi("rss", "RSS p50", SCN.kpi.rss, "+3.1 dB");
  setKpi("edge", "Edge users <0 dB", SCN.kpi.edge, "−13 pp");

  $("#reco-why").innerHTML =
    "The new building threw a <b>diffraction shadow</b> across sector A. Ray-tracing the actual " +
    "geometry shows a 6° down-tilt cuts overshoot interference, re-pointing azimuth to <b>138°</b> " +
    "fills the street-canyon gap behind the obstruction, and <b>+2 dB</b> offsets penetration loss — " +
    "<b>projected to restore</b> edge reliability without a new site. Recommended for engineer review.";

  $("#reco").classList.remove("hidden");
  requestAnimationFrame(() => $("#reco").classList.add("show"));

  // animate bars after the card is visible
  setTimeout(() => {
    ["sinr", "rss", "edge"].forEach((k) => {
      const card = document.querySelector(`.kpi[data-k="${k}"] .kpi-bar i`);
      const pct = barPct(k);
      card.style.width = pct.before + "%";
      setTimeout(() => (card.style.width = pct.after + "%"), 650);
    });
  }, 500);
}
function setKpi(k, _label, m, delta) {
  const c = document.querySelector(`.kpi[data-k="${k}"]`);
  c.querySelector(".before").textContent = m.before + (m.unit === "%" ? "%" : " " + m.unit);
  c.querySelector(".after").textContent = m.after + (m.unit === "%" ? "%" : " " + m.unit);
  c.querySelector(".delta").textContent = delta;
}

// --------------------------------------------------------------------------
// Hero pulse marker
// --------------------------------------------------------------------------
function pulseHero() {
  if (heroMarker) return;
  const el = document.createElement("div");
  el.className = "hero-pulse";
  el.innerHTML = '<span class="ring"></span><span class="ring r2"></span><span class="dot"></span><span class="hero-label">⚠ NR-978 · sector A</span>';
  heroMarker = new maplibregl.Marker({ element: el, anchor: "center" })
    .setLngLat([hero.lon, hero.lat]).addTo(map);
}

// --------------------------------------------------------------------------
// Apply + reset/replay
// --------------------------------------------------------------------------
function applyConfig() {
  const b = $("#apply-btn");
  b.textContent = "✓ Sent to RF engineer · awaiting review";
  b.classList.add("done");
  paint("sector-fill", "fill-opacity", 0.26);
  if (heroMarker) { const l = heroMarker.getElement().querySelector(".hero-label"); if (l) { l.textContent = "NR-978 · change proposed"; l.style.background = "rgba(52,232,158,.92)"; } }
  $("#chapter-name").textContent = "Handed to RF engineer";
}

function reset() {
  ["#alarm", "#console", "#reco"].forEach((s) => { $(s).classList.remove("show"); $(s).classList.add("hidden"); });
  $("#replay").classList.add("hidden");
  logEl().innerHTML = "";
  const b = $("#apply-btn"); b.textContent = "✉ Send recommendation to RF engineer"; b.classList.remove("done");
  if (heroMarker) { heroMarker.remove(); heroMarker = null; }
  ["cover-fill", "cover-line", "sector-fill", "bldg-fill", "bldg-line"].forEach((l) => {
    const p = l.endsWith("line") ? "line-opacity" : "fill-opacity"; paint(l, p, 0);
  });
  paint("sector-fill", "fill-color", "#e20074");
  ["cover", "sector", "bldg"].forEach((s) => setData(s, empty()));
}

// --------------------------------------------------------------------------
// Wire up
// --------------------------------------------------------------------------
initMap();
$("#start-btn").addEventListener("click", () => {
  const tc = $("#titlecard");
  tc.style.opacity = "0";
  setTimeout(() => tc.classList.add("hidden"), 800);
  runScenario();
});
$("#apply-btn").addEventListener("click", applyConfig);
$("#replay").addEventListener("click", () => { if (!running) runScenario(); });
