"""Seattle story presets — the curated gallery for the Downtown neighborhood.

Mirrors the etoile demo's A–G "stories": each preset holds everything steady and flips one
knob so a viewer compares like-for-like. The difference is the transmitters are **real
Downtown Seattle towers** (loaded by ``towers.load_towers`` from Unity Catalog and projected
to local meters), not a synthetic 7-cell ring.

Because a Sionna radio-map solve is monochromatic + single-array (``scene.frequency`` and
``scene.tx_array`` are scene-level), each story fixes those at the scene level. A story can
also restrict which towers participate via ``tower_filter`` (e.g. the 5G story renders only
``NR`` towers). Per-tower power and antenna height still come from the random per-tower
config, so even the "same" tower set looks realistically heterogeneous.

All stories render over the **Downtown core tile** (the single densest ~800 m tile) so each
is one render = one cache row, keeping the gallery to ~7 instant-load entries. Approximation
knobs (samples 1e6, depth 3, cell_size 5 m) keep each render in the seconds–minutes range.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

# Approximation defaults — the knobs that make ~30-tower tiles tractable (vs etoile's 1e7/5/1m).
APPROX_SAMPLES_PER_TX = 10 ** 6
APPROX_MAX_DEPTH = 3
APPROX_CELL_SIZE = 5.0


@dataclass
class SeattleStory:
    """A scene-level render config for one Downtown story."""

    name: str
    story_key: str
    num_rows_tx: int = 8
    num_cols_tx: int = 2
    num_rows_rx: int = 2
    num_cols_rx: int = 2
    frequency_hz: float = 3.5e9          # 5G C-band — Downtown's dominant band
    bandwidth_hz: float = 1e8            # 100 MHz
    max_depth: int = APPROX_MAX_DEPTH
    samples_per_tx: int = APPROX_SAMPLES_PER_TX
    cell_size_x: float = APPROX_CELL_SIZE
    cell_size_y: float = APPROX_CELL_SIZE
    pattern: str = "tr38901"
    polarization: str = "V"
    num_user_samples: int = 50
    min_sinr_db: float = 3.0
    min_user_dist_m: float = 10.0
    max_user_dist_m: float = 400.0
    neighborhood: str = "Downtown"
    # Restrict participating towers to this tower_type (None = all).
    tower_filter: Optional[str] = None
    # Uniform per-cell power override (None = keep each tower's random power).
    cell_power_override_dbm: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Story S1 — Baseline coverage. All Downtown towers, C-band, 8×2 macro array.
S1_BASELINE = SeattleStory(
    "S1 · Downtown baseline — all towers @ 3.5 GHz, 8×2", "s1_baseline",
)

# Story S2 — Massive-MIMO densification. Same towers, 16×16 array.
S2_DENSIFY = SeattleStory(
    "S2 · Densification — 16×16 massive MIMO", "s2_densify",
    num_rows_tx=16, num_cols_tx=16,
)

# Story S3 — 5G NR mmWave layer. Only NR towers, 28 GHz.
S3_NR_MMWAVE = SeattleStory(
    "S3 · 5G NR mmWave layer — NR towers @ 28 GHz", "s3_nr_mmwave",
    frequency_hz=28e9, bandwidth_hz=4e8, num_rows_tx=16, num_cols_tx=16,
    tower_filter="NR",
)

# Story S4 — LTE coverage layer. Only LTE towers, 1.8 GHz.
S4_LTE_COVERAGE = SeattleStory(
    "S4 · LTE coverage layer — LTE towers @ 1.8 GHz", "s4_lte_coverage",
    frequency_hz=1.8e9, bandwidth_hz=2e7, tower_filter="LTE",
)

# Story S5 — Power optimization. All towers forced to 50 dBm (gap-fill scenario).
S5_HIGH_POWER = SeattleStory(
    "S5 · High power — all towers @ 50 dBm", "s5_high_power",
    cell_power_override_dbm=50.0,
)

# Story S6 — Ray-tracing fidelity. All towers, deeper bounces (3 → 5).
S6_FIDELITY = SeattleStory(
    "S6 · Fidelity — max_depth 5 (more multipath)", "s6_fidelity",
    max_depth=5,
)

# Story S7 — Wide bandwidth. All towers, 400 MHz mmWave-style channel.
S7_WIDEBAND = SeattleStory(
    "S7 · Wide bandwidth — 400 MHz channel", "s7_wideband",
    bandwidth_hz=4e8,
)


STORIES: dict[str, SeattleStory] = {
    s.story_key: s for s in (
        S1_BASELINE, S2_DENSIFY, S3_NR_MMWAVE, S4_LTE_COVERAGE,
        S5_HIGH_POWER, S6_FIDELITY, S7_WIDEBAND,
    )
}

# Config the on-demand neighborhood dropdown renders (one coverage pass per tile).
NEIGHBORHOOD_DEFAULT_STORY = S1_BASELINE


def apply_story_to_cells(story: SeattleStory, cells: List[dict]) -> List[dict]:
    """Filter + power-override a tower list for a story; re-index cell_id contiguously."""
    out = [c for c in cells
           if story.tower_filter is None
           or str(c.get("tower_type", "")).upper() == story.tower_filter.upper()]
    out = [dict(c) for c in out]
    if story.cell_power_override_dbm is not None:
        for c in out:
            c["power_dbm"] = float(story.cell_power_override_dbm)
    for k, c in enumerate(sorted(out, key=lambda c: c.get("tower_id", c["cell_id"]))):
        c["cell_id"] = k
    return out
