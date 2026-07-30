"""Region presets for the large-scale radio-map tab.

A "region" is a geographic bounding box (WGS84 lat/lon) plus the radio and
tiling parameters the NVlabs `sionna_lrm` pipeline needs. Defaults mirror
sionna_lrm/constants.py where applicable (cell sizes, RSS colour range).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class RegionConfig:
    name: str
    south: float
    west: float
    north: float
    east: float
    frequency_hz: float = 3.5e9          # 5G C-band, typical macro deployment
    tx_power_dbm: float = 49.0           # ~49 dBm EIRP macro cell
    num_base_stations: int = 24          # demo synthetic layout size
    min_cell_size_m: float = 5.0         # sionna_lrm DEFAULT_MIN_CELL_SIZE
    max_cell_size_m: float = 100.0       # sionna_lrm DEFAULT_MAX_CELL_SIZE
    samples: int = 20_000_000            # per-tile ray samples (real path)
    demo_grid: int = 320                 # demo raster resolution per side
    coverage_threshold_dbm: float = -100.0

    def to_dict(self) -> dict:
        return asdict(self)


# Seattle downtown / SLU — matches the existing Seattle RF Digital Twin work.
SEATTLE = RegionConfig(
    name="Seattle (downtown + SLU)",
    south=47.5980, west=-122.3520, north=47.6300, east=-122.3150,
)

SAN_FRANCISCO = RegionConfig(
    name="San Francisco (downtown)",
    south=37.7800, west=-122.4300, north=37.8000, east=-122.3900,
)

PARIS_ETOILE = RegionConfig(
    name="Paris (Arc de Triomphe / étoile)",
    south=48.8680, west=2.2870, north=48.8790, east=2.3050,
    frequency_hz=28e9, tx_power_dbm=44.0,
)


REGION_PRESETS: dict[str, RegionConfig] = {
    "seattle":        SEATTLE,
    "san_francisco":  SAN_FRANCISCO,
    "paris_etoile":   PARIS_ETOILE,
}

DEFAULT_REGION = SEATTLE
