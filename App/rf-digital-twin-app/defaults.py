"""Default scene configurations and 7-cell layout for the etoile (Arc de Triomphe) scene.

The cell positions below approximate macro cells placed along the radial avenues that
fan out from the Arc. They are reasonable starting values for the Sionna `etoile`
scene and match the 7-cell network described in the Medium article. Customers can
edit them freely from the app UI.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List


# ---------------------------------------------------------------------------
# Per-cell layout (positions in scene coordinates, meters)
# ---------------------------------------------------------------------------

DEFAULT_CELLS: List[dict] = [
    {"cell_id": 0, "name": "tx0", "x":  -150.0, "y":     0.0, "z": 25.0,
     "look_at_x":  -300.0, "look_at_y":     0.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 1, "name": "tx1", "x":  -100.0, "y":   100.0, "z": 25.0,
     "look_at_x":  -200.0, "look_at_y":   200.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 2, "name": "tx2", "x":     0.0, "y":   150.0, "z": 25.0,
     "look_at_x":     0.0, "look_at_y":   300.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 3, "name": "tx3", "x":   100.0, "y":   100.0, "z": 25.0,
     "look_at_x":   200.0, "look_at_y":   200.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 4, "name": "tx4", "x":   150.0, "y":     0.0, "z": 25.0,
     "look_at_x":   300.0, "look_at_y":     0.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 5, "name": "tx5", "x":   100.0, "y":  -100.0, "z": 25.0,
     "look_at_x":   200.0, "look_at_y":  -200.0, "look_at_z": 0.0, "power_dbm": 44.0},
    {"cell_id": 6, "name": "tx6", "x":  -100.0, "y":  -100.0, "z": 25.0,
     "look_at_x":  -200.0, "look_at_y":  -200.0, "look_at_z": 0.0, "power_dbm": 44.0},
]


# ---------------------------------------------------------------------------
# Scene-level config
# ---------------------------------------------------------------------------

@dataclass
class SceneConfig:
    """Everything the app exposes for editing at the scene level."""

    name: str
    num_rows_tx: int
    num_cols_tx: int
    num_rows_rx: int
    num_cols_rx: int
    frequency_hz: float = 28e9          # 28 GHz, per the article
    bandwidth_hz: float = 1e8           # 100 MHz
    max_depth: int = 5
    samples_per_tx: int = 10 ** 7
    cell_size_x: float = 1.0
    cell_size_y: float = 1.0
    pattern: str = "tr38901"
    polarization: str = "V"
    num_user_samples: int = 50
    min_sinr_db: float = 3.0
    min_user_dist_m: float = 10.0
    max_user_dist_m: float = 200.0

    def to_dict(self) -> dict:
        return asdict(self)


# Config 1: 8x2 UPA TX, 2x2 UPA RX  (baseline)
CONFIG_1 = SceneConfig(
    name="Config 1 (8x2 TX / 2x2 RX)",
    num_rows_tx=8,
    num_cols_tx=2,
    num_rows_rx=2,
    num_cols_rx=2,
)

# Config 2: 16x16 UPA TX, 2x2 UPA RX  (the article upgrade scenario)
CONFIG_2 = SceneConfig(
    name="Config 2 (16x16 TX / 2x2 RX)",
    num_rows_tx=16,
    num_cols_tx=16,
    num_rows_rx=2,
    num_cols_rx=2,
)


PRESETS: dict[str, SceneConfig] = {
    "config_1": CONFIG_1,
    "config_2": CONFIG_2,
}


def preset_cells() -> List[dict]:
    """Return a deep-ish copy of the default 7-cell layout."""
    return [dict(c) for c in DEFAULT_CELLS]
