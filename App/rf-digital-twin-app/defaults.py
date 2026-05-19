"""Default scene configurations and 7-cell layout for the etoile (Arc de Triomphe) scene.

The cell positions below approximate macro cells placed along the radial avenues that
fan out from the Arc. They are reasonable starting values for the Sionna `etoile`
scene and match the 7-cell network described in the Medium article. Customers can
edit them freely from the app UI.

Presets are grouped into "stories" — each story holds the other variables steady
and varies one knob so the demo viewer can compare like-for-like:

  A. Antenna densification (TX UPA size)
  B. Frequency band ladder (carrier frequency)
  C. Antenna pattern (beam shape)
  D. Polarization
  E. Power level (uniform per-cell)
  F. Bandwidth
  G. Ray tracing fidelity (max_depth)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional


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

    # When set, every cell's power_dbm is replaced by this value in
    # cells_for_preset(). Lets a preset say "all 7 cells at 38 dBm" without
    # duplicating the cell list. Hash still varies because the override
    # propagates into the per-cell power_dbm field.
    cell_power_override_dbm: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def preset_cells() -> List[dict]:
    """Return a deep-ish copy of the default 7-cell layout."""
    return [dict(c) for c in DEFAULT_CELLS]


def cells_for_preset(cfg: SceneConfig) -> List[dict]:
    """Apply preset-level overrides (e.g. uniform TX power) to the cell list."""
    cells = preset_cells()
    if cfg.cell_power_override_dbm is not None:
        for c in cells:
            c["power_dbm"] = float(cfg.cell_power_override_dbm)
    return cells


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# Story A — Antenna densification. Same defaults otherwise (28 GHz, 100 MHz,
# tr38901, V, 44 dBm, max_depth=5).
CONFIG_DENS_2X2  = SceneConfig("A · Densification — 2×2 TX (baseline)",   2,  2, 2, 2)
CONFIG_DENS_4X4  = SceneConfig("A · Densification — 4×4 TX",              4,  4, 2, 2)
CONFIG_1         = SceneConfig("A · Densification — 8×2 TX (Config 1)",   8,  2, 2, 2)
CONFIG_DENS_8X8  = SceneConfig("A · Densification — 8×8 TX",              8,  8, 2, 2)
CONFIG_2         = SceneConfig("A · Densification — 16×16 TX (Config 2)", 16, 16, 2, 2)
CONFIG_DENS_32X8 = SceneConfig("A · Densification — 32×8 TX (elongated)", 32, 8, 2, 2)

# Story B — Frequency band ladder. 8×2 TX held constant. Bandwidth scaled to
# realistic deployment for each band.
# Note: the etoile scene's ITU `marble` material is only defined for 1–100 GHz,
# so the low end of this ladder is 1.8 GHz (LTE band 3), not 700 MHz.
CONFIG_FREQ_1P8G = SceneConfig(
    "B · Frequency — 8×2 @ 1.8 GHz (LTE coverage band)",
    8, 2, 2, 2, frequency_hz=1.8e9, bandwidth_hz=2e7,
)
CONFIG_FREQ_2P6G = SceneConfig(
    "B · Frequency — 8×2 @ 2.6 GHz (LTE/NR FR1)",
    8, 2, 2, 2, frequency_hz=2.6e9, bandwidth_hz=2e7,
)
CONFIG_FREQ_3P5G = SceneConfig(
    "B · Frequency — 8×2 @ 3.5 GHz (5G C-band)",
    8, 2, 2, 2, frequency_hz=3.5e9, bandwidth_hz=1e8,
)
# (Config 1 covers 8×2 @ 28 GHz already.)
CONFIG_FREQ_39G  = SceneConfig(
    "B · Frequency — 8×2 @ 39 GHz (high mmWave / FWA)",
    8, 2, 2, 2, frequency_hz=3.9e10, bandwidth_hz=4e8,
)

# Story C — Antenna pattern. 16×16 TX held constant.
# (Config 2 covers tr38901.)
CONFIG_PAT_ISO    = SceneConfig("C · Pattern — 16×16 isotropic",  16, 16, 2, 2, pattern="iso")
CONFIG_PAT_DIPOLE = SceneConfig("C · Pattern — 16×16 dipole",     16, 16, 2, 2, pattern="dipole")

# Story D — Polarization diversity. 16×16 TX held constant.
# (Config 2 covers V.)
CONFIG_POL_VH = SceneConfig("D · Polarization — 16×16 cross (VH)", 16, 16, 2, 2, polarization="VH")

# Story E — Power optimization. 16×16 TX held constant, uniform power across cells.
CONFIG_PWR_LOW  = SceneConfig(
    "E · Power — 16×16 @ 38 dBm (low / dense urban)",
    16, 16, 2, 2, cell_power_override_dbm=38.0,
)
# (Config 2 = 44 dBm default.)
CONFIG_PWR_HIGH = SceneConfig(
    "E · Power — 16×16 @ 50 dBm (high / gap-fill)",
    16, 16, 2, 2, cell_power_override_dbm=50.0,
)

# Story F — Bandwidth. 16×16 TX held constant.
CONFIG_BW_20M  = SceneConfig("F · Bandwidth — 16×16 @ 20 MHz (LTE-like)",   16, 16, 2, 2, bandwidth_hz=2e7)
# (Config 2 = 100 MHz default.)
CONFIG_BW_400M = SceneConfig("F · Bandwidth — 16×16 @ 400 MHz (mmWave NR)", 16, 16, 2, 2, bandwidth_hz=4e8)

# Story G — Ray tracing fidelity. 16×16 TX held constant.
CONFIG_DEPTH_3 = SceneConfig("G · Fidelity — 16×16 max_depth=3 (LOS-heavy)", 16, 16, 2, 2, max_depth=3)
# (Config 2 = max_depth=5 default.)
CONFIG_DEPTH_8 = SceneConfig("G · Fidelity — 16×16 max_depth=8 (detailed)",  16, 16, 2, 2, max_depth=8)


PRESETS: dict[str, SceneConfig] = {
    # Story A
    "dens_2x2":   CONFIG_DENS_2X2,
    "dens_4x4":   CONFIG_DENS_4X4,
    "config_1":   CONFIG_1,           # densification 8×2
    "dens_8x8":   CONFIG_DENS_8X8,
    "config_2":   CONFIG_2,           # densification 16×16
    "dens_32x8":  CONFIG_DENS_32X8,

    # Story B
    "freq_1p8g":  CONFIG_FREQ_1P8G,
    "freq_2p6g":  CONFIG_FREQ_2P6G,
    "freq_3p5g":  CONFIG_FREQ_3P5G,
    "freq_39g":   CONFIG_FREQ_39G,

    # Story C
    "pat_iso":    CONFIG_PAT_ISO,
    "pat_dipole": CONFIG_PAT_DIPOLE,

    # Story D
    "pol_vh":     CONFIG_POL_VH,

    # Story E
    "pwr_low":    CONFIG_PWR_LOW,
    "pwr_high":   CONFIG_PWR_HIGH,

    # Story F
    "bw_20m":     CONFIG_BW_20M,
    "bw_400m":    CONFIG_BW_400M,

    # Story G
    "depth_3":    CONFIG_DEPTH_3,
    "depth_8":    CONFIG_DEPTH_8,
}
