"""Seattle metro neighborhoods + lat/lon → local-meters projection.

The source tower table (``cmegdemos_catalog.network_analytics_enablement.cell_towers``)
holds 2,312 real T-Mobile towers across the Seattle metro in WGS-84 lat/lon. Sionna RT
works in a local Cartesian (meters) frame, and a single scene cannot hold the whole
44 × 28 km metro — so we cut the metro into named **neighborhoods**, each a bounding box
small enough to render (after tiling) with the approximation knobs.

Each neighborhood carries its own local-ENU **origin** (its center), so the projected
tower coordinates are centered on (0, 0) and stay small — which keeps the Sionna scene,
the radio-map tensor, and the top-down camera framing well-behaved.

Projection is a cheap equirectangular approximation, which is accurate to well under a
metre over a few-km box at Seattle's latitude — far below the ray-tracing resolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Meters per degree. Latitude is ~constant; longitude shrinks by cos(lat).
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQ = 111_320.0


@dataclass(frozen=True)
class Neighborhood:
    """A named bounding box over the Seattle metro.

    ``Downtown`` is the one we pre-render end-to-end; the rest are render-on-demand
    from the app's neighborhood dropdown.
    """

    name: str
    lat_lo: float
    lat_hi: float
    lon_lo: float
    lon_hi: float
    blurb: str = ""

    @property
    def center_lat(self) -> float:
        return (self.lat_lo + self.lat_hi) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.lon_lo + self.lon_hi) / 2.0

    @property
    def origin(self) -> Tuple[float, float]:
        """Local-ENU origin (lat0, lon0) — the box center."""
        return (self.center_lat, self.center_lon)

    @property
    def extent_m(self) -> Tuple[float, float]:
        """(east_west_m, north_south_m) physical size of the box."""
        ew = (self.lon_hi - self.lon_lo) * _M_PER_DEG_LON_EQ * math.cos(math.radians(self.center_lat))
        ns = (self.lat_hi - self.lat_lo) * _M_PER_DEG_LAT
        return (ew, ns)

    def sql_bbox_filter(self, lat_col: str = "latitude", lon_col: str = "longitude") -> str:
        """A SQL ``WHERE`` fragment selecting towers inside this box."""
        return (
            f"{lat_col} BETWEEN {self.lat_lo} AND {self.lat_hi} "
            f"AND {lon_col} BETWEEN {self.lon_lo} AND {self.lon_hi}"
        )


def project_lonlat(lat: float, lon: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    """Project (lat, lon) to local (x_east, y_north) meters about ``origin``.

    Equirectangular approximation — exact enough (sub-metre) for a few-km neighborhood.
    """
    lat0, lon0 = origin
    x_east = (lon - lon0) * _M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    y_north = (lat - lat0) * _M_PER_DEG_LAT
    return (x_east, y_north)


# ---------------------------------------------------------------------------
# Neighborhood registry
# ---------------------------------------------------------------------------
# Boxes chosen against the real tower distribution. Tower counts (verified
# against the UC table) are noted so demo owners know the render weight.
#   Downtown      ~125 towers (47 NR / 71 LTE / ~7 GSM-UMTS)  <- pre-rendered
#   Capitol Hill  ~260 | Belltown/SLU dense
#   Bellevue       ~66 | West Seattle ~27 | Ballard ~11

NEIGHBORHOODS: Dict[str, Neighborhood] = {
    "Downtown": Neighborhood(
        "Downtown", 47.600, 47.618, -122.345, -122.332,
        "Seattle CBD / financial district — densest tower cluster (~125 towers). Pre-rendered.",
    ),
    "Capitol Hill": Neighborhood(
        "Capitol Hill", 47.610, 47.640, -122.330, -122.300,
        "Dense residential + nightlife east of I-5.",
    ),
    "South Lake Union": Neighborhood(
        "South Lake Union", 47.618, 47.640, -122.345, -122.328,
        "Tech campuses (Amazon) north of downtown.",
    ),
    "SODO": Neighborhood(
        "SODO", 47.578, 47.600, -122.345, -122.325,
        "Stadium / industrial district south of downtown.",
    ),
    "Ballard": Neighborhood(
        "Ballard", 47.660, 47.690, -122.400, -122.360,
        "NW residential + maritime.",
    ),
    "West Seattle": Neighborhood(
        "West Seattle", 47.550, 47.590, -122.410, -122.360,
        "Across the Duwamish, lower tower density.",
    ),
    "U-District": Neighborhood(
        "U-District", 47.650, 47.675, -122.325, -122.295,
        "University of Washington campus + housing.",
    ),
    "Bellevue": Neighborhood(
        "Bellevue", 47.590, 47.630, -122.210, -122.170,
        "Eastside downtown across Lake Washington.",
    ),
}

DEFAULT_NEIGHBORHOOD = "Downtown"


def get(name: str) -> Neighborhood:
    return NEIGHBORHOODS[name]


def names() -> List[str]:
    """Neighborhood names, Downtown first (it's the pre-rendered default)."""
    rest = sorted(n for n in NEIGHBORHOODS if n != DEFAULT_NEIGHBORHOOD)
    return [DEFAULT_NEIGHBORHOOD, *rest]
