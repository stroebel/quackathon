"""Configuration for a single-ward microgrid siting pilot."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# UTM zone 35S: metric CRS covering Ntabankulu (~29.3°E, 30.9°S).
METRIC_CRS = "EPSG:32735"
MUNICIPALITY_CODE = "EC444"  # Ntabankulu Local Municipality


@dataclass(frozen=True)
class PilotConfig:
    ward_no: int = 1

    # Demand focus: settlement clusters further than this from the grid.
    grid_distance_m: float = 2_000.0
    # "osm" (mapped lines, very incomplete here) or "gridfinder" (predicted MV grid,
    # Arderne et al. 2020, needs data/raw/grid.gpkg from zenodo.org/records/3628142).
    grid_source: str = "osm"

    # Siting: build exactly this many microgrids.
    n_sites: int = 3

    # Qubit budget: one binary variable per candidate site.
    n_candidates: int = 12

    # A microgrid serves demand clusters within this distance of its site.
    service_radius_m: float = 1_500.0

    # Buildings are aggregated into square cells of this size to form demand nodes.
    cluster_cell_m: float = 500.0
    min_buildings_per_cluster: int = 3

    # Demand model (ESMAP MTF Tier 2-3 is roughly 0.2-1.0 kWh/household/day).
    households_per_building: float = 1.0
    kwh_per_household_day: float = 0.5

    # Economics: minimise sum(site_cost) - value_per_kwh_day * served_demand.
    site_cost: float = 0.0
    value_per_kwh_day: float = 1.0

    data_dir: Path = field(default=PROJECT_ROOT / "data")
    shapefile_dir: Path = field(default=PROJECT_ROOT / "shapefiles")

    def __post_init__(self):
        if not 1 <= self.n_sites <= self.n_candidates:
            raise ValueError("need 1 <= n_sites <= n_candidates")
        if self.grid_distance_m < 0 or self.service_radius_m <= 0:
            raise ValueError("distances must be non-negative (radius positive)")
        if self.grid_source not in ("osm", "gridfinder"):
            raise ValueError(f"unknown grid_source {self.grid_source!r}")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"
