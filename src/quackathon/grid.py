"""Existing electricity grid, from whichever source the config selects."""

import geopandas as gpd

from quackathon.config import METRIC_CRS, PilotConfig
from quackathon.osm import fetch_power_lines

GRID_SEARCH_BUFFER_M = 20_000


def load_grid(area: gpd.GeoDataFrame, cfg: PilotConfig) -> gpd.GeoDataFrame:
    if cfg.grid_source == "osm":
        return fetch_power_lines(area, cfg.raw_dir, buffer_m=GRID_SEARCH_BUFFER_M)

    path = cfg.raw_dir / "grid.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download grid.gpkg from https://zenodo.org/records/3628142")
    bbox = tuple(area.buffer(GRID_SEARCH_BUFFER_M).to_crs("EPSG:4326").total_bounds)
    return gpd.read_file(path, bbox=bbox).to_crs(METRIC_CRS)
