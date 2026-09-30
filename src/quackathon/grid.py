"""Existing electricity grid, from whichever source the config selects."""

import geopandas as gpd

from quackathon.config import METRIC_CRS, MUNICIPALITY_CODE, PilotConfig
from quackathon.osm import fetch_power_lines
from quackathon.region import load_wards

GRID_SEARCH_BUFFER_M = 20_000
GRIDFINDER_URL = "https://zenodo.org/records/3628142"


def load_grid(area: gpd.GeoDataFrame, cfg: PilotConfig) -> gpd.GeoDataFrame:
    if cfg.grid_source == "osm":
        return fetch_power_lines(area, cfg.raw_dir, buffer_m=GRID_SEARCH_BUFFER_M)
    grid = load_gridfinder(cfg)
    return grid[grid.intersects(area.buffer(GRID_SEARCH_BUFFER_M).union_all())].reset_index(drop=True)


def load_gridfinder(cfg: PilotConfig) -> gpd.GeoDataFrame:
    """Gridfinder MV grid (Arderne et al. 2020) around the municipality.

    The global file is 725 MB, too big for git, so on first use it is clipped to the
    municipality plus a buffer and the clip is saved to data/raw, which is committed.
    `source` is "openstreetmap" for mapped lines and "gridfinder" for predicted ones.
    """
    clip = cfg.raw_dir / f"gridfinder_{MUNICIPALITY_CODE}.gpkg"
    if not clip.exists():
        full = next((p for p in (cfg.data_dir / "grid.gpkg", cfg.raw_dir / "grid.gpkg") if p.exists()), None)
        if full is None:
            raise FileNotFoundError(f"put grid.gpkg from {GRIDFINDER_URL} in {cfg.data_dir}")
        region = load_wards(cfg).buffer(GRID_SEARCH_BUFFER_M).union_all()
        bbox = tuple(gpd.GeoSeries([region], crs=METRIC_CRS).to_crs("EPSG:4326").total_bounds)
        grid = gpd.read_file(full, bbox=bbox).to_crs(METRIC_CRS)
        grid = grid[grid.intersects(region)].reset_index(drop=True)
        clip.parent.mkdir(parents=True, exist_ok=True)
        grid.to_file(clip)
    return gpd.read_file(clip)
