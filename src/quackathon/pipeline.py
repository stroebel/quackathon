"""Ward -> buildings -> far-from-grid demand nodes -> candidate sites -> problem."""

from dataclasses import dataclass

import geopandas as gpd

from quackathon.config import PilotConfig
from quackathon.demand import candidate_sites, cluster_buildings, far_from_grid
from quackathon.grid import load_grid
from quackathon.osm import fetch_buildings
from quackathon.problem import MicrogridProblem
from quackathon.region import load_ward
from quackathon.synthetic import synthetic_buildings


@dataclass
class Pilot:
    cfg: PilotConfig
    ward: gpd.GeoDataFrame
    grid: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    all_nodes: gpd.GeoDataFrame
    nodes: gpd.GeoDataFrame     # far-from-grid demand nodes only
    sites: gpd.GeoDataFrame
    problem: MicrogridProblem


def build_pilot(cfg: PilotConfig, buildings_source: str = "osm") -> Pilot:
    ward = load_ward(cfg)
    grid = load_grid(ward, cfg)
    if buildings_source == "osm":
        buildings = fetch_buildings(ward, cfg.raw_dir)
    elif buildings_source == "synthetic":
        buildings = synthetic_buildings(ward)
    else:
        raise ValueError(f"unknown buildings_source {buildings_source!r}")

    all_nodes = cluster_buildings(buildings, cfg)
    nodes = far_from_grid(all_nodes, grid, cfg)
    if len(nodes) < cfg.n_candidates:
        raise ValueError(
            f"only {len(nodes)} demand nodes beyond {cfg.grid_distance_m:.0f} m of the grid; "
            f"need at least n_candidates={cfg.n_candidates}"
        )
    sites = candidate_sites(nodes, cfg)
    problem = MicrogridProblem.from_geodata(nodes, sites, cfg)
    return Pilot(cfg, ward, grid, buildings, all_nodes, nodes, sites, problem)
