import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from quackathon.config import PilotConfig
from quackathon.demand import candidate_sites, cluster_buildings, far_from_grid

CRS = "EPSG:32735"


def points(xy):
    xy = np.asarray(xy, float)
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS)


def test_clusters_drop_sparse_cells_and_use_building_centroid():
    cfg = PilotConfig(cluster_cell_m=500, min_buildings_per_cluster=3, households_per_building=1, kwh_per_household_day=1)
    nodes = cluster_buildings(points([[10, 10], [20, 10], [30, 40], [5_000, 5_000]]), cfg)
    assert len(nodes) == 1
    assert nodes.geometry[0].x == 20 and nodes.geometry[0].y == 20
    assert nodes["demand_kwh_day"][0] == 3


def test_far_from_grid_uses_configured_distance():
    nodes = points([[0, 500], [0, 1_500], [0, 3_000]])
    nodes["demand_kwh_day"] = 1.0
    grid = gpd.GeoDataFrame(geometry=[LineString([(-10_000, 0), (10_000, 0)])], crs=CRS)
    assert len(far_from_grid(nodes, grid, PilotConfig(grid_distance_m=1_000))) == 2
    assert len(far_from_grid(nodes, grid, PilotConfig(grid_distance_m=2_000))) == 1


def test_candidates_are_spread_out():
    cfg = PilotConfig(n_sites=1, n_candidates=2, service_radius_m=1_000)
    nodes = points([[0, 0], [100, 0], [5_000, 0]])
    nodes["demand_kwh_day"] = [10.0, 9.0, 1.0]
    sites = candidate_sites(nodes, cfg)
    assert sorted(sites.geometry.x) == [0, 5_000]
