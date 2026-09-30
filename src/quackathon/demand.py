"""Buildings -> demand nodes, filtered to those far from the grid."""

import geopandas as gpd
import numpy as np

from quackathon.config import PilotConfig


def cluster_buildings(buildings: gpd.GeoDataFrame, cfg: PilotConfig) -> gpd.GeoDataFrame:
    """Aggregate building points into square cells; each non-trivial cell is a demand node.

    Node location is the mean of its buildings, not the cell centre, so it sits on
    the settlement.
    """
    xy = np.column_stack([buildings.geometry.x, buildings.geometry.y])
    cells = np.floor(xy / cfg.cluster_cell_m).astype(np.int64)
    _, cell_idx, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    cell_idx = cell_idx.ravel()

    sums = np.zeros((len(counts), 2))
    np.add.at(sums, cell_idx, xy)
    centroids = sums / counts[:, None]

    keep = counts >= cfg.min_buildings_per_cluster
    n_buildings = counts[keep]
    nodes = gpd.GeoDataFrame(
        {
            "n_buildings": n_buildings,
            "demand_kwh_day": n_buildings * cfg.households_per_building * cfg.kwh_per_household_day,
        },
        geometry=gpd.points_from_xy(centroids[keep, 0], centroids[keep, 1]),
        crs=buildings.crs,
    )
    return nodes.sort_values("demand_kwh_day", ascending=False).reset_index(drop=True)


def far_from_grid(nodes: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, cfg: PilotConfig) -> gpd.GeoDataFrame:
    """Nodes further than `cfg.grid_distance_m` from any grid line."""
    nodes = nodes.copy()
    nodes["grid_distance_m"] = nodes.distance(grid.union_all()) if not grid.empty else np.inf
    return nodes[nodes["grid_distance_m"] > cfg.grid_distance_m].reset_index(drop=True)


def candidate_sites(nodes: gpd.GeoDataFrame, cfg: PilotConfig) -> gpd.GeoDataFrame:
    """Candidate microgrid sites: the demand nodes that could serve the most demand.

    Picking greedily by reachable demand while skipping sites too close to an
    already-picked one keeps candidates spread out, so the qubit budget isn't spent
    on near-duplicates.
    """
    xy = np.column_stack([nodes.geometry.x, nodes.geometry.y])
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    reach = (dist <= cfg.service_radius_m) @ nodes["demand_kwh_day"].to_numpy()

    min_spacing = cfg.service_radius_m / 2
    picked: list[int] = []
    for i in np.argsort(-reach, kind="stable"):
        if all(dist[i, j] >= min_spacing for j in picked):
            picked.append(int(i))
        if len(picked) == cfg.n_candidates:
            break
    sites = nodes.iloc[picked][["geometry"]].reset_index(drop=True)
    sites["reachable_kwh_day"] = reach[picked]
    return sites
