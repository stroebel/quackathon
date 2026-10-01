"""Ward selection: solve the siting problem of every ward classically and compare.

Each ward is solved two ways:
- the candidate problem, choosing K of the `n_candidates` candidate sites (what the QUBO
  and QAOA see), solved exactly by enumeration;
- the full problem, where every far-from-grid demand node is a candidate site, solved
  exactly as a MILP. This is the best the ward could do without the qubit budget.
"""

from dataclasses import dataclass, replace

import geopandas as gpd
import numpy as np
import pandas as pd

from quackathon.classical import solve_exact, solve_greedy, solve_milp, solve_qubo_brute_force
from quackathon.config import PilotConfig
from quackathon.demand import candidate_sites, cluster_buildings, far_from_grid
from quackathon.grid import load_grid
from quackathon.overture import fetch_building_centroids
from quackathon.problem import MicrogridProblem
from quackathon.region import load_wards


@dataclass
class Municipality:
    wards: gpd.GeoDataFrame
    grid: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame  # with the WardNo each building falls in
    grid_source: str

    @classmethod
    def load(cls, cfg: PilotConfig) -> "Municipality":
        wards = load_wards(cfg)
        buildings = fetch_building_centroids(wards, cfg.raw_dir)
        buildings = buildings.sjoin(wards[["WardNo", "geometry"]], predicate="within").drop(columns="index_right")
        return cls(wards, load_grid(wards, cfg), buildings.reset_index(drop=True), cfg.grid_source)

    def far_nodes(self, cfg: PilotConfig, mapped_only: bool = False) -> gpd.GeoDataFrame:
        """Far-from-grid demand nodes of every ward, clustered ward by ward as in `build_pilot`.

        `mapped_only` ignores gridfinder's predicted lines, to see what demand they exclude.
        """
        if cfg.grid_source != self.grid_source:
            raise ValueError(f"loaded with grid_source={self.grid_source!r}, config has {cfg.grid_source!r}")
        grid = self.grid
        if mapped_only and "source" in grid:
            grid = grid[grid["source"] != "gridfinder"]
        per_ward = []
        for ward_no, b in self.buildings.groupby("WardNo"):
            nodes = far_from_grid(cluster_buildings(b, cfg), grid, cfg)
            per_ward.append(nodes.assign(WardNo=ward_no))
        return pd.concat(per_ward, ignore_index=True)


def all_nodes_problem(nodes: gpd.GeoDataFrame, cfg: PilotConfig) -> MicrogridProblem:
    """Siting problem where every demand node is also a candidate site."""
    return MicrogridProblem.from_geodata(nodes, nodes[["geometry"]], cfg)


def screen_wards(muni: Municipality, cfg: PilotConfig) -> pd.DataFrame:
    """One row per ward: demand pool, candidate optimum, full optimum and greedy, all in kWh/day."""
    far = muni.far_nodes(cfg)
    # Demand that is "near the grid" only because of a predicted line: missed if gridfinder is wrong.
    mapped_only_demand = muni.far_nodes(cfg, mapped_only=True).groupby("WardNo")["demand_kwh_day"].sum()
    n_buildings = muni.buildings["WardNo"].value_counts()
    rows = []
    for ward_no in sorted(muni.wards["WardNo"]):
        nodes = far[far["WardNo"] == ward_no].reset_index(drop=True)
        row = {"ward": ward_no, "buildings": int(n_buildings.get(ward_no, 0)), "far_nodes": len(nodes),
               "far_buildings": int(nodes["n_buildings"].sum()), "far_demand": float(nodes["demand_kwh_day"].sum()),
               "eligible": len(nodes) >= cfg.n_candidates}
        row["far_households"] = row["far_buildings"] * cfg.households_per_building
        row["predicted_excluded_demand"] = float(mapped_only_demand.get(ward_no, 0.0)) - row["far_demand"]
        if len(nodes) >= cfg.n_sites:
            full = all_nodes_problem(nodes, cfg)
            row["full_served"] = full.served_demand(solve_milp(full).x)
        if row["eligible"]:
            problem = MicrogridProblem.from_geodata(nodes, candidate_sites(nodes, cfg), cfg)
            row["served"] = problem.served_demand(solve_exact(problem).x)
            row["greedy_served"] = problem.served_demand(solve_greedy(problem).x)
            row["qubo_served"] = problem.served_demand(solve_qubo_brute_force(*problem.to_qubo()).x)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("ward")
    df["served_share"] = df["served"] / df["far_demand"]
    df["candidate_gap"] = 1 - df["served"] / df["full_served"]  # lost to the qubit budget
    df["greedy_gap"] = 1 - df["greedy_served"] / df["served"]   # > 0: greedy is not enough
    df["rank"] = df["served"].rank(ascending=False, method="min").astype("Int64")
    return df.sort_values("rank")


def rank_sweep(muni: Municipality, cfg: PilotConfig, **grid: list) -> pd.DataFrame:
    """Rank of every ward under each combination of config values, e.g. service_radius_m=[...].

    Returns a frame indexed by ward with one column per combination (a MultiIndex if more
    than one parameter is swept).
    """
    names = list(grid)
    combos = pd.MultiIndex.from_product(list(grid.values()), names=names)
    ranks = {}
    for values in combos:
        values = values if isinstance(values, tuple) else (values,)
        ranks[values] = screen_wards(muni, replace(cfg, **dict(zip(names, values))))["rank"]
    out = pd.DataFrame(ranks)
    out.columns = combos
    return out.sort_index()


def allocate_sites(muni: Municipality, cfg: PilotConfig, n_total: int,
                   time_limit_s: float = 120.0) -> gpd.GeoDataFrame:
    """Best `n_total` sites anywhere in the municipality, ignoring ward boundaries.

    Returns the chosen sites with the ward each falls in and the demand each serves first
    (demand reachable from several chosen sites is credited to the nearest one).
    """
    far = muni.far_nodes(cfg)
    problem = replace(all_nodes_problem(far, cfg), n_select=n_total)
    chosen = np.flatnonzero(solve_milp(problem, time_limit_s=time_limit_s).x)
    dist = np.where(problem.coverage[:, chosen], problem.distance[:, chosen], np.inf)
    covered = np.isfinite(dist).any(axis=1)
    nearest = np.argmin(dist, axis=1)
    serves = [float(far["demand_kwh_day"][covered & (nearest == k)].sum()) for k in range(len(chosen))]
    sites = far.iloc[chosen][["WardNo", "geometry"]].reset_index(drop=True)
    sites["serves_kwh_day"] = serves
    return sites.sort_values("serves_kwh_day", ascending=False).reset_index(drop=True)
