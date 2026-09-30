"""The microgrid siting problem and its QUBO form.

Choose exactly K of N candidate sites (x_j in {0,1}) to

    minimise   sum_j c_j x_j  -  alpha * served(x)
    served(x) = sum_i d_i * [some chosen site j has dist(i, j) <= R]

`served` is not quadratic, so the QUBO uses the second-order inclusion-exclusion
approximation

    served(x) ~= sum_j D_j x_j - sum_{j<k} O_jk x_j x_k
    D_j  = demand within R of site j
    O_jk = demand within R of both j and k

which is exact whenever no demand node is covered by three or more chosen sites.
The cardinality constraint becomes the penalty P * (sum_j x_j - K)^2.
"""

from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import numpy as np

from quackathon.config import PilotConfig


@dataclass(frozen=True)
class MicrogridProblem:
    demand: np.ndarray       # (n_nodes,) kWh/day
    distance: np.ndarray     # (n_nodes, n_sites) metres
    site_cost: np.ndarray    # (n_sites,)
    n_select: int
    service_radius_m: float
    value_per_kwh_day: float = 1.0

    @classmethod
    def from_geodata(cls, nodes: gpd.GeoDataFrame, sites: gpd.GeoDataFrame, cfg: PilotConfig) -> "MicrogridProblem":
        node_xy = np.column_stack([nodes.geometry.x, nodes.geometry.y])
        site_xy = np.column_stack([sites.geometry.x, sites.geometry.y])
        return cls(
            demand=nodes["demand_kwh_day"].to_numpy(float),
            distance=np.linalg.norm(node_xy[:, None, :] - site_xy[None, :, :], axis=-1),
            site_cost=np.full(len(sites), cfg.site_cost),
            n_select=cfg.n_sites,
            service_radius_m=cfg.service_radius_m,
            value_per_kwh_day=cfg.value_per_kwh_day,
        )

    @property
    def n_sites(self) -> int:
        return self.distance.shape[1]

    @cached_property
    def coverage(self) -> np.ndarray:
        """(n_nodes, n_sites) bool: node i is within reach of site j."""
        return self.distance <= self.service_radius_m

    def served_demand(self, x: np.ndarray) -> float:
        """Exact demand served by selection `x`."""
        return float(self.demand @ self.coverage[:, np.asarray(x, bool)].any(axis=1))

    def objective(self, x: np.ndarray) -> float:
        """Exact objective, ignoring the cardinality constraint."""
        x = np.asarray(x)
        return float(self.site_cost @ x - self.value_per_kwh_day * self.served_demand(x))

    def is_feasible(self, x: np.ndarray) -> bool:
        return int(np.sum(x)) == self.n_select

    def to_qubo(self, penalty: float | None = None) -> tuple[np.ndarray, float]:
        """Upper-triangular Q and offset such that x^T Q x + offset is the QUBO energy.

        Diagonal entries hold the linear terms (x_j^2 = x_j for binaries).
        """
        a = self.value_per_kwh_day
        cov = self.coverage.astype(float)
        linear = self.site_cost - a * (self.demand @ cov)
        overlap = a * (cov.T * self.demand) @ cov

        if penalty is None:
            # Adding or dropping one site changes the unpenalised energy by at most
            # |c_j| + alpha * D_j; the penalty for being one site off must beat that.
            penalty = 1.0 + float(np.max(np.abs(self.site_cost) + a * (self.demand @ cov)))

        # P(sum x - K)^2 = P[sum_j (1-2K) x_j + 2 sum_{j<k} x_j x_k + K^2]
        k = self.n_select
        q = np.triu(overlap, k=1) + np.triu(np.full_like(overlap, 2 * penalty), k=1)
        q[np.diag_indices_from(q)] = linear + penalty * (1 - 2 * k)
        return q, penalty * k * k

    def qubo_energy(self, x: np.ndarray, penalty: float | None = None) -> float:
        q, offset = self.to_qubo(penalty)
        x = np.asarray(x, float)
        return float(x @ q @ x + offset)
