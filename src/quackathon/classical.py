"""Exact classical solvers, used as ground truth for the quantum solver."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from quackathon.problem import MicrogridProblem

MAX_BRUTE_FORCE_VARS = 24


@dataclass(frozen=True)
class Solution:
    x: np.ndarray
    energy: float


def all_bitstrings(n: int) -> np.ndarray:
    """(2^n, n) array; row b is the little-endian bits of b (matches Qiskit ordering)."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def qubo_energies(q: np.ndarray, offset: float) -> np.ndarray:
    """Energy of every bitstring, indexed like `all_bitstrings`."""
    n = q.shape[0]
    if n > MAX_BRUTE_FORCE_VARS:
        raise ValueError(f"{n} variables is too many to enumerate")
    x = all_bitstrings(n).astype(float)
    return np.einsum("bi,ij,bj->b", x, q, x) + offset


def solve_qubo_brute_force(q: np.ndarray, offset: float) -> Solution:
    energies = qubo_energies(q, offset)
    best = int(np.argmin(energies))
    return Solution(all_bitstrings(q.shape[0])[best], float(energies[best]))


def solve_exact(problem: MicrogridProblem) -> Solution:
    """Optimum of the true (non-approximated) objective over all K-subsets."""
    best: Solution | None = None
    for chosen in combinations(range(problem.n_sites), problem.n_select):
        x = np.zeros(problem.n_sites, dtype=np.int8)
        x[list(chosen)] = 1
        energy = problem.objective(x)
        if best is None or energy < best.energy:
            best = Solution(x, energy)
    return best


def solve_greedy(problem: MicrogridProblem) -> Solution:
    """Add the site with the largest drop in objective, K times. The usual max-coverage baseline."""
    x = np.zeros(problem.n_sites, dtype=np.int8)
    for _ in range(problem.n_select):
        trial = np.where(x == 0)[0]
        energies = [problem.objective(x + np.eye(problem.n_sites, dtype=np.int8)[j]) for j in trial]
        x[trial[int(np.argmin(energies))]] = 1
    return Solution(x, problem.objective(x))


def solve_milp(problem: MicrogridProblem, time_limit_s: float = 60.0) -> Solution:
    """Optimum of the true objective as a mixed-integer program (HiGHS), for any number of sites.

        minimise  c.x - alpha * d.y
        s.t.      y_i <= sum_j a_ij x_j,   sum_j x_j = K,   x binary,   0 <= y <= 1

    y can stay continuous: once x is integral, the best y_i is 1 if node i is covered and 0
    otherwise.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_array, hstack, identity

    n_nodes, n_sites = problem.coverage.shape
    c = np.concatenate([problem.site_cost, -problem.value_per_kwh_day * problem.demand])
    covered = LinearConstraint(hstack([-csr_array(problem.coverage.astype(float)), identity(n_nodes)]), -np.inf, 0)
    cardinality = LinearConstraint(np.concatenate([np.ones(n_sites), np.zeros(n_nodes)])[None, :],
                                   problem.n_select, problem.n_select)
    res = milp(c, constraints=[covered, cardinality], bounds=Bounds(0, 1),
               integrality=np.concatenate([np.ones(n_sites), np.zeros(n_nodes)]),
               options={"time_limit": time_limit_s})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = np.round(res.x[:n_sites]).astype(np.int8)
    return Solution(x, problem.objective(x))
