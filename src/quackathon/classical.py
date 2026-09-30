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
