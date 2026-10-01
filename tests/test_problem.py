from itertools import combinations

import numpy as np
import pytest

from quackathon.classical import (all_bitstrings, qubo_energies, solve_exact, solve_greedy, solve_milp,
                                  solve_qubo_brute_force)
from quackathon.config import PilotConfig
from quackathon.problem import MicrogridProblem


def random_problem(seed=0, n_nodes=30, n_sites=8, k=3, radius=1_500.0):
    rng = np.random.default_rng(seed)
    nodes = rng.uniform(0, 6_000, (n_nodes, 2))
    sites = rng.uniform(0, 6_000, (n_sites, 2))
    return MicrogridProblem(
        demand=rng.uniform(1, 50, n_nodes),
        distance=np.linalg.norm(nodes[:, None] - sites[None], axis=-1),
        site_cost=np.zeros(n_sites),
        n_select=k,
        service_radius_m=radius,
    )


def test_qubo_matches_inclusion_exclusion_on_feasible_points():
    p = random_problem()
    q, offset = p.to_qubo()
    cov = p.coverage.astype(float)
    for chosen in combinations(range(p.n_sites), p.n_select):
        x = np.zeros(p.n_sites)
        x[list(chosen)] = 1
        approx = x @ (p.demand @ cov) - sum(p.demand @ (cov[:, j] * cov[:, k]) for j, k in combinations(chosen, 2))
        assert x @ q @ x + offset == pytest.approx(-approx)


def test_qubo_exact_when_no_node_covered_three_times():
    # Nodes on a line at 0, 1, ..., 5 km; sites at 0, 2, 4 km with 1.2 km reach, so
    # nodes are covered by at most two sites.
    nodes = np.arange(6) * 1_000.0
    sites = np.array([0.0, 2_000.0, 4_000.0])
    p = MicrogridProblem(
        demand=np.arange(1.0, 7.0),
        distance=np.abs(nodes[:, None] - sites[None]),
        site_cost=np.zeros(3),
        n_select=2,
        service_radius_m=1_200.0,
    )
    for x in ([1, 1, 0], [1, 0, 1], [0, 1, 1]):
        assert p.qubo_energy(np.array(x)) == pytest.approx(p.objective(np.array(x)))


@pytest.mark.parametrize("seed", range(5))
def test_qubo_optimum_is_feasible(seed):
    p = random_problem(seed)
    x = solve_qubo_brute_force(*p.to_qubo()).x
    assert p.is_feasible(x)


def test_qubo_optimum_close_to_exact():
    for seed in range(5):
        p = random_problem(seed)
        qubo_x = solve_qubo_brute_force(*p.to_qubo()).x
        assert p.served_demand(qubo_x) >= 0.9 * p.served_demand(solve_exact(p).x)


def test_bitstring_ordering_is_little_endian():
    bits = all_bitstrings(3)
    assert bits[1].tolist() == [1, 0, 0]
    assert bits[4].tolist() == [0, 0, 1]
    q = np.diag([1.0, 2.0, 4.0])
    assert qubo_energies(q, 0.0).tolist() == list(range(8))


def test_config_rejects_more_sites_than_candidates():
    with pytest.raises(ValueError):
        PilotConfig(n_sites=5, n_candidates=4)


@pytest.mark.parametrize("seed", range(5))
def test_milp_matches_enumeration(seed):
    p = random_problem(seed, n_sites=10, k=3)
    assert solve_milp(p).energy == pytest.approx(solve_exact(p).energy)
    assert p.is_feasible(solve_milp(p).x)


@pytest.mark.parametrize("seed", range(5))
def test_greedy_is_feasible_and_no_better_than_optimum(seed):
    p = random_problem(seed, n_sites=10, k=3)
    greedy = solve_greedy(p)
    assert p.is_feasible(greedy.x)
    assert greedy.energy >= solve_exact(p).energy - 1e-9
