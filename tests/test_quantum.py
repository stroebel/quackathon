import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from quackathon.classical import qubo_energies, solve_exact
from quackathon.quantum import _interp, bitstring_to_x, dicke_state, qaoa_circuit, qubo_to_ising, solve_qaoa
from tests.test_problem import random_problem


def test_ising_diagonal_matches_qubo_energies():
    p = random_problem(n_sites=6, k=2)
    q, offset = p.to_qubo()
    h, constant = qubo_to_ising(q, offset)
    # Qiskit's basis index b has qubit j as bit j, the same ordering as all_bitstrings.
    assert np.allclose(np.diag(h.to_matrix()).real + constant, qubo_energies(q, offset))


@pytest.mark.parametrize("n,k", [(3, 1), (5, 2), (6, 3), (7, 5)])
def test_dicke_state_is_uniform_over_weight_k(n, k):
    probs = Statevector(dicke_state(n, k)).probabilities()
    weights = np.array([bin(i).count("1") for i in range(2**n)])
    assert np.allclose(probs[weights == k], 1 / np.sum(weights == k))


def test_xy_mixer_stays_in_feasible_subspace():
    p = random_problem(n_sites=6, k=2)
    h, _ = qubo_to_ising(*p.to_qubo(0.0))
    qc = qaoa_circuit(h, reps=2, mixer="xy", n_select=2)
    probs = Statevector(qc.assign_parameters(np.random.default_rng(1).uniform(-2, 2, 4))).probabilities()
    weights = np.array([bin(i).count("1") for i in range(2**6)])
    assert probs[weights != 2].sum() == pytest.approx(0, abs=1e-9)


def test_bitstring_to_x_reverses_qiskit_order():
    assert bitstring_to_x("0011").tolist() == [1, 1, 0, 0]


def test_interp_keeps_endpoints_and_grows_depth():
    new = _interp(np.array([0.4, 0.8, 0.6, 0.2]))
    assert new.shape == (6,)
    assert new[0] == pytest.approx(0.4) and new[2] == pytest.approx(0.8)


@pytest.mark.parametrize("mixer", ["xy", "x"])
def test_qaoa_beats_random_guessing_on_small_problem(mixer):
    p = random_problem(seed=3, n_nodes=20, n_sites=5, k=2)
    res = solve_qaoa(p, reps=2, mixer=mixer, shots=1024, restarts=2)
    assert res.p_ground > 1.5 * res.p_ground_random
    assert p.served_demand(res.x) == pytest.approx(p.served_demand(solve_exact(p).x))


def test_fake_backend_run_returns_feasible_optimum_on_tiny_problem():
    from quackathon.hardware import run_on_backend

    p = random_problem(seed=3, n_nodes=12, n_sites=4, k=2)
    res = solve_qaoa(p, reps=1, shots=512, restarts=1)
    hw = run_on_backend(p, res, "fake_torino", shots=2048)
    assert hw.shots == 2048
    assert hw.two_qubit_gates > 0
    assert 0 < hw.p_feasible <= 1
    assert p.served_demand(hw.x) == pytest.approx(p.served_demand(solve_exact(p).x))
