"""QAOA for the microgrid siting QUBO, on Qiskit's local statevector simulator.

Pipeline: QUBO -> Ising Hamiltonian (SparsePauliOp) -> QAOA circuit -> optimise the
angles against StatevectorEstimator -> sample the tuned circuit with StatevectorSampler.

Two mixers:
- "xy" (default): start in the Dicke state (equal superposition of every K-site
  selection) and mix with XY swaps on a ring. Both preserve the number of 1s, so the
  circuit never leaves the feasible subspace and needs no cardinality penalty.
- "x": textbook QAOA from |+>^n with an X mixer; the "exactly K" constraint lives in
  the QUBO penalty, which makes the landscape steep and wastes amplitude on
  infeasible states.

Bit convention: qubit j is site j, and |1> means "build". Qiskit bitstrings put qubit 0
rightmost, so int(bitstring, 2) is the same index `classical.all_bitstrings` uses.
"""

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RYGate, XXPlusYYGate
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from quackathon.classical import all_bitstrings, qubo_energies
from quackathon.problem import MicrogridProblem


def qubo_to_ising(q: np.ndarray, offset: float) -> tuple[SparsePauliOp, float]:
    """Hamiltonian H and constant c with <x|H|x> + c = x^T Q x + offset.

    Substitutes x_i = (1 - Z_i) / 2 into the upper-triangular QUBO.
    """
    n = q.shape[0]
    h = np.zeros(n)
    terms = []
    constant = offset
    for i in range(n):
        constant += q[i, i] / 2
        h[i] -= q[i, i] / 2
        for j in range(i + 1, n):
            if q[i, j] == 0:
                continue
            constant += q[i, j] / 4
            h[i] -= q[i, j] / 4
            h[j] -= q[i, j] / 4
            terms.append(("ZZ", [i, j], q[i, j] / 4))
    terms += [("Z", [i], h[i]) for i in range(n) if h[i] != 0]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify(), constant


def dicke_state(n: int, k: int) -> QuantumCircuit:
    """Prepare |D^n_k>, the uniform superposition of all weight-k bitstrings.

    Bärtschi & Eidenbenz (2019), "Deterministic Preparation of Dicke States": start from
    k ones in the top qubits and apply split-and-cyclic-shift blocks. O(nk) gates.
    """
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    qc.x(range(n - k, n))
    for m in range(n, k, -1):
        _split_cyclic_shift(qc, m, k)
    for m in range(k, 1, -1):
        _split_cyclic_shift(qc, m, m - 1)
    return qc


def _split_cyclic_shift(qc: QuantumCircuit, m: int, k: int) -> None:
    """SCS_{m,k} on qubits 0..m-1 (paper's qubits 1..m map to 0..m-1)."""
    top = m - 1
    qc.cx(top - 1, top)
    qc.cry(2 * np.arccos(np.sqrt(1 / m)), top, top - 1)
    qc.cx(top - 1, top)
    for l in range(2, k + 1):
        target, control = top - l, top - l + 1
        qc.cx(target, top)
        qc.append(_ccry(2 * np.arccos(np.sqrt(l / m))), [top, control, target])
        qc.cx(target, top)


def _ccry(theta: float):
    return RYGate(theta).control(2, annotated=False)


def qaoa_circuit(hamiltonian: SparsePauliOp, reps: int, mixer: str = "xy", n_select: int | None = None) -> QuantumCircuit:
    """QAOA ansatz: initial state, then `reps` rounds of exp(-iγH) and the mixer exp(-iβB).

    Parameters are ordered [γ_1..γ_p, β_1..β_p].
    """
    n = hamiltonian.num_qubits
    gammas = ParameterVector("γ", reps)
    betas = ParameterVector("β", reps)
    qc = QuantumCircuit(n)
    if mixer == "xy":
        qc.compose(dicke_state(n, n_select), inplace=True)
    elif mixer == "x":
        qc.h(range(n))
    else:
        raise ValueError(f"unknown mixer {mixer!r}")

    ring = [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]
    if n > 2:
        ring.append((n - 1, 0))
    for layer in range(reps):
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
            qubits = [k for k in range(n) if pauli.z[k]]
            if len(qubits) == 1:
                qc.rz(2 * coeff * gammas[layer], qubits[0])
            elif len(qubits) == 2:
                qc.rzz(2 * coeff * gammas[layer], *qubits)
        if mixer == "xy":
            # XXPlusYYGate(θ) = exp(-iθ(XX+YY)/4); θ = 4β gives exp(-iβ(XX+YY)/2)
            for i, j in ring:
                qc.append(XXPlusYYGate(4 * betas[layer]), [i, j])
        else:
            qc.rx(2 * betas[layer], range(n))
    return qc


@dataclass
class QAOAResult:
    x: np.ndarray                  # best feasible bitstring sampled (by true objective)
    params: np.ndarray             # optimised [γ..., β...]
    reps: int
    mixer: str
    counts: dict[str, int]         # Qiskit bitstring -> shots
    probabilities: np.ndarray      # (2^n,) exact output distribution of the tuned circuit
    energies: np.ndarray           # (2^n,) QUBO energy of each bitstring (Hamiltonian used)
    feasible: np.ndarray           # (2^n,) bool: exactly K sites
    expected_energy: float         # <E> under the tuned distribution
    ground_state_index: int        # lowest-energy feasible bitstring
    history: list[float] = field(default_factory=list)  # <E> per optimiser evaluation
    circuit: QuantumCircuit | None = None  # tuned circuit (parameters bound, no measurement)

    @property
    def ground_energy(self) -> float:
        return float(self.energies[self.ground_state_index])

    @property
    def p_ground(self) -> float:
        """Probability of measuring the optimum."""
        return float(self.probabilities[self.ground_state_index])

    @property
    def shots(self) -> int:
        return sum(self.counts.values())

    @property
    def p_feasible(self) -> float:
        return float(self.probabilities[self.feasible].sum())

    @property
    def p_ground_random(self) -> float:
        """P(optimum) for the mixer's starting distribution, i.e. a random guess."""
        n_start = self.feasible.sum() if self.mixer == "xy" else len(self.feasible)
        return 1 / float(n_start)

    @property
    def approximation_ratio(self) -> float:
        """(E_max - <E>) / (E_max - E_min) over feasible states: 1 = always optimal."""
        e = self.energies[self.feasible]
        return float((e.max() - self.expected_energy) / (e.max() - e.min()))


def solve_qaoa(problem: MicrogridProblem, reps: int = 2, **kwargs) -> QAOAResult:
    """Tune a depth-`reps` QAOA circuit on the local statevector simulator and sample it."""
    return qaoa_depth_sweep(problem, reps, **kwargs)[-1]


def qaoa_depth_sweep(
    problem: MicrogridProblem,
    max_reps: int,
    mixer: str = "xy",
    penalty: float | None = None,
    shots: int = 4096,
    maxiter: int = 300,
    restarts: int = 3,
    seed: int = 0,
) -> list[QAOAResult]:
    """Results for depth 1..max_reps.

    Each depth is seeded from the previous depth's optimum (INTERP, Zhou et al. 2020).
    Without this, deeper circuits often end up worse than shallow ones because COBYLA
    stalls in poor local minima. `penalty` only applies to the "x" mixer; the "xy"
    mixer enforces K by construction.
    """
    q, offset = problem.to_qubo(0.0 if mixer == "xy" else penalty)
    hamiltonian, constant = qubo_to_ising(q, offset)
    energies = qubo_energies(q, offset)
    feasible = all_bitstrings(problem.n_sites).sum(axis=1) == problem.n_select
    ground = int(np.flatnonzero(feasible)[np.argmin(energies[feasible])])

    # QUBO coefficients are in kWh/day (hundreds); rescale so useful γ values are O(1).
    scale = float(np.max(np.abs(hamiltonian.coeffs.real)))
    h_norm = SparsePauliOp(hamiltonian.paulis, hamiltonian.coeffs.real / scale)

    estimator = StatevectorEstimator(seed=seed)
    sampler = StatevectorSampler(seed=seed)
    rng = np.random.default_rng(seed)

    results: list[QAOAResult] = []
    best_params = None
    for depth in range(1, max_reps + 1):
        circuit = qaoa_circuit(h_norm, depth, mixer, problem.n_select)

        def expected_energy(params: np.ndarray, circuit=circuit) -> float:
            ev = estimator.run([(circuit, h_norm, params)]).result()[0].data.evs
            return float(ev) * scale + constant

        if best_params is None:
            starts = _grid_starts(expected_energy, restarts)
        else:
            seed_params = _interp(best_params)
            starts = [seed_params] + [seed_params + rng.normal(0, 0.1, seed_params.shape)
                                      for _ in range(restarts - 1)]
        best_energy, best_history = np.inf, []
        for start in starts:
            history: list[float] = []
            res = minimize(
                lambda p: history.append(e := expected_energy(p)) or e,
                start, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2},
            )
            if res.fun < best_energy:
                best_params, best_energy, best_history = res.x, float(res.fun), history

        tuned = circuit.assign_parameters(best_params)
        measured = tuned.copy()
        measured.measure_all()
        counts = sampler.run([measured], shots=shots).result()[0].data.meas.get_counts()
        results.append(QAOAResult(
            x=_best_feasible_sample(problem, counts),
            params=best_params,
            reps=depth,
            mixer=mixer,
            counts=counts,
            probabilities=Statevector(tuned).probabilities(),
            energies=energies,
            feasible=feasible,
            expected_energy=best_energy,
            ground_state_index=ground,
            history=best_history,
            circuit=tuned,
        ))
    return results


def _grid_starts(expected_energy, restarts: int) -> list[np.ndarray]:
    """The `restarts` best (γ, β) points of a coarse depth-1 grid."""
    grid = np.array([(g, b) for g in np.linspace(0.05, 1.5, 12) for b in np.linspace(0.05, 1.5, 8)])
    scores = np.array([expected_energy(gb) for gb in grid])
    return list(grid[np.argsort(scores)[:restarts]])


def _interp(params: np.ndarray) -> np.ndarray:
    """Depth-p optimum [γ, β] -> depth-(p+1) starting point by linear interpolation."""
    p = len(params) // 2

    def stretch(v):
        padded = np.concatenate([[0.0], v, [0.0]])
        i = np.arange(1, p + 2)
        return (i - 1) / p * padded[i - 1] + (p - i + 1) / p * padded[i]

    return np.concatenate([stretch(params[:p]), stretch(params[p:])])


def _best_feasible_sample(problem: MicrogridProblem, counts: dict[str, int]) -> np.ndarray:
    """Among sampled bitstrings with exactly K sites, the one with the best true objective."""
    best_x, best_obj = None, np.inf
    for key in counts:
        x = bitstring_to_x(key)
        if problem.is_feasible(x) and (obj := problem.objective(x)) < best_obj:
            best_x, best_obj = x, obj
    if best_x is None:
        raise RuntimeError("no feasible bitstring sampled; increase shots, reps or penalty")
    return best_x


def bitstring_to_x(key: str) -> np.ndarray:
    """Qiskit bitstring (qubit 0 rightmost) -> x with x[j] = site j."""
    return np.array([int(b) for b in reversed(key)], dtype=np.int8)
