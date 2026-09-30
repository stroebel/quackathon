"""Run a tuned QAOA circuit on IBM Quantum hardware (or a local noisy copy of one).

The angles are tuned on the local statevector simulator (`quantum.solve_qaoa`) and only
the final sampling runs on the device. This takes one job instead of the hundreds a
variational loop on hardware would need, which fits within Open Plan limits and avoids
queueing once per optimiser step.

Authentication (either one):
- environment variables QISKIT_IBM_TOKEN (API key) and QISKIT_IBM_INSTANCE (CRN), or
- a saved account: QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
  token=..., instance=..., set_as_default=True), stored in ~/.qiskit/qiskit-ibm.json.

Backend names:
- "fake_<device>" (e.g. "fake_torino"): local Aer simulation with that device's
  calibrated noise model, no account needed, for dry runs;
- "least_busy": the least busy real device your instance can use;
- anything else: a device by name, e.g. "ibm_torino".
"""

from dataclasses import dataclass

import numpy as np
from qiskit.transpiler import generate_preset_pass_manager

from quackathon.problem import MicrogridProblem
from quackathon.quantum import QAOAResult, bitstring_to_x


def get_backend(name: str):
    if name.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        base = "Fake" + name.removeprefix("fake_").capitalize()
        for cls_name in (base, base + "V2"):
            if hasattr(fake_provider, cls_name):
                return getattr(fake_provider, cls_name)()
        raise ValueError(f"no fake backend {base} in qiskit_ibm_runtime.fake_provider")

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    if name == "least_busy":
        return service.least_busy(operational=True, simulator=False)
    return service.backend(name)


@dataclass
class HardwareResult:
    backend: str
    job_id: str | None
    counts: dict[str, int]
    x: np.ndarray | None          # best feasible sample by true objective (None if none)
    p_ground: float               # empirical P(optimum) over all shots
    p_feasible: float             # share of shots with exactly K sites
    two_qubit_gates: int          # after transpilation to the device
    depth: int

    @property
    def shots(self) -> int:
        return sum(self.counts.values())


def transpile_for(backend, result: QAOAResult, optimization_level: int = 3, seed: int = 0):
    circuit = result.circuit.copy()
    circuit.measure_all()
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      seed_transpiler=seed)
    return pm.run(circuit)


def two_qubit_gate_count(circuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


def run_on_backend(
    problem: MicrogridProblem,
    result: QAOAResult,
    backend_name: str,
    shots: int = 4096,
    optimization_level: int = 3,
) -> HardwareResult:
    """Sample the tuned circuit from `result` on `backend_name`. Blocks until the job finishes."""
    from qiskit_ibm_runtime.executor_sampler import Sampler

    backend = get_backend(backend_name)
    isa = transpile_for(backend, result, optimization_level)

    sampler = Sampler(mode=backend)
    if not backend_name.startswith("fake_"):
        # Cheap error suppression: idle qubits get dynamical decoupling, and Pauli
        # twirling turns coherent gate errors into less harmful random noise.
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True

    job = sampler.run([isa], shots=shots)
    print(f"Submitted job {job.job_id()} to {backend.name}; waiting for results...")
    counts = job.result()[0].data.meas.get_counts()

    ground = format(result.ground_state_index, f"0{problem.n_sites}b")
    best_x, best_obj, feasible_shots = None, np.inf, 0
    for key, c in counts.items():
        x = bitstring_to_x(key)
        if problem.is_feasible(x):
            feasible_shots += c
            if (obj := problem.objective(x)) < best_obj:
                best_x, best_obj = x, obj
    total = sum(counts.values())
    return HardwareResult(
        backend=backend.name,
        job_id=job.job_id(),
        counts=counts,
        x=best_x,
        p_ground=counts.get(ground, 0) / total,
        p_feasible=feasible_shots / total,
        two_qubit_gates=two_qubit_gate_count(isa),
        depth=isa.depth(),
    )
