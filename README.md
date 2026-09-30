# quackathon: microgrid siting in Ntabankulu

Choose where to build solar and battery microgrids in one ward of Ntabankulu Local
Municipality (EC444, Eastern Cape), serving settlements that are far from the
existing grid. The problem is set up as a QUBO so it can be solved with Qiskit (QAOA).

## Problem

- **Demand nodes** `i`: Overture building footprints grouped into 500 m cells. Only
  nodes further than `grid_distance_m` from the grid are kept.
  Demand `d_i = buildings × households_per_building × kWh_per_household_day`.
- **Candidate sites** `j` (one qubit each): up to `n_candidates` demand nodes that
  can reach the most demand, spaced at least `service_radius_m / 2` apart.
- **Decision**: `x_j ∈ {0,1}`. Build exactly `n_sites` microgrids.
- **Objective**: minimise `Σ c_j x_j − α · served(x)`, where a node is served if a chosen
  site is within `service_radius_m`.

QUBO form (see `src/quackathon/problem.py`):

    E(x) = Σ_j (c_j − α D_j) x_j + α Σ_{j<k} O_jk x_j x_k + P (Σ_j x_j − K)²

`D_j` is the demand reachable from site `j`. `O_jk` is the demand reachable from both
`j` and `k`. The approximation is exact when no node is covered by three or more
chosen sites. `classical.solve_exact` computes the true optimum for comparison.

## Quantum solver

`src/quackathon/quantum.py` runs QAOA on Qiskit's local statevector simulator:

1. The QUBO becomes an Ising Hamiltonian (`SparsePauliOp`) through `x_j = (1 − Z_j)/2`.
2. The default **XY mixer** starts from the Dicke state, an equal superposition of every
   K-site selection (Bärtschi & Eidenbenz circuit), and mixes with ring XY swaps. Every
   sample is a valid plan, so the cardinality penalty is dropped. `--mixer x` gives
   textbook QAOA from `|+⟩ⁿ`, with the penalty enforcing K instead.
3. COBYLA tunes the angles against `StatevectorEstimator`. Each depth is initialised
   from the previous one (INTERP), and the tuned circuit is sampled with `StatevectorSampler`.

Ward 14 results (12 qubits, K = 3, depth 1–4):

| | XY mixer | X mixer |
|---|---|---|
| P(optimum) | 0.9–1.2% (random 0.45%) | 0.0–0.2% |
| P(feasible) | 100% | 2–28% |
| Approximation ratio | 0.72–0.79 | < 0 |

On the 8-qubit toy (`n_candidates=8`), P(optimum) rises from 4.0% to 7.0% over depths
1–4, against 1.8% for random.

## IBM Quantum hardware

The angles are tuned on the local simulator, and only the final sampling job goes to
IBM (one job, a few seconds of QPU time). See `src/quackathon/hardware.py`.

1. Get an API key and your instance CRN from https://quantum.cloud.ibm.com.
2. Authenticate, either with environment variables:

       export QISKIT_IBM_TOKEN=...       # API key
       export QISKIT_IBM_INSTANCE=...    # instance CRN

   or by saving the account once. It is stored in `~/.qiskit/qiskit-ibm.json`,
   outside the repo:

       uv run python -c "from getpass import getpass; from qiskit_ibm_runtime import QiskitRuntimeService as S; \
         S.save_account(channel='ibm_quantum_platform', token=getpass('API key: '), \
         instance=input('Instance CRN: '), set_as_default=True, overwrite=True)"

3. Do a dry run against a local noisy copy of a device (no account needed), then run it
   for real:

       uv run quackathon --ward 14 --candidates 8 --qaoa-reps 1 --backend fake_torino
       uv run quackathon --ward 14 --candidates 8 --qaoa-reps 1 --backend least_busy

**Expect noise to dominate.** The Dicke state preparation and the dense ZZ couplings
transpile to a lot of two-qubit gates on the device: about 120 (6 qubits), 280
(8 qubits) and 570 (12 qubits) at depth 1. On the noisy Torino model, P(optimum)
drops below random guessing among valid plans, and 25–65% of shots remain valid.
The best sample still finds the optimum, but at this size that comes from classically
checking every sample. The 6–8 qubit toys at depth 1 are the realistic hardware
targets.

## Run

    uv run quackathon --ward 14 --grid-distance 2000 --sites 3 --candidates 12 --radius 1500 \
        --qaoa-reps 3 --mixer xy
    uv run pytest

The first run takes about 3 minutes: it pulls every building in the municipality
(about 312k) from Overture on S3 with DuckDB and caches them in `data/raw/`.

## Data sources and caveats

| Layer | Source | Caveat |
|---|---|---|
| Wards | MDB Wards 2026 (`shapefiles/`) | Newer boundaries than Census 2022 |
| Buildings | Overture Maps `2026-09-23.1` (Google, Microsoft, OSM) | Several structures per homestead; calibrate `households_per_building` against the census |
| Grid (`--grid-source gridfinder`, default) | Arderne et al. 2020 predicted MV grid plus OSM lines, clipped to `data/raw/gridfinder_EC444.gpkg` | Predicted from night-time lights and roads (2020), so an estimate. The global 725 MB `grid.gpkg` from https://zenodo.org/records/3628142 is only needed to rebuild the clip and is gitignored |
| Grid (`--grid-source osm`) | OSM power lines via Overpass | Mostly 132 kV transmission; the MV/LV distribution network is largely unmapped, so nearly everything counts as "far from the grid" |
