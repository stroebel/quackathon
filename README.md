# quackathon: microgrid siting in Ntabankulu

This repo selects sites for solar and battery microgrids in one ward of Ntabankulu Local
Municipality (EC444, Eastern Cape). It serves settlements far from the existing grid.
The problem is a QUBO. Qiskit solves it with QAOA, and classical solvers check the result.

## Setup

You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

    just setup

This installs the dependencies and unzips the MDB ward and district shapefiles into `shapefiles/`.

## Run the app

    uv run quackathon-app

The app opens a Dear PyGui window. Use it to explore one ward at a time.

1. Go to **Wards**. Screen every ward and load one into the settings.
2. Set the options in **Settings** (left). Click **Run**.
3. Look at the result in the other tabs. The map shows the classical result in about a
   second. QAOA results come later.

Tabs:

- **Map**: buildings, grid, demand nodes, candidate sites, and the selected plan. Click a site
  to add or remove it. **Export** writes the plan to `data/exports/*.geojson`.
- **Sites**: the reach and grid distance of each candidate. Also the change in served demand
  if you toggle it.
- **Compare**: served demand, gap to the optimum, and QUBO error for each plan.
- **QAOA**: the output distribution over valid plans (click a bar to map that plan), the
  optimiser trace, and P(optimum), P(feasible) and approximation ratio by depth.

Two worker processes do the runs, so the window stays responsive. **Cancel** stops queued
runs only. If a run has started, the app discards its result when it finishes.

## Run from the command line

    uv run quackathon --ward 14 --sites 3 --candidates 12 --radius 1500 --qaoa-reps 3

Main options (`uv run quackathon --help` shows all):

| Option            | Default      | Meaning                                         |
|-------------------|--------------|-------------------------------------------------|
| `--ward`          | 1            | Ward number                                     |
| `--sites`         | 3            | Number of microgrids to build (K)               |
| `--candidates`    | 12           | Number of candidate sites (qubits)              |
| `--radius`        | 1500         | Service radius in metres                        |
| `--grid-distance` | 2000         | Keep only demand further than this from the grid (metres) |
| `--grid-source`   | `gridfinder` | `gridfinder` or `osm`                           |
| `--qaoa-reps`     | 3            | QAOA depth. 0 skips QAOA                        |
| `--mixer`         | `xy`         | `xy` (valid plans only) or `x` (textbook QAOA)  |
| `--backend`       | none         | IBM device, `least_busy`, or `fake_<device>`    |

The first run takes about 3 minutes. It downloads all buildings in the municipality (about
312k) from Overture on S3 and caches them in `data/raw/`. Later runs use the cache.

## Run the tests

    uv run pytest

Some tests need the shapefiles. Run `just setup` first.

## Run on IBM Quantum hardware

The local simulator tunes the QAOA angles. Only the final sampling job goes to IBM. This is
one job of a few seconds of QPU time.

1. Get an API key and your instance CRN from https://quantum.cloud.ibm.com.
2. Set your credentials:

       export QISKIT_IBM_TOKEN=...       # API key
       export QISKIT_IBM_INSTANCE=...    # instance CRN

   Or save them once to `~/.qiskit/qiskit-ibm.json`:

       uv run python -c "from getpass import getpass; from qiskit_ibm_runtime import QiskitRuntimeService as S; \
         S.save_account(channel='ibm_quantum_platform', token=getpass('API key: '), \
         instance=input('Instance CRN: '), set_as_default=True, overwrite=True)"

3. Do a dry run on a local noisy model. This needs no account:

       uv run quackathon --ward 14 --candidates 8 --qaoa-reps 1 --backend fake_torino

4. Run on a real device:

       uv run quackathon --ward 14 --candidates 8 --qaoa-reps 1 --backend least_busy

Use 6 to 8 qubits at depth 1 on hardware. Larger problems give mostly noise. At depth 1, the
circuit has about 120 two-qubit gates at 6 qubits, 280 at 8 qubits, and 570 at 12 qubits.
On the noisy Torino model, P(optimum) is lower than random choice among valid plans. Only
25–65% of shots are valid plans.

## Notebooks

- `notebooks/shapefiles.ipynb`: the ward and district boundaries.
- `notebooks/ward_choice.ipynb`: selects the pilot ward with classical solvers.
- `notebooks/pilot.ipynb`: the full pipeline for one ward.

Start Jupyter with `uv run jupyter lab`.

## Method

### Problem

- **Demand nodes** `i`: Overture buildings grouped into 500 m cells. Only nodes further than
  `grid_distance_m` from the grid are kept. Demand is
  `d_i = buildings × households_per_building × kWh_per_household_day`.
- **Candidate sites** `j`: one qubit each. These are the `n_candidates` demand nodes that
  reach the most demand, at least `service_radius_m / 2` apart.
- **Decision**: `x_j ∈ {0,1}`. Build exactly `n_sites` microgrids.
- **Objective**: minimise `Σ c_j x_j − α · served(x)`. A node is served if a selected site is
  within `service_radius_m`.

QUBO form (`src/quackathon/problem.py`):

    E(x) = Σ_j (c_j − α D_j) x_j + α Σ_{j<k} O_jk x_j x_k + P (Σ_j x_j − K)²

`D_j` is the demand that site `j` reaches. `O_jk` is the demand that both `j` and `k` reach.
The QUBO is exact if no node is covered by three or more selected sites.
`classical.solve_exact` finds the true optimum for comparison.

### Ward selection

`notebooks/ward_choice.ipynb` and `src/quackathon/screening.py` select the pilot ward. They
solve each ward's candidate problem exactly. They also solve the problem with all demand
nodes as candidates as a MILP (`classical.solve_milp`, HiGHS). Then they check that the
ranking stays the same for different service radius and grid distance values. With the
default config, ward 4 serves the most demand (about 2× ward 19). Ward 1 is a close second.

### Quantum solver

`src/quackathon/quantum.py` runs QAOA on the Qiskit statevector simulator:

1. Convert the QUBO to an Ising Hamiltonian (`SparsePauliOp`) with `x_j = (1 − Z_j)/2`.
2. The **XY mixer** (default) starts from the Dicke state. This is an equal superposition of
   all K-site plans (Bärtschi & Eidenbenz circuit). It mixes with ring XY swaps. All samples
   are valid plans, so the cardinality penalty is not used. The **X mixer** (`--mixer x`) is
   textbook QAOA from `|+⟩ⁿ`. It uses the penalty to enforce K.
3. COBYLA tunes the angles with `StatevectorEstimator`. Each depth starts from the angles of
   the previous depth (INTERP). `StatevectorSampler` samples the tuned circuit.

### Results

Ward 14, 12 qubits, K = 3, depth 1–4:

|                     | XY mixer                | X mixer  |
|---------------------|-------------------------|----------|
| P(optimum)          | 0.9–1.2% (random 0.45%) | 0.0–0.2% |
| P(feasible)         | 100%                    | 2–28%    |
| Approximation ratio | 0.72–0.79               | < 0      |

With 8 qubits (`--candidates 8`), P(optimum) goes from 4.0% to 7.0% over depths 1–4.
Random choice gives 1.8%.

## Data sources

| Layer                    | Source                                                    | Limits                                                    |
|--------------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| Wards                    | MDB Wards 2026 (`shapefiles/`)                            | Newer boundaries than Census 2022                         |
| Buildings                | Overture Maps `2026-09-23.1` (Google, Microsoft, OSM)     | One homestead has several buildings. Calibrate `households_per_building` against the census |
| Grid (`gridfinder`)      | Arderne et al. 2020 predicted MV grid plus OSM lines. Clip in `data/raw/gridfinder_EC444.gpkg` | An estimate from night-time lights and roads (2020) |
| Grid (`osm`)             | OSM power lines from Overpass                             | Mostly 132 kV lines. The MV/LV network is not mapped, so almost all demand is "far from the grid" |

You need the global gridfinder file (`grid.gpkg`, 725 MB, https://zenodo.org/records/3628142)
only to rebuild the clip. Git ignores it.
