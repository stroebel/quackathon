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

## Run

    uv run quackathon --ward 14 --grid-distance 2000 --sites 3 --candidates 12 --radius 1500
    uv run pytest

The first run takes about 3 minutes: it pulls every building in the municipality
(about 312k) from Overture on S3 with DuckDB and caches them in `data/raw/`.

## Data sources and caveats

| Layer | Source | Caveat |
|---|---|---|
| Wards | MDB Wards 2026 (`shapefiles/`) | Newer boundaries than Census 2022 |
| Buildings | Overture Maps `2026-09-23.1` (Google, Microsoft, OSM) | Several structures per homestead; calibrate `households_per_building` against the census |
| Grid (`--grid-source osm`) | OSM power lines via Overpass | Mostly 132 kV transmission; the MV/LV distribution network is largely unmapped, so nearly everything counts as "far from the grid" |
| Grid (`--grid-source gridfinder`) | Arderne et al. 2020 predicted MV grid | Put `grid.gpkg` (724 MB) from https://zenodo.org/records/3628142 in `data/raw/` |
