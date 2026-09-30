import argparse

from quackathon.config import PilotConfig


def main() -> None:
    from quackathon.classical import solve_exact, solve_qubo_brute_force
    from quackathon.pipeline import build_pilot

    defaults = PilotConfig()
    parser = argparse.ArgumentParser(description="Microgrid siting pilot for one Ntabankulu ward")
    parser.add_argument("--ward", type=int, default=defaults.ward_no)
    parser.add_argument("--grid-distance", type=float, default=defaults.grid_distance_m, help="metres")
    parser.add_argument("--grid-source", choices=["osm", "gridfinder"], default=defaults.grid_source)
    parser.add_argument("--sites", type=int, default=defaults.n_sites, help="microgrids to build")
    parser.add_argument("--candidates", type=int, default=defaults.n_candidates, help="qubits")
    parser.add_argument("--radius", type=float, default=defaults.service_radius_m, help="metres")
    parser.add_argument("--buildings", choices=["overture", "osm", "synthetic"], default="overture")
    args = parser.parse_args()

    cfg = PilotConfig(
        ward_no=args.ward,
        grid_distance_m=args.grid_distance,
        grid_source=args.grid_source,
        n_sites=args.sites,
        n_candidates=args.candidates,
        service_radius_m=args.radius,
    )
    pilot = build_pilot(cfg, buildings_source=args.buildings)
    p = pilot.problem

    print(f"Ward {cfg.ward_no}: {len(pilot.buildings)} buildings, {len(pilot.grid)} grid lines ({cfg.grid_source})")
    print(f"Demand nodes: {len(pilot.all_nodes)} total, {len(pilot.nodes)} beyond {cfg.grid_distance_m:.0f} m "
          f"of grid ({p.demand.sum():.1f} kWh/day)")
    print(f"Choose {p.n_select} of {p.n_sites} candidate sites, service radius {cfg.service_radius_m:.0f} m")

    exact = solve_exact(p)
    q, offset = p.to_qubo()
    qubo = solve_qubo_brute_force(q, offset)
    for name, sol in [("exact", exact), ("QUBO", qubo)]:
        print(f"  {name:5s} x={''.join(map(str, sol.x))} served={p.served_demand(sol.x):.1f} kWh/day "
              f"feasible={p.is_feasible(sol.x)}")
