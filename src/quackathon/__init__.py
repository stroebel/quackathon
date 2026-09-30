import argparse

from quackathon.config import PilotConfig


def main() -> None:
    from quackathon.classical import solve_exact, solve_qubo_brute_force
    from quackathon.pipeline import build_pilot
    from quackathon.quantum import solve_qaoa

    defaults = PilotConfig()
    parser = argparse.ArgumentParser(description="Microgrid siting pilot for one Ntabankulu ward")
    parser.add_argument("--ward", type=int, default=defaults.ward_no)
    parser.add_argument("--grid-distance", type=float, default=defaults.grid_distance_m, help="metres")
    parser.add_argument("--grid-source", choices=["gridfinder", "osm"], default=defaults.grid_source)
    parser.add_argument("--sites", type=int, default=defaults.n_sites, help="microgrids to build")
    parser.add_argument("--candidates", type=int, default=defaults.n_candidates, help="qubits")
    parser.add_argument("--radius", type=float, default=defaults.service_radius_m, help="metres")
    parser.add_argument("--buildings", choices=["overture", "osm", "synthetic"], default="overture")
    parser.add_argument("--qaoa-reps", type=int, default=3, help="QAOA depth (0 to skip)")
    parser.add_argument("--mixer", choices=["xy", "x"], default="xy")
    parser.add_argument("--backend", default=None,
                        help="also sample the tuned circuit on IBM hardware: a device name, "
                             "'least_busy', or 'fake_<device>' for a local noisy dry run")
    parser.add_argument("--shots", type=int, default=4096)
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

    if args.qaoa_reps > 0:
        res = solve_qaoa(p, reps=args.qaoa_reps, mixer=args.mixer, shots=args.shots)
        print(f"  QAOA  x={''.join(map(str, res.x))} served={p.served_demand(res.x):.1f} kWh/day "
              f"(best of {res.shots} shots, {args.mixer} mixer, p={res.reps})")
        print(f"        P(optimum)={res.p_ground:.3f} vs random {res.p_ground_random:.3f}, "
              f"P(feasible)={res.p_feasible:.2f}, approximation ratio={res.approximation_ratio:.2f}")

        if args.backend:
            from quackathon.hardware import run_on_backend

            hw = run_on_backend(p, res, args.backend, shots=args.shots)
            best = f"served={p.served_demand(hw.x):.1f} kWh/day" if hw.x is not None else "no feasible sample"
            print(f"  {hw.backend}: {best}  P(optimum)={hw.p_ground:.3f}  P(feasible)={hw.p_feasible:.2f}  "
                  f"({hw.two_qubit_gates} two-qubit gates, depth {hw.depth}, job {hw.job_id})")
