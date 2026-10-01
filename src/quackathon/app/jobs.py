"""Work the app runs in worker processes, and the run record the GUI keeps.

Nothing here imports dearpygui: worker processes are spawned and import this module, and
they should not pull in the GUI.
"""

import time
from dataclasses import dataclass, field

import pandas as pd

from quackathon.classical import Solution, solve_exact, solve_greedy, solve_qubo_brute_force
from quackathon.config import PilotConfig
from quackathon.pipeline import Pilot, build_pilot
from quackathon.problem import MicrogridProblem
from quackathon.quantum import QAOAResult, qaoa_depth_sweep

PROGRESS_EVERY = 10  # optimiser evaluations per progress message


@dataclass(frozen=True)
class QAOAOptions:
    reps: int = 3
    mixer: str = "xy"
    shots: int = 4096
    maxiter: int = 300
    restarts: int = 3
    seed: int = 0


@dataclass
class BuildResult:
    pilot: Pilot
    exact: Solution
    greedy: Solution
    qubo: Solution
    elapsed_s: float


def build_run(cfg: PilotConfig, buildings_source: str = "overture") -> BuildResult:
    """Stage 1: the ward's problem and its classical solutions (about a second)."""
    start = time.perf_counter()
    pilot = build_pilot(cfg, buildings_source=buildings_source)
    p = pilot.problem
    return BuildResult(pilot, solve_exact(p), solve_greedy(p), solve_qubo_brute_force(*p.to_qubo()),
                       time.perf_counter() - start)


def qaoa_run(problem: MicrogridProblem, opts: QAOAOptions, run_id: int, queue=None) -> list[QAOAResult]:
    """Stage 2: QAOA depth sweep 1..opts.reps, sending ("progress", run_id, depth, n_evals, energies)
    messages to `queue` with the energies evaluated since the last message."""
    pending: list[float] = []
    current = [1, 0]  # depth, evaluations of the energies in `pending`

    def flush() -> None:
        if pending:
            queue.put(("progress", run_id, current[0], current[1], pending.copy()))
            pending.clear()

    def report(depth: int, n_evals: int, energy: float) -> None:
        if depth != current[0]:
            flush()
        current[:] = depth, n_evals
        pending.append(energy)
        if len(pending) >= PROGRESS_EVERY:
            flush()

    sweep = qaoa_depth_sweep(problem, opts.reps, mixer=opts.mixer, shots=opts.shots, maxiter=opts.maxiter,
                             restarts=opts.restarts, seed=opts.seed,
                             callback=report if queue is not None else None)
    if queue is not None:
        flush()
    return sweep


def screen_run(cfg: PilotConfig) -> pd.DataFrame:
    """Every ward's candidate and full problem, solved classically (see `screening.screen_wards`)."""
    from quackathon.screening import Municipality, screen_wards

    return screen_wards(Municipality.load(cfg), cfg)


def allocate_run(cfg: PilotConfig, n_total: int) -> pd.DataFrame:
    """Best `n_total` sites anywhere in the municipality (see `screening.allocate_sites`)."""
    from quackathon.screening import Municipality, allocate_sites

    return allocate_sites(Municipality.load(cfg), cfg, n_total)


def load_context(cfg: PilotConfig) -> dict:
    """OSM roads and facilities for the municipality. The first call downloads them (minutes)."""
    from quackathon.osm import fetch_facilities, fetch_roads
    from quackathon.region import load_wards

    wards = load_wards(cfg)
    return {"roads": fetch_roads(wards, cfg.raw_dir), "facilities": fetch_facilities(wards, cfg.raw_dir)}


@dataclass
class Run:
    """One optimisation run, as the GUI tracks it."""
    id: int
    cfg: PilotConfig
    buildings_source: str
    opts: QAOAOptions | None
    status: str = "queued"          # queued, building, qaoa queued, qaoa, done, failed, cancelled
    build: BuildResult | None = None
    sweep: list[QAOAResult] = field(default_factory=list)
    # Live optimiser trace while QAOA runs: energies of every evaluation (all restarts) per depth.
    live: dict[int, list[float]] = field(default_factory=dict)
    error: str | None = None
    cancelled: bool = False
    started: float = field(default_factory=time.perf_counter)
    finished: float | None = None

    @property
    def problem(self) -> MicrogridProblem | None:
        return self.build.pilot.problem if self.build else None

    @property
    def elapsed_s(self) -> float:
        return (self.finished or time.perf_counter()) - self.started

    @property
    def progress(self) -> str:
        if self.status != "qaoa" or not self.live:
            return self.status
        depth = max(self.live)
        return f"p {depth}/{self.opts.reps}, eval {len(self.live[depth])}"
