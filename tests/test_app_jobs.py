import time

import numpy as np
import pytest

from quackathon.app.jobs import QAOAOptions, build_run, qaoa_run
from quackathon.app.runner import JobRunner
from quackathon.classical import solve_exact
from quackathon.config import PilotConfig
from quackathon.quantum import qaoa_depth_sweep
from tests.test_problem import random_problem


class ListQueue(list):
    put = list.append


def test_sweep_callback_reports_every_evaluation_by_depth():
    calls = []
    qaoa_depth_sweep(random_problem(n_sites=5, k=2), 2, maxiter=20, restarts=2,
                     callback=lambda depth, n, e: calls.append((depth, n, e)))
    depths = [d for d, _, _ in calls]
    assert depths == sorted(depths) and set(depths) == {1, 2}
    for depth in (1, 2):
        assert [n for d, n, _ in calls if d == depth] == list(range(1, depths.count(depth) + 1))


def test_qaoa_run_sends_every_energy_once_without_mixing_depths():
    calls, queue = [], ListQueue()
    p = random_problem(n_sites=5, k=2)
    opts = QAOAOptions(reps=2, maxiter=20, restarts=2)
    qaoa_depth_sweep(p, 2, maxiter=20, restarts=2, callback=lambda d, n, e: calls.append((d, e)))
    sweep = qaoa_run(p, opts, run_id=7, queue=queue)
    assert all(kind == "progress" and run_id == 7 for kind, run_id, *_ in queue)
    received = [(depth, e) for _, _, depth, _, energies in queue for e in energies]
    assert received == calls  # same seed, so the same evaluations
    assert [r.reps for r in sweep] == [1, 2]


@pytest.mark.skipif(not (PilotConfig().shapefile_dir / "MDBWards2026").exists(), reason="ward shapefile missing")
def test_build_run_matches_exact_solver():
    result = build_run(PilotConfig(ward_no=4, n_candidates=6, n_sites=2), buildings_source="synthetic")
    p = result.pilot.problem
    assert p.n_sites == 6
    assert np.array_equal(result.exact.x, solve_exact(p).x)
    assert p.served_demand(result.greedy.x) <= p.served_demand(result.exact.x)


def test_runner_runs_qaoa_in_a_worker_process_with_progress():
    runner = JobRunner(max_workers=1)
    try:
        p = random_problem(n_sites=6, k=2)
        runner.submit("qaoa", qaoa_run, p, QAOAOptions(reps=1, maxiter=30), 1, runner.queue)
        messages, done, deadline = [], [], time.time() + 120
        while not done and time.time() < deadline:
            new, done = runner.poll()
            messages += new
            time.sleep(0.05)
        messages += runner.poll()[0]
        (key, future), = done
        assert key == "qaoa"
        assert p.is_feasible(future.result()[0].x)
        assert messages and all(m[:2] == ("progress", 1) for m in messages)
    finally:
        runner.shutdown()
