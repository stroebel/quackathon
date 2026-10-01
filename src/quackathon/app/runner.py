"""Background jobs for the GUI: a process pool plus a progress queue, polled every frame.

Optimisation is CPU-bound Python (COBYLA driving a statevector simulator) that holds the
GIL, so it runs in separate processes rather than threads. The GUI never blocks on a job:
it calls `poll()` once per frame, which returns progress messages and finished jobs, and
does all Dear PyGui calls itself on the main thread.
"""

import multiprocessing as mp
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ProcessPoolExecutor
from queue import Empty


class JobRunner:
    def __init__(self, max_workers: int = 2):
        ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        self._manager = ctx.Manager()
        self.queue = self._manager.Queue()  # picklable proxy, so it can be passed to jobs
        self._jobs: dict[Future, Hashable] = {}

    def submit(self, key: Hashable, fn: Callable, *args) -> Future:
        future = self._pool.submit(fn, *args)
        self._jobs[future] = key
        return future

    def cancel(self, future: Future) -> bool:
        """Cancel a job that has not started yet. A running job cannot be stopped."""
        if future.cancel():
            self._jobs.pop(future, None)
            return True
        return False

    def poll(self, max_messages: int = 200) -> tuple[list[tuple], list[tuple[Hashable, Future]]]:
        """Progress messages so far and the jobs that finished since the last poll."""
        messages = []
        for _ in range(max_messages):
            try:
                messages.append(self.queue.get_nowait())
            except Empty:
                break
        done = [(key, f) for f, key in self._jobs.items() if f.done()]
        for _, f in done:
            del self._jobs[f]
        return messages, done

    @property
    def n_running(self) -> int:
        return sum(f.running() for f in self._jobs)

    @property
    def n_queued(self) -> int:
        return len(self._jobs) - self.n_running

    def shutdown(self) -> None:
        """Stop at once, killing jobs in progress (otherwise closing the app waits for them)."""
        self._pool.terminate_workers()
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._manager.shutdown()
