"""Background execution for builds.

A bounded in-process thread pool: `max_workers` builds run at once and at
most `max_pending` builds (running + waiting) are accepted. Builds are
CPU-light but slow (LLM calls, subprocess test runs), so threads are enough.

The interface is just submit()/has_capacity()/shutdown(), so this can later
be replaced by an external worker queue without touching callers. State is
in memory: jobs do not survive a restart (interrupted builds are marked
failed on the next startup; see SqlBuildStore.fail_interrupted).
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


class JobManager:
    def __init__(self, max_workers: int, max_pending: int):
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="metaagent-build")
        self._max_pending = max(1, max_pending)
        self._pending = 0
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def has_capacity(self) -> bool:
        return self.pending < self._max_pending

    def submit(self, name: str, fn: Callable[[], None]) -> None:
        with self._lock:
            if self._pending >= self._max_pending:
                raise QueueFull(f"{self._pending} builds already queued or running")
            self._pending += 1
        try:
            self._executor.submit(self._run, name, fn)
        except RuntimeError:  # executor already shut down
            with self._lock:
                self._pending -= 1
            raise QueueFull("server is shutting down")

    def _run(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("Background job %s crashed", name)
        finally:
            with self._lock:
                self._pending -= 1

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
