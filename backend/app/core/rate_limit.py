"""In-process rate limiting foundation.

Sliding-window counters per (API key, action), plus a non-blocking
concurrency gate for long-running operations. State lives in memory, so
limits apply per server process; a shared store (e.g. Redis) can replace
`RateLimiter` later without changing callers.
"""

import math
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Deque, Dict, Tuple

from backend.app.core.errors import TooManyRequests


class RateLimiter:
    def __init__(self):
        self._hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, subject: str, action: str, limit: int, window_s: float) -> None:
        """Record one request or raise TooManyRequests (HTTP 429 with Retry-After)."""
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(subject, action)]
            while hits and now - hits[0] >= window_s:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, math.ceil(window_s - (now - hits[0])))
                raise TooManyRequests(
                    f"Rate limit exceeded for {action}: {limit} per {int(window_s)}s. Retry in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)


class ConcurrencyGate:
    def __init__(self, limit: int, what: str):
        self._semaphore = threading.BoundedSemaphore(max(1, limit))
        self._what = what

    @contextmanager
    def slot(self):
        if not self._semaphore.acquire(blocking=False):
            raise TooManyRequests(f"Too many concurrent {self._what}; try again shortly",
                                  headers={"Retry-After": "5"})
        try:
            yield
        finally:
            self._semaphore.release()
