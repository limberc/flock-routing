from __future__ import annotations

import threading
from typing import Literal


class StatsTracker:
    """Thread-safe tracker for local vs. remote routing decisions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local: int = 0
        self._remote: int = 0

    def record(self, destination: Literal["local", "remote"]) -> None:
        """Record that a request was routed to *destination*."""
        if destination not in ("local", "remote"):
            raise ValueError(f"Invalid destination '{destination}'. Must be 'local' or 'remote'.")
        with self._lock:
            if destination == "local":
                self._local += 1
            else:
                self._remote += 1

    def snapshot(self) -> dict[str, int | float]:
        """Return a point-in-time copy of all counters."""
        with self._lock:
            total = self._local + self._remote
            local_rate = self._local / total if total > 0 else 0.0
            return {
                "local": self._local,
                "remote": self._remote,
                "total": total,
                "local_rate": local_rate,
            }
