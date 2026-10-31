"""Drift canaries: two fixed reference workloads timed around every run.

  compute  repeated np.exp over a 4096-element float32 array that stays
           in L1. Tracks effective core frequency (turbo, throttling,
           frequency scaling).
  memory   a streaming sum over a buffer of at least 4x the last-level
           cache, capped at 512 MiB. Tracks memory bandwidth and
           contention (other processes, noisy neighbors).

Each measurement is the minimum of 5 repetitions (the least-disturbed
observation). Both use only NumPy, so they work without the native
kernels. If the canaries slow down between baseline and current while
the code is unchanged, the environment changed, not the code; see the
drift-bounded correction in stats/drift.py.

Environment:
  RGLNS_CANARY=0               disable canaries (no drift correction)
  RGLNS_CANARY_MEM_BYTES=N     override the memory buffer size (tests)
"""
import os
import time

from . import hw

REPS = 5
COMPUTE_ELEMENTS = 4096
COMPUTE_ITERATIONS = 64
MIN_MEM_BYTES = 16 * 1024 * 1024
MAX_MEM_BYTES = 512 * 1024 * 1024
# Long runs take an extra sample this often (seconds).
PERIOD_S = 30.0


def enabled():
    return os.environ.get("RGLNS_CANARY", "1") != "0"


def mem_bytes():
    override = os.environ.get("RGLNS_CANARY_MEM_BYTES")
    if override:
        return max(1024 * 1024, int(override))
    return max(MIN_MEM_BYTES, min(MAX_MEM_BYTES, 4 * hw.cache_sizes()["l3"]))


class Canary:
    def __init__(self, buffer_bytes=None):
        import numpy as np
        self._np = np
        self._x = np.linspace(-1.0, 1.0, COMPUTE_ELEMENTS, dtype=np.float32)
        self._y = np.empty_like(self._x)
        self._buffer_bytes = buffer_bytes or mem_bytes()
        self._buf = None

    def compute_ns(self):
        np, x, y = self._np, self._x, self._y
        best = None
        for _ in range(REPS):
            t0 = time.perf_counter_ns()
            for _ in range(COMPUTE_ITERATIONS):
                np.exp(x, out=y)
            dt = time.perf_counter_ns() - t0
            best = dt if best is None or dt < best else best
        return float(best)

    def mem_ns(self):
        np = self._np
        if self._buf is None:
            # ones(), not empty(): the pages must be really mapped, or
            # the first reps would measure page faults.
            self._buf = np.ones(self._buffer_bytes // 8, dtype=np.float64)
        buf = self._buf
        best = None
        for _ in range(REPS):
            t0 = time.perf_counter_ns()
            buf.sum()
            dt = time.perf_counter_ns() - t0
            best = dt if best is None or dt < best else best
        return float(best)

    def measure(self, where):
        """One sample: {"compute_ns", "mem_ns", "where", "t", "cost_ns"}.
        cost_ns is how long the measurement itself took, so a process can
        subtract its own canary overhead from wall time."""
        t0 = time.perf_counter_ns()
        sample = {"compute_ns": self.compute_ns(), "mem_ns": self.mem_ns(),
                  "where": where, "t": time.time()}
        sample["cost_ns"] = float(time.perf_counter_ns() - t0)
        return sample


_shared = None


def shared():
    """Process-wide canary (allocates the memory buffer once)."""
    global _shared
    if _shared is None:
        _shared = Canary()
    return _shared
