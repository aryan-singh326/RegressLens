"""Fifth-kernel gate (WP8): is a native `diff` kernel worth building?

diff_log is log followed by diff. A native diff kernel is only worth
building if diff is a large share of diff_log (gate: more than 30%).
This measures, on 1M float64 values:

  raw share       np.diff / (np.diff(np.log(x)))     the plan's gate
  compute share   subtraction into a preallocated output / diff_log

np.diff allocates a fresh output, and on some platforms page-faulting
that memory is most of its cost. A native kernel must allocate the same
output, so only the compute share is something a kernel could
remove. Both are reported; the decision uses the compute share.

    python tools/kernel_gate.py
"""
import platform
import statistics
import sys
import time

import numpy as np

GATE = 0.30


def median_ns(fn, reps=60):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples)


def main():
    x = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(2e-4, 1e-2, 1_000_000)))
    lx = np.log(x)
    out = np.empty(lx.size - 1)
    raw, compute = [], []
    for _ in range(9):
        total = median_ns(lambda: np.diff(np.log(x)))
        raw.append(median_ns(lambda: np.diff(lx)) / total)
        compute.append(median_ns(lambda: np.subtract(lx[1:], lx[:-1], out=out)) / total)
    raw_share, compute_share = statistics.median(raw), statistics.median(compute)
    print(f"{platform.system()} {platform.machine()}, NumPy {np.__version__}")
    print(f"  raw diff share of diff_log:     {raw_share:.1%} (plan gate: > {GATE:.0%})")
    print(f"  compute share (no allocation):  {compute_share:.1%}")
    build = compute_share > GATE
    print(f"  decision: {'BUILD' if build else 'do not build'} a native diff kernel")
    if raw_share > GATE >= compute_share:
        print("  (the raw share passes only because of output allocation, which a native "
              "kernel would pay too)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
