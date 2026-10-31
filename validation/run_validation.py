"""The Phase 4 validation harness: RegressLens vs pure NumPy on the
validation pipeline (prices -> diff_log -> rolling_mean -> zscore ->
filter_gt -> mean).

Reports, over 100 executions: output correctness on EVERY run (not
sampled), absolute runtime, a run-level BCa interval for RegressLens vs
NumPy (the same engine as `rglns check`), the per-stage breakdown with
the accelerated vs NumPy-fallback share, and the kernels chosen per
call site.

There is no "vs hand-tuned heuristic" comparison: only one selection
policy exists (the characterization-derived heuristic), so it is
reported as N/A rather than omitted.

    python validation/run_validation.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import pipeline  # noqa: E402
from regresslens import observe, store  # noqa: E402
from regresslens.stats import bootstrap  # noqa: E402

N_RUNS = 100
ARRAY_SIZE = 1_000_000
WINDOW = 20
FILTER_THRESHOLD = 0.5
STAGES = ["diff_log_ns", "rolling_mean_ns", "zscore_mean_ns", "zscore_std_ns",
          "zscore_affine_ns", "filter_gt_ns", "final_mean_ns"]
ACCELERATED = {"rolling_mean_ns", "zscore_mean_ns", "zscore_affine_ns", "filter_gt_ns",
               "final_mean_ns"}


def main():
    print(f"RegressLens validation pipeline: n={ARRAY_SIZE}, window={WINDOW}, runs={N_RUNS}")
    failures, numpy_ns, rl_ns, stages = [], [], [], []
    for i in range(N_RUNS):
        prices = pipeline.generate_prices(n=ARRAY_SIZE, seed=i)
        t0 = time.perf_counter_ns()
        ref = pipeline.run_numpy_baseline(prices, WINDOW, FILTER_THRESHOLD)
        numpy_ns.append(time.perf_counter_ns() - t0)
        timing = {}
        t0 = time.perf_counter_ns()
        got = pipeline.run_regresslens(prices, WINDOW, FILTER_THRESHOLD, timing=timing)
        rl_ns.append(time.perf_counter_ns() - t0)
        stages.append(timing)
        if not np.isclose(ref, got, rtol=1e-9, atol=1e-12):
            failures.append((i, ref, got))

    print("\nCORRECTNESS")
    if failures:
        print(f"  FAILED: {len(failures)}/{N_RUNS} runs differ from NumPy")
        for i, a, b in failures[:5]:
            print(f"    run {i}: numpy={a!r} regresslens={b!r}")
    else:
        print(f"  all {N_RUNS}/{N_RUNS} runs match NumPy (rtol 1e-9)")

    print("\nRUNTIME")
    print(f"  NumPy median:       {np.median(numpy_ns) / 1e6:.3f} ms")
    print(f"  RegressLens median: {np.median(rl_ns) / 1e6:.3f} ms")
    cmp = bootstrap.compare(bootstrap.Side([np.log(numpy_ns)]), bootstrap.Side([np.log(rl_ns)]),
                            theta=0.0, B=4000, rng=np.random.default_rng(0))
    pct = lambda v: f"{math.expm1(v) * 100:+.1f}%"
    print(f"  RegressLens vs NumPy: {pct(cmp.estimate)} [{pct(cmp.raw_ci_low)}, "
          f"{pct(cmp.raw_ci_high)}] (run-level BCa, 95%)")
    print("  vs hand-tuned heuristic: N/A (one selection policy in this build)")

    print("\nSTAGES (median per run)")
    acc_total = fb_total = 0
    for stage in STAGES:
        values = [t[stage] for t in stages if stage in t]
        if not values:
            continue
        tag = "accelerated" if stage in ACCELERATED else "NumPy (not accelerated)"
        print(f"  {stage:18s} {np.median(values) / 1e6:8.3f} ms  {tag}")
        if stage in ACCELERATED:
            acc_total += sum(values)
        else:
            fb_total += sum(values)
    total = acc_total + fb_total
    print(f"  accelerated share of pipeline time: {100 * acc_total / total:.1f}%")

    print("\nKERNELS BY CALL SITE")
    agg = observe.get_aggregator()
    observe.flush()
    conn = store.connect()
    rows = conn.execute(
        "SELECT s.path, s.line, c.operator, c.dtype, c.kernel, SUM(c.count), MIN(c.n), MAX(c.n) "
        "FROM calls c JOIN call_sites s USING (call_site_id) WHERE c.run_id = ? "
        "GROUP BY s.path, s.line, c.operator, c.dtype, c.kernel ORDER BY s.line",
        (agg.run_id,)).fetchall()
    for path, line, op, dtype, kernel, count, nmin, nmax in rows:
        n = f"n={nmin}" if nmin == nmax else f"n={nmin}..{nmax}"
        print(f"  {path}:{line:<4} {op:10s} {dtype} {n}: {kernel} x{count}")
    if cmp.raw_ci_low > 0:
        print("\nRegressLens was SLOWER than NumPy on this pipeline; see the stage table for "
              "where (per-call overhead is fixed, so it matters most for small arrays).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
