"""
validation/pipeline.py: the Phase 4 validation pipeline.

Implements the exact pipeline shape from the project brief:
    prices -> diff_log -> rolling_mean -> zscore -> filter_gt -> mean

Two implementations sharing the same seeded data generator:
  - run_regresslens(): uses regresslens.array where the four v0.1
    primitives apply (rolling_mean, filter_gt, reduction-based mean),
    and honest NumPy fallback where they don't (log, diff, std).
  - run_numpy_baseline(): pure NumPy, no regresslens involvement at
    all, independently written (not derived from the accelerated
    version by find-replace) so it serves as real ground truth, not
    a copy that could share a bug.

DESIGN NOTE on diff_log and zscore: neither is one of the four v0.1
accelerated operations (projection, filter, reduction, rolling).
diff_log needs np.log + np.diff (no accelerated equivalent exists).
zscore needs mean AND std — mean is accelerated (reduction), std is
not (no variance/stddev kernel exists in v0.1). zscore's actual
elementwise step, (x - mean) / std, IS expressible as
project_affine(x, scale=1/std, offset=-mean/std) and IS accelerated.
This is a genuinely realistic partial-acceleration pipeline, not a
contrived one — most real pipelines mix supported and unsupported
operations, which is exactly what the project brief's honesty
requirement is about.
"""
import time

import numpy as np

import regresslens as rl


def generate_prices(n=1_000_000, seed=42):
    """Synthetic price series: a random walk with positive drift,
    always strictly positive (required for log)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0002, scale=0.01, size=n)
    log_prices = np.cumsum(log_returns) + np.log(100.0)
    return np.exp(log_prices).astype(np.float64)


def run_numpy_baseline(prices, window=20, filter_threshold=0.5):
    """Pure NumPy. No regresslens involvement. This is the ground
    truth run_regresslens() is checked against every single run, not
    just once — per the project brief's correctness-first
    requirement."""
    returns = np.diff(np.log(prices))
    rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
    mean = np.mean(rolling)
    std = np.std(rolling)
    zscore = (rolling - mean) / std
    signals = zscore[zscore > filter_threshold]
    if len(signals) == 0:
        return 0.0
    factor = np.mean(signals)
    return factor


def run_regresslens(prices, window=20, filter_threshold=0.5, timing=None):
    """RegressLens-accelerated version. `timing`, if given a dict, is
    populated with wall-clock durations for each stage — including
    the UNACCELERATED stages (log, diff, std), which are NOT captured
    by regresslens.trace since they never call into a regresslens.array
    method at all. Without this manual instrumentation, the
    'intercepted fraction vs fallback fraction' report would only see
    the accelerated portion and silently miss the rest of the
    pipeline's real cost.
    """
    def _record(key, t0):
        if timing is not None:
            timing[key] = time.perf_counter_ns() - t0

    # --- diff_log: NOT accelerated, no regresslens involvement ---
    t0 = time.perf_counter_ns()
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    _record("diff_log_ns", t0)

    # returns is a fresh contiguous array from np.diff — safe to wrap.
    returns_arr = rl.array(returns)

    # --- rolling_mean: ACCELERATED (v0.1 primitive) ---
    t0 = time.perf_counter_ns()
    rolling_arr = returns_arr.rolling_mean(window)
    _record("rolling_mean_ns", t0)

    # --- zscore: mean is ACCELERATED (reduction), std is NOT ---
    t0 = time.perf_counter_ns()
    mean = rolling_arr.mean()
    _record("zscore_mean_ns", t0)

    t0 = time.perf_counter_ns()
    std = float(np.std(rolling_arr.to_numpy()))
    _record("zscore_std_ns", t0)

    t0 = time.perf_counter_ns()
    zscore_arr = rolling_arr.project_affine(1.0 / std, -mean / std)
    _record("zscore_affine_ns", t0)

    # --- filter_gt: ACCELERATED ---
    t0 = time.perf_counter_ns()
    signals_arr = zscore_arr.filter_gt(filter_threshold)
    _record("filter_gt_ns", t0)

    if len(signals_arr) == 0:
        return 0.0

    # --- mean: ACCELERATED (reduction) ---
    t0 = time.perf_counter_ns()
    factor = signals_arr.mean()
    _record("final_mean_ns", t0)

    return factor
