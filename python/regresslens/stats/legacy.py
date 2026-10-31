"""The v0.1 check engine, kept only so validation can run old and new
engines on identical data (docs/validation.md).

v0.1 pooled every individual call from every run of a session, resampled
calls independently in a percentile bootstrap of the ratio of means,
applied Bonferroni across operations, and flagged a regression when the
interval's lower bound exceeded the threshold. This is a vectorized
port of exactly that algorithm (same statistic, same interval, same
decision). It is not used by any command.
"""
import numpy as np


def ci_relative_slowdown(baseline, current, B=2000, ci=0.95, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    point = (c.mean() - b.mean()) / b.mean()
    rb = b[rng.integers(0, len(b), size=(B, len(b)))].mean(axis=1)
    rc = c[rng.integers(0, len(c), size=(B, len(c)))].mean(axis=1)
    reps = np.sort((rc - rb) / rb)
    a = 1 - ci
    lo = reps[int(B * (a / 2))]
    hi = reps[min(int(B * (1 - a / 2)), B - 1)]
    return point, lo, hi


def evaluate(baseline, current, threshold=0.05, num_comparisons=1, B=2000, rng=None):
    """v0.1 decision: 'regression' when the Bonferroni-widened interval's
    lower bound exceeds the practical threshold, else 'within_noise'."""
    corrected = 1 - 0.05 / max(num_comparisons, 1)
    point, lo, hi = ci_relative_slowdown(baseline, current, B, corrected, rng)
    return ("regression" if lo > threshold else "within_noise"), point, lo, hi
