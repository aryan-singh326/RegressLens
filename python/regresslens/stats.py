"""
regresslens.stats: regression detection statistics.

Per the project brief: bootstrap confidence intervals on relative
slowdown, minimum sample size before reporting, a practical effect
threshold (not just statistical significance), and Bonferroni
correction across multiple operations compared in one report.

METHODOLOGICAL NOTE — read before trusting this for anything real:
The brief specifies INTERLEAVED baseline/current measurement pairs
within a single session, to cancel out thermal throttling and
background-activity drift. This module does NOT do that. `rglns
baseline` captures N runs at one point in time; `rglns check` captures
N NEW runs at a later point in time (possibly a different session,
different day, different thermal state) and compares against the
STORED baseline samples. True interleaving would require re-executing
the exact baseline code path at check time, which needs the baseline's
source available and runnable — that's git-integration territory, out
of scope for v0.1. This is a real, documented limitation, not
something to discover the hard way from a false regression report.
"""
import random


class InsufficientDataError(Exception):
    pass


def bootstrap_ci_relative_slowdown(baseline_samples, current_samples,
                                    n_bootstrap=10_000, ci=0.95, seed=None):
    """
    Returns (point_estimate, ci_low, ci_high) for the relative
    slowdown of current vs baseline: (mean(current) - mean(baseline))
    / mean(baseline). Positive means current is SLOWER.

    Uses a standard two-sample bootstrap (independently resample each
    group with replacement, recompute the statistic, repeat) — NOT
    the interleaved-pairs design the brief specifies; see module
    docstring.
    """
    if len(baseline_samples) < 2 or len(current_samples) < 2:
        raise InsufficientDataError(
            f"need at least 2 samples per group, got "
            f"baseline={len(baseline_samples)} current={len(current_samples)}"
        )

    rng = random.Random(seed)
    n_b, n_c = len(baseline_samples), len(current_samples)

    def relative_slowdown(b, c):
        mean_b = sum(b) / len(b)
        mean_c = sum(c) / len(c)
        if mean_b == 0:
            return 0.0
        return (mean_c - mean_b) / mean_b

    point_estimate = relative_slowdown(baseline_samples, current_samples)

    resampled = []
    for _ in range(n_bootstrap):
        b_resample = [baseline_samples[rng.randrange(n_b)] for _ in range(n_b)]
        c_resample = [current_samples[rng.randrange(n_c)] for _ in range(n_c)]
        resampled.append(relative_slowdown(b_resample, c_resample))

    resampled.sort()
    alpha = 1 - ci
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    ci_low = resampled[lo_idx]
    ci_high = resampled[min(hi_idx, n_bootstrap - 1)]

    return point_estimate, ci_low, ci_high


def evaluate_regression(baseline_samples, current_samples, min_pairs=20,
                         practical_threshold=0.05, num_comparisons=1,
                         n_bootstrap=10_000, seed=None):
    """
    Evaluates one operation's regression status. Returns a dict:
      status: "insufficient_data" | "regression" | "within_noise"
      point_estimate, ci_low, ci_high: relative slowdown (or None if
        insufficient_data)
      n_baseline, n_current: sample counts

    Per the project brief:
    - Below min_pairs samples, reports insufficient_data rather than
      treating noise as signal.
    - A practical_threshold (default 5%) means statistically
      detectable but operationally meaningless slowdowns aren't
      flagged as regressions — the CI's lower bound must exceed the
      threshold, not just be positive.
    - Bonferroni correction (num_comparisons) widens the effective CI
      when multiple operations are being checked in the same report,
      to control the family-wise false positive rate.
    """
    n_baseline, n_current = len(baseline_samples), len(current_samples)
    if n_baseline < min_pairs or n_current < min_pairs:
        return {
            "status": "insufficient_data",
            "point_estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n_baseline": n_baseline,
            "n_current": n_current,
        }

    # Bonferroni: divide alpha by the number of comparisons, i.e.
    # widen the confidence level per comparison so the FAMILY-WISE
    # error rate across all operations stays at the nominal 5%.
    corrected_ci = 1 - (0.05 / max(num_comparisons, 1))

    point_estimate, ci_low, ci_high = bootstrap_ci_relative_slowdown(
        baseline_samples, current_samples, n_bootstrap=n_bootstrap,
        ci=corrected_ci, seed=seed,
    )

    # A regression requires the ENTIRE CI to be above the practical
    # threshold — not just the point estimate. This is what makes the
    # test conservative: noisy-but-plausibly-fine results don't get
    # flagged just because the point estimate happens to land above
    # threshold.
    is_regression = ci_low > practical_threshold

    return {
        "status": "regression" if is_regression else "within_noise",
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_baseline": n_baseline,
        "n_current": n_current,
    }
