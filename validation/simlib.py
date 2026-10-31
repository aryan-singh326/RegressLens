"""Simulation library for the statistical validation (docs/validation.md).

The same functions back the fast unit tests (few trials) and the full
validation runs (thousands of trials):

  WP3  check engine: false-regression rate, CI coverage, BH vs Bonferroni,
       canary drift, size normalization; the v0.1 engine on the same data
  WP4  compare: always-valid confidence sequence vs naive repeated looks
  WP5  bisect: probabilistic bisection vs fixed-sample baselines
  WP6  watch: BOCPD false alarms and detection delay
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from regresslens.stats import bocpd, bootstrap, fdr, legacy, verdict  # noqa: E402
from regresslens.stats.bisection import Bisector  # noqa: E402
from regresslens.stats.sequential import ConfidenceSequence, halfwidth  # noqa: E402

# --- WP3: check engine -------------------------------------------------------

CALLS_PER_RUN = 20
SIGMA_CALL = 0.10  # per-call log noise within a run


def simulate_side(rng, runs, sigma_run, effect, sessions=1, sigma_session=0.0,
                  canary_shift=None):
    """One side: `sessions` sessions of `runs` runs, each run with
    CALLS_PER_RUN calls. Returns (Side of run-level log medians, pooled
    raw call times for the v0.1 engine). `canary_shift=(c, m)` adds a
    machine slowdown seen by the canaries; the workload slows by the
    memory share (a memory-bound operation)."""
    values, log_c, log_m, raw = [], [], [], []
    base = math.log(1e6)
    for _ in range(sessions):
        s_eff = rng.normal(0, sigma_session)
        run_eff = rng.normal(0, sigma_run, runs) + s_eff
        mach_c, mach_m = (canary_shift or (0.0, 0.0))
        calls = base + run_eff[:, None] + rng.normal(0, SIGMA_CALL, (runs, CALLS_PER_RUN))
        calls += math.log1p(effect) + mach_m
        values.append(np.median(calls, axis=1))
        noise = 0.01
        log_c.append(math.log(1e5) + mach_c + rng.normal(0, noise, runs))
        log_m.append(math.log(2e7) + mach_m + rng.normal(0, noise, runs))
        raw.append(np.exp(calls).ravel())
    return bootstrap.Side(values, log_c, log_m), np.concatenate(raw)


def check_cell(sigma_run, effect, runs, trials, theta=0.05, alpha=0.05, B=1000, seed=0,
               legacy_engine=True):
    """New vs v0.1 engine on identical simulated data (single session per
    side, no drift). Returns rates."""
    rng = np.random.default_rng(seed)
    theta_log = math.log1p(theta)
    new_reg = new_cov = old_reg = old_cov = 0
    for _ in range(trials):
        b, b_raw = simulate_side(rng, runs, sigma_run, 0.0)
        c, c_raw = simulate_side(rng, runs, sigma_run, effect)
        r = bootstrap.compare(b, c, theta, alpha, B, rng)
        v = verdict.assign([r], theta, alpha)[0]
        new_reg += v.verdict == verdict.REGRESSION
        true_log = math.log1p(effect)
        new_cov += r.raw_ci_low <= true_log <= r.raw_ci_high
        if legacy_engine:
            dec, _p, lo, hi = legacy.evaluate(b_raw, c_raw, theta, 1, B, rng)
            old_reg += dec == "regression"
            old_cov += lo <= effect <= hi
    return {"new_regression": new_reg / trials, "new_coverage": new_cov / trials,
            "old_regression": old_reg / trials if legacy_engine else None,
            "old_coverage": old_cov / trials if legacy_engine else None,
            "is_null": effect <= theta, "theta_log": theta_log}


def canary_cell(trials, mem_slowdown=0.15, compute_slowdown=0.0, runs=10, sigma_run=0.02,
                theta=0.05, B=1000, seed=0):
    """Current session on a machine slowed by a bandwidth hog: the
    workload and the memory canary both slow by `mem_slowdown`, code
    unchanged. Returns the fraction of verdicts per class (new engine)
    and the v0.1 engine's regression rate."""
    rng = np.random.default_rng(seed)
    shift = (math.log1p(compute_slowdown), math.log1p(mem_slowdown))
    counts = {}
    old_reg = 0
    for _ in range(trials):
        b, b_raw = simulate_side(rng, runs, sigma_run, 0.0)
        c, c_raw = simulate_side(rng, runs, sigma_run, 0.0, canary_shift=shift)
        r = bootstrap.compare(b, c, theta, 0.05, B, rng)
        v = verdict.assign([r], theta)[0].verdict
        counts[v] = counts.get(v, 0) + 1
        old_reg += legacy.evaluate(b_raw, c_raw, theta, 1, B, rng)[0] == "regression"
    return {k: n / trials for k, n in counts.items()}, old_reg / trials


def bh_trial(rng, m=200, k=10, effect=0.10, sigma=0.03, runs=20, theta=0.05, alpha=0.05,
             B=600):
    """One family of m operations, k with a true effect. Returns
    (bh_false, bh_true, bonf_true, discoveries)."""
    pvals, truth = [], []
    for j in range(m):
        eff = effect if j < k else 0.0
        b = bootstrap.Side([rng.normal(0, sigma, runs)])
        c = bootstrap.Side([rng.normal(math.log1p(eff), sigma, runs)])
        pvals.append(bootstrap.compare(b, c, theta, alpha, B, rng, use_canaries=False).p_value)
        truth.append(j < k)
    truth = np.array(truth)
    rej_bh = fdr.benjamini_hochberg(pvals, alpha)
    rej_bf = fdr.bonferroni(pvals, alpha)
    return (int((rej_bh & ~truth).sum()), int((rej_bh & truth).sum()),
            int((rej_bf & truth).sum()), int(rej_bh.sum()))


def bh_experiment(trials, seed=0, **kw):
    rng = np.random.default_rng(seed)
    fdp, power_bh, power_bf = [], [], []
    k = kw.get("k", 10)
    for _ in range(trials):
        false, true, bf_true, disc = bh_trial(rng, **kw)
        fdp.append(false / max(disc, 1))
        power_bh.append(true / k)
        power_bf.append(bf_true / k)
    return {"fdr": float(np.mean(fdp)), "power_bh": float(np.mean(power_bh)),
            "power_bonferroni": float(np.mean(power_bf))}


# --- WP4: sequential compare ---------------------------------------------------

def sequential_trial(rng, effect, sigma, theta=0.05, alpha=0.05, max_blocks=100,
                     min_blocks=4):
    """Block differences d ~ N(log(1 + effect), sigma^2). Returns
    (decision, blocks used)."""
    cs = ConfidenceSequence(theta, alpha, None, min_blocks, max_blocks)
    mu = math.log1p(effect)
    for n in range(1, max_blocks + 1):
        cs.add(rng.normal(mu, sigma))
        d = cs.decision()
        if d is not None:
            return d, n
    return verdict.INCONCLUSIVE, max_blocks


def naive_trial(rng, effect, sigma, theta=0.05, alpha=0.05, max_blocks=100, min_blocks=4,
                B=400):
    """The tempting alternative: a fixed-n bootstrap CI recomputed after
    every block, stopping as soon as it excludes the boundaries."""
    theta_log = math.log1p(theta)
    mu = math.log1p(effect)
    d = []
    for n in range(1, max_blocks + 1):
        d.append(rng.normal(mu, sigma))
        if n < max(min_blocks, 3):
            continue
        arr = np.array(d)
        means = arr[rng.integers(0, n, (B, n))].mean(axis=1)
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        if lo > theta_log:
            return verdict.REGRESSION, n
        if hi < -theta_log:
            return verdict.IMPROVEMENT, n
    return verdict.INCONCLUSIVE, max_blocks


def sequential_null(trials, sigma, theta=0.05, seed=0, naive=False):
    rng = np.random.default_rng(seed)
    fn = naive_trial if naive else sequential_trial
    bad = 0
    for _ in range(trials):
        dec, _ = fn(rng, 0.0, sigma, theta)
        bad += dec in (verdict.REGRESSION, verdict.IMPROVEMENT)
    return bad / trials


# --- WP5: bisection ------------------------------------------------------------

ENDPOINT_RUNS = 20


def _bisect_data(rng, N, k, effect, sigma):
    mu = np.where(np.arange(N + 1) >= k, math.log1p(effect), 0.0)
    return lambda i, r: rng.normal(mu[i], sigma, r)


def probabilistic_bisect(rng, N, k, effect, sigma, prior=None, batch=3, confidence=0.95,
                         max_runs=4000):
    draw = _bisect_data(rng, N, k, effect, sigma)
    bis = Bisector(N, prior)
    bis.add(0, draw(0, ENDPOINT_RUNS))
    bis.add(N, draw(N, ENDPOINT_RUNS))
    runs = 2 * ENDPOINT_RUNS
    while runs < max_runs:
        best, p = bis.best()
        if p >= confidence:
            break
        i = bis.next_commit()
        if i is None:
            break
        bis.add(i, draw(i, batch))
        runs += batch
    best, p = bis.best()
    return {"correct": best == k, "runs": runs, "in_credible": k in bis.credible_set(0.95)}


def naive_bisect(rng, N, k, effect, sigma, runs_per_step):
    """Classic bisection, fixed runs per step, midpoint threshold between
    the endpoint means."""
    draw = _bisect_data(rng, N, k, effect, sigma)
    g, b = draw(0, ENDPOINT_RUNS).mean(), draw(N, ENDPOINT_RUNS).mean()
    cut = (g + b) / 2
    lo, hi = 0, N  # lo good, hi bad
    runs = 2 * ENDPOINT_RUNS
    while hi - lo > 1:
        mid = (lo + hi) // 2
        y = draw(mid, runs_per_step).mean()
        runs += runs_per_step
        if y > cut:
            hi = mid
        else:
            lo = mid
    return {"correct": hi == k, "runs": runs}


def _mwu_p(x, y):
    """Two-sided Mann-Whitney U p-value (normal approximation)."""
    allv = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    n1, n2 = len(x), len(y)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sd if sd > 0 else 0.0
    return 2 * (1 - bootstrap.norm_cdf(abs(z)))


def mwu_bisect(rng, N, k, effect, sigma, runs_per_step):
    """Fixed-sample bisection deciding each step by which endpoint the
    midpoint's samples are statistically closer to (Pinpoint-style
    Mann-Whitney comparison)."""
    draw = _bisect_data(rng, N, k, effect, sigma)
    good, bad = draw(0, ENDPOINT_RUNS), draw(N, ENDPOINT_RUNS)
    lo, hi = 0, N
    runs = 2 * ENDPOINT_RUNS
    while hi - lo > 1:
        mid = (lo + hi) // 2
        y = draw(mid, runs_per_step)
        runs += runs_per_step
        if _mwu_p(y, good) < _mwu_p(y, bad):
            hi = mid
        else:
            lo = mid
    return {"correct": hi == k, "runs": runs}


def runs_to_95(method, rng, N, effect, sigma, trials, steps=(1, 2, 3, 5, 8, 13, 21, 34,
                                                             55, 89, 144)):
    """Smallest fixed runs-per-step reaching >= 95% correct localization,
    with its mean total runs (None if no setting reaches 95%)."""
    for r in steps:
        res = [method(rng, N, int(rng.integers(1, N + 1)), effect, sigma, r)
               for _ in range(trials)]
        acc = np.mean([x["correct"] for x in res])
        if acc >= 0.95:
            return {"runs_per_step": r, "accuracy": float(acc),
                    "runs": float(np.mean([x["runs"] for x in res]))}
    return None


def prior_vector(rng, N, k, informative_p, base_rate=0.1):
    """Attribution flags a_k: the culprit is flagged with probability
    informative_p; other commits touch the file at base_rate."""
    a = (rng.random(N) < base_rate).astype(float)
    a[k - 1] = 1.0 if rng.random() < informative_p else 0.0
    return 1.0 + 4.0 * a


# --- WP6: BOCPD -------------------------------------------------------------------

def bocpd_false_alarms(series_count, length=500, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    alarms = 0
    for _ in range(series_count):
        alerts, _ = bocpd.detect(rng.normal(0, sigma, length))
        alarms += len(alerts)
    return alarms / series_count  # alarms per `length` null sessions


def bocpd_delays(trials, shift=0.10, sigma=0.03, change_at=60, length=120, seed=0):
    rng = np.random.default_rng(seed)
    delays = []
    for _ in range(trials):
        x = rng.normal(0, sigma, length)
        x[change_at:] += math.log1p(shift)
        alerts, _ = bocpd.detect(x)
        after = [a.index for a in alerts if a.index >= change_at]
        delays.append(after[0] - change_at if after else math.inf)
    return np.array(delays, dtype=float)


__all__ = [name for name in dir() if not name.startswith("_")] + ["halfwidth"]
