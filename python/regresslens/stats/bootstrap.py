"""Run-level resampling with drift-bounded effects and BCa intervals.

Unit of analysis: the RUN. Each run contributes one value per call site
(its median log time), never one value per call. Calls within a run
share that run's machine state, so resampling calls would understate
the uncertainty (pseudo-replication). That was the v0.1 engine's flaw.

Hierarchy: with >= 3 sessions on both sides, replicates resample
sessions first, then runs within each chosen session, so
session-to-session variation (a different machine state each
invocation) enters the interval. With fewer sessions that variation
cannot be estimated. Runs are then pooled, and callers must label the
verdict advisory (cross-session).

Drift-bounded correction (see drift.py): when compute- and
memory-canary log ratios dc and dm are available, each replicate
computes
    lower = d - max(dc, dm)     upper = d - min(dc, dm)
with d the log-ratio of run-level means (current - baseline). The
effect interval is [BCa lower bound of `lower`, BCa upper bound of
`upper`], and the regression p-value is P*(lower <= theta).
"""
import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

_N01 = NormalDist()


def norm_cdf(x):
    return _N01.cdf(x)


def norm_ppf(p):
    return _N01.inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


@dataclass
class Side:
    """Run-level observations for one side of a comparison: one array of
    log values per session, with optional aligned per-run log canaries
    (NaN where a run has no canary sample)."""
    values: list
    log_c: list = None
    log_m: list = None

    def __post_init__(self):
        values = [np.asarray(v, dtype=float) for v in self.values]
        log_c = _align(self.log_c, values)
        log_m = _align(self.log_m, values)
        keep = [i for i, v in enumerate(values) if len(v)]  # drop empty sessions
        self.values = [values[i] for i in keep]
        self.log_c = [log_c[i] for i in keep]
        self.log_m = [log_m[i] for i in keep]

    @property
    def n_sessions(self):
        return len(self.values)

    @property
    def n_runs(self):
        return int(sum(len(v) for v in self.values))

    def flat(self, attr="values"):
        arrs = getattr(self, attr)
        return np.concatenate(arrs) if arrs else np.empty(0)

    def has_canaries(self):
        return (np.isfinite(self.flat("log_c")).any() and np.isfinite(self.flat("log_m")).any())


def _align(arrs, values):
    """Per-session canary arrays matching `values` in shape (NaN when a
    session has no canaries or a mismatched length)."""
    out = []
    for i, v in enumerate(values):
        a = None
        if arrs is not None and i < len(arrs) and arrs[i] is not None:
            a = np.asarray(arrs[i], dtype=float)
        out.append(a if a is not None and len(a) == len(v) else np.full(len(v), np.nan))
    return out


@dataclass
class Comparison:
    estimate: float              # d: log ratio of run-level means, uncorrected
    lower_est: float             # drift-bounded point estimates
    upper_est: float
    ci_low: float                # effect interval (drift-bounded, BCa)
    ci_high: float
    raw_ci_low: float            # uncorrected BCa interval for d
    raw_ci_high: float
    p_value: float               # P*(lower <= theta): regression p-value
    p_raw: float                 # P*(d <= theta), uncorrected
    hierarchical: bool
    drift_c: float               # canary log shifts (current - baseline)
    drift_m: float
    sigma_diff: float            # SD of a current-minus-baseline run pair
    n_base_runs: int
    n_cur_runs: int
    n_base_sessions: int
    n_cur_sessions: int
    reps: dict = field(default_factory=dict, repr=False)

    def runs_needed(self, theta_log, alpha=0.05):
        """Runs per side so the interval half-width shrinks to the gap
        between the estimate and the nearest decision boundary:
        n ~ (z * sigma / margin)^2 (sigma of a run-pair difference)."""
        z = norm_ppf(1 - alpha / 2)
        margin = min(abs(self.estimate - theta_log), abs(self.estimate + theta_log))
        margin = max(margin, 0.05 * theta_log, 1e-6)
        n = (z * self.sigma_diff / margin) ** 2
        return int(min(max(math.ceil(n), 2), 1_000_000))


def _session_padded(side, attr):
    arrs = getattr(side, attr)
    rmax = max(len(a) for a in arrs)
    out = np.full((len(arrs), rmax), np.nan)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def _resample_means(side, B, rng, hierarchical):
    """(B,) bootstrap means of values, log_c, log_m with joint resampling
    (the same runs are drawn for all three)."""
    if hierarchical:
        counts = np.array([len(v) for v in side.values])
        S, rmax = len(counts), counts.max()
        sess = rng.integers(0, S, size=(B, S))
        r = counts[sess]                                  # (B, S)
        idx = np.floor(rng.random((B, S, rmax)) * r[..., None]).astype(np.intp)
        valid = np.arange(rmax)[None, None, :] < r[..., None]
        out = []
        for attr in ("values", "log_c", "log_m"):
            pad = _session_padded(side, attr)
            draws = pad[sess[..., None], idx]
            draws = np.where(valid, draws, np.nan)
            out.append(_nanmean_rows(draws.reshape(B, -1)))
        return out
    flat = [side.flat(a) for a in ("values", "log_c", "log_m")]
    n = len(flat[0])
    idx = rng.integers(0, n, size=(B, n))
    return [_nanmean_rows(f[idx]) for f in flat]


def _nanmean_rows(x):
    finite = np.isfinite(x)
    cnt = finite.sum(axis=1)
    s = np.where(finite, x, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _jackknife_means(side, hierarchical):
    """Leave-one-unit-out means (units: sessions or runs) for each attr."""
    out = []
    for attr in ("values", "log_c", "log_m"):
        arrs = getattr(side, attr)
        if hierarchical:
            sums = np.array([np.nansum(a) for a in arrs])
            cnts = np.array([np.isfinite(a).sum() for a in arrs], dtype=float)
        else:
            flat = side.flat(attr)
            sums = np.where(np.isfinite(flat), flat, 0.0)
            cnts = np.isfinite(flat).astype(float)
        tot_s, tot_c = sums.sum(), cnts.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            loo = np.where(tot_c - cnts > 0, (tot_s - sums) / (tot_c - cnts), np.nan)
            full = tot_s / tot_c if tot_c > 0 else np.nan
        out.append((loo, full))
    return out


def _bounds(d, dc, dm, use_canaries):
    if not use_canaries:
        return d, d
    hi_shift = np.fmax(dc, dm)
    lo_shift = np.fmin(dc, dm)
    return d - hi_shift, d - lo_shift


def _acceleration(jack):
    jack = jack[np.isfinite(jack)]
    if len(jack) < 3:
        return 0.0
    dev = jack.mean() - jack
    den = 6.0 * (np.sum(dev ** 2) ** 1.5)
    return float(np.sum(dev ** 3) / den) if den > 0 else 0.0


def bca_interval(theta_hat, reps, jack, alpha, expand_n=None):
    """Two-sided (1 - alpha) BCa interval. `expand_n` (smallest sample
    size) widens the nominal level for small samples (Hesterberg's
    expanded percentile adjustment) before the BCa correction."""
    reps = reps[np.isfinite(reps)]
    B = len(reps)
    if B < 10 or not np.isfinite(theta_hat):
        return float("nan"), float("nan")
    prop = (np.sum(reps < theta_hat) + 0.5 * np.sum(reps == theta_hat)) / B
    prop = min(max(prop, 1.0 / (B + 1)), B / (B + 1.0))
    z0 = norm_ppf(prop)
    a = _acceleration(jack)
    z_alpha = norm_ppf(1 - alpha / 2)
    if expand_n and expand_n > 2:
        z_alpha *= expansion_factor(expand_n, alpha)
    out = []
    for z in (-z_alpha, z_alpha):
        denom = 1 - a * (z0 + z)
        q = norm_cdf(z0 + (z0 + z) / denom) if denom > 0 else (0.0 if z < 0 else 1.0)
        out.append(float(np.quantile(reps, min(max(q, 0.0), 1.0))))
    return out[0], out[1]


def expansion_factor(n, alpha):
    """sqrt(n / (n - 1)) * t_{1-alpha/2, n-1} / z_{1-alpha/2}: corrects a
    bootstrap interval's narrowness for small n (Hesterberg 2015)."""
    z = norm_ppf(1 - alpha / 2)
    return math.sqrt(n / (n - 1)) * t_ppf(1 - alpha / 2, n - 1) / z


def t_ppf(p, df):
    """Student-t quantile via the Cornish-Fisher expansion (accurate to
    ~1e-3 for df >= 3, which is all the expansion factor needs)."""
    z = norm_ppf(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def compare(base, cur, theta=0.05, alpha=0.05, B=2000, rng=None, use_canaries=True,
            expand=True):
    """Bootstrap comparison of two Sides. theta is the practical threshold
    as a fraction (0.05 = 5%); it is converted to log scale."""
    rng = rng if rng is not None else np.random.default_rng()
    theta_log = math.log1p(theta)
    hierarchical = base.n_sessions >= 3 and cur.n_sessions >= 3
    use_c = use_canaries and base.has_canaries() and cur.has_canaries()

    bv, bc, bm = (np.nanmean(base.flat(a)) if np.isfinite(base.flat(a)).any() else np.nan
                  for a in ("values", "log_c", "log_m"))
    cv, cc, cm = (np.nanmean(cur.flat(a)) if np.isfinite(cur.flat(a)).any() else np.nan
                  for a in ("values", "log_c", "log_m"))
    d_hat = cv - bv
    lo_hat, hi_hat = _bounds(d_hat, cc - bc, cm - bm, use_c)

    rb = _resample_means(base, B, rng, hierarchical)
    rc = _resample_means(cur, B, rng, hierarchical)
    d = rc[0] - rb[0]
    lo, hi = _bounds(d, rc[1] - rb[1], rc[2] - rb[2], use_c)

    # Jackknife over units of both sides (for the BCa acceleration).
    (b_loo_v, b_v), (b_loo_c, b_c), (b_loo_m, b_m) = _jackknife_means(base, hierarchical)
    (c_loo_v, c_v), (c_loo_c, c_c), (c_loo_m, c_m) = _jackknife_means(cur, hierarchical)
    jd = np.concatenate([c_v - b_loo_v, c_loo_v - b_v])
    jdc = np.concatenate([c_c - b_loo_c, c_loo_c - b_c])
    jdm = np.concatenate([c_m - b_loo_m, c_loo_m - b_m])
    j_lo, j_hi = _bounds(jd, jdc, jdm, use_c)
    jack = {"d": jd, "lo": j_lo, "hi": j_hi}

    def jack_stats(which):
        return jack[which]

    n_small = min(base.n_sessions, cur.n_sessions) if hierarchical else min(base.n_runs,
                                                                             cur.n_runs)
    expand_n = n_small if expand else None
    raw_lo, raw_hi = bca_interval(d_hat, d, jack_stats("d"), alpha, expand_n)
    ci_lo, _ = bca_interval(lo_hat, lo, jack_stats("lo"), alpha, expand_n)
    _, ci_hi = bca_interval(hi_hat, hi, jack_stats("hi"), alpha, expand_n)

    p_value = (1 + np.sum(lo[np.isfinite(lo)] <= theta_log)) / (1 + np.isfinite(lo).sum())
    p_raw = (1 + np.sum(d[np.isfinite(d)] <= theta_log)) / (1 + np.isfinite(d).sum())

    sd_b = np.nanstd(base.flat(), ddof=1) if base.n_runs > 1 else np.nan
    sd_c = np.nanstd(cur.flat(), ddof=1) if cur.n_runs > 1 else np.nan
    sigma_diff = float(np.sqrt(np.nansum([sd_b ** 2, sd_c ** 2])))

    return Comparison(
        estimate=float(d_hat), lower_est=float(lo_hat), upper_est=float(hi_hat),
        ci_low=ci_lo, ci_high=ci_hi, raw_ci_low=raw_lo, raw_ci_high=raw_hi,
        p_value=float(p_value), p_raw=float(p_raw), hierarchical=hierarchical,
        drift_c=float(cc - bc) if use_c else 0.0, drift_m=float(cm - bm) if use_c else 0.0,
        sigma_diff=sigma_diff, n_base_runs=base.n_runs, n_cur_runs=cur.n_runs,
        n_base_sessions=base.n_sessions, n_cur_sessions=cur.n_sessions,
        reps={"d": d, "lower": lo, "upper": hi})
