"""Comparison of two groups of sessions, per call site (rglns check,
and the exploratory drill-down of rglns compare).

For every call site seen on both sides:
  1. One value per run: the median log time of that run's calls at the
     site (reservoir samples, weighted by call count). When the two sides
     ran different array sizes, each sample is first divided by the
     baseline cost model's prediction at its n (stats/costmodel.py).
  2. A run-level bootstrap with drift-bounded correction and BCa
     intervals (stats/bootstrap.py), hierarchical when both sides have
     >= 3 sessions.
  3. Benjamini-Hochberg across sites, then verdicts (stats/verdict.py).
  4. Diagnosis hypotheses for every site that changed (diagnosis.py).
The whole-program wall time is compared the same way, as "<wall time>".
"""
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import diagnosis, hw
from .stats import bootstrap, costmodel, verdict

WALL = "<wall time>"


@dataclass
class SiteResult:
    key: str
    display: str
    operators: list
    comparison: object = None
    verdict: object = None
    size_note: str = None
    alpha_beta: dict = None
    hypotheses: list = field(default_factory=list)
    only_in: str = None   # "baseline" | "current" when the site exists on one side


@dataclass
class Report:
    sites: list
    theta: float
    alpha: float
    base_sessions: list
    cur_sessions: list
    warnings: list
    exploratory: bool = False

    @property
    def regressions(self):
        return [s for s in self.sites if s.verdict and s.verdict.verdict == verdict.REGRESSION]


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, cum[-1] / 2.0)])


def _run_value(rows, model):
    vals, weights = [], []
    for r in rows:
        if not r.samples:
            continue
        w = r.count / len(r.samples)
        adj = math.log(model.predict(r.n)) if (model is not None and r.n) else 0.0
        for s in r.samples:
            vals.append(math.log(max(s, 1.0)) - adj)
            weights.append(w)
    return _weighted_median(vals, weights) if vals else None


def _log_or_nan(x):
    return math.log(x) if x and x > 0 else float("nan")


def _by_site(sessions):
    """{site_id: [[(run, [rows])] per session]} with sessions aligned."""
    out = defaultdict(lambda: [[] for _ in sessions])
    for si, s in enumerate(sessions):
        for run in s.runs:
            rows_by_site = defaultdict(list)
            for c in run.calls:
                rows_by_site[c.call_site_id].append(c)
            for site_id, rows in rows_by_site.items():
                out[site_id][si].append((run, rows))
    return out


def _side(per_session, model):
    values, log_c, log_m = [], [], []
    for runs in per_session:
        v, c, m = [], [], []
        for run, rows in runs:
            x = _run_value(rows, model)
            if x is None:
                continue
            v.append(x)
            c.append(_log_or_nan(run.canary_compute_ns))
            m.append(_log_or_nan(run.canary_mem_ns))
        values.append(v)
        log_c.append(c)
        log_m.append(m)
    return bootstrap.Side(values, log_c, log_m)


def _wall_side(sessions):
    values, log_c, log_m = [], [], []
    for s in sessions:
        runs = [r for r in s.runs if r.wall_ns]
        values.append([math.log(r.wall_ns) for r in runs])
        log_c.append([_log_or_nan(r.canary_compute_ns) for r in runs])
        log_m.append([_log_or_nan(r.canary_mem_ns) for r in runs])
    return bootstrap.Side(values, log_c, log_m)


def _all_rows(per_session):
    return [c for runs in per_session for _, rows in runs for c in rows]


def _size_model(base_rows, cur_rows, caches):
    """(model, note, low_confidence) for size normalization, or
    (None, note, None) when no normalization applies."""
    nb = {r.n for r in base_rows if r.n}
    nc = {r.n for r in cur_rows if r.n}
    if not nb or not nc or nb == nc:
        return None, None, None
    dtype = max(((r.dtype, r.count) for r in base_rows if r.dtype),
                key=lambda t: t[1], default=("float64", 0))[0]
    try:
        itemsize = np.dtype(dtype).itemsize
    except TypeError:
        itemsize = 8
    points = []
    for r in base_rows:
        if r.n and r.samples:
            points.append((r.n, float(np.median(r.samples))))
    model = costmodel.fit(points, itemsize, caches)
    if model is None:
        return None, ("array sizes differ but the baseline has a single size, so the "
                      "per-call overhead and per-element cost cannot be separated; "
                      "comparing raw times"), None
    low = None
    if any(model.extrapolated(n) for n in nc):
        low = "extrapolated, low confidence: current sizes are >2x outside the baseline range"
    return model, f"size-normalized (baseline n {min(nb):,}..{max(nb):,}, current n {min(nc):,}..{max(nc):,})", low


def _site_display(info, site_id):
    if not info:
        return site_id[:10]
    line = f":{info['line']}" if info.get("line") else ""
    qual = f" {info['qualname']}" if info.get("qualname") and info["qualname"] != "<module>" else ""
    return f"{info['path']}{line}{qual}"


def compare_sessions(base_sessions, cur_sessions, site_info, theta=0.05, alpha=0.05,
                     B=2000, seed=None, use_canaries=True, exploratory=False,
                     min_runs=3, with_diagnosis=True):
    rng = np.random.default_rng(seed)
    caches = hw.cache_sizes()
    base_by_site = _by_site(base_sessions)
    cur_by_site = _by_site(cur_sessions)
    base_runs = [r for s in base_sessions for r in s.runs]
    cur_runs = [r for s in cur_sessions for r in s.runs]
    warnings = _environment_warnings(base_sessions, cur_sessions)

    sites, comparisons, low_conf = [], [], []
    wall_b, wall_c = _wall_side(base_sessions), _wall_side(cur_sessions)
    if wall_b.n_runs >= min_runs and wall_c.n_runs >= min_runs:
        res = SiteResult(WALL, WALL, ["process"])
        res.comparison = bootstrap.compare(wall_b, wall_c, theta, alpha, B, rng, use_canaries)
        sites.append(res)
        comparisons.append(res.comparison)
        low_conf.append(None)

    for site_id in sorted(set(base_by_site) | set(cur_by_site)):
        info = site_info.get(site_id)
        b_rows = _all_rows(base_by_site.get(site_id, []))
        c_rows = _all_rows(cur_by_site.get(site_id, []))
        ops = sorted({r.operator for r in b_rows + c_rows})
        res = SiteResult(site_id, _site_display(info, site_id), ops)
        if not b_rows or not c_rows:
            res.only_in = "baseline" if b_rows else "current"
            sites.append(res)
            continue
        model, note, low = _size_model(b_rows, c_rows, caches)
        res.size_note = note
        b_side = _side(base_by_site[site_id], model)
        c_side = _side(cur_by_site[site_id], model)
        if b_side.n_runs < min_runs or c_side.n_runs < min_runs:
            res.size_note = (f"insufficient runs (baseline {b_side.n_runs}, current "
                             f"{c_side.n_runs}; need {min_runs} per side)")
            sites.append(res)
            continue
        res.comparison = bootstrap.compare(b_side, c_side, theta, alpha, B, rng, use_canaries)
        if model is not None:
            cur_model = costmodel.fit([(r.n, float(np.median(r.samples))) for r in c_rows
                                       if r.n and r.samples], model.itemsize, caches)
            res.alpha_beta = costmodel.shift(model, cur_model,
                                             int(np.median([r.n for r in c_rows if r.n])))
        if with_diagnosis:
            res.hypotheses = diagnosis.diagnose(
                diagnosis.SiteView(b_rows, base_runs, res.display),
                diagnosis.SiteView(c_rows, cur_runs, res.display), site_id)
        sites.append(res)
        comparisons.append(res.comparison)
        low_conf.append(low)

    verdicts = verdict.assign(comparisons, theta, alpha, low_conf)
    for res, v in zip([s for s in sites if s.comparison is not None], verdicts):
        res.verdict = v
    return Report(sites, theta, alpha, base_sessions, cur_sessions, warnings, exploratory)


def _environment_warnings(base, cur):
    out = []
    hb = {s.hw_fingerprint for s in base if s.hw_fingerprint}
    hc = {s.hw_fingerprint for s in cur if s.hw_fingerprint}
    if hb and hc and hb != hc:
        out.append("baseline and current ran on DIFFERENT hardware: "
                   f"{', '.join(sorted(hw.describe(h) for h in hb))} vs "
                   f"{', '.join(sorted(hw.describe(h) for h in hc))}. Any change below may "
                   "be the machine, not the code.")
    for attr, label in (("python", "Python"), ("numpy", "NumPy")):
        vb = {getattr(s, attr) for s in base if getattr(s, attr)}
        vc = {getattr(s, attr) for s in cur if getattr(s, attr)}
        if vb and vc and vb != vc:
            out.append(f"{label} version differs: {', '.join(sorted(vb))} vs "
                       f"{', '.join(sorted(vc))}")
    return out


# --- rendering -------------------------------------------------------------

def pct(x):
    return f"{math.expm1(x) * 100:+.1f}%" if x is not None and math.isfinite(x) else "n/a"


def _interval(c):
    return f"{pct(c.estimate)} [{pct(c.ci_low)}, {pct(c.ci_high)}]"


_ORDER = {verdict.REGRESSION: 0, verdict.ENVIRONMENT_DRIFT: 1, verdict.INCONCLUSIVE: 2,
          verdict.IMPROVEMENT: 3, verdict.EQUIVALENT: 4}


def _sorted_sites(report):
    def key(s):
        v = s.verdict.verdict if s.verdict else "zz"
        return (0 if s.key == WALL else 1, _ORDER.get(v, 9),
                -(s.comparison.estimate if s.comparison else 0))
    return sorted(report.sites, key=key)


def render_text(report, title="Performance comparison"):
    lines = []
    nb = sum(len(s.runs) for s in report.base_sessions)
    nc = sum(len(s.runs) for s in report.cur_sessions)
    lines.append(f"{title}: baseline {len(report.base_sessions)} session(s) / {nb} runs "
                 f"vs current {len(report.cur_sessions)} session(s) / {nc} runs")
    lines.append(f"  threshold {report.theta:.0%}, alpha {report.alpha}, Benjamini-Hochberg "
                 f"across {sum(1 for s in report.sites if s.comparison)} comparisons, "
                 "effect = drift-bounded run-level ratio (BCa)")
    if report.exploratory:
        lines.append("  EXPLORATORY per-call-site drill-down (single session per side)")
    for w in report.warnings:
        lines.append(f"  WARNING: {w}")
    lines.append("")
    for s in _sorted_sites(report):
        if s.only_in:
            lines.append(f"  {'only in ' + s.only_in:<18} {'':<34} {s.display}")
            continue
        if s.comparison is None:
            lines.append(f"  {'not compared':<18} {'':<34} {s.display}: {s.size_note}")
            continue
        v = s.verdict
        tag = v.verdict + (" (advisory)" if v.advisory and not report.exploratory else "")
        lines.append(f"  {tag:<18} {_interval(s.comparison):<34} {s.display}  "
                     f"[{', '.join(s.operators)}]")
        if v.verdict == verdict.INCONCLUSIVE and v.runs_needed:
            lines.append(f"      ~{v.runs_needed:,} runs per side needed to decide"
                         + (f" ({v.reason})" if v.reason else ""))
        elif v.reason:
            lines.append(f"      {v.reason}")
        if s.size_note and s.key != WALL:
            lines.append(f"      {s.size_note}")
        if s.alpha_beta:
            ab = s.alpha_beta
            lines.append(f"      fixed overhead (alpha) {ab['alpha']:+.0%}, per-element cost "
                         f"(beta) {ab['beta']:+.0%}")
        if s.hypotheses and v.verdict != verdict.EQUIVALENT:
            lines.append("      diagnosis (naive-Bayes combination of independent detectors):")
            for h in s.hypotheses[:3]:
                lines.append(f"        {h.posterior:>4.0%}  {h.name}: {h.evidence}")
                if h.remediation:
                    lines.append(f"              -> {h.remediation}")
    if not report.exploratory and any(s.verdict and s.verdict.advisory for s in report.sites):
        lines.append("")
        lines.append("  (advisory) fewer than 3 sessions on a side: session-to-session "
                     "variation is not in the interval. Record more baseline sessions, or "
                     "use `rglns compare` (interleaved) for a decisive answer.")
    return "\n".join(lines)


def to_dict(report):
    out = {"theta": report.theta, "alpha": report.alpha, "exploratory": report.exploratory,
           "warnings": report.warnings, "sites": []}
    for s in report.sites:
        d = {"key": s.key, "display": s.display, "operators": s.operators,
             "only_in": s.only_in, "note": s.size_note}
        if s.comparison is not None:
            c = s.comparison
            d.update(effect=math.expm1(c.estimate), ci_low=math.expm1(c.ci_low),
                     ci_high=math.expm1(c.ci_high), p_value=c.p_value,
                     hierarchical=c.hierarchical, runs=[c.n_base_runs, c.n_cur_runs],
                     verdict=s.verdict.verdict, advisory=s.verdict.advisory,
                     runs_needed=s.verdict.runs_needed)
            d["diagnosis"] = [{"name": h.name, "evidence": h.evidence,
                               "likelihood_ratio": h.likelihood_ratio,
                               "posterior": h.posterior, "remediation": h.remediation}
                              for h in s.hypotheses]
        out["sites"].append(d)
    return out


def render_json(report):
    return json.dumps(to_dict(report), indent=2, default=float)
