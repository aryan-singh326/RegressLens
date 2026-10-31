"""Diagnosis: independent detectors that each propose a cause.

    Detector.analyze(baseline, current, key) -> [Hypothesis(name, evidence,
                                                 likelihood_ratio, remediation)]

Each detector compares one call site's baseline and current SiteView
and returns hypotheses with a likelihood ratio LR = P(observations |
cause) / P(observations | no specific cause). Detectors:

  contiguity_loss     the site started receiving non-contiguous arrays
  dtype_promotion     the dominant dtype changed (e.g. float32 -> float64)
  kernel_selection    a different native kernel ran (e.g. selectivity
                      drifted out of the AVX2 band)
  numpy_fallback      more calls fell back to NumPy for another reason
  size_regime         the arrays got much larger or crossed a cache level
  environment_drift   the drift canaries moved (machine, not code)

Combination is naive Bayes: log-LRs of the same hypothesis are summed,
as if the detectors' evidence were independent. That is an
approximation, stated here and in the report, not a claim. Posteriors
use prior odds 1:1 between "no specific cause" and the set of causes.
Hypotheses with LR < REPORT_LR are not reported, so an unchanged site
yields no diagnosis.
"""
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import hw

REPORT_LR = 10.0
_FALLBACK_KERNELS = ("numpy_fallback",)
_NATIVE_KERNELS = ("scalar", "avx2", "mt_avx2", "scalar_nf")
FILTER_AVX2_BAND = (0.3, 0.7)


@dataclass
class Hypothesis:
    name: str
    evidence: str
    likelihood_ratio: float
    remediation: str = None
    posterior: float = None


@dataclass
class SiteView:
    """Everything recorded for one call site on one side."""
    rows: list                      # store.CallRow across all runs
    runs: list = field(default_factory=list)  # store.Run of that side (canaries)
    display: str = ""

    def _weighted(self, attr):
        c = Counter()
        for r in self.rows:
            c[getattr(r, attr)] += r.count
        return c

    @property
    def calls(self):
        return sum(r.count for r in self.rows)

    def fraction(self, pred):
        total = self.calls
        return sum(r.count for r in self.rows if pred(r)) / total if total else 0.0

    def dominant(self, attr):
        c = self._weighted(attr)
        return c.most_common(1)[0][0] if c else None

    def median_n(self):
        pairs = sorted((r.n, r.count) for r in self.rows if r.n is not None)
        if not pairs:
            return None
        half, acc = sum(c for _, c in pairs) / 2, 0
        for n, c in pairs:
            acc += c
            if acc >= half:
                return n
        return pairs[-1][0]

    def selectivity(self):
        num = sum(r.extra.get("sel_sum", 0.0) for r in self.rows)
        den = sum(r.extra.get("sel_n", 0) for r in self.rows)
        return num / den if den else None

    def samples(self, pred=lambda r: True):
        return [s for r in self.rows if pred(r) for s in r.samples]

    def canaries(self):
        c = np.array([r.canary_compute_ns for r in self.runs if r.canary_compute_ns], float)
        m = np.array([r.canary_mem_ns for r in self.runs if r.canary_mem_ns], float)
        return np.log(c), np.log(m)


class Detector:
    name = "detector"

    def analyze(self, baseline, current, key):
        raise NotImplementedError


def _lr_from_shift(delta, scale):
    """LR growing smoothly with a shift in [0, 1] (fractions)."""
    return 1.0 + scale * max(delta, 0.0) ** 2


class ContiguityLossDetector(Detector):
    name = "contiguity_loss"

    def analyze(self, baseline, current, key):
        noncontig = lambda r: r.contiguous is False
        fb, fc = baseline.fraction(noncontig), current.fraction(noncontig)
        if fc - fb < 0.2:
            return []
        evidence = (f"{fc:.0%} of calls at {current.display or key} received a "
                    f"non-contiguous array (baseline {fb:.0%}). RegressLens identifies "
                    "the first affected call site, not the upstream operation that "
                    "produced the strided view.")
        remediation = ("Make the array contiguous before this call "
                       "(np.ascontiguousarray), ideally where the strided view is created.")
        est = _copy_estimate(baseline, current)
        if est:
            remediation += " " + est
        return [Hypothesis(self.name, evidence, _lr_from_shift(fc - fb, 100.0), remediation)]


def _copy_estimate(baseline, current):
    """Cost/benefit of the contiguity fix from observed timings."""
    from . import remediation
    n = current.median_n()
    dtype = current.dominant("dtype")
    fallback = current.samples(lambda r: r.contiguous is False)
    accelerated = baseline.samples(lambda r: r.contiguous is True)
    if not n or not dtype or not fallback:
        return None
    try:
        itemsize = np.dtype(dtype).itemsize
    except TypeError:
        return None
    rem = remediation.estimate_remediation(n, itemsize, fallback, accelerated)
    if rem["reason"]:
        return None
    verdict = ("worth applying" if rem["recommend_apply"]
               else "only worth it if the array is reused several times")
    return (f"Estimated copy cost {rem['copy_cost_ns'] / 1e6:.2f} ms vs savings "
            f"{rem['estimated_savings_ns'] / 1e6:.2f} ms per call: {verdict}.")


class DtypePromotionDetector(Detector):
    name = "dtype_promotion"

    def analyze(self, baseline, current, key):
        db, dc = baseline.dominant("dtype"), current.dominant("dtype")
        if not db or not dc or db == dc:
            return []
        try:
            grew = np.dtype(dc).itemsize > np.dtype(db).itemsize
        except TypeError:
            grew = False
        share = current.fraction(lambda r: r.dtype == dc)
        lr = (50.0 if grew else 20.0) * share
        return [Hypothesis(
            self.name, f"dtype changed from {db} to {dc} ({share:.0%} of current calls)",
            max(lr, 1.0),
            f"Find where the array becomes {dc} upstream (astype, np.asarray without dtype=, "
            f"mixing with {dc} constants or arrays) and keep it {db} if that precision "
            "is enough.")]


class KernelSelectionDetector(Detector):
    name = "kernel_selection"

    def analyze(self, baseline, current, key):
        kb = Counter({k: v for k, v in baseline._weighted("kernel").items() if k in _NATIVE_KERNELS})
        kc = Counter({k: v for k, v in current._weighted("kernel").items() if k in _NATIVE_KERNELS})
        if not kb or not kc:
            return []
        top_b, top_c = kb.most_common(1)[0][0], kc.most_common(1)[0][0]
        if top_b == top_c:
            return []
        share = kc[top_c] / sum(kc.values())
        evidence = f"kernel changed from {top_b} to {top_c} ({share:.0%} of native calls)"
        sb, sc = baseline.selectivity(), current.selectivity()
        lr = 30.0 * share
        if sb is not None and sc is not None:
            lo, hi = FILTER_AVX2_BAND
            evidence += f"; filter selectivity moved {sb:.2f} -> {sc:.2f}"
            if (lo <= sb <= hi) != (lo <= sc <= hi):
                evidence += f" (crossing the AVX2 band [{lo}, {hi}])"
                lr *= 2
        return [Hypothesis(self.name, evidence, max(lr, 1.0),
                           "The kernel selector reacted to different data at this site. "
                           "If the new kernel is slower here, the selector thresholds "
                           "(cpp/src/kernel_selector.cpp) need re-deriving for this case.")]


class FallbackRateDetector(Detector):
    name = "numpy_fallback"

    def analyze(self, baseline, current, key):
        # Non-contiguous fallbacks are contiguity_loss; this covers the rest
        # (multi-dimensional input, overflow-risk rolling input, missing
        # native module) at sites that used native kernels before.
        fallback = lambda r: r.contiguous is not False and (
            r.kernel in _FALLBACK_KERNELS or r.kernel == "numpy")
        if not any(r.kernel in _NATIVE_KERNELS for r in baseline.rows):
            return []
        fb, fc = baseline.fraction(fallback), current.fraction(fallback)
        if fc - fb < 0.2:
            return []
        return [Hypothesis(self.name,
                           f"{fc:.0%} of contiguous calls fell back to NumPy (baseline {fb:.0%})",
                           _lr_from_shift(fc - fb, 100.0),
                           "Check that regresslens-accel is installed and importable "
                           "(rl.accel_info()), and that inputs are 1-D.")]


def _regime(nbytes, caches):
    for name, limit in (("L1", caches["l1d"]), ("L2", caches["l2"]), ("L3", caches["l3"])):
        if nbytes <= limit:
            return name
    return "DRAM"


class SizeRegimeDetector(Detector):
    name = "size_regime"

    def __init__(self, caches=None):
        self.caches = caches or hw.cache_sizes()

    def analyze(self, baseline, current, key):
        nb, nc = baseline.median_n(), current.median_n()
        if not nb or not nc:
            return []
        ratio = nc / nb
        try:
            itemsize = np.dtype(current.dominant("dtype") or "float64").itemsize
        except TypeError:
            itemsize = 8
        rb, rc = _regime(nb * itemsize, self.caches), _regime(nc * itemsize, self.caches)
        if 0.5 < ratio < 2.0 and rb == rc:
            return []
        lr = 1.0 + 10.0 * abs(math.log2(ratio))
        if rb != rc:
            lr *= 2
        return [Hypothesis(
            self.name,
            f"typical n changed from {nb:,} to {nc:,} ({ratio:.1f}x; {rb} -> {rc})",
            lr, "More data, not slower code: compare per-element cost (the size-normalized "
                "effect in this report accounts for it).")]


class EnvironmentDriftDetector(Detector):
    name = "environment_drift"
    MIN_SHIFT = math.log1p(0.05)

    def analyze(self, baseline, current, key):
        (cb, mb), (cc, mc) = baseline.canaries(), current.canaries()
        found = []
        for label, b, c in (("compute", cb, cc), ("memory", mb, mc)):
            if len(b) < 2 or len(c) < 2:
                continue
            shift = c.mean() - b.mean()
            se = math.sqrt(b.var(ddof=1) / len(b) + c.var(ddof=1) / len(c)) or 1e-9
            if abs(shift) >= self.MIN_SHIFT and abs(shift) / se >= 3:
                found.append((label, shift))
        if not found:
            return []
        worst = max(abs(s) for _, s in found)
        desc = ", ".join(f"{label} canary {math.expm1(s):+.1%}" for label, s in found)
        return [Hypothesis(self.name, desc, 1.0 + 50.0 * min(worst / math.log1p(0.10), 4.0),
                           "The machine changed between sessions (frequency, contention, "
                           "noisy neighbors). Re-measure on a quiet machine, or use "
                           "`rglns compare`, which interleaves both sides.")]


DEFAULT_DETECTORS = (ContiguityLossDetector, DtypePromotionDetector, KernelSelectionDetector,
                     FallbackRateDetector, SizeRegimeDetector, EnvironmentDriftDetector)


def diagnose(baseline, current, key, detectors=None):
    """Ranked hypotheses (LR >= REPORT_LR) with naive-Bayes posteriors."""
    detectors = detectors or [cls() for cls in DEFAULT_DETECTORS]
    combined = {}
    for det in detectors:
        for h in det.analyze(baseline, current, key):
            if h.name in combined:
                prev = combined[h.name]
                prev.likelihood_ratio *= h.likelihood_ratio  # sum of log-LRs
                prev.evidence += "; " + h.evidence
            else:
                combined[h.name] = h
    # Prior: P(no specific cause) = 1/2, the rest shared equally among
    # the detectors' causes. Every detected hypothesis enters the
    # normalization, including ones too weak to report.
    k = max(len(detectors), 1)
    weights = {name: (0.5 / k) * h.likelihood_ratio for name, h in combined.items()}
    total = 0.5 + sum(weights.values())
    for name, h in combined.items():
        h.posterior = weights[name] / total
    hyps = [h for h in combined.values() if h.likelihood_ratio >= REPORT_LR]
    return sorted(hyps, key=lambda h: -h.likelihood_ratio)
