"""Size-normalized cost model (WP3e).

Per (call site, dtype), the baseline's time-vs-size points are fitted as

    t(n) = alpha + beta * n

piecewise by cache regime (L1, L2, L3, DRAM; breakpoints are the
detected cache sizes in bytes), each piece by Theil-Sen (the median of
pairwise slopes), which ignores the outliers that timing data always
has. A current observation at size n is then compared with the
baseline's prediction at that n, so a run that simply processed 2x more
data is not reported as a 2x regression. Before this model, differing
sizes made the engine drop the operation ("no common keys").

alpha is fixed per-call overhead and beta per-element cost, so knowing
which one moved is a diagnosis in itself (e.g. dispatch overhead vs. a
slower kernel). Predictions more than 2x outside the baseline's size
range are labeled "extrapolated, low confidence".
"""
import math
from dataclasses import dataclass, field

import numpy as np

MAX_POINTS = 1500  # Theil-Sen is O(n^2) in points; subsample above this
EXTRAPOLATION_FACTOR = 2.0


def theil_sen(x, y):
    """(alpha, beta) robust line fit. Requires >= 2 distinct x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return None
    beta = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    alpha = float(np.median(y - beta * x))
    return alpha, beta


@dataclass
class Piece:
    lo_bytes: float
    hi_bytes: float
    alpha: float
    beta: float
    n_points: int


@dataclass
class CostModel:
    pieces: list
    itemsize: int
    n_min: int
    n_max: int
    breakpoints: list = field(default_factory=list)

    def _piece(self, n):
        nbytes = n * self.itemsize
        fitted = [p for p in self.pieces if p.n_points]
        inside = [p for p in fitted if p.lo_bytes <= nbytes < p.hi_bytes]
        if inside:
            return inside[0]
        # No fit for this regime: use the nearest fitted one.
        return min(fitted, key=lambda p: min(abs(math.log(max(nbytes, 1) / max(p.lo_bytes, 1))),
                                             abs(math.log(max(nbytes, 1) / max(p.hi_bytes, 1)))))

    def predict(self, n):
        p = self._piece(n)
        t = p.alpha + p.beta * n
        # A negative intercept can make tiny-n predictions nonsensical;
        # never predict below 1% of the per-element cost line.
        return max(t, 0.01 * abs(p.beta) * n, 1.0)

    def extrapolated(self, n):
        return n > EXTRAPOLATION_FACTOR * self.n_max or n < self.n_min / EXTRAPOLATION_FACTOR


def _regimes(caches):
    bounds = [0.0, float(caches["l1d"]), float(caches["l2"]), float(caches["l3"]), float("inf")]
    return [(bounds[k], bounds[k + 1]) for k in range(4)]


def fit(points, itemsize, caches, rng=None):
    """points: iterable of (n, t_ns). Returns a CostModel, or None when the
    baseline has fewer than two distinct sizes (alpha and beta are then
    not separable and nothing can be normalized)."""
    pts = [(int(n), float(t)) for n, t in points if n and t > 0 and math.isfinite(t)]
    if len({n for n, _ in pts}) < 2:
        return None
    if len(pts) > MAX_POINTS:
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = rng.choice(len(pts), MAX_POINTS, replace=False)
        pts = [pts[i] for i in idx]
    pieces = []
    for lo, hi in _regimes(caches):
        sub = [(n, t) for n, t in pts if lo <= n * itemsize < hi]
        line = theil_sen(*zip(*sub)) if len({n for n, _ in sub}) >= 2 else None
        if line is None:
            pieces.append(Piece(lo, hi, 0.0, 0.0, 0))
        else:
            pieces.append(Piece(lo, hi, line[0], line[1], len(sub)))
    if not any(p.n_points for p in pieces):
        # Sizes span regimes with < 2 distinct sizes each: fit one line.
        a, b = theil_sen(*zip(*pts))
        pieces = [Piece(0.0, float("inf"), a, b, len(pts))]
    ns = [n for n, _ in pts]
    return CostModel(pieces, itemsize, min(ns), max(ns),
                     [lo for lo, _ in _regimes(caches)][1:])


def shift(base_model, cur_model, n):
    """Relative change of (alpha, beta) in the regime containing n, or None
    when the current side has too few sizes to fit its own model."""
    if cur_model is None:
        return None
    b, c = base_model._piece(n), cur_model._piece(n)
    def rel(new, old):
        return (new - old) / abs(old) if old else float("inf") if new else 0.0
    return {"alpha": rel(c.alpha, b.alpha), "beta": rel(c.beta, b.beta),
            "alpha_base": b.alpha, "alpha_cur": c.alpha,
            "beta_base": b.beta, "beta_cur": c.beta}
