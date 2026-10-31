"""Exact reference results for accuracy verification (rglns verify).

Every finite float is m * 2**e with an integer m (< 2**53), so sums,
window sums and window variances can be computed EXACTLY with Python
integers and Fractions. No rounding happens anywhere in a reference. The
error of a kernel's output is then measured in ulps of the exact value:

    error = |output - exact| / ulp(exact)

which is what the accuracy policy (docs/accuracy.md) is stated in.
`math.fsum` gives a correctly rounded sum, but rolling windows and
variance need exact intermediate values, so this module uses integer
prefix sums instead.
"""
import math
from fractions import Fraction

import numpy as np


def _mantissas(x):
    """(ints M, int exponent offset emin, shifts) with x == M * 2**(E)
    exactly, E = shifts + emin; inputs must be finite."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    M = (m * float(2 ** 53)).astype(np.int64)  # exact: m has <= 53 significant bits
    E = e.astype(np.int64) - 53
    E[M == 0] = E.max() if len(E) else 0
    emin = int(E.min()) if len(E) else 0
    return M, E - emin, emin


def exact_sum(x):
    """Exact sum of finite values, as a Fraction."""
    if len(x) == 0:
        return Fraction(0)
    M, shift, emin = _mantissas(x)
    total = 0
    for s in np.unique(shift):
        sel = M[shift == s]
        # Split mantissas so per-exponent int64 sums cannot overflow.
        hi = int((sel >> 26).sum())
        lo = int((sel & ((1 << 26) - 1)).sum())
        total += ((hi << 26) + lo) << int(s)
    return Fraction(total) * Fraction(2) ** emin


def _prefix(values_int):
    out = [0]
    acc = 0
    for v in values_int:
        acc += v
        out.append(acc)
    return out


def _scaled_ints(x):
    """Python ints v_i with x_i == v_i * 2**emin exactly (nonfinite -> 0)."""
    x = np.where(np.isfinite(x), x, 0.0)
    M, shift, emin = _mantissas(x)
    return [int(m) << int(s) for m, s in zip(M, shift)], emin


def rolling_sums(x, window):
    """Exact window sums (Fractions) of every complete window. Windows
    containing non-finite values are returned as None."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < window:
        return []
    ints, emin = _scaled_ints(x)
    p = _prefix(ints)
    bad = _prefix([0 if math.isfinite(v) else 1 for v in x])
    scale = Fraction(2) ** emin
    return [None if bad[i + window] - bad[i] else (p[i + window] - p[i]) * scale
            for i in range(n - window + 1)]


def rolling_vars(x, window, ddof=0):
    """Exact window variances (np.var semantics with ddof)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < window or window - ddof <= 0:
        return []
    ints, emin = _scaled_ints(x)
    p1 = _prefix(ints)
    p2 = _prefix([v * v for v in ints])
    bad = _prefix([0 if math.isfinite(v) else 1 for v in x])
    out = []
    for i in range(n - window + 1):
        if bad[i + window] - bad[i]:
            out.append(None)
            continue
        s1 = p1[i + window] - p1[i]
        s2 = p2[i + window] - p2[i]
        # (s2 - s1^2 / w) / (w - ddof), in units of 2**(2*emin)
        num = Fraction(s2 * window - s1 * s1, window)
        out.append(num / (window - ddof) * Fraction(2) ** (2 * emin))
    return out


def affine(x, scale, offset):
    """Exact scale * x + offset per element (Fractions)."""
    s, o = Fraction(float(scale)), Fraction(float(offset))
    return [s * Fraction(float(v)) + o if math.isfinite(v) else None for v in x]


def ulp(exact, dtype):
    """ulp of the exact value's magnitude in `dtype` (smallest subnormal
    step for an exact zero)."""
    v = abs(float(exact))
    if dtype == np.float32:
        return float(np.spacing(np.float32(v))) if v else float(np.spacing(np.float32(0)))
    return math.ulp(v) if v else math.ulp(0.0)


def ulp_error(value, exact, dtype):
    if exact is None:
        return 0.0
    value = float(value)
    if not math.isfinite(value):
        return math.inf
    return float(abs(Fraction(value) - exact) / Fraction(ulp(exact, dtype)))
