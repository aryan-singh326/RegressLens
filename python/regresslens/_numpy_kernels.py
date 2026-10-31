"""Pure-NumPy implementations of every rl.array operation.

Used when the native kernels are unavailable (kernel name ``numpy``)
and for inputs the native kernels do not accept, such as
non-contiguous or multi-dimensional arrays (kernel name
``numpy_fallback``). Their semantics define what the native kernels
must reproduce. NaN/Inf behavior in particular is exactly NumPy's,
because these functions are NumPy.
"""
import numpy as np


def sum_(x):
    return np.sum(x)


def project_affine(x, scale, offset):
    return x * scale + offset


def filter_gt(x, threshold):
    return x[x > threshold]


def _empty_like_dtype(x):
    return np.empty(0, dtype=x.dtype)


def rolling_sum(x, window):
    """Sum of each complete window: np.convolve(x, ones, "valid") for
    n >= window, and an empty array when there is no complete window
    (np.convolve would silently swap its arguments in that case)."""
    if x.shape[0] < window:
        return _empty_like_dtype(x)
    return np.convolve(x, np.ones(window, dtype=x.dtype), mode="valid")


def rolling_mean(x, window):
    return rolling_sum(x, window) / x.dtype.type(window)


def rolling_var(x, window, ddof=0):
    if x.shape[0] < window:
        return _empty_like_dtype(x)
    view = np.lib.stride_tricks.sliding_window_view(x, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.var(view, axis=-1, ddof=ddof).astype(x.dtype, copy=False)
