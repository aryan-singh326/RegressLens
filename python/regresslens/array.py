"""
regresslens.array: the NumPy integration layer.

Per the project brief: minimal integration, not invisible. Wrap an
array once with regresslens.array(), then call supported operations
(currently: sum/mean) either as methods or via NumPy's dispatch
protocols. Unsupported operations fall back to plain NumPy
transparently — this is NOT a full NumPy-compatible array type, and
is not trying to be one.
"""
import logging
import numpy as np

from . import _regresslens_native as _native

logger = logging.getLogger("regresslens")

# v0.1 scope: float32/float64 only. Anything else is rejected at
# wrap time with a clear error, per the project brief — not silently
# coerced, which would hide correctness issues.
_SUPPORTED_DTYPES = (np.float32, np.float64)


class UnsupportedDTypeError(TypeError):
    pass


class array:
    """
    Wraps a NumPy ndarray so supported operations (currently: sum)
    can be intercepted and routed through RegressLens's kernel
    selector. Unsupported operations and non-contiguous arrays fall
    back to plain NumPy transparently.

    Identity through operations: `regresslens.array` does NOT
    currently return a `regresslens.array` from operations — `.sum()`
    returns a plain float, matching NumPy's own `.sum()` return type
    for a 1D array. There is no chained-operation identity to lose
    yet, because only reduction is wired up so far. This will matter
    once projection/filter/rolling are added (Phase 3 continuation)
    and is called out explicitly here so it isn't assumed silently
    correct later — see the project brief's note on `rd.array`
    identity preservation being a correctness requirement, not a
    nice-to-have.
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        arr = np.asarray(data)
        if arr.dtype.type not in _SUPPORTED_DTYPES:
            raise UnsupportedDTypeError(
                f"regresslens.array: unsupported dtype {arr.dtype}. "
                f"v0.1 supports float32 and float64 only."
            )
        self._data = arr

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    def to_numpy(self):
        """Escape hatch back to a plain ndarray. Always available —
        this is how unsupported operations get applied without the
        caller having to know that's what's happening."""
        return self._data

    def __repr__(self):
        return f"regresslens.array({self._data!r})"

    def __len__(self):
        return len(self._data)

    def sum(self, threads=None):
        return _dispatch_sum(self._data, threads)

    def mean(self, threads=None):
        n = self._data.shape[0]
        if n == 0:
            raise ValueError("mean of empty array")
        return _dispatch_sum(self._data, threads) / n

    # --- NumPy protocol handlers ---
    #
    # This is what makes `np.sum(rl_array)` work without the caller
    # writing `rl_array.sum()` explicitly. Unsupported functions
    # return NotImplemented, which tells NumPy to fall back to its
    # own implementation on the unwrapped array — the graceful
    # fallback the project brief requires, implemented via the
    # actual protocol mechanism rather than a manual dispatch table.
    _HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        handler = self._HANDLED_FUNCTIONS.get(func)
        if handler is None:
            logger.debug(
                "regresslens: %s not intercepted, falling back to NumPy "
                "on unwrapped array", func.__module__ + "." + func.__name__
            )
            # Unwrap any regresslens.array arguments to plain ndarrays
            # and let NumPy's default implementation handle it.
            unwrapped_args = tuple(
                a.to_numpy() if isinstance(a, array) else a for a in args
            )
            return func(*unwrapped_args, **kwargs)
        return handler(*args, **kwargs)

    def __array__(self, dtype=None, copy=None):
        # Lets plain np.* functions that DON'T go through
        # __array_function__ (e.g. np.asarray(rl_arr)) still work by
        # falling back to the underlying data.
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # No ufuncs are accelerated in this checkpoint — reduction is
        # the only operation wired up so far, and it isn't a ufunc.
        # This handler exists so ufunc calls (np.sqrt, np.add, etc.)
        # go through explicit, LOGGED fallback rather than silently
        # coercing via __array__ with no visibility into what
        # happened. Without this override, NumPy would still produce
        # correct results (via __array__), but with zero debug
        # logging — which matters once real users are trying to
        # understand why their pipeline did or didn't get
        # accelerated.
        logger.debug(
            "regresslens: ufunc %s not accelerated, falling back to NumPy",
            ufunc.__name__,
        )
        unwrapped_inputs = tuple(
            a.to_numpy() if isinstance(a, array) else a for a in inputs
        )
        return getattr(ufunc, method)(*unwrapped_inputs, **kwargs)


def _dispatch_sum(data, threads):
    if not data.flags["C_CONTIGUOUS"]:
        logger.debug(
            "regresslens: sum() on non-contiguous array, falling back to "
            "NumPy. This is exactly the pattern the contiguity-loss "
            "diagnosis feature (Phase 3 continuation) will flag instead "
            "of silently falling back."
        )
        return float(np.sum(data))
    threads_override = -1 if threads is None else int(threads)
    value, kernel = _native.reduce_sum_native(data, threads_override)
    logger.debug("regresslens: sum() used kernel=%s", kernel)
    return value


def _np_sum_handler(a, axis=None, dtype=None, out=None, **kwargs):
    if axis is not None or out is not None:
        # v0.1 only supports whole-array reduction. Multi-axis or
        # out= usage falls back rather than pretending to support it.
        logger.debug("regresslens: np.sum with axis/out, falling back to NumPy")
        unwrapped = a.to_numpy() if isinstance(a, array) else a
        return np.sum(unwrapped, axis=axis, dtype=dtype, out=out, **kwargs)
    data = a._data if isinstance(a, array) else np.asarray(a)
    return _dispatch_sum(data, None)


array._HANDLED_FUNCTIONS[np.sum] = _np_sum_handler
