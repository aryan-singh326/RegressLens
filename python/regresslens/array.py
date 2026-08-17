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
import os
import time
import numpy as np

from . import _regresslens_native as _native
from . import trace as _trace

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

    Identity through operations: project_affine(), filter_gt(), and
    rolling_sum()/rolling_mean() all return a NEW regresslens.array,
    not a plain ndarray — so chained calls
    (arr.project_affine(...).filter_gt(...)) stay accelerated at
    every step instead of silently dropping back to plain NumPy after
    the first operation. sum()/mean() return plain floats, matching
    NumPy's own reduction return type — there's nothing to preserve
    identity of once you're down to a scalar. This is tested
    explicitly in test_array.py's TestIdentityPreservation, per the
    project brief's requirement that silent acceleration loss is a
    correctness bug, not a performance footnote.
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

    def project_affine(self, scale, offset):
        """out[i] = scale * self[i] + offset. Returns a new
        regresslens.array — this is the representative elementwise
        transform underlying ops like diff_log/zscore in the eventual
        public API (see project brief); those aren't implemented as
        named functions yet, this is the primitive they'll build on.
        """
        if not self._data.flags["C_CONTIGUOUS"]:
            logger.debug(
                "regresslens: project_affine() on non-contiguous array, "
                "falling back to NumPy"
            )
            t0 = time.perf_counter_ns()
            result = self._data * scale + offset
            t1 = time.perf_counter_ns()
            _trace.record_trace(
                operator="projection",
                row_count=self._data.shape[0],
                dtype=str(self._data.dtype),
                contiguous=False,
                selected_kernel="numpy_fallback",
                runtime_ns=float(t1 - t0),
                available_cores=os.cpu_count() or 1,
                call_site=_trace.capture_call_site(),
            )
            return array(result)
        result, kernel, runtime_ns = _native.project_affine_native(
            self._data, scale, offset
        )
        logger.debug("regresslens: project_affine() used kernel=%s", kernel)
        _trace.record_trace(
            operator="projection",
            row_count=self._data.shape[0],
            dtype=str(self._data.dtype),
            contiguous=True,
            selected_kernel=kernel,
            runtime_ns=runtime_ns,
            available_cores=os.cpu_count() or 1,
        )
        return array(result)

    def filter_gt(self, threshold):
        """Elements strictly greater than threshold, in order.
        Returns a new regresslens.array sized to the actual match
        count — not the worst-case allocation the native layer uses
        internally, which is sliced away here before the caller ever
        sees it."""
        if not self._data.flags["C_CONTIGUOUS"]:
            logger.debug(
                "regresslens: filter_gt() on non-contiguous array, "
                "falling back to NumPy"
            )
            t0 = time.perf_counter_ns()
            result = self._data[self._data > threshold]
            t1 = time.perf_counter_ns()
            n = self._data.shape[0]
            _trace.record_trace(
                operator="filter",
                row_count=n,
                dtype=str(self._data.dtype),
                contiguous=False,
                selected_kernel="numpy_fallback",
                runtime_ns=float(t1 - t0),
                available_cores=os.cpu_count() or 1,
                selectivity=(len(result) / n) if n > 0 else None,
                call_site=_trace.capture_call_site(),
            )
            return array(result)
        n = self._data.shape[0]
        # Feedback loop: use real historical selectivity at this call
        # shape if we have it, instead of always assuming 0.5. Falls
        # back to -1 ("unknown") when there's no history yet, which
        # the native layer treats as the neutral 0.5 default.
        historical_selectivity = _trace.get_average_selectivity(
            str(self._data.dtype), n
        )
        estimated_selectivity = (
            historical_selectivity if historical_selectivity is not None else -1.0
        )
        full, count, kernel, runtime_ns = _native.filter_gt_native(
            self._data, threshold, -1, estimated_selectivity
        )
        logger.debug(
            "regresslens: filter_gt() used kernel=%s (matched %d/%d, "
            "estimated_selectivity=%s)",
            kernel, count, len(self._data),
            f"{historical_selectivity:.3f}" if historical_selectivity is not None
            else "unknown",
        )
        _trace.record_trace(
            operator="filter",
            row_count=n,
            dtype=str(self._data.dtype),
            contiguous=True,
            selected_kernel=kernel,
            runtime_ns=runtime_ns,
            available_cores=os.cpu_count() or 1,
            # This is the REAL observed selectivity, unlike the
            # hardcoded 0.5 assumption the native layer's kernel
            # selector currently uses (see kernel_selector.cpp). This
            # is exactly the trace data a future selectivity-aware
            # heuristic would read from, once enough history exists.
            selectivity=(count / n) if n > 0 else None,
        )
        return array(full[:count])

    def rolling_sum(self, window):
        """Fixed-window rolling sum. Output length is
        max(0, n - window + 1) — no partial windows, matching v0.1's
        no-NaN-handling scope (a partial window would need a
        sentinel). Returns a new regresslens.array."""
        if not self._data.flags["C_CONTIGUOUS"]:
            logger.debug(
                "regresslens: rolling_sum() on non-contiguous array, "
                "falling back to NumPy"
            )
            n = self._data.shape[0]
            call_site = _trace.capture_call_site()
            if window == 0 or n < window:
                _trace.record_trace(
                    operator="rolling", row_count=n, dtype=str(self._data.dtype),
                    contiguous=False, selected_kernel="numpy_fallback",
                    runtime_ns=0.0, available_cores=os.cpu_count() or 1,
                    window=window, call_site=call_site,
                )
                return array(np.array([], dtype=self._data.dtype))
            # np.convolve is the standard NumPy-only way to compute
            # this; used only on the fallback path, never the
            # accelerated one.
            t0 = time.perf_counter_ns()
            kernel_win = np.ones(window, dtype=self._data.dtype)
            result = np.convolve(self._data, kernel_win, mode="valid")
            t1 = time.perf_counter_ns()
            _trace.record_trace(
                operator="rolling",
                row_count=n,
                dtype=str(self._data.dtype),
                contiguous=False,
                selected_kernel="numpy_fallback",
                runtime_ns=float(t1 - t0),
                available_cores=os.cpu_count() or 1,
                window=window,
                call_site=call_site,
            )
            return array(result)
        result, kernel, runtime_ns = _native.rolling_sum_native(self._data, window)
        logger.debug("regresslens: rolling_sum() used kernel=%s", kernel)
        _trace.record_trace(
            operator="rolling",
            row_count=self._data.shape[0],
            dtype=str(self._data.dtype),
            contiguous=True,
            selected_kernel=kernel,
            runtime_ns=runtime_ns,
            available_cores=os.cpu_count() or 1,
            window=window,
        )
        return array(result)

    def rolling_mean(self, window):
        """Fixed-window rolling mean. Returns a new regresslens.array."""
        summed = self.rolling_sum(window)
        return array(summed._data / window)

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
            "NumPy."
        )
        t0 = time.perf_counter_ns()
        result = float(np.sum(data))
        t1 = time.perf_counter_ns()
        _trace.record_trace(
            operator="reduction",
            row_count=data.shape[0],
            dtype=str(data.dtype),
            contiguous=False,
            selected_kernel="numpy_fallback",
            runtime_ns=float(t1 - t0),
            available_cores=os.cpu_count() or 1,
            call_site=_trace.capture_call_site(),
        )
        return result
    threads_override = -1 if threads is None else int(threads)
    value, kernel, runtime_ns = _native.reduce_sum_native(data, threads_override)
    logger.debug("regresslens: sum() used kernel=%s", kernel)
    _trace.record_trace(
        operator="reduction",
        row_count=data.shape[0],
        dtype=str(data.dtype),
        contiguous=True,
        selected_kernel=kernel,
        runtime_ns=runtime_ns,
        available_cores=os.cpu_count() or 1,
    )
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
