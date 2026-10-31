"""rl.array: explicit opt-in acceleration for NumPy arrays.

Wrap an array once with ``rl.array(x)`` and call the supported
operations (sum/mean, project_affine, filter_gt, rolling_sum/mean/var)
as methods, or reach sum through ``np.sum`` via NumPy's dispatch
protocol. Everything else falls back to plain NumPy on the wrapped
array. This is not a full NumPy-compatible array type and does not try
to be one.

Kernel choice per call:
  * native kernel (scalar / avx2 / mt_avx2, picked by the selector) for
    1-D C-contiguous float32/float64 input when regresslens-accel is
    installed;
  * ``numpy_fallback`` for input the native kernels do not take
    (non-contiguous or multi-dimensional);
  * ``numpy`` when the native kernels are not installed at all.

NaN/Inf: results match NumPy's IEEE semantics on every path (rolling
windows containing NaN are NaN; other windows are unaffected).

Every call is recorded with its call site, kernel and runtime in the
in-process aggregator (flushed once at exit), never written per call.
Operations that return arrays return a new rl.array, so chained calls
stay accelerated at every step.
"""
import logging
import time

import numpy as np

from . import _accel
from . import _numpy_kernels as _np_kernels
from . import observe as _observe
from . import store as _store

logger = logging.getLogger("regresslens")

_SUPPORTED_DTYPES = (np.float32, np.float64)
_perf = time.perf_counter_ns


class UnsupportedDTypeError(TypeError):
    pass


def _native_ok(data):
    return data.ndim == 1 and data.flags.c_contiguous


def _record(callee, operator, data, kernel, ns, selectivity=None, window=None, site=None):
    site = site or _observe.caller_site(callee)
    if site is None:
        return
    _observe.get_aggregator().record(site.id, operator, data.dtype.name, int(data.size),
                                 bool(data.flags.c_contiguous), kernel, float(ns),
                                 selectivity=selectivity, window=window)


def _historical_selectivity(site):
    """Expected selectivity at this call site, or None without history.

    This process's own calls at the site win as soon as there are two:
    data can drift between runs, and a long average over old runs would
    keep the selector on a stale kernel for many runs after the
    selectivity left the AVX2 band. Earlier runs in the trace database
    only seed the first calls of a process."""
    total, count = _observe.get_aggregator().selectivity_estimate(site.id)
    if count >= 2:
        return total / count
    if site.id not in _selectivity_db_cache and _store.tracing_enabled():
        try:
            conn = _store.connect()
            try:
                _selectivity_db_cache[site.id] = _store.site_selectivity_history(conn, site.id)
            finally:
                conn.close()
        except Exception:
            _selectivity_db_cache[site.id] = (0.0, 0)
    db_total, db_count = _selectivity_db_cache.get(site.id, (0.0, 0))
    total += db_total
    count += db_count
    return total / count if count else None


_selectivity_db_cache = {}


class array:
    """A NumPy array wrapper whose supported operations run on RegressLens
    kernels. See the module docstring for kernel choice and fallbacks."""

    __slots__ = ("_data",)

    def __init__(self, data):
        arr = np.asarray(data)
        if arr.dtype.type not in _SUPPORTED_DTYPES:
            raise UnsupportedDTypeError(
                f"regresslens.array: unsupported dtype {arr.dtype}; "
                f"float32 and float64 are supported.")
        self._data = arr

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    def to_numpy(self):
        """The underlying ndarray."""
        return self._data

    def __repr__(self):
        return f"regresslens.array({self._data!r})"

    def __len__(self):
        return len(self._data)

    # --- reductions -----------------------------------------------------

    def sum(self, threads=None):
        """Sum of all elements, as a NumPy scalar of the array's dtype
        (like np.sum). The native kernel accumulates in double with a
        compensation term, so it is at least as accurate as np.sum."""
        return _dispatch_sum(self._data, threads, "sum")

    def mean(self, threads=None):
        """Mean of all elements, as a NumPy scalar of the array's dtype.
        The native kernel divides the compensated sum with a single
        rounding."""
        if self._data.size == 0:
            raise ValueError("mean of empty array")
        return _dispatch_sum(self._data, threads, "mean", mean=True)

    # --- elementwise --------------------------------------------------------

    def project_affine(self, scale, offset):
        """out[i] = scale * self[i] + offset, as a new rl.array."""
        data = self._data
        native = _accel.native()
        if native is not None and _native_ok(data):
            out, kernel, ns = native.project_affine(data, float(scale), float(offset),
                                                    _accel.kernel_arg())
        else:
            kernel = "numpy" if native is None else "numpy_fallback"
            t0 = _perf()
            out = _np_kernels.project_affine(data, scale, offset)
            ns = _perf() - t0
        _record("project_affine", "projection", data, kernel, ns)
        return array(out)

    def filter_gt(self, threshold):
        """Elements strictly greater than threshold, in order, as a new
        rl.array. NaN never passes (like data[data > threshold])."""
        data = self._data
        native = _accel.native()
        site = _observe.caller_site("filter_gt")
        n = data.size
        if native is not None and _native_ok(data):
            # Selectivity feedback: the kernel selector chooses between
            # scalar and AVX2 by expected selectivity, taken from this
            # call site's history (-1 = no history yet).
            hist = _historical_selectivity(site) if site is not None else None
            full, count, kernel, ns = native.filter_gt(
                data, float(threshold), _accel.kernel_arg(), 0, -1.0 if hist is None else hist)
            out = full[:count].copy() if count < n // 2 else full[:count]
        else:
            kernel = "numpy" if native is None else "numpy_fallback"
            t0 = _perf()
            out = _np_kernels.filter_gt(data, threshold)
            ns = _perf() - t0
            count = out.size
        _record("filter_gt", "filter", data, kernel, ns,
                selectivity=(count / n) if n else None, site=site)
        return array(out)

    # --- rolling windows ----------------------------------------------------

    def rolling_sum(self, window):
        """Sum of each complete window (n - window + 1 outputs), as a new
        rl.array. Equivalent to np.convolve(x, np.ones(window), "valid")
        for n >= window; empty when n < window."""
        return self._rolling("rolling_sum", window)

    def rolling_mean(self, window):
        """Mean of each complete window, as a new rl.array."""
        return self._rolling("rolling_mean", window)

    def rolling_var(self, window, ddof=0):
        """Variance of each complete window (np.var semantics, with
        ddof), as a new rl.array."""
        return self._rolling("rolling_var", window, ddof)

    def _rolling(self, op, window, ddof=0):
        window = int(window)
        if window < 1:
            raise ValueError(f"{op}: window must be >= 1, got {window}")
        data = self._data
        native = _accel.native()
        out = None
        if native is not None and _native_ok(data) and not (op == "rolling_var"
                                                            and ddof >= window):
            if op == "rolling_var":
                out, kernel, ns = native.rolling_var(data, window, int(ddof))
            else:
                out, kernel, ns = getattr(native, op)(data, window, _accel.kernel_arg())
            # None: finite input large enough to risk accumulator
            # overflow; NumPy's own overflow behavior applies instead.
        if out is None:
            kernel = "numpy" if native is None else "numpy_fallback"
            t0 = _perf()
            if op == "rolling_var":
                out = _np_kernels.rolling_var(data.ravel() if data.ndim != 1 else data,
                                              window, ddof)
            else:
                out = getattr(_np_kernels, op)(data.ravel() if data.ndim != 1 else data,
                                               window)
            ns = _perf() - t0
        _record(op, "rolling", data, kernel, ns, window=window)
        return array(out)

    # --- NumPy protocols ------------------------------------------------------
    #
    # __array_function__ makes np.sum(rl_arr) use the RegressLens kernel.
    # Everything not in _HANDLED_FUNCTIONS (and every ufunc) is applied
    # by NumPy to the unwrapped arrays.
    _HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        handler = self._HANDLED_FUNCTIONS.get(func)
        if handler is None:
            logger.debug("regresslens: %s not accelerated; NumPy fallback", func.__name__)
            return func(*_unwrap(args), **_unwrap_kwargs(kwargs))
        return handler(*args, **kwargs)

    def __array__(self, dtype=None, copy=None):
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        logger.debug("regresslens: ufunc %s not accelerated; NumPy fallback", ufunc.__name__)
        return getattr(ufunc, method)(*_unwrap(inputs), **_unwrap_kwargs(kwargs))


def _unwrap(seq):
    return tuple(a.to_numpy() if isinstance(a, array) else a for a in seq)


def _unwrap_kwargs(kwargs):
    if "out" in kwargs and isinstance(kwargs["out"], tuple):
        kwargs = dict(kwargs, out=_unwrap(kwargs["out"]))
    return kwargs


def _dispatch_sum(data, threads, callee, mean=False):
    native = _accel.native()
    if native is not None and _native_ok(data):
        fn = native.reduce_mean if mean else native.reduce_sum
        value, kernel, ns = fn(data, _accel.kernel_arg(), 0 if threads is None else int(threads))
        result = data.dtype.type(value)
    else:
        kernel = "numpy" if native is None else "numpy_fallback"
        t0 = _perf()
        result = np.mean(data) if mean else _np_kernels.sum_(data)
        ns = _perf() - t0
    _record(callee, "reduction", data, kernel, ns)
    return result


def _np_sum_handler(a, axis=None, dtype=None, out=None, **kwargs):
    if axis is not None or dtype is not None or out is not None or kwargs:
        # Whole-array reduction only; anything else is NumPy's.
        return np.sum(a.to_numpy() if isinstance(a, array) else a, axis=axis, dtype=dtype,
                      out=out, **kwargs)
    data = a._data if isinstance(a, array) else np.asarray(a)
    return _dispatch_sum(data, None, "sum")


array._HANDLED_FUNCTIONS[np.sum] = _np_sum_handler
