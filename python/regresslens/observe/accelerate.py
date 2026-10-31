"""`rglns run --accelerate`: route eligible module-level NumPy calls to
the native kernels.

Scope is deliberately narrow, because ndarray is an immutable C type
(x.sum() cannot be patched) and operators bypass call hooks entirely:

  * only module-level functions are replaced: numpy.sum and numpy.mean;
  * only when called directly from user code (NumPy's own internal
    calls always get the original);
  * only for exactly-ndarray, 1-D, C-contiguous float32/float64 input,
    non-empty, with no other arguments (default kwargs);
  * everything else is delegated to the original function unchanged.

The result type matches NumPy's (np.float64 / np.float32 scalar). Values
can differ in the last bits because the native sum is compensated, which
is at least as accurate as NumPy's pairwise summation (docs/accuracy.md).
Each accelerated call is recorded with the native kernel's name.
"""
import functools
import sys
import time

from .. import _accel, callsite
from . import aggregator as _agg

_installed = {}


def _eligible(np, a, args, kwargs):
    return (not args and not kwargs and type(a) is np.ndarray and a.ndim == 1
            and a.size > 0 and a.flags.c_contiguous
            and (a.dtype == np.float64 or a.dtype == np.float32))


def _describe(np, a):
    if isinstance(a, np.ndarray):
        return a.dtype.name, a.size, a.flags.c_contiguous
    return None, None, None


def _make_wrapper(np, original, operator, native_fn):
    native = _accel.native()

    @functools.wraps(original)
    def wrapper(a, *args, **kwargs):
        caller = sys._getframe(1)
        if not callsite.is_user_file(caller.f_code.co_filename):
            return original(a, *args, **kwargs)
        # The observation backend skips this wrapper (it is marked
        # __rglns_accelerated__), so every user call is recorded here,
        # accelerated or not.
        t0 = time.perf_counter_ns()
        if native is not None and _eligible(np, a, args, kwargs):
            value, kernel, _ns = getattr(native, native_fn)(a)
            result = a.dtype.type(value)
        else:
            result = original(a, *args, **kwargs)
            kernel = "numpy"
        elapsed = float(time.perf_counter_ns() - t0)
        dtype, n, contig = _describe(np, a)
        agg = _agg.get()
        site = callsite.resolve_frame(caller, callee=original.__name__)
        agg.note_site(site, callsite.user_stack(caller))
        agg.record(site.id, operator, dtype, n, contig, kernel, elapsed)
        return result

    wrapper.__rglns_accelerated__ = operator
    wrapper.__rglns_original__ = original
    return wrapper


def install(np):
    """Patches numpy.sum/numpy.mean. Idempotent."""
    if _installed or _accel.native() is None:
        return False
    for name, operator, native_fn in (("sum", "np.sum", "reduce_sum"),
                                      ("mean", "np.mean", "reduce_mean")):
        original = getattr(np, name)
        _installed[name] = original
        setattr(np, name, _make_wrapper(np, original, operator, native_fn))
    return True


def uninstall():
    np = sys.modules.get("numpy")
    if np is not None:
        for name, original in _installed.items():
            setattr(np, name, original)
    _installed.clear()
