"""Is a callable NumPy-related, and what array is it operating on?

Shared by both observation backends. Never imports NumPy itself: until
the program's own `import numpy` has completed, nothing is a NumPy
callable.
"""
import sys
import types

# kind values
UFUNC = "ufunc"        # np.add, np.exp: ufunc objects
FUNC = "func"          # np.sum, np.convolve: C-level callables (incl. dispatchers)
PYFUNC = "pyfunc"      # NumPy functions implemented in Python (need PY_RETURN timing)
METHOD = "method"      # x.sum(): builtin method bound to an ndarray (or other NumPy object)

_dtype_names = {}


class Callee:
    __slots__ = ("kind", "operator", "name", "pycode")

    def __init__(self, kind, operator, name, pycode=None):
        self.kind = kind
        self.operator = operator
        self.name = name
        self.pycode = pycode


_np = None


def numpy_ready():
    """NumPy once its import has completed, else None. While NumPy is
    still importing, its partially initialized module lacks attributes
    (np.ufunc, np.ndarray), and code it exec()s looks like user code."""
    global _np
    if _np is None:
        np = sys.modules.get("numpy")
        if np is None or getattr(getattr(np, "__spec__", None), "_initializing", False):
            return None
        _np = np
    return _np


def _is_numpy_module(modname):
    return bool(modname) and (modname == "numpy" or modname.startswith("numpy."))


def classify(obj):
    """Callee for a NumPy-related callable, else None. Callables marked
    with __rglns_accelerated__ (the --accelerate wrappers) and anything
    from regresslens itself return None: they record their own calls."""
    np = numpy_ready()
    if np is None:
        return None
    if getattr(obj, "__rglns_accelerated__", None):
        return None
    t = type(obj)
    if t is np.ufunc:
        return Callee(UFUNC, "np." + obj.__name__, obj.__name__)
    mod = getattr(obj, "__module__", None)
    if mod and mod.startswith("regresslens"):
        return None
    if t is types.BuiltinMethodType or t is types.BuiltinFunctionType:
        self_obj = getattr(obj, "__self__", None)
        if self_obj is not None and not isinstance(self_obj, types.ModuleType):
            st = type(self_obj)
            if isinstance(self_obj, np.ndarray):
                return Callee(METHOD, "ndarray." + obj.__name__, obj.__name__)
            if _is_numpy_module(st.__module__):
                return Callee(METHOD, f"{st.__name__}.{obj.__name__}", obj.__name__)
            return None
        if _is_numpy_module(mod) or (isinstance(self_obj, types.ModuleType)
                                     and _is_numpy_module(self_obj.__name__)):
            return Callee(FUNC, "np." + obj.__name__, obj.__name__)
        return None
    if t.__name__ == "_ArrayFunctionDispatcher" and _is_numpy_module(t.__module__):
        return Callee(FUNC, "np." + obj.__name__, obj.__name__)
    if t is types.FunctionType and _is_numpy_module(mod):
        return Callee(PYFUNC, "np." + obj.__name__, obj.__name__, obj.__code__)
    if t is types.MethodDescriptorType and getattr(obj, "__objclass__", None) is np.ndarray:
        return Callee(FUNC, "ndarray." + obj.__name__, obj.__name__)
    return None


_NONE3 = (None, None, None)
_ndarray = None


def describe(arr):
    """(dtype name, n elements, C-contiguous) for an ndarray, else
    (None, None, None). On the per-call hot path, hence the caching:
    dtype names by id (dtype hashing is slow), ndarray type once."""
    global _ndarray
    if _ndarray is None:
        np = numpy_ready()
        if np is None:
            return _NONE3
        _ndarray = np.ndarray
    if not isinstance(arr, _ndarray):
        return _NONE3
    dt = arr.dtype
    hit = _dtype_names.get(id(dt))
    if hit is None or hit[0] is not dt:
        hit = _dtype_names[id(dt)] = (dt, dt.name)
    return hit[1], arr.size, arr.flags.c_contiguous
