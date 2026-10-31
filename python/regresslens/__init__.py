"""RegressLens: performance regression intelligence for recurring
scientific Python / NumPy pipelines.

  rglns run -- python script.py     observe unmodified NumPy code
  rglns compare A B --cmd ...       interleaved, anytime-valid A/B comparison
  rglns bisect GOOD BAD --cmd ...   find the commit that introduced a slowdown
  rglns check / baseline / watch    track a pipeline against its history

The optional native kernels (``pip install "regresslens[accel]"``) are
loaded lazily; without them every operation runs on NumPy and is
recorded with kernel ``numpy``.
"""
__all__ = ["array", "UnsupportedDTypeError", "flush", "accel_info"]
__version__ = "0.2.0"


def __getattr__(name):
    # Lazy, so that importing regresslens (which the injected
    # sitecustomize does at interpreter start-up) never imports NumPy:
    # a program that sets OMP_NUM_THREADS etc. before its own
    # `import numpy` must behave the same under rglns.
    if name in ("array", "UnsupportedDTypeError"):
        from . import _array
        value = getattr(_array, name)
    elif name == "flush":
        from .observe import flush as value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def accel_info():
    """Which kernels this process uses: {"native": bool, "isa": ...}."""
    from . import _accel
    return _accel.info()
