"""Lazy loader for the optional native kernels (regresslens-accel).

``import regresslens`` never requires the native module. When the
``regresslens[accel]`` extra is not installed, cannot be imported, or
was built against a different ABI, every operation runs on the NumPy
implementations in ``_numpy_kernels`` and records ``numpy`` as its
kernel.

Environment:
  RGLNS_DISABLE_ACCEL=1     never load the native module (forces NumPy)
  RGLNS_FORCE_KERNEL=...    passed through to the native module, which
                            then uses only that kernel set. If it is set
                            and the module fails to load, the error is
                            raised instead of silently falling back,
                            because a test that asked for a specific
                            kernel must not run on NumPy by accident.
"""
import contextlib
import logging
import os
import threading

logger = logging.getLogger("regresslens")

ABI_VERSION = 2

_state = {"loaded": False, "module": None, "error": None}
_kernel = threading.local()


def kernel_arg():
    """The kernel rl.array requests from the native module: "auto" (the
    selector) unless overridden with using_kernel()."""
    return getattr(_kernel, "name", "auto")


@contextlib.contextmanager
def using_kernel(name):
    """Forces rl.array operations in this thread onto one kernel set
    ("scalar", "avx2"; "auto" restores the selector). For tests and
    verification on a single machine; RGLNS_FORCE_KERNEL does the same
    process-wide."""
    previous = kernel_arg()
    _kernel.name = name
    try:
        yield
    finally:
        _kernel.name = previous


def native():
    """The native module, or None when unavailable."""
    if not _state["loaded"]:
        _state["loaded"] = True
        _state["module"], _state["error"] = _load()
    return _state["module"]


def _load():
    if os.environ.get("RGLNS_DISABLE_ACCEL") == "1":
        return None, "disabled by RGLNS_DISABLE_ACCEL=1"
    forced = os.environ.get("RGLNS_FORCE_KERNEL", "")
    try:
        from regresslens_accel import _native as mod
    except ImportError as exc:
        if forced:
            raise
        return None, f"regresslens-accel not importable ({exc})"
    abi = getattr(mod, "ABI_VERSION", None)
    if abi != ABI_VERSION:
        msg = (f"regresslens-accel ABI {abi} does not match regresslens ABI "
               f"{ABI_VERSION}; reinstall matching versions")
        if forced:
            raise ImportError(msg)
        logger.warning("regresslens: %s. Using NumPy kernels.", msg)
        return None, msg
    return mod, None


def info():
    """Describes which kernels this process uses (for reports and the
    hardware fingerprint)."""
    mod = native()
    if mod is None:
        return {"native": False, "isa": "numpy", "reason": _state["error"]}
    d = dict(mod.info())
    d["native"] = True
    return d


def _reset_for_tests():
    _state.update(loaded=False, module=None, error=None)
