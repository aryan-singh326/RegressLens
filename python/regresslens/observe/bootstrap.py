"""Runs inside every Python process launched by the rglns CLI.

Called from the injected sitecustomize.py when RGLNS_ACTIVE=1:
  * observation (RGLNS_OBSERVE=1, the default): picks the backend for
    this interpreter, optionally --accelerate and the Tier D import hook;
  * drift canaries (RGLNS_CHILD_CANARY=1): a sample right after the
    program's own `import numpy` (never earlier, so code that sets
    thread-count variables before importing NumPy is unaffected), one at
    exit, and periodic ones between observed calls on long runs. They
    are taken here, in the measured process, because a sample taken in
    the waiting CLI process measures that process's scheduling instead
    (on hybrid CPUs an idle parent lands on an efficiency core, and its
    canary read 3-9x slower while the workload itself was unchanged);
  * an exit handler that prints the coverage report (stderr) and writes
    the aggregated calls and canary samples (one write).

Nothing here may break the user's program: any failure is reported on
stderr and the program runs unobserved.
"""
import atexit
import os
import sys
import time

from .. import canary
from . import aggregator as _agg

_state = {"installed": False, "backend": None, "started": None}


def _choose_backend(agg):
    forced = os.environ.get("RGLNS_BACKEND", "")
    if forced == "profile" or (not forced and sys.version_info < (3, 12)):
        from .backend_profile import ProfileBackend
        return ProfileBackend(agg)
    from .backend_monitoring import MonitoringBackend
    return MonitoringBackend(agg)


def install():
    if _state["installed"]:
        return
    _state["installed"] = True
    agg = _agg.get()
    observe = os.environ.get("RGLNS_OBSERVE", "1") == "1"
    if observe:
        backend = _choose_backend(agg)
        backend.install()
        _state["backend"] = backend
        agg.backend_name = backend.name
        agg.unobserved = list(backend.unobserved)

    if canary.enabled() and os.environ.get("RGLNS_CHILD_CANARY") == "1":
        _state["canaries"] = True
        _after_numpy_import(lambda np: _canary_sample(agg, "start"))
        if observe:
            _start_periodic_canaries(agg)

    if observe and os.environ.get("RGLNS_ACCELERATE") == "1":
        _after_numpy_import(_install_accelerate)
    if observe and os.environ.get("RGLNS_TIERD") == "1":
        from . import tierd
        tierd.install_import_hook()
        agg.unobserved = [u for u in agg.unobserved
                          if not u.startswith(("arithmetic operators", "indexing"))]
        agg.unobserved.append("operators and indexing outside your own modules")

    atexit.register(_at_exit)
    _state["started"] = time.perf_counter_ns()  # the program starts now


def _canary_sample(agg, where):
    """One canary sample with a fresh buffer, released right after, so the
    measured program never carries the 4x-LLC memory buffer around."""
    try:
        agg.canary_samples.append(canary.Canary().measure(where))
    except Exception as exc:  # never break the measured program
        print(f"rglns: canary failed ({exc})", file=sys.stderr)


def _install_accelerate(np):
    from . import accelerate
    if not accelerate.install(np):
        print("rglns: --accelerate requested but regresslens-accel is not installed; "
              "running unaccelerated", file=sys.stderr)


def _after_numpy_import(callback):
    """Runs callback(numpy) once numpy has been imported (now, if it
    already is), without importing it ourselves."""
    if "numpy" in sys.modules:
        callback(sys.modules["numpy"])
        return
    import importlib.abc
    import importlib.machinery

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "numpy":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                callback(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())


def _start_periodic_canaries(agg):
    """Long runs get an extra canary sample every PERIOD_S seconds; the CLI
    takes the start/end samples. Samples are taken inline, between two
    observed calls (see Aggregator.record). A background thread would
    measure contention from the program's own concurrent work instead of
    the environment."""
    agg.enable_periodic_canaries(canary.PERIOD_S, lambda: canary.Canary().measure("periodic"))


def coverage_report(agg, wall_ns):
    calls, sites, outer_ns = agg.totals()
    lines = [f"rglns: observed {calls:,} NumPy calls at {sites} call sites "
             f"[{agg.backend_name or 'rl.array only'}]"]
    if wall_ns > 0:
        share = 100.0 * outer_ns / wall_ns
        lines.append(f"rglns: {share:.1f}% of wall time ({outer_ns / 1e9:.3f} s of "
                     f"{wall_ns / 1e9:.3f} s) was inside observed calls")
    if agg.unobserved:
        lines.append("rglns: not observed: " + ", ".join(agg.unobserved))
    return "\n".join(lines)


def _at_exit():
    backend = _state["backend"]
    wall_ns = time.perf_counter_ns() - (_state["started"] or time.perf_counter_ns())
    if backend is not None:
        try:
            backend.uninstall()
        except Exception:
            pass
    agg = _agg.get()
    # Canaries taken while the program ran are excluded from its time; the
    # end sample below is taken after wall_ns was measured.
    canary_cost = sum(s.get("cost_ns", 0.0) for s in agg.canary_samples)
    program_ns = max(wall_ns - canary_cost, 0.0)
    if _state.get("canaries") and "numpy" in sys.modules:
        _canary_sample(agg, "end")
    if backend is not None and os.environ.get("RGLNS_QUIET") != "1":
        print(coverage_report(agg, program_ns), file=sys.stderr)
    # The program's own run time: from the end of this bootstrap to exit
    # (after the program's own atexit handlers, which run first), minus
    # canary time. It excludes interpreter start-up, this bootstrap and
    # the database write below, which are constant per run and would
    # only dilute relative effects. The CLI uses it as the run's wall time.
    agg.flush(wall_ns=float(program_ns) if os.environ.get("RGLNS_RUN_ID") else None)
