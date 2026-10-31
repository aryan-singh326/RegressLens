"""Observation backend for Python 3.12+: sys.monitoring (PEP 669).

CALL fires before every call. The callback classifies the callable. If
the caller is not user code, or the callable is not NumPy-related, it
returns DISABLE, and CPython never fires CALL at that bytecode location
again. Steady-state overhead is therefore paid only at NumPy call sites.

Timing: C-level callables (ufuncs, NumPy's dispatcher objects, ndarray
methods) end with C_RETURN/C_RAISE at the same location. The few NumPy
functions implemented in Python end with PY_RETURN, which is enabled
locally on just those functions' code objects.

Two implementations of the per-call hot path:
  * C (regresslens-accel installed): the callbacks are METH_FASTCALL
    functions in regresslens_accel._native (cpp/python_bindings/
    monitor.cpp). Only the once-per-location work (classification,
    call-site resolution) calls back into `_slow_path` here.
  * Pure Python, otherwise (or with RGLNS_PURE_OBSERVER=1). Several
    times slower per call, see docs/observation.md.

What it cannot see: operators (a + b compiles to BINARY_OP, not a
call), indexing (x[mask]), and implicit calls made by C code.
"""
import os
import sys
import threading
import time

from .. import _accel, callsite
from . import classify as _cls

_mon = sys.monitoring
_E = _mon.events
DISABLE = _mon.DISABLE
_perf = time.perf_counter_ns
_KIND_CODE = {_cls.UFUNC: 0, _cls.FUNC: 0, _cls.METHOD: 1, _cls.PYFUNC: 2}


class _Location:
    __slots__ = ("code", "site_id", "callee", "fixed")

    def __init__(self, code, site_id, callee, fixed):
        self.code = code    # keeps id(code) from being recycled while cached
        self.site_id = site_id
        self.callee = callee
        self.fixed = fixed  # the identical callable object every time (functions, ufuncs)


class MonitoringBackend:
    unobserved = ("arithmetic operators (a + b)", "indexing (x[mask])",
                  "non-NumPy libraries")

    def __init__(self, aggregator):
        self.agg = aggregator
        self.tool = None
        self._locs = {}
        self._tls = threading.local()
        self._py_codes = set()
        native = None if os.environ.get("RGLNS_PURE_OBSERVER") == "1" else _accel.native()
        self.fast = native if native is not None and hasattr(native, "mon_call") else None
        self.name = "sys.monitoring" + (" (C fast path)" if self.fast else "")
        self._site_ids = []      # fast path: site index -> call_site_id
        self._site_index = {}
        self._ops = []           # fast path: operator index -> operator name
        self._op_index = {}
        self._ndarray_set = False

    # -- lifecycle -------------------------------------------------------

    def install(self):
        for tool in (_mon.PROFILER_ID, 3, 4):
            if _mon.get_tool(tool) is None:
                _mon.use_tool_id(tool, "regresslens")
                self.tool = tool
                break
        else:
            raise RuntimeError("no free sys.monitoring tool id")
        reg = _mon.register_callback
        if self.fast is not None:
            self.fast.mon_init(self._slow_path, DISABLE)
            reg(self.tool, _E.CALL, self.fast.mon_call)
            reg(self.tool, _E.C_RETURN, self.fast.mon_c_return)
            reg(self.tool, _E.C_RAISE, self.fast.mon_c_raise)
            reg(self.tool, _E.PY_RETURN, self.fast.mon_py_return)
            self.agg.pre_flush.append(self._drain)
        else:
            reg(self.tool, _E.CALL, self._on_call)
            reg(self.tool, _E.C_RETURN, self._on_c_return)
            reg(self.tool, _E.C_RAISE, self._on_c_raise)
            reg(self.tool, _E.PY_RETURN, self._on_py_return)
        _mon.set_events(self.tool, _E.CALL | _E.C_RETURN | _E.C_RAISE)

    def uninstall(self):
        if self.tool is None:
            return
        _mon.set_events(self.tool, 0)
        for code in self._py_codes:
            _mon.set_local_events(self.tool, code, 0)
        for ev in (_E.CALL, _E.C_RETURN, _E.C_RAISE, _E.PY_RETURN):
            _mon.register_callback(self.tool, ev, None)
        _mon.free_tool_id(self.tool)
        self.tool = None
        if self.fast is not None:
            self._drain()

    # -- C fast path ----------------------------------------------------------

    def _slow_path(self, code, offset, func):
        """Called by the C callback at a new (or polymorphic) location.
        Returns None (not observed) or (site index, kind, fixed callable,
        callee code, method name, operator index)."""
        loc = self._new_location(code, offset, func, depth=2)
        if loc is None:
            return None
        if not self._ndarray_set:
            self.fast.mon_set_ndarray(_cls.numpy_ready().ndarray)  # classify() saw NumPy
            self._ndarray_set = True
        callee = loc.callee
        idx = self._site_index.get(loc.site_id)
        if idx is None:
            idx = self._site_index[loc.site_id] = len(self._site_ids)
            self._site_ids.append(loc.site_id)
        op = self._op_index.get(callee.operator)
        if op is None:
            op = self._op_index[callee.operator] = len(self._ops)
            self._ops.append(callee.operator)
        method = callee.name if callee.kind == _cls.METHOD else None
        return (idx, _KIND_CODE[callee.kind], loc.fixed, callee.pycode, method, op)

    def _drain(self):
        if self.fast is None:
            return
        rows, dtypes, _calls, outer_ns = self.fast.mon_drain()
        names = [d.name for d in dtypes]
        self.agg.ingest(
            [(self._site_ids[site], self._ops[op], names[dt] if dt >= 0 else None,
              n if n >= 0 else None, None if contig < 0 else bool(contig), "numpy",
              count, s, sl, samples)
             for site, op, dt, n, contig, count, s, sl, samples in rows], outer_ns)

    # -- pure-Python path --------------------------------------------------------

    def _stack(self):
        st = getattr(self._tls, "stack", None)
        if st is None:
            st = self._tls.stack = []
            self._tls.table = self.agg.table()
        return st

    def _on_call(self, code, offset, func, arg0):
        # Keyed by id(code): hashing a code object costs ~1.4 us.
        try:
            loc = self._locs.get((id(code), offset))
            if loc is None or loc.code is not code:
                loc = self._new_location(code, offset, func, depth=2)
                if loc is None:
                    return DISABLE
                callee = loc.callee
            elif loc.fixed is func:
                callee = loc.callee
            else:
                # Same location, different callable object: bound methods
                # are recreated on every call, so re-check cheaply.
                callee = loc.callee if loc.fixed is None else None
                if callee is None or (callee.kind == _cls.METHOD and
                                      getattr(func, "__name__", None) != callee.name):
                    callee = _cls.classify(func)
                    if callee is None:
                        return None  # polymorphic site; keep monitoring it
            if callee.kind == _cls.METHOD:
                arr = getattr(func, "__self__", None)
            else:
                arr = arg0
            dtype, n, contig = _cls.describe(arr)
            stack = self._stack()
            stack.append((code, offset, loc.site_id, callee, dtype, n, contig, _perf()))
            return None
        except Exception as exc:  # never break the user's program
            self._fail(exc)
            return DISABLE

    def _new_location(self, code, offset, func, depth):
        if not callsite.is_user_file(code.co_filename):
            return None
        callee = _cls.classify(func)
        if callee is None:
            return None
        site = callsite.resolve(code, offset=offset, callee=callee.name)
        frame = sys._getframe(depth)  # the user frame making the call
        self.agg.note_site(site, callsite.user_stack(frame))
        if callee.kind == _cls.PYFUNC and callee.pycode not in self._py_codes:
            self._py_codes.add(callee.pycode)
            _mon.set_local_events(self.tool, callee.pycode, _E.PY_RETURN)
        fixed = None if callee.kind == _cls.METHOD else func
        loc = _Location(code, site.id, callee, fixed)
        self._locs[(id(code), offset)] = loc
        return loc

    def _finish(self, code, offset, record):
        t1 = _perf()
        try:
            stack = getattr(self._tls, "stack", None)
            if not stack:
                return
            top = stack[-1]
            if top[0] is not code or top[1] != offset:
                # A stranded entry (e.g. a Python-implemented callable that
                # raised, so no PY_RETURN arrived) may sit on top.
                for i in range(len(stack) - 2, max(-1, len(stack) - 8), -1):
                    if stack[i][0] is code and stack[i][1] == offset:
                        del stack[i + 1:]
                        top = stack[-1]
                        break
                else:
                    return
            stack.pop()
            if record:
                self._record(top, t1 - top[7], not stack)
        except Exception as exc:  # never break the user's program
            self._fail(exc)

    def _record(self, entry, ns, outermost):
        _code, _off, site_id, callee, dtype, n, contig, _t0 = entry
        table = self._tls.table
        if outermost:
            table.outer_ns += ns
        self.agg.record(site_id, callee.operator, dtype, n, contig, "numpy", float(ns),
                        table=table)

    def _on_c_return(self, code, offset, func, arg0):
        self._finish(code, offset, True)

    def _on_c_raise(self, code, offset, func, arg0):
        self._finish(code, offset, False)

    def _on_py_return(self, code, offset, retval):
        t1 = _perf()
        try:
            stack = getattr(self._tls, "stack", None)
            if not stack:
                return None
            top = stack[-1]
            callee = top[3]
            if callee.kind == _cls.PYFUNC and callee.pycode is code:
                stack.pop()
                self._record(top, t1 - top[7], not stack)
        except Exception as exc:  # never break the user's program
            self._fail(exc)
        return None

    def _fail(self, exc):
        """An exception raised by a sys.monitoring callback propagates into
        the monitored program, so the pure-Python callbacks catch
        everything and stop observing instead. (The C fast path clears
        errors itself.)"""
        try:
            self.uninstall()
        except Exception:
            pass
        print(f"rglns: observation stopped after an internal error "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)
