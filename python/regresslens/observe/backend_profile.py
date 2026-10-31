"""Observation backend for Python 3.9-3.11: sys.setprofile.

The profile hook sees every Python call/return and every call to a
builtin C function ('c_call'), in every thread (threading.setprofile
covers threads started later). It records:

  * c_call from user code to a NumPy builtin (x.sum(), np.dot, np.empty)
  * 'call' events entering a NumPy Python function directly from user
    code (np.sum, np.mean, np.convolve: on NumPy >= 1.25 a C dispatcher
    sits in between, which the profiler does not report, so the Python
    implementation appears to be called by the user frame; on older
    NumPy the dispatcher is Python code in the pseudo-file
    "<__array_function__ internals>", which counts as NumPy)

Calls nested inside an already-recorded NumPy call are ignored, so
x.sum() (a C method that calls back into numpy/_core/_methods.py) is
counted once.

Blind spots, beyond operators and indexing: ufunc objects (np.add(a, b),
np.exp(x)). A ufunc is neither a Python function nor a builtin C
function, so the profiler never reports calling one. This backend is
also much slower than sys.monitoring because the hook runs on every
call in the program, not just at NumPy call sites.
"""
import os
import sys
import threading
import time

from .. import callsite
from . import classify as _cls

_perf = time.perf_counter_ns


class ProfileBackend:
    name = "sys.setprofile"
    unobserved = ("arithmetic operators (a + b)", "indexing (x[mask])",
                  "ufunc objects (np.add(a, b), np.exp(x))", "non-NumPy libraries")

    def __init__(self, aggregator):
        self.agg = aggregator
        self._tls = threading.local()
        self._numpy_dirs = None
        self._file_kind = {}  # co_filename -> "user" | "numpy" | "other"
        self._sites = {}

    def install(self):
        sys.setprofile(self._hook)
        threading.setprofile(self._hook)

    def uninstall(self):
        sys.setprofile(None)
        threading.setprofile(None)

    def _kind(self, filename):
        k = self._file_kind.get(filename)
        if k is None:
            np = sys.modules.get("numpy")
            if self._numpy_dirs is None and np is not None:
                self._numpy_dirs = os.path.normcase(os.path.dirname(os.path.abspath(np.__file__)))
            norm = os.path.normcase(os.path.abspath(filename)) if not filename.startswith("<") else ""
            if filename == callsite.NUMPY_PSEUDO_FILE or (self._numpy_dirs and
                                                          norm.startswith(self._numpy_dirs)):
                k = "numpy"
            elif callsite.is_user_file(filename):
                k = "user"
            else:
                k = "other"
            # Cache only once NumPy is known, or the answer may change.
            if self._numpy_dirs is not None:
                self._file_kind[filename] = k
        return k

    def _state(self):
        st = getattr(self._tls, "stack", None)
        if st is None:
            st = self._tls.stack = []
            self._tls.table = self.agg.table()
        return st

    def _site(self, frame, name):
        site = callsite.resolve_frame(frame, callee=name)
        if site.id not in self.agg.sites:
            self.agg.note_site(site, callsite.user_stack(frame))
        return site.id

    def _hook(self, frame, event, arg):
        try:
            if event == "c_call":
                if self._kind(frame.f_code.co_filename) != "user":
                    return
                stack = self._state()
                callee = _cls.classify(arg)
                if callee is None:
                    return
                arr = getattr(arg, "__self__", None) if callee.kind == _cls.METHOD else None
                dtype, n, contig = _cls.describe(arr)
                stack.append(("c", frame, arg, self._site(frame, callee.name), callee.operator,
                              dtype, n, contig, _perf()))
            elif event == "c_return" or event == "c_exception":
                stack = getattr(self._tls, "stack", None)
                # Matched by frame only: a frame cannot make another call while
                # a C call it made is running, and on 3.12+ c_return may carry
                # a different object than the matching c_call.
                if stack and stack[-1][0] == "c" and stack[-1][1] is frame:
                    t1 = _perf()
                    entry = stack.pop()
                    if event == "c_return":
                        self._record(entry, t1 - entry[8], not stack)
            elif event == "call":
                code = frame.f_code
                if self._kind(code.co_filename) != "numpy":
                    return
                caller = frame.f_back
                if caller is None or self._kind(caller.f_code.co_filename) != "user":
                    return
                if code.co_name.endswith("_dispatcher"):
                    return  # NumPy >= 1.25 calls an argument-dispatcher helper first
                stack = self._state()
                if stack and stack[-1][0] == "c" and stack[-1][1] is caller:
                    return  # NumPy C method calling back into NumPy Python code
                arr = frame.f_locals.get("a") if code.co_argcount else None
                if arr is None and code.co_argcount:
                    arr = frame.f_locals.get(code.co_varnames[0])
                dtype, n, contig = _cls.describe(arr)
                name = code.co_name
                stack.append(("py", frame, None, self._site(caller, name), "np." + name,
                              dtype, n, contig, _perf()))
            elif event == "return":
                stack = getattr(self._tls, "stack", None)
                if stack and stack[-1][0] == "py" and stack[-1][1] is frame:
                    t1 = _perf()
                    entry = stack.pop()
                    self._record(entry, t1 - entry[8], not stack)
        except Exception as exc:  # never break the user's program
            self._fail(exc)

    def _fail(self, exc):
        self.uninstall()
        print(f"rglns: observation stopped after an internal error "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)

    def _record(self, entry, ns, outermost):
        table = self._tls.table
        if outermost:
            table.outer_ns += ns
        self.agg.record(entry[3], entry[4], entry[5], entry[6], entry[7], "numpy",
                        float(ns), table=table)
