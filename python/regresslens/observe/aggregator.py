"""In-process aggregation of observed calls, flushed once at exit.

Per (call site, operator, dtype, n, contiguous, kernel) the aggregator
keeps count, sum of ns, sum of log ns, and a 64-sample reservoir
(Algorithm R). Nothing is written to disk per call. `flush()` writes one
`calls` row per key, the call sites (with stacks), and the run's canary
samples in a single transaction.

Each thread records into its own table, so the hot path takes no lock;
tables are merged at flush. If a call site sees more than
MAX_DISTINCT_N distinct sizes in one process, further sizes are
bucketed to quarter-octaves (marked n_bucketed) to bound memory.
"""
import atexit
import heapq
import math
import os
import platform
import random
import sys
import threading
import time

from .. import store

RESERVOIR = store.RESERVOIR_SIZE
MAX_DISTINCT_N = 64


class Stats:
    __slots__ = ("count", "sum_ns", "sum_log", "samples", "sel_sum", "sel_n", "window",
                 "bucketed")

    def __init__(self):
        self.count = 0
        self.sum_ns = 0.0
        self.sum_log = 0.0
        self.samples = []
        self.sel_sum = 0.0
        self.sel_n = 0
        self.window = None
        self.bucketed = False

    def add(self, ns, rng):
        self.count += 1
        self.sum_ns += ns
        self.sum_log += math.log(ns if ns > 1.0 else 1.0)
        if len(self.samples) < RESERVOIR:
            self.samples.append(ns)
        else:
            j = int(rng.random() * self.count)
            if j < RESERVOIR:
                self.samples[j] = ns

    def merge(self, other, rng):
        """Combines two aggregates; the merged reservoir is a weighted
        sample (Efraimidis-Spirakis) so each source contributes in
        proportion to its call count."""
        weighted = []
        for st in (self, other):
            if st.samples:
                w = st.count / len(st.samples)
                weighted.extend((rng.random() ** (1.0 / w), s) for s in st.samples)
        self.samples = [s for _, s in heapq.nlargest(RESERVOIR, weighted)]
        self.count += other.count
        self.sum_ns += other.sum_ns
        self.sum_log += other.sum_log
        self.sel_sum += other.sel_sum
        self.sel_n += other.sel_n
        self.window = self.window if self.window is not None else other.window
        self.bucketed = self.bucketed or other.bucketed


def _bucket(n):
    if n <= 0:
        return n
    return int(round(2 ** (round(4 * math.log2(n)) / 4)))


class _ThreadTable:
    __slots__ = ("stats", "distinct_n", "rng", "outer_ns", "depth")

    def __init__(self, seed):
        self.stats = {}
        self.distinct_n = {}
        self.rng = random.Random(seed)
        self.outer_ns = 0
        self.depth = 0


class Aggregator:
    def __init__(self, run_id=None, session_id=None, label="adhoc"):
        self.run_id = run_id
        self.session_id = session_id
        self.label = label
        self.sites = {}          # call_site_id -> CallSite
        self.stacks = {}         # call_site_id -> normalized stack
        self.canary_samples = []
        self.backend_name = None
        self.unobserved = []
        self.started_ns = time.perf_counter_ns()
        self.pre_flush = []      # callables run before summarizing (e.g. drain a C fast path)
        self._tables = []
        self._lock = threading.Lock()
        self._local = threading.local()
        self._seed = hash((run_id, os.getpid())) & 0xFFFFFFFF
        self._canary_fn = None
        self._canary_period_ns = 0
        self._canary_due_ns = 0
        self._countdown = 1 << 62

    def enable_periodic_canaries(self, period_s, measure):
        """Every `period_s`, take one canary sample between two recorded
        calls. The clock is checked only every 1024 records."""
        self._canary_fn = measure
        self._canary_period_ns = int(period_s * 1e9)
        self._canary_due_ns = time.perf_counter_ns() + self._canary_period_ns
        self._countdown = 1024

    def _maybe_canary(self):
        self._countdown = 1024
        now = time.perf_counter_ns()
        if self._canary_fn is None or now < self._canary_due_ns:
            return
        try:
            self.canary_samples.append(self._canary_fn())
        except Exception:
            self._canary_fn = None
        self._canary_due_ns = time.perf_counter_ns() + self._canary_period_ns

    # -- hot path ---------------------------------------------------------

    def table(self):
        t = getattr(self._local, "t", None)
        if t is None:
            t = _ThreadTable(self._seed + len(self._tables))
            self._local.t = t
            with self._lock:
                self._tables.append(t)
        return t

    def record(self, site_id, operator, dtype, n, contiguous, kernel, ns,
               selectivity=None, window=None, table=None):
        t = table or self.table()
        key = (site_id, operator, dtype, n, contiguous, kernel)
        st = t.stats.get(key)
        if st is None:
            st, key = self._new_key(t, key)
        st.add(ns, t.rng)
        if selectivity is not None:
            st.sel_sum += selectivity
            st.sel_n += 1
        if window is not None:
            st.window = window
        self._countdown -= 1
        if self._countdown <= 0:
            self._maybe_canary()

    def _new_key(self, t, key):
        site_id, operator, dtype, n, contiguous, kernel = key
        if n is not None:
            seen = t.distinct_n.setdefault((site_id, operator), set())
            if n not in seen and len(seen) >= MAX_DISTINCT_N:
                key = (site_id, operator, dtype, _bucket(n), contiguous, kernel)
                st = t.stats.get(key)
                if st is None:
                    st = t.stats[key] = Stats()
                    st.bucketed = True
                return st, key
            seen.add(n)
        st = t.stats[key] = Stats()
        return st, key

    def note_site(self, site, stack=None):
        if site.id not in self.sites:
            self.sites[site.id] = site
            if stack is not None:
                self.stacks[site.id] = stack

    # -- summaries ----------------------------------------------------------

    def ingest(self, rows, outer_ns=0):
        """Adds pre-aggregated rows (from the C fast path):
        (site_id, operator, dtype, n, contiguous, kernel, count, sum_ns,
        sum_log_ns, samples)."""
        t = self.table()
        t.outer_ns += outer_ns
        rng = t.rng
        for site_id, op, dtype, n, contig, kernel, count, sum_ns, sum_log, samples in rows:
            st = Stats()
            st.count, st.sum_ns, st.sum_log, st.samples = count, sum_ns, sum_log, list(samples)
            key = (site_id, op, dtype, n, contig, kernel)
            cur = t.stats.get(key)
            if cur is None:
                t.stats[key] = st
            else:
                cur.merge(st, rng)

    def _run_pre_flush(self):
        for fn in list(self.pre_flush):
            try:
                fn()
            except Exception as exc:  # pragma: no cover
                print(f"rglns: warning: {exc}", file=sys.stderr)

    def merged(self):
        self._run_pre_flush()
        rng = random.Random(self._seed)
        out = {}
        with self._lock:
            tables = list(self._tables)
        for t in tables:
            for key, st in list(t.stats.items()):
                cur = out.get(key)
                if cur is None:
                    copy = Stats()
                    copy.merge(st, rng)
                    out[key] = copy
                else:
                    cur.merge(st, rng)
        return out

    def totals(self):
        self._run_pre_flush()
        with self._lock:
            tables = list(self._tables)
        calls = sum(st.count for t in tables for st in t.stats.values())
        sites = {k[0] for t in tables for k in t.stats}
        outer = sum(t.outer_ns for t in tables)
        return calls, len(sites), outer

    def selectivity_estimate(self, site_id):
        total, count = 0.0, 0
        with self._lock:
            tables = list(self._tables)
        for t in tables:
            for key, st in t.stats.items():
                if key[0] == site_id and st.sel_n:
                    total += st.sel_sum
                    count += st.sel_n
        return total, count

    def reset(self):
        with self._lock:
            for t in self._tables:
                t.stats.clear()
                t.distinct_n.clear()
                t.outer_ns = 0
        self.canary_samples = []

    # -- persistence -----------------------------------------------------------

    def flush(self, wall_ns=None):
        """Writes everything recorded so far and clears it. `wall_ns` is the
        program's own run time (see bootstrap._at_exit). Errors are
        reported on stderr, never raised: observation must not break the
        measured program."""
        if not store.tracing_enabled():
            self.reset()
            return
        merged = self.merged()
        if not merged and not self.canary_samples and wall_ns is None:
            return
        try:
            conn = store.connect()
            try:
                if self.session_id is None:
                    self.session_id = store.create_session(conn, self.label)
                if self.run_id is None:
                    self.run_id = store.new_id()
                np_mod = sys.modules.get("numpy")
                store.fill_session_versions(conn, self.session_id,
                                            platform.python_version(),
                                            getattr(np_mod, "__version__", None))
                store.upsert_run(conn, self.run_id, self.session_id, wall_ns=wall_ns,
                                 canary_samples=self.canary_samples)
                rows = []
                for (site_id, op, dtype, n, contig, kernel), st in merged.items():
                    extra = {}
                    if st.sel_n:
                        extra["sel_sum"] = st.sel_sum
                        extra["sel_n"] = st.sel_n
                    if st.window is not None:
                        extra["window"] = st.window
                    if st.bucketed:
                        extra["n_bucketed"] = True
                    rows.append((site_id, op, dtype, n, contig, kernel, st.count,
                                 st.sum_ns, st.sum_log, st.samples, extra))
                store.insert_calls(conn, self.run_id, rows)
                store.upsert_call_sites(conn, [
                    (s.id, s.path, s.qualname, s.ordinal, s.line, self.stacks.get(s.id))
                    for s in self.sites.values()])
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"rglns: warning: could not write traces ({exc})", file=sys.stderr)
        self.reset()


_global = None
_global_lock = threading.Lock()


def get():
    """The process-wide aggregator. Uses RGLNS_RUN_ID/RGLNS_SESSION_ID
    when launched by the rglns CLI; otherwise an ad-hoc session and run
    are created at the first flush, and a flush is registered at exit."""
    global _global
    if _global is None:
        with _global_lock:
            if _global is None:
                run_id = os.environ.get("RGLNS_RUN_ID")
                session_id = os.environ.get("RGLNS_SESSION_ID")
                _global = Aggregator(run_id, session_id)
                atexit.register(_global.flush)
    return _global


def _after_fork_in_child():
    # A forked child inherits the parent's unflushed records; drop them
    # so they are not written twice. The child keeps the run id.
    if _global is not None:
        _global._lock = threading.Lock()
        _global._local = threading.local()
        _global._tables = []
        _global.canary_samples = []


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
