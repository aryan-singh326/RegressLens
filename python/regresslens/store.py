"""SQLite persistence, schema v2 (PRAGMA user_version = 2).

Schema v1 was one flat `traces` table: one row per kernel call, grouped
by a free-form run_label that lumped every subprocess run of a CLI
invocation together, with (operator, dtype, row_count) standing in for
call-site identity. v2 records the identities that run-level statistics,
bisection and cross-revision comparison need:

  sessions    one CLI invocation (RGLNS_SESSION_ID): label, when, where
              (hardware fingerprint, git sha/dirty, Python/NumPy versions)
  runs        one subprocess run (RGLNS_RUN_ID): wall time, drift
              canaries, exit code
  calls       per-run aggregate for one call site and argument shape:
              count, sum of ns, sum of log ns, a reservoir of up to 64
              raw samples, plus operation-specific extras
  call_sites  display and attribution data for a call_site_id: path,
              enclosing qualname, ordinal, current line, and up to 8
              normalized user stack frames (used by bisect's prior)

calls are written once per process at exit (never per call), so the
write cost is independent of how many NumPy calls a run makes.

Opening a v1 database migrates it in place: the old table is kept
verbatim as `legacy_traces_v1`, and its rows are converted into v2
sessions/runs/calls (one session per v1 run_label, one run per session
because v1 never recorded run boundaries).
"""
import hashlib
import json
import math
import os
import sqlite3
import statistics
import time
import uuid
from dataclasses import dataclass, field

SCHEMA_VERSION = 2
RESERVOIR_SIZE = 64

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    label          TEXT NOT NULL,
    started_at     REAL NOT NULL,
    hw_fingerprint TEXT,
    git_sha        TEXT,
    git_dirty      INTEGER,
    python         TEXT,
    numpy          TEXT,
    meta_json      TEXT
);
CREATE INDEX IF NOT EXISTS sessions_by_label ON sessions(label, started_at);

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    wall_ns           REAL,
    canary_compute_ns REAL,
    canary_mem_ns     REAL,
    exit_code         INTEGER,
    started_at        REAL,
    canary_json       TEXT
);
CREATE INDEX IF NOT EXISTS runs_by_session ON runs(session_id);

CREATE TABLE IF NOT EXISTS calls (
    run_id         TEXT NOT NULL,
    call_site_id   TEXT NOT NULL,
    operator       TEXT NOT NULL,
    dtype          TEXT,
    n              INTEGER,
    contiguous     INTEGER,
    kernel         TEXT,
    count          INTEGER NOT NULL,
    sum_ns         REAL NOT NULL,
    sum_log_ns     REAL NOT NULL,
    sample_ns_json TEXT,
    extra_json     TEXT
);
CREATE INDEX IF NOT EXISTS calls_by_run ON calls(run_id);
CREATE INDEX IF NOT EXISTS calls_by_site ON calls(call_site_id);

CREATE TABLE IF NOT EXISTS call_sites (
    call_site_id TEXT PRIMARY KEY,
    path         TEXT,
    qualname     TEXT,
    ordinal      TEXT,
    line         INTEGER,
    stack_json   TEXT
);
"""


def db_path():
    """RGLNS_DB_PATH, else the pre-0.2 REGRESSLENS_DB_PATH, else
    ~/.regresslens/traces.db."""
    return (os.environ.get("RGLNS_DB_PATH")
            or os.environ.get("REGRESSLENS_DB_PATH")
            or os.path.expanduser(os.path.join("~", ".regresslens", "traces.db")))


def tracing_enabled():
    return not (os.environ.get("RGLNS_NO_TRACE") or os.environ.get("REGRESSLENS_NO_TRACE"))


def new_id():
    return uuid.uuid4().hex


def connect(path=None):
    """Opens (creating or migrating as needed) the trace database."""
    path = path or db_path()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    # WAL + synchronous=NORMAL: commits no longer fsync, which measured
    # ~10x lower write latency in Phase 4. A power loss can drop the
    # last few commits; a process crash cannot corrupt the database.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn):
    if conn.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
        return
    # One write transaction: the migration happens fully or not at all,
    # and a second process opening the same fresh database waits here,
    # then sees the finished schema when it re-checks below.
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            has_v1 = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='traces'"
            ).fetchone() is not None
            for stmt in _DDL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            if has_v1:
                _migrate_v1(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


# --- v1 migration -------------------------------------------------------

def _v1_session_id(label):
    return "v1-" + hashlib.sha1(label.encode()).hexdigest()[:16]


def _v1_call_site_id(call_site, operator):
    return hashlib.sha1(f"v1|{call_site}|{operator}".encode()).hexdigest()


def _evenly_spaced(values, k):
    if len(values) <= k:
        return list(values)
    step = len(values) / k
    return [values[int(i * step)] for i in range(k)]


def _migrate_v1(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
    conn.execute("ALTER TABLE traces RENAME TO legacy_traces_v1")
    run_label = "run_label" if "run_label" in cols else "NULL"
    call_site = "call_site" if "call_site" in cols else "NULL"
    rows = conn.execute(
        f"SELECT operator, row_count, dtype, contiguous, selectivity, window, "
        f"selected_kernel, runtime_ns, hardware_fingerprint, timestamp, "
        f"{run_label}, {call_site} FROM legacy_traces_v1 ORDER BY id"
    ).fetchall()

    sessions = {}
    groups = {}
    for (op, n, dtype, contig, sel, window, kernel, ns, hw, ts, label, site) in rows:
        label = label or "legacy:unlabeled"
        site = site or "unknown"
        s = sessions.setdefault(label, {"start": ts, "hw": hw})
        s["start"] = min(s["start"], ts)
        key = (label, site, op, dtype, n, contig, kernel)
        g = groups.setdefault(key, {"ns": [], "sel_sum": 0.0, "sel_n": 0, "window": window})
        g["ns"].append(float(ns))
        if sel is not None:
            g["sel_sum"] += sel
            g["sel_n"] += 1

    meta = json.dumps({"migrated_from": "v1",
                       "note": "v1 did not record run boundaries; all of this "
                               "label's traces are stored as one run"})
    for label, s in sessions.items():
        sid = _v1_session_id(label)
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, label, started_at, "
            "hw_fingerprint, meta_json) VALUES (?, ?, ?, ?, ?)",
            (sid, label, s["start"], s["hw"], meta))
        conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, session_id, started_at) VALUES (?, ?, ?)",
            (sid + "-r0", sid, s["start"]))

    for (label, site, op, dtype, n, contig, kernel), g in groups.items():
        sid = _v1_session_id(label)
        cid = _v1_call_site_id(site, op)
        ns = g["ns"]
        extra = {"legacy_call_site": site}
        if g["sel_n"]:
            extra.update(sel_sum=g["sel_sum"], sel_n=g["sel_n"])
        if g["window"] is not None:
            extra["window"] = g["window"]
        conn.execute(
            "INSERT INTO calls (run_id, call_site_id, operator, dtype, n, contiguous, "
            "kernel, count, sum_ns, sum_log_ns, sample_ns_json, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid + "-r0", cid, op, dtype, n, contig, kernel, len(ns), sum(ns),
             sum(math.log(max(v, 1.0)) for v in ns),
             json.dumps(_evenly_spaced(ns, RESERVOIR_SIZE)), json.dumps(extra)))
        path, _, line = site.rpartition(":")
        conn.execute(
            "INSERT OR IGNORE INTO call_sites (call_site_id, path, qualname, ordinal, "
            "line, stack_json) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, path or site, None, "v1", int(line) if line.isdigit() else None, "[]"))


# --- writers -----------------------------------------------------------------

def create_session(conn, label, *, session_id=None, hw_fingerprint=None, git_sha=None,
                   git_dirty=None, python=None, numpy=None, meta=None, started_at=None):
    session_id = session_id or new_id()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, label, started_at, hw_fingerprint, "
            "git_sha, git_dirty, python, numpy, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, label, started_at or time.time(), hw_fingerprint, git_sha,
             None if git_dirty is None else int(bool(git_dirty)), python, numpy,
             json.dumps(meta or {})))
    return session_id


def fill_session_versions(conn, session_id, python, numpy):
    """The CLI may run under a different interpreter than the measured
    program, so the child process records its own versions."""
    with conn:
        conn.execute(
            "UPDATE sessions SET python = COALESCE(python, ?), numpy = COALESCE(numpy, ?) "
            "WHERE session_id = ?", (python, numpy, session_id))


def _median(values):
    return statistics.median(values) if values else None


def upsert_run(conn, run_id, session_id, *, wall_ns=None, exit_code=None, started_at=None,
               canary_samples=()):
    """Creates or updates a run. Canary samples ({"compute_ns", "mem_ns",
    "where", "t"}) accumulate across calls: the CLI adds start/end
    samples and the measured process adds periodic ones. The run's
    canary columns hold the median of all samples."""
    with conn:
        row = conn.execute("SELECT canary_json FROM runs WHERE run_id = ?",
                           (run_id,)).fetchone()
        samples = json.loads(row[0]) if row and row[0] else []
        samples.extend(canary_samples)
        compute = _median([s["compute_ns"] for s in samples if s.get("compute_ns")])
        mem = _median([s["mem_ns"] for s in samples if s.get("mem_ns")])
        conn.execute(
            "INSERT INTO runs (run_id, session_id, wall_ns, canary_compute_ns, canary_mem_ns, "
            "exit_code, started_at, canary_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "wall_ns = COALESCE(excluded.wall_ns, runs.wall_ns), "
            "exit_code = COALESCE(excluded.exit_code, runs.exit_code), "
            "started_at = COALESCE(runs.started_at, excluded.started_at), "
            "canary_compute_ns = excluded.canary_compute_ns, "
            "canary_mem_ns = excluded.canary_mem_ns, "
            "canary_json = excluded.canary_json",
            (run_id, session_id, wall_ns, compute, mem, exit_code, started_at,
             json.dumps(samples) if samples else None))


def insert_calls(conn, run_id, rows):
    """rows: iterables of (call_site_id, operator, dtype, n, contiguous,
    kernel, count, sum_ns, sum_log_ns, samples_list, extra_dict)."""
    with conn:
        conn.executemany(
            "INSERT INTO calls (run_id, call_site_id, operator, dtype, n, contiguous, kernel, "
            "count, sum_ns, sum_log_ns, sample_ns_json, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, cid, op, dtype, n, None if contig is None else int(contig), kernel,
              count, sum_ns, sum_log, json.dumps(samples), json.dumps(extra or {}))
             for (cid, op, dtype, n, contig, kernel, count, sum_ns, sum_log, samples, extra)
             in rows])


def upsert_call_sites(conn, sites):
    """sites: iterables of (call_site_id, path, qualname, ordinal, line, stack).
    The display line is refreshed (it moves between commits); a stored
    stack is only filled in when missing."""
    with conn:
        conn.executemany(
            "INSERT INTO call_sites (call_site_id, path, qualname, ordinal, line, stack_json) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(call_site_id) DO UPDATE SET "
            "line = excluded.line, path = excluded.path, qualname = excluded.qualname, "
            "stack_json = CASE WHEN call_sites.stack_json IS NULL OR call_sites.stack_json = '[]' "
            "THEN excluded.stack_json ELSE call_sites.stack_json END",
            [(cid, path, qual, str(ordinal), line, json.dumps(stack or []))
             for (cid, path, qual, ordinal, line, stack) in sites])


# --- readers -----------------------------------------------------------------

@dataclass
class CallRow:
    call_site_id: str
    operator: str
    dtype: str
    n: int
    contiguous: bool
    kernel: str
    count: int
    sum_ns: float
    sum_log_ns: float
    samples: list
    extra: dict


@dataclass
class Run:
    run_id: str
    session_id: str
    wall_ns: float
    canary_compute_ns: float
    canary_mem_ns: float
    exit_code: int
    started_at: float
    calls: list = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    label: str
    started_at: float
    hw_fingerprint: str
    git_sha: str
    git_dirty: bool
    python: str
    numpy: str
    meta: dict
    runs: list = field(default_factory=list)


_SESSION_COLS = ("session_id, label, started_at, hw_fingerprint, git_sha, git_dirty, "
                 "python, numpy, meta_json")


def _session_from_row(r):
    return Session(r[0], r[1], r[2], r[3], r[4],
                   None if r[5] is None else bool(r[5]), r[6], r[7],
                   json.loads(r[8]) if r[8] else {})


def find_sessions(conn, label=None, label_prefix=None, git_sha=None, limit=None):
    """Sessions (without runs) matching the filters, oldest first."""
    where, args = [], []
    if label is not None:
        where.append("label = ?")
        args.append(label)
    if label_prefix is not None:
        where.append("label LIKE ? ESCAPE '\\'")
        args.append(label_prefix.replace("%", "\\%").replace("_", "\\_") + "%")
    if git_sha is not None:
        where.append("git_sha = ?")
        args.append(git_sha)
    sql = f"SELECT {_SESSION_COLS} FROM sessions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_at"
    rows = conn.execute(sql, args).fetchall()
    sessions = [_session_from_row(r) for r in rows]
    return sessions[-limit:] if limit else sessions


def load_sessions(conn, session_ids, with_calls=True, only_successful=True):
    """Full Session objects (runs, and calls if requested), in the order
    given. Runs whose process exited non-zero are excluded unless
    only_successful is False, because their timings are not comparable."""
    out = []
    for sid in session_ids:
        row = conn.execute(f"SELECT {_SESSION_COLS} FROM sessions WHERE session_id = ?",
                           (sid,)).fetchone()
        if row is None:
            continue
        s = _session_from_row(row)
        for r in conn.execute(
                "SELECT run_id, session_id, wall_ns, canary_compute_ns, canary_mem_ns, "
                "exit_code, started_at FROM runs WHERE session_id = ? ORDER BY started_at",
                (sid,)):
            run = Run(*r)
            if only_successful and run.exit_code not in (None, 0):
                continue
            s.runs.append(run)
        if with_calls and s.runs:
            by_run = {r.run_id: r for r in s.runs}
            placeholders = ",".join("?" * len(by_run))
            for c in conn.execute(
                    "SELECT run_id, call_site_id, operator, dtype, n, contiguous, kernel, "
                    "count, sum_ns, sum_log_ns, sample_ns_json, extra_json FROM calls "
                    f"WHERE run_id IN ({placeholders})", list(by_run)):
                by_run[c[0]].calls.append(CallRow(
                    c[1], c[2], c[3], c[4], None if c[5] is None else bool(c[5]), c[6],
                    c[7], c[8], c[9], json.loads(c[10]) if c[10] else [],
                    json.loads(c[11]) if c[11] else {}))
        out.append(s)
    return out


def call_site_info(conn, call_site_ids):
    ids = list(call_site_ids)
    if not ids:
        return {}
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        rows = conn.execute(
            "SELECT call_site_id, path, qualname, ordinal, line, stack_json FROM call_sites "
            f"WHERE call_site_id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        for cid, path, qual, ordinal, line, stack in rows:
            out[cid] = {"path": path, "qualname": qual, "ordinal": ordinal, "line": line,
                        "stack": json.loads(stack) if stack else []}
    return out


def site_selectivity_history(conn, call_site_id, limit=200):
    """(sum of selectivities, number of calls) recorded at a call site
    across past runs, most recent rows first."""
    rows = conn.execute(
        "SELECT extra_json FROM calls WHERE call_site_id = ? AND operator = 'filter' "
        "ORDER BY rowid DESC LIMIT ?", (call_site_id, limit)).fetchall()
    total, count = 0.0, 0
    for (extra,) in rows:
        e = json.loads(extra) if extra else {}
        total += e.get("sel_sum", 0.0)
        count += e.get("sel_n", 0)
    return total, count


def session_labels(conn):
    return [r[0] for r in conn.execute(
        "SELECT label FROM sessions GROUP BY label ORDER BY MAX(started_at) DESC")]
