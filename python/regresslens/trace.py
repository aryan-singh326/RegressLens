"""
regresslens.trace: execution trace persistence.

Per the project brief: SQLite database at ~/.regresslens/traces.db.
Schema per trace: operator type, row count, data type, contiguity
flag, selectivity, available cores, selected kernel, observed
runtime, hardware fingerprint, timestamp.

Traces persist across sessions — this is what eventually lets
`rglns check` compare a pipeline's current performance against its
own history, and what lets filter's selectivity assumption (currently
hardcoded to 0.5 in the native layer) become a real runtime-derived
estimate instead, once enough history exists per call site.
"""
import json
import logging
import os
import platform
import sqlite3
import threading
import time

logger = logging.getLogger("regresslens")

_DEFAULT_DB_PATH = os.environ.get(
    "REGRESSLENS_DB_PATH", os.path.expanduser("~/.regresslens/traces.db")
)

# Tracing can be disabled entirely (e.g. for tests, or a user who
# doesn't want disk writes on every call) without touching call
# sites — this is the one flag that matters.
_TRACING_ENABLED = os.environ.get("REGRESSLENS_NO_TRACE", "") == ""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operator TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    contiguous INTEGER NOT NULL,
    selectivity REAL,
    window INTEGER,
    available_cores INTEGER NOT NULL,
    selected_kernel TEXT NOT NULL,
    runtime_ns REAL NOT NULL,
    hardware_fingerprint TEXT NOT NULL,
    timestamp REAL NOT NULL,
    run_label TEXT,
    call_site TEXT
);
"""

# One connection per thread — sqlite3 connections aren't safe to
# share across threads, and RegressLens's own MT-AVX2 kernels mean
# calls could plausibly originate from multiple threads in a future
# integration. A lock still guards writes because SQLite itself
# serializes writers regardless; this avoids "database is locked"
# errors under any concurrent use rather than assuming single-
# threaded callers.
_local = threading.local()
_write_lock = threading.Lock()


def _migrate(conn):
    """Adds columns introduced after the initial schema, for
    databases created by an earlier version of RegressLens. A fresh
    CREATE TABLE IF NOT EXISTS won't add columns to an existing
    table — this is what actually handles that case."""
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
    if "run_label" not in existing_cols:
        conn.execute("ALTER TABLE traces ADD COLUMN run_label TEXT")
        conn.commit()
    if "call_site" not in existing_cols:
        conn.execute("ALTER TABLE traces ADD COLUMN call_site TEXT")
        conn.commit()


def _get_connection(db_path):
    if not hasattr(_local, "conn") or getattr(_local, "db_path", None) != db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute(_SCHEMA)
        conn.commit()
        _migrate(conn)
        _local.conn = conn
        _local.db_path = db_path
    return _local.conn


def _hardware_fingerprint():
    """A stable-ish descriptor of the current machine, for detecting
    when a regression report's baseline and current runs weren't
    actually measured on comparable hardware. Per the project brief,
    differences here should be flagged as a potential confound, not
    silently absorbed — this function just captures the data; the
    comparison logic is Phase 3's regression-detection piece, not
    yet implemented.

    Deliberately does NOT try to detect AVX2/compiler flags here —
    those are build-time facts about the native extension, not
    Python-runtime-observable ones. Revisit if that turns out to
    matter once real regression reports are being generated.
    """
    cpu_model = "unknown"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    fingerprint = {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }
    try:
        import numpy as np
        fingerprint["numpy_version"] = np.__version__
    except ImportError:
        pass

    return json.dumps(fingerprint, sort_keys=True)


# Computed once per process, not per trace — this doesn't change
# mid-run and re-reading /proc/cpuinfo on every single kernel call
# would add real overhead to exactly the interception-overhead
# measurement the project brief cares about keeping small.
_CACHED_FINGERPRINT = None


def get_hardware_fingerprint():
    global _CACHED_FINGERPRINT
    if _CACHED_FINGERPRINT is None:
        _CACHED_FINGERPRINT = _hardware_fingerprint()
    return _CACHED_FINGERPRINT


def capture_call_site():
    """Returns 'file.py:line' for the first stack frame OUTSIDE the
    regresslens package itself — i.e. the user's own code that called
    into a regresslens.array method. This is the attribution the
    project brief specifies: identifies WHERE the call that received
    non-contiguous data was made, not WHY the array became
    non-contiguous upstream (that would require tracking provenance
    through the user's entire pipeline, which v0.1 does not attempt
    — see the project brief's honest scope statement on this exact
    point).
    """
    import inspect
    import os as _os

    package_dir = _os.path.dirname(_os.path.abspath(__file__))
    for frame_info in inspect.stack():
        frame_path = _os.path.abspath(frame_info.filename)
        if not frame_path.startswith(package_dir):
            return f"{_os.path.basename(frame_info.filename)}:{frame_info.lineno}"
    return "unknown"


def record_trace(
    operator,
    row_count,
    dtype,
    contiguous,
    selected_kernel,
    runtime_ns,
    available_cores,
    selectivity=None,
    window=None,
    db_path=None,
    run_label=None,
    call_site=None,
):
    """Writes one trace row. Failures here are logged, not raised —
    a broken trace database should never be the reason a user's
    actual computation fails. This is observability infrastructure,
    not the computation itself.

    run_label: identifies which baseline/check session this trace
    belongs to. Defaults to the REGRESSLENS_RUN_LABEL environment
    variable if not passed explicitly — this is how the CLI tags a
    subprocess-run user script as part of a baseline or check session
    without requiring any change to the user's own code.
    """
    if not _TRACING_ENABLED:
        return

    if run_label is None:
        run_label = os.environ.get("REGRESSLENS_RUN_LABEL")

    path = db_path or _DEFAULT_DB_PATH
    try:
        with _write_lock:
            conn = _get_connection(path)
            conn.execute(
                "INSERT INTO traces (operator, row_count, dtype, contiguous, "
                "selectivity, window, available_cores, selected_kernel, "
                "runtime_ns, hardware_fingerprint, timestamp, run_label, "
                "call_site) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    operator,
                    row_count,
                    dtype,
                    1 if contiguous else 0,
                    selectivity,
                    window,
                    available_cores,
                    selected_kernel,
                    runtime_ns,
                    get_hardware_fingerprint(),
                    time.time(),
                    run_label,
                    call_site,
                ),
            )
            conn.commit()
    except Exception:
        logger.warning("regresslens: failed to write trace", exc_info=True)


def get_hardware_fingerprints_for_label(run_label, db_path=None):
    """Returns the set of distinct hardware_fingerprint values seen
    across all traces under run_label. Used to detect two real
    confounds: (1) baseline and current sessions run on different
    hardware, and (2) a single session's own runs weren't even
    consistent with each other (e.g. thermal throttling changed the
    CPU's effective clock mid-session on shared/noisy infrastructure).
    Per the project brief, these should be flagged, not silently
    absorbed into the regression report as if they didn't happen.
    """
    path = db_path or _DEFAULT_DB_PATH
    if not os.path.exists(path):
        return set()
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    _migrate(conn)
    rows = conn.execute(
        "SELECT DISTINCT hardware_fingerprint FROM traces WHERE run_label = ?",
        (run_label,),
    ).fetchall()
    return {r[0] for r in rows}


def get_samples(run_label, db_path=None):
    """Returns {(operator, dtype, row_count): [runtime_ns, ...]} for
    all traces tagged with the given run_label. This groups by
    (operator, dtype, row_count) as an APPROXIMATION of "same call
    site" — v0.1 does not yet do real call-site attribution via stack
    capture (that's a separate, harder feature; see the project
    brief's contiguity-attribution note for the closest analogue).
    Two different call sites that happen to use the same operator,
    dtype, and array size will be conflated here. This is a known,
    documented limitation, not an oversight — worth revisiting once
    real stack-based attribution exists.
    """
    path = db_path or _DEFAULT_DB_PATH
    if not os.path.exists(path):
        return {}
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    _migrate(conn)
    rows = conn.execute(
        "SELECT operator, dtype, row_count, runtime_ns FROM traces "
        "WHERE run_label = ?",
        (run_label,),
    ).fetchall()

    samples = {}
    for operator, dtype, row_count, runtime_ns in rows:
        key = (operator, dtype, row_count)
        samples.setdefault(key, []).append(runtime_ns)
    return samples


def get_first_contiguity_loss(operator, dtype, row_count, run_label, db_path=None):
    """Returns (call_site, timestamp) for the EARLIEST non-contiguous
    trace matching (operator, dtype, row_count) under run_label, or
    None if there isn't one. 'Earliest' matches the project brief's
    'first rd-observed call receiving non-contiguous array' framing
    -- later occurrences at the same call site are the same root
    cause repeating, not new information.
    """
    path = db_path or _DEFAULT_DB_PATH
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    _migrate(conn)
    row = conn.execute(
        "SELECT call_site, timestamp FROM traces WHERE run_label = ? AND "
        "operator = ? AND dtype = ? AND row_count = ? AND contiguous = 0 "
        "ORDER BY timestamp ASC LIMIT 1",
        (run_label, operator, dtype, row_count),
    ).fetchone()
    return row


def get_contiguous_runtime_samples(operator, dtype, row_count, db_path=None):
    """Returns ALL recorded runtime_ns for CONTIGUOUS (accelerated)
    traces matching (operator, dtype, row_count), across every
    run_label -- used to estimate what this call would cost if it
    weren't hitting the non-contiguous fallback path, for the
    remediation cost/benefit estimate. Deliberately not scoped to one
    run_label: any historical accelerated run at this shape is a
    reasonable reference point, and restricting to one session would
    often mean no data at all.
    """
    path = db_path or _DEFAULT_DB_PATH
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    _migrate(conn)
    rows = conn.execute(
        "SELECT runtime_ns FROM traces WHERE operator = ? AND dtype = ? AND "
        "row_count = ? AND contiguous = 1",
        (operator, dtype, row_count),
    ).fetchall()
    return [r[0] for r in rows]
