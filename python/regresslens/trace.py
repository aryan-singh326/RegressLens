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

_DEFAULT_DB_PATH = os.path.expanduser("~/.regresslens/traces.db")

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
    timestamp REAL NOT NULL
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


def _get_connection(db_path):
    if not hasattr(_local, "conn") or getattr(_local, "db_path", None) != db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute(_SCHEMA)
        conn.commit()
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
):
    """Writes one trace row. Failures here are logged, not raised —
    a broken trace database should never be the reason a user's
    actual computation fails. This is observability infrastructure,
    not the computation itself."""
    if not _TRACING_ENABLED:
        return

    path = db_path or _DEFAULT_DB_PATH
    try:
        with _write_lock:
            conn = _get_connection(path)
            conn.execute(
                "INSERT INTO traces (operator, row_count, dtype, contiguous, "
                "selectivity, window, available_cores, selected_kernel, "
                "runtime_ns, hardware_fingerprint, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )
            conn.commit()
    except Exception:
        logger.warning("regresslens: failed to write trace", exc_info=True)
