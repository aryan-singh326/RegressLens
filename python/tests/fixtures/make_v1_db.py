"""Regenerates v1_traces.db: a trace database written by the v0.1 code.

The v0.1 trace module is loaded straight from git history (commit
c7aa79b, the last v1-schema release), so the fixture is exactly what
a 0.1.0.dev0 installation left on users' disks:

    python tests/fixtures/make_v1_db.py
"""
import importlib.util
import os
import sqlite3
import subprocess
import sys
import tempfile

V1_COMMIT = "c7aa79b"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "v1_traces.db")


def load_v1_trace_module(db_path):
    source = subprocess.run(["git", "show", f"{V1_COMMIT}:python/regresslens/trace.py"],
                            cwd=HERE, capture_output=True, text=True, check=True).stdout
    os.environ["REGRESSLENS_DB_PATH"] = db_path
    path = os.path.join(tempfile.mkdtemp(), "trace_v1.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("trace_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    v1 = load_v1_trace_module(OUT)
    labels = ["baseline:pipeline", "check:pipeline:1760000000", "profile:1760000100", None]
    i = 0
    for label in labels:
        for run in range(5):
            for op, n, kernel, contiguous, sel, window, site in (
                    ("reduction", 1_000_000, "avx2", True, None, None, "pipeline.py:41"),
                    ("filter", 1_000_000, "avx2", True, 0.31, None, "pipeline.py:45"),
                    ("rolling", 999_981, "scalar", True, None, 20, "pipeline.py:38"),
                    ("projection", 999_981, "numpy_fallback", False, None, None,
                     "pipeline.py:43")):
                i += 1
                v1.record_trace(operator=op, row_count=n, dtype="float64",
                                contiguous=contiguous, selected_kernel=kernel,
                                runtime_ns=1000.0 + 37 * (i % 11) + 400 * run,
                                available_cores=8, selectivity=sel, window=window,
                                run_label=label, call_site=site)
    conn = v1._get_connection(OUT)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")  # single self-contained file
    conn.close()
    with sqlite3.connect(OUT) as c:
        print(OUT, c.execute("SELECT COUNT(*) FROM traces").fetchone()[0], "v1 rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
