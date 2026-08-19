"""Tests for regresslens.trace — the execution trace persistence layer.

Run with: python3 -m pytest test_trace.py -v
"""
import json
import os
import sqlite3
import tempfile

import numpy as np
import pytest

import regresslens as rl
import regresslens.trace as trace


@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "traces.db")
    yield db_path


class TestRecordTrace:
    def test_creates_database_and_table(self, temp_db):
        trace.record_trace(
            operator="reduction", row_count=100, dtype="float64",
            contiguous=True, selected_kernel="avx2", runtime_ns=123.0,
            available_cores=4, db_path=temp_db,
        )
        assert os.path.exists(temp_db)
        conn = sqlite3.connect(temp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert ("traces",) in tables

    def test_row_fields_are_correct(self, temp_db):
        trace.record_trace(
            operator="filter", row_count=5000, dtype="float32",
            contiguous=True, selected_kernel="mt_avx2", runtime_ns=4567.0,
            available_cores=8, selectivity=0.35, db_path=temp_db,
        )
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT operator, row_count, dtype, contiguous, selectivity, "
            "available_cores, selected_kernel, runtime_ns FROM traces"
        ).fetchone()
        assert row == ("filter", 5000, "float32", 1, 0.35, 8, "mt_avx2", 4567.0)

    def test_multiple_calls_append_rows(self, temp_db):
        for i in range(5):
            trace.record_trace(
                operator="reduction", row_count=100 * i, dtype="float64",
                contiguous=True, selected_kernel="avx2", runtime_ns=100.0,
                available_cores=4, db_path=temp_db,
            )
        conn = sqlite3.connect(temp_db)
        count = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        assert count == 5

    def test_null_fields_for_operations_without_selectivity_or_window(self, temp_db):
        trace.record_trace(
            operator="projection", row_count=100, dtype="float64",
            contiguous=True, selected_kernel="scalar", runtime_ns=100.0,
            available_cores=4, db_path=temp_db,
        )
        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT selectivity, window FROM traces").fetchone()
        assert row == (None, None)

    def test_hardware_fingerprint_is_valid_json(self, temp_db):
        trace.record_trace(
            operator="reduction", row_count=100, dtype="float64",
            contiguous=True, selected_kernel="avx2", runtime_ns=100.0,
            available_cores=4, db_path=temp_db,
        )
        conn = sqlite3.connect(temp_db)
        fp = conn.execute("SELECT hardware_fingerprint FROM traces").fetchone()[0]
        parsed = json.loads(fp)  # must not raise
        assert "system" in parsed
        assert "machine" in parsed

    def test_broken_path_does_not_raise(self):
        # Trace failures must degrade gracefully, never break the
        # caller's actual computation — this is observability
        # infrastructure, not the product itself.
        trace.record_trace(
            operator="reduction", row_count=10, dtype="float64",
            contiguous=True, selected_kernel="avx2", runtime_ns=100.0,
            available_cores=1,
            db_path="/this/path/cannot/possibly/be/created/traces.db",
        )
        # No assertion needed beyond "didn't raise" — reaching this
        # line is the test.


class TestTracingDisableFlag:
    def test_no_trace_env_var_prevents_db_creation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REGRESSLENS_NO_TRACE", "1")
        # Reload the module so it re-reads the env var at import time.
        import importlib
        importlib.reload(trace)

        db_path = str(tmp_path / "traces.db")
        trace.record_trace(
            operator="reduction", row_count=10, dtype="float64",
            contiguous=True, selected_kernel="avx2", runtime_ns=100.0,
            available_cores=1, db_path=db_path,
        )
        assert not os.path.exists(db_path)

        # Restore normal behavior for subsequent tests.
        monkeypatch.delenv("REGRESSLENS_NO_TRACE", raising=False)
        importlib.reload(trace)


class TestArrayIntegration:
    """Confirms actual rl.array operations write real trace rows —
    not just that record_trace() works in isolation."""

    def test_sum_writes_reduction_trace(self, temp_db, monkeypatch):
        monkeypatch.setattr(trace, "_DEFAULT_DB_PATH", temp_db)
        data = np.random.default_rng(1).uniform(-10, 10, 1000).astype(np.float64)
        arr = rl.array(data)
        arr.sum()

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT operator, row_count, dtype FROM traces WHERE operator='reduction'"
        ).fetchone()
        assert row == ("reduction", 1000, "float64")

    def test_filter_trace_records_actual_selectivity(self, temp_db, monkeypatch):
        monkeypatch.setattr(trace, "_DEFAULT_DB_PATH", temp_db)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        arr = rl.array(data)
        arr.filter_gt(5.0)  # 5 of 10 elements pass -> selectivity 0.5

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT selectivity FROM traces WHERE operator='filter'"
        ).fetchone()
        assert row[0] == pytest.approx(0.5)

    def test_rolling_trace_records_window(self, temp_db, monkeypatch):
        monkeypatch.setattr(trace, "_DEFAULT_DB_PATH", temp_db)
        data = np.random.default_rng(2).uniform(-10, 10, 500).astype(np.float64)
        arr = rl.array(data)
        arr.rolling_sum(25)

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT window FROM traces WHERE operator='rolling'"
        ).fetchone()
        assert row[0] == 25


class TestWALMode:
    """Locks in the WAL-mode fix — added after measuring that the
    default SQLite rollback-journal mode's per-commit fsync cost
    (~0.5ms median) was a meaningful fraction of the Phase 4
    validation pipeline's total per-call time. Confirms the pragma
    actually took effect, not just that record_trace() doesn't
    crash."""

    def test_journal_mode_is_wal(self, temp_db):
        trace.record_trace(
            operator="reduction", row_count=100, dtype="float64",
            contiguous=True, selected_kernel="avx2", runtime_ns=100.0,
            available_cores=4, db_path=temp_db,
        )
        conn = sqlite3.connect(temp_db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
