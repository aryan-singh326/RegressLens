"""Schema v2, the v1 migration, and the store read/write paths (WP0)."""
import json
import os
import shutil
import sqlite3

import pytest

from conftest import FIXTURES, run_rglns
from regresslens import store

V1_DB = os.path.join(FIXTURES, "v1_traces.db")


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


class TestSchema:
    def test_fresh_database_has_v2_schema_and_wal(self, tmp_path):
        conn = store.connect(str(tmp_path / "x.db"))
        assert {"sessions", "runs", "calls", "call_sites"} <= _tables(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        cols = {r[1] for r in conn.execute("PRAGMA table_info(calls)")}
        assert {"run_id", "call_site_id", "operator", "dtype", "n", "contiguous", "kernel",
                "count", "sum_ns", "sum_log_ns", "sample_ns_json", "extra_json"} <= cols
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
        assert {"session_id", "label", "started_at", "hw_fingerprint", "git_sha", "git_dirty",
                "python", "numpy"} <= cols
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert {"run_id", "session_id", "wall_ns", "canary_compute_ns", "canary_mem_ns",
                "exit_code"} <= cols

    def test_round_trip(self):
        conn = store.connect()
        sid = store.create_session(conn, "lbl", git_sha="abc", git_dirty=True)
        store.upsert_run(conn, "r1", sid, canary_samples=[{"compute_ns": 10.0, "mem_ns": 20.0}])
        store.upsert_run(conn, "r1", sid, wall_ns=5e6, exit_code=0,
                         canary_samples=[{"compute_ns": 30.0, "mem_ns": 40.0}])
        store.insert_calls(conn, "r1", [("site", "np.sum", "float64", 100, True, "numpy", 3,
                                         30.0, 6.9, [9.0, 10.0, 11.0], {"window": 5})])
        store.upsert_call_sites(conn, [("site", "a.py", "f", "0", 12, [{"path": "a.py"}])])
        [s] = store.load_sessions(conn, [sid])
        assert s.label == "lbl" and s.git_dirty is True
        [run] = s.runs
        assert run.wall_ns == 5e6
        assert run.canary_compute_ns == 20.0 and run.canary_mem_ns == 30.0  # medians
        [call] = run.calls
        assert call.samples == [9.0, 10.0, 11.0] and call.extra == {"window": 5}
        assert store.call_site_info(conn, ["site"])["site"]["stack"] == [{"path": "a.py"}]

    def test_failed_runs_are_excluded_from_statistics(self):
        conn = store.connect()
        sid = store.create_session(conn, "x")
        store.upsert_run(conn, "ok", sid, wall_ns=1.0, exit_code=0)
        store.upsert_run(conn, "bad", sid, wall_ns=1.0, exit_code=3)
        assert [r.run_id for r in store.load_sessions(conn, [sid])[0].runs] == ["ok"]

    def test_find_sessions_by_label_prefix_escapes_wildcards(self):
        conn = store.connect()
        store.create_session(conn, "baseline:a_b")
        store.create_session(conn, "baseline:aXb")
        assert [s.label for s in store.find_sessions(conn, label_prefix="baseline:a_")] == [
            "baseline:a_b"]


class TestV1Migration:
    @pytest.fixture
    def v1_copy(self, tmp_path, monkeypatch):
        path = str(tmp_path / "v1.db")
        shutil.copy(V1_DB, path)
        monkeypatch.setenv("RGLNS_DB_PATH", path)
        return path

    def test_migration_preserves_every_v1_row(self, v1_copy):
        raw = sqlite3.connect(v1_copy)
        v1_rows = raw.execute("SELECT * FROM traces ORDER BY id").fetchall()
        labels = {r[0] or "legacy:unlabeled" for r in raw.execute("SELECT run_label FROM traces")}
        raw.close()

        conn = store.connect(v1_copy)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        # The original table survives verbatim...
        assert conn.execute("SELECT * FROM legacy_traces_v1 ORDER BY id").fetchall() == v1_rows
        # ...and every row is accounted for in v2 calls.
        assert conn.execute("SELECT SUM(count) FROM calls").fetchone()[0] == len(v1_rows)
        assert {s.label for s in store.find_sessions(conn)} == labels
        total_ns = conn.execute("SELECT SUM(sum_ns) FROM calls").fetchone()[0]
        assert total_ns == pytest.approx(sum(r[9] for r in v1_rows))

    def test_migration_is_idempotent(self, v1_copy):
        store.connect(v1_copy).close()
        conn = store.connect(v1_copy)
        assert conn.execute("SELECT SUM(count) FROM calls").fetchone()[0] == 80

    def test_selectivity_and_call_sites_carried_over(self, v1_copy):
        conn = store.connect(v1_copy)
        extras = [json.loads(e) for (e,) in conn.execute(
            "SELECT extra_json FROM calls WHERE operator='filter'")]
        assert all(abs(e["sel_sum"] / e["sel_n"] - 0.31) < 1e-9 for e in extras)
        paths = {r[0] for r in conn.execute("SELECT path FROM call_sites")}
        assert paths == {"pipeline.py"}

    def test_rglns_profile_runs_on_migrated_database(self, v1_copy):
        proc = run_rglns("profile", "--label", "baseline:pipeline")
        assert proc.returncode == 0, proc.stderr
        assert "pipeline.py" in proc.stdout
        proc = run_rglns("sessions")
        assert proc.returncode == 0 and "baseline:pipeline" in proc.stdout
