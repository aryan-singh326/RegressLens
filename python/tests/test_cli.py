"""End-to-end CLI tests: real measured subprocesses, real (temporary)
trace database, no mocks."""
import os
import sys
import textwrap

import pytest

from regresslens import analysis, cli, store, telemetry

PIPELINE = textwrap.dedent("""
    import os
    import numpy as np
    import regresslens as rl

    rng = np.random.default_rng(7)
    data = rng.uniform(-100, 100, 400_000)
    arr = rl.array(data)
    # SIMULATE_REGRESSION makes the filter keep ~100% instead of ~0.5% of
    # the data, so filter and the following sum do far more work.
    threshold = -99.0 if os.environ.get("SIMULATE_REGRESSION") == "1" else 99.0
    for _ in range(5):
        kept = arr.filter_gt(threshold)
        kept.sum()
""")

CONTIGUITY = textwrap.dedent("""
    import os
    import numpy as np
    import regresslens as rl

    n = 1_000_000  # the strided copy must stream from DRAM, not sit in cache
    flat = np.random.default_rng(3).uniform(-100, 100, n)
    if os.environ.get("SIMULATE_CONTIGUITY_LOSS") == "1":
        padded = np.zeros((n, 8))
        padded[:, 0] = flat
        data = padded[:, 0]          # strided view, same n
    else:
        data = flat
    arr = rl.array(data)
    for _ in range(5):
        arr.filter_gt(0.0).sum()
        arr.rolling_mean(20)
        arr.sum()  # ~5x slower strided, with or without native kernels
""")


@pytest.fixture
def script(tmp_path):
    def make(text, name="pipeline.py"):
        p = tmp_path / name
        p.write_text(text)
        return str(p)
    return make


def rglns(*args):
    return cli.main(list(args))


class TestRunAndProfile:
    def test_run_propagates_exit_code_and_records_session(self, script, capsys):
        path = script("import sys, numpy as np\nnp.sum(np.ones(10))\nsys.exit(3)\n")
        assert rglns("run", "--label", "smoke", "--", sys.executable, path) == 3
        conn = store.connect()
        [s] = store.find_sessions(conn, label="smoke")
        [run] = store.load_sessions(conn, [s.session_id], only_successful=False)[0].runs
        assert run.exit_code == 3 and run.wall_ns > 0
        assert run.canary_compute_ns > 0 and run.canary_mem_ns > 0

    def test_profile_reports_call_sites(self, script, capsys):
        assert rglns("profile", "--runs", "2", "--", "python", script(PIPELINE)) == 0
        out = capsys.readouterr().out
        assert "pipeline.py" in out and "filter" in out and "share" in out

    def test_legacy_script_flag_and_missing_script(self, script):
        assert rglns("profile", "--runs", "1", "--script", script(PIPELINE)) == 0
        with pytest.raises(SystemExit):
            rglns("profile", "--runs", "1", "--script", "/nonexistent/x.py")

    def test_sessions_lists_labels(self, script, capsys):
        rglns("run", "--label", "listed", "--", "python", script("pass\n"))
        assert rglns("sessions") == 0
        assert "listed" in capsys.readouterr().out


class TestBaselineCheck:
    def test_detects_real_regression(self, script, monkeypatch, capsys):
        path = script(PIPELINE)
        monkeypatch.delenv("SIMULATE_REGRESSION", raising=False)
        assert rglns("baseline", "--name", "v1", "--runs", "6", "--", "python", path) == 0
        monkeypatch.setenv("SIMULATE_REGRESSION", "1")
        rc = rglns("check", "--baseline", "v1", "--runs", "6", "--seed", "1", "--",
                   "python", path)
        out = capsys.readouterr().out
        assert rc == 1, out
        assert "regression" in out and "pipeline.py" in out

    def test_unchanged_code_is_not_a_regression(self, script, monkeypatch, capsys):
        path = script(PIPELINE)
        monkeypatch.delenv("SIMULATE_REGRESSION", raising=False)
        rglns("baseline", "--name", "v1", "--runs", "6", "--", "python", path)
        # A generous threshold keeps this deterministic on noisy CI hosts;
        # the false-positive rate itself is validated statistically.
        rc = rglns("check", "--baseline", "v1", "--runs", "6", "--theta", "0.5", "--seed",
                   "2", "--", "python", path)
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "regression " not in out.replace("regression (advisory)", "")

    def test_missing_baseline_is_a_usage_error(self, script):
        assert rglns("check", "--baseline", "nope", "--runs", "2", "--", "python",
                     script(PIPELINE)) == 2

    def test_insufficient_runs_reported_not_fatal(self, script, capsys):
        path = script(PIPELINE)
        rglns("baseline", "--name", "v1", "--runs", "2", "--", "python", path)
        rc = rglns("check", "--baseline", "v1", "--runs", "2", "--", "python", path)
        out = capsys.readouterr().out
        assert rc == 0 and "insufficient runs" in out

    def test_contiguity_loss_is_diagnosed_with_remediation(self, script, monkeypatch, capsys):
        # No canaries: the test-sized memory canary (4 MiB) fits in cache, and
        # on shared CI runners it can move by +70% between two sessions a few
        # seconds apart, which correctly turns every verdict into advisory
        # environment drift. Drift handling has its own tests (test_stats,
        # test_diagnosis).
        path = script(CONTIGUITY, "contiguity_pipeline.py")
        monkeypatch.delenv("SIMULATE_CONTIGUITY_LOSS", raising=False)
        rglns("baseline", "--name", "v1", "--runs", "7", "--no-canary", "--", "python", path)
        monkeypatch.setenv("SIMULATE_CONTIGUITY_LOSS", "1")
        rc = rglns("check", "--baseline", "v1", "--runs", "7", "--no-canary", "--seed", "3",
                   "--json", str(os.path.dirname(path) + "/report.json"), "--", "python", path)
        out = capsys.readouterr().out
        assert rc == 1, out
        assert "contiguity_loss" in out and "contiguity_pipeline.py" in out
        assert "np.ascontiguousarray" in out
        import json
        report = json.load(open(os.path.dirname(path) + "/report.json"))
        names = {h["name"] for s in report["sites"] for h in s.get("diagnosis", [])}
        assert "contiguity_loss" in names


class TestEnvironmentWarnings:
    def _session(self, hw="{}", python="3.12.1", numpy="2.0.0"):
        return store.Session("s", "l", 0.0, hw, None, None, python, numpy, {})

    def test_flags_different_hardware_and_versions(self):
        w = analysis._environment_warnings([self._session('{"cpu_model": "A"}')],
                                           [self._session('{"cpu_model": "B"}', numpy="2.1.0")])
        assert any("DIFFERENT hardware" in x for x in w)
        assert any("NumPy version differs" in x for x in w)

    def test_silent_when_identical(self):
        s = self._session('{"cpu_model": "A"}')
        assert analysis._environment_warnings([s], [s]) == []


class TestTelemetryCommand:
    def test_on_off_status(self, capsys, monkeypatch):
        monkeypatch.delenv("RGLNS_TELEMETRY", raising=False)
        assert rglns("telemetry", "status") == 0
        assert "undecided" in capsys.readouterr().out
        rglns("telemetry", "on")
        assert telemetry.state() == "on"
        rglns("telemetry", "off")
        assert telemetry.state() == "off"
