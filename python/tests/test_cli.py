"""Tests for regresslens.cli — the rglns command-line tool.

Run with: python3 -m pytest test_cli.py -v

These tests actually invoke the CLI's command functions against a
real synthetic pipeline script and a real (temp-directory) SQLite
database — not mocks. Slower than the other test files as a result
(each test runs a subprocess N times), which is why run counts here
are kept low; correctness, not speed, is what's being verified.
"""
import os
import sys
import textwrap

import pytest

from regresslens import cli, remediation, trace


@pytest.fixture
def pipeline_script(tmp_path):
    """A minimal RegressLens pipeline script, with a selectivity-
    driven regression toggle — same design as the manual testing
    that validated this CLI, kept here so the behavior stays covered
    by the automated suite, not just ad hoc runs."""
    script = tmp_path / "pipeline.py"
    script.write_text(textwrap.dedent("""
        import os
        import numpy as np
        import regresslens as rl

        rng = np.random.default_rng(7)
        data = rng.uniform(-100, 100, 20_000).astype(np.float64)
        arr = rl.array(data)

        if os.environ.get("SIMULATE_REGRESSION") == "1":
            threshold = -99.0  # ~100% selectivity: much more work
        else:
            threshold = 99.0   # ~0% selectivity: much less work

        result = arr.filter_gt(threshold).sum()
    """))
    return str(script)


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "traces.db")
    # Two separate things need to point at this path:
    # 1. The env var, so the CLI's subprocess-spawned pipeline script
    #    (a completely separate Python process) writes here instead
    #    of the real ~/.regresslens/traces.db — attribute patches
    #    don't cross process boundaries, only env vars do.
    # 2. The module attribute, for the PARENT test process's own
    #    calls (trace.get_samples etc.) — _DEFAULT_DB_PATH is read
    #    from the env var once at import time, so setting the env
    #    var alone wouldn't affect an already-imported module.
    monkeypatch.setenv("REGRESSLENS_DB_PATH", path)
    monkeypatch.setattr(trace, "_DEFAULT_DB_PATH", path)
    return path


class Args(dict):
    """Lets tests build an argparse.Namespace-like object without
    going through argparse itself."""
    def __getattr__(self, k):
        return self[k]


class TestBaseline:
    def test_baseline_captures_samples(self, pipeline_script, db_path, capsys):
        args = Args(name="v1", runs=5, script=pipeline_script)
        rc = cli.cmd_baseline(args)
        assert rc == 0

        samples = trace.get_samples("baseline:v1", db_path=db_path)
        assert ("filter", "float64", 20000) in samples
        assert len(samples[("filter", "float64", 20000)]) == 5

    def test_missing_script_returns_error(self, db_path):
        args = Args(name="v1", runs=5, script="/nonexistent/script.py")
        rc = cli.cmd_baseline(args)
        assert rc == 1


class TestCheck:
    def test_detects_real_regression(self, pipeline_script, db_path, monkeypatch):
        # Capture baseline WITHOUT the regression.
        monkeypatch.delenv("SIMULATE_REGRESSION", raising=False)
        cli.cmd_baseline(Args(name="v1", runs=15, script=pipeline_script))

        # Check WITH the regression active.
        monkeypatch.setenv("SIMULATE_REGRESSION", "1")
        rc = cli.cmd_check(Args(
            baseline="v1", runs=15, script=pipeline_script,
            min_pairs=10, threshold=0.05,
        ))
        # Non-zero exit specifically signals a detected regression —
        # this matters for eventual CI integration.
        assert rc == 1

    def test_no_regression_when_nothing_changed(self, pipeline_script, db_path,
                                                   monkeypatch):
        monkeypatch.delenv("SIMULATE_REGRESSION", raising=False)
        cli.cmd_baseline(Args(name="v1", runs=15, script=pipeline_script))
        rc = cli.cmd_check(Args(
            baseline="v1", runs=15, script=pipeline_script,
            min_pairs=10, threshold=0.05,
        ))
        assert rc == 0

    def test_missing_baseline_returns_error(self, pipeline_script, db_path):
        rc = cli.cmd_check(Args(
            baseline="does_not_exist", runs=5, script=pipeline_script,
            min_pairs=20, threshold=0.05,
        ))
        assert rc == 1

    def test_insufficient_data_does_not_crash(self, pipeline_script, db_path,
                                                 monkeypatch):
        monkeypatch.delenv("SIMULATE_REGRESSION", raising=False)
        # Fewer runs than min_pairs requires.
        cli.cmd_baseline(Args(name="v1", runs=3, script=pipeline_script))
        rc = cli.cmd_check(Args(
            baseline="v1", runs=3, script=pipeline_script,
            min_pairs=20, threshold=0.05,
        ))
        # Must not crash; insufficient data is reported, not fatal,
        # and doesn't count as a detected regression.
        assert rc == 0


class TestHardwareConfoundDetection:
    """This is the check that was MISSING until real end-to-end
    testing surfaced a false-positive regression report caused by a
    genuine hardware/frequency change between sessions on noisy
    infrastructure. Locking this in so it can't silently regress.

    Tests the extracted _detect_hardware_confound() function directly
    against synthetic fingerprint sets, rather than trying to force
    two real subprocesses onto different hardware — that isn't
    controllable from a test, and an earlier version of this test
    tried exactly that and failed for the wrong reason (a monkeypatch
    in the test process has no effect on a subprocess's own fresh
    fingerprint computation)."""

    def test_flags_different_fingerprints_between_sessions(self):
        warning = cli._detect_hardware_confound(
            {"machine A fingerprint"}, {"machine B fingerprint"}
        )
        assert warning is not None
        assert "DIFFERENT hardware fingerprints" in warning

    def test_no_warning_when_fingerprints_match(self):
        warning = cli._detect_hardware_confound(
            {"same fingerprint"}, {"same fingerprint"}
        )
        assert warning is None

    def test_flags_inconsistent_fingerprints_within_current_session(self):
        # The current session's OWN runs weren't even consistent with
        # each other — e.g. thermal throttling mid-session.
        warning = cli._detect_hardware_confound(
            {"fingerprint A"}, {"fingerprint A", "fingerprint B"}
        )
        assert warning is not None
        assert "more than one hardware fingerprint" in warning

    def test_no_warning_when_either_set_is_empty(self):
        # Empty sets happen if no traces were recorded — a different
        # error path handles that (no data at all), this function
        # shouldn't also complain about it.
        assert cli._detect_hardware_confound(set(), {"fp"}) is None
        assert cli._detect_hardware_confound({"fp"}, set()) is None

    def test_end_to_end_no_warning_on_same_machine(
        self, pipeline_script, db_path, monkeypatch, capsys
    ):
        # Real end-to-end run, actually on the same machine (this
        # sandbox), confirming no false-positive warning in the
        # normal case.
        monkeypatch.delenv("SIMULATE_REGRESSION", raising=False)
        cli.cmd_baseline(Args(name="v1", runs=10, script=pipeline_script))
        cli.cmd_check(Args(
            baseline="v1", runs=10, script=pipeline_script,
            min_pairs=5, threshold=0.05,
        ))
        captured = capsys.readouterr()
        assert "DIFFERENT hardware fingerprints" not in captured.out


class TestProfile:
    def test_profile_reports_breakdown(self, pipeline_script, db_path, capsys):
        rc = cli.cmd_profile(Args(script=pipeline_script, runs=5))
        assert rc == 0
        captured = capsys.readouterr()
        assert "filter" in captured.out
        assert "% of total" in captured.out

    def test_missing_script_returns_error(self, db_path):
        rc = cli.cmd_profile(Args(script="/nonexistent/script.py", runs=5))
        assert rc == 1


class TestContiguityAttributionAndRemediation:
    """Validates the diagnosis + remediation feature added after the
    core CLI: when a regression coincides with a non-contiguous
    array, the report should identify WHERE that happened and
    estimate whether fixing it is worth the copy cost."""

    @pytest.fixture
    def contiguity_pipeline_script(self, tmp_path):
        script = tmp_path / "contiguity_pipeline.py"
        script.write_text(textwrap.dedent("""
            import os
            import numpy as np
            import regresslens as rl

            n = 5000
            rng = np.random.default_rng(3)
            flat_data = rng.uniform(-100, 100, n).astype(np.float64)

            if os.environ.get("SIMULATE_CONTIGUITY_LOSS") == "1":
                # Strided view: same row count, non-contiguous --
                # simulates slicing a column out of a 2D array, a
                # real-world pattern, while keeping the trace
                # grouping key (row_count) identical to baseline.
                padded = np.zeros((n, 2), dtype=np.float64)
                padded[:, 0] = flat_data
                data = padded[:, 0]
            else:
                data = flat_data

            arr = rl.array(data)
            result = arr.filter_gt(0.0)
            total = result.sum()
        """))
        return str(script)

    def test_regression_report_includes_diagnosis(
        self, contiguity_pipeline_script, db_path, monkeypatch, capsys
    ):
        monkeypatch.delenv("SIMULATE_CONTIGUITY_LOSS", raising=False)
        cli.cmd_baseline(Args(name="v1", runs=15, script=contiguity_pipeline_script))

        monkeypatch.setenv("SIMULATE_CONTIGUITY_LOSS", "1")
        rc = cli.cmd_check(Args(
            baseline="v1", runs=15, script=contiguity_pipeline_script,
            min_pairs=10, threshold=0.05, remediation_margin=2.0,
        ))
        captured = capsys.readouterr()

        assert rc == 1  # a real regression should be detected
        assert "Diagnosis:" in captured.out
        assert "contiguity_pipeline.py" in captured.out
        assert "Remediation" in captured.out

    def test_no_diagnosis_when_nothing_is_non_contiguous(
        self, contiguity_pipeline_script, db_path, monkeypatch, capsys
    ):
        monkeypatch.delenv("SIMULATE_CONTIGUITY_LOSS", raising=False)
        cli.cmd_baseline(Args(name="v1", runs=15, script=contiguity_pipeline_script))
        cli.cmd_check(Args(
            baseline="v1", runs=15, script=contiguity_pipeline_script,
            min_pairs=10, threshold=0.05, remediation_margin=2.0,
        ))
        captured = capsys.readouterr()
        assert "Diagnosis:" not in captured.out


class TestRemediationEstimation:
    """Unit tests for the remediation cost/benefit math itself,
    independent of the CLI plumbing."""

    def test_recommends_apply_when_savings_far_exceed_cost(self):
        result = remediation.estimate_remediation(
            row_count=1000, dtype_bytes=8,
            fallback_runtime_samples=[50_000_000.0] * 10,
            accelerated_runtime_samples=[1000.0] * 10,
        )
        assert result["recommend_apply"] is True

    def test_does_not_recommend_when_cost_exceeds_savings(self):
        result = remediation.estimate_remediation(
            row_count=100_000_000, dtype_bytes=8,
            fallback_runtime_samples=[100_000.0] * 10,
            accelerated_runtime_samples=[99_000.0] * 10,
        )
        assert result["recommend_apply"] is False

    def test_no_historical_data_does_not_crash(self):
        result = remediation.estimate_remediation(
            row_count=1000, dtype_bytes=8,
            fallback_runtime_samples=[1000.0] * 5,
            accelerated_runtime_samples=[],
        )
        assert result["recommend_apply"] is False
        assert result["reason"] is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
