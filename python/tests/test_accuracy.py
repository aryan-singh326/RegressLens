"""Accuracy policy and `rglns verify` (WP8)."""
import math
from fractions import Fraction

import numpy as np
import pytest

from regresslens import cli, exact
from regresslens.commands import verify


class TestExactReferences:
    def test_exact_sum_is_exact(self):
        x = np.array([1e16, 1.0, -1e16, 3.5, 2.0 ** -40], dtype=np.float64)
        assert exact.exact_sum(x) == sum(Fraction(v) for v in x.tolist())

    def test_exact_sum_matches_fsum_rounding(self):
        x = np.random.default_rng(0).normal(size=10_000) * 10.0 ** np.random.default_rng(1).uniform(-8, 8, 10_000)
        assert float(exact.exact_sum(x)) == math.fsum(x.tolist())

    def test_rolling_sums_and_vars_match_brute_force(self):
        x = np.random.default_rng(2).normal(100, 3, 40)
        x[7] = np.nan
        sums = exact.rolling_sums(x, 5)
        varis = exact.rolling_vars(x, 5, ddof=1)
        for i in range(len(x) - 4):
            window = x[i:i + 5]
            if np.isnan(window).any():
                assert sums[i] is None and varis[i] is None
                continue
            fr = [Fraction(v) for v in window.tolist()]
            mean = sum(fr) / 5
            assert sums[i] == sum(fr)
            assert varis[i] == sum((f - mean) ** 2 for f in fr) / 4

    def test_ulp_error(self):
        assert exact.ulp_error(1.0, Fraction(1), np.float64) == 0.0
        assert exact.ulp_error(1.0 + 2 ** -52, Fraction(1), np.float64) == pytest.approx(1.0)


class TestPolicy:
    @pytest.mark.parametrize("op", verify.OPS)
    def test_policy_holds_on_a_small_corpus(self, native, op):
        cases = verify.make_cases(60, seed=99, ops=(op,), dtypes=("float32", "float64"),
                                  max_n=3000)
        for case in cases:
            for kernel in verify.kernels_for(native, op):
                ok, ours, theirs, msg = verify.check_case(native, case, kernel)
                assert ok, (case, kernel, msg)

    @pytest.mark.parametrize("run_seed,case_seed", [(12, 1797722556), (14, 271030069)])
    def test_cases_found_by_the_nightly_verify(self, native, run_seed, case_seed):
        """Windows of ~1e-8 values after 1e8 ones: the sliding accumulator
        kept those values' rounding residue (4 and 256 ulp off; NumPy 0.5)."""
        [case] = [c for c in verify.make_cases(7000, run_seed, verify.OPS,
                                               ("float32", "float64"), 20000)
                  if c.seed == case_seed and c.dtype == "float64"]
        for kernel in verify.kernels_for(native, case.op):
            ok, ours, theirs, msg = verify.check_case(native, case, kernel)
            assert ok, (case, kernel, msg)

    @pytest.mark.parametrize("nonfinite", [False, True])
    def test_window_of_one_returns_the_input(self, native, nonfinite):
        """Whatever has left the window before, a window of one is its value."""
        x = verify.Case("rolling_sum", "float64", "wide", 5000, 1, 3, nonfinite).data()
        for kernel in verify.kernels_for(native, "rolling_sum"):
            for op in ("rolling_sum", "rolling_mean"):
                got = getattr(native, op)(x, 1, kernel)[0]
                assert np.array_equal(got, x, equal_nan=True), (op, kernel)

    def test_float32_reductions_accumulate_in_double(self, native):
        x = np.full(10_000_001, 0.1, dtype=np.float32)
        got = native.reduce_sum(x, "scalar")[0]
        assert got == pytest.approx(10_000_001 * float(np.float32(0.1)), rel=1e-14)

    def test_rolling_variance_adversarial_case(self, native):
        """Mean 1e6, std 1e-2: relative error vs np.var <= 1e-9."""
        x = np.random.default_rng(3).normal(1e6, 1e-2, 50_000)
        got = native.rolling_var(x, 20, 0)[0]
        ref = np.var(np.lib.stride_tricks.sliding_window_view(x, 20), axis=-1)
        assert np.max(np.abs(got - ref) / ref) <= 1e-9

    def test_constant_windows_have_exactly_zero_variance(self, native):
        x = np.repeat([0.1, 3.7, -2.25], 50)
        got = native.rolling_var(x, 10, 0)[0]
        assert (got[np.array([i % 50 <= 40 for i in range(len(got))])] == 0.0).all()


class TestVerifyCommand:
    def test_runs_and_reports(self, native, capsys, tmp_path):
        rc = cli.main(["verify", "--cases", "70", "--seed", "5", "--max-n", "2000",
                       "--failures-dir", str(tmp_path / "fail")])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "rolling_var" in out and "NaN/Inf semantics" in out and "FAIL" not in out

    def test_failing_cases_are_shrunk_and_saved(self, native, tmp_path, monkeypatch):
        """A deliberately broken kernel must be caught, shrunk and saved."""
        case = verify.Case("sum", "float64", "uniform", 4096, 1, 7, False, {})
        x = case.data()
        broken = lambda nat, c, xx, k: np.float64(np.sum(xx) + (1.0 if len(xx) > 3 else 0.0))
        monkeypatch.setattr(verify, "_native_result", broken)
        ok, *_ = verify.check_case(native, case, "scalar", x)
        assert not ok
        small_case, small_x = verify.shrink(native, case, "scalar", x)
        assert len(small_x) <= 8
        path = verify.save_failure(str(tmp_path), small_case, "scalar", small_x, "broken")
        assert np.load(path)["x"].tolist() == small_x.tolist()
