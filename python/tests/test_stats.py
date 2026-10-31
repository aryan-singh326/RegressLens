"""Measurement core (WP3): run-level bootstrap, verdicts, drift bounds,
Benjamini-Hochberg, and the size-normalized cost model. Full Monte Carlo
grids are in validation/ (docs/validation.md); these use fewer trials."""
import math
import os
import sys

import numpy as np
import pytest

from regresslens import analysis, hw, store
from regresslens.stats import bootstrap, costmodel, fdr, legacy, verdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "validation"))
import simlib  # noqa: E402


class TestBootstrap:
    def test_detects_large_effect(self):
        rng = np.random.default_rng(0)
        b = bootstrap.Side([rng.normal(0, 0.02, 15)])
        c = bootstrap.Side([rng.normal(math.log(1.3), 0.02, 15)])
        r = bootstrap.compare(b, c, 0.05, 0.05, 1000, rng)
        assert r.ci_low > math.log(1.05) and r.p_value < 0.01
        assert verdict.assign([r])[0].verdict == verdict.REGRESSION

    def test_equivalent_when_tight_and_unchanged(self):
        rng = np.random.default_rng(1)
        b = bootstrap.Side([rng.normal(0, 0.005, 30)])
        c = bootstrap.Side([rng.normal(0, 0.005, 30)])
        v = verdict.assign([bootstrap.compare(b, c, 0.05, 0.05, 1000, rng)])[0]
        assert v.verdict == verdict.EQUIVALENT and v.advisory

    def test_inconclusive_estimates_runs_needed(self):
        rng = np.random.default_rng(2)
        b = bootstrap.Side([rng.normal(0, 0.10, 5)])
        c = bootstrap.Side([rng.normal(math.log(1.04), 0.10, 5)])
        v = verdict.assign([bootstrap.compare(b, c, 0.05, 0.05, 1000, rng)])[0]
        assert v.verdict == verdict.INCONCLUSIVE and v.runs_needed > 5

    def test_hierarchical_with_three_sessions_per_side(self):
        rng = np.random.default_rng(3)
        mk = lambda mu: bootstrap.Side([rng.normal(mu, 0.02, 6) for _ in range(3)])
        r = bootstrap.compare(mk(0.0), mk(0.0), 0.05, 0.05, 800, rng)
        assert r.hierarchical
        assert not verdict.assign([r])[0].advisory

    def test_session_variation_widens_hierarchical_interval(self):
        rng = np.random.default_rng(4)
        def side(mu):
            return bootstrap.Side([rng.normal(mu + rng.normal(0, 0.05), 0.01, 8)
                                   for _ in range(4)])
        b, c = side(0.0), side(0.0)
        hier = bootstrap.compare(b, c, 0.05, 0.05, 1000, rng)
        flat = bootstrap.compare(bootstrap.Side([b.flat()]), bootstrap.Side([c.flat()]),
                                 0.05, 0.05, 1000, rng)
        assert (hier.raw_ci_high - hier.raw_ci_low) > 1.5 * (flat.raw_ci_high - flat.raw_ci_low)

    def test_t_quantile_approximation(self):
        assert bootstrap.t_ppf(0.975, 9) == pytest.approx(2.262, abs=2e-3)
        assert bootstrap.t_ppf(0.975, 39) == pytest.approx(2.023, abs=1e-3)


class TestDriftBoundedCorrection:
    def test_environment_drift_not_regression(self):
        """Memory-bandwidth contention slows the workload and the memory
        canary equally; the code is unchanged."""
        rates, old_rate = simlib.canary_cell(trials=40, seed=5)
        assert rates.get(verdict.ENVIRONMENT_DRIFT, 0) >= 0.95
        assert rates.get(verdict.REGRESSION, 0) == 0
        assert old_rate >= 0.9  # the v0.1 engine calls it a regression

    def test_real_regression_survives_drift_bounds(self):
        rng = np.random.default_rng(6)
        b, _ = simlib.simulate_side(rng, 12, 0.02, 0.0)
        c, _ = simlib.simulate_side(rng, 12, 0.02, 0.30, canary_shift=(0.0, math.log(1.05)))
        r = bootstrap.compare(b, c, 0.05, 0.05, 1000, rng)
        assert verdict.assign([r])[0].verdict == verdict.REGRESSION
        assert r.lower_est < r.estimate  # the bound subtracted the canary shift

    def test_interval_spans_compute_and_memory_attribution(self):
        """effect in [r/c, r/m]: bounds use the larger and smaller shift."""
        lo, hi = bootstrap._bounds(np.array([0.3]), np.array([0.1]), np.array([0.2]), True)
        assert lo[0] == pytest.approx(0.1) and hi[0] == pytest.approx(0.2)


class TestMonteCarloSmall:
    """Small versions of the WP3 grid; the full 2000-trial grid is in
    validation/check_engine.py."""

    @pytest.mark.parametrize("sigma,effect", [(0.03, 0.0), (0.08, 0.03)])
    def test_false_regression_rate_and_coverage(self, sigma, effect):
        cell = simlib.check_cell(sigma, effect, runs=10, trials=150, seed=7)
        assert cell["new_regression"] <= 0.065
        assert 0.89 <= cell["new_coverage"] <= 0.99  # 93-97% at 2000 trials

    def test_v01_engine_undercovers(self):
        cell = simlib.check_cell(0.08, 0.0, runs=10, trials=100, seed=8)
        assert cell["old_coverage"] < 0.8  # pooled calls: pseudo-replication


class TestFDR:
    def test_bh_step_up(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        assert fdr.benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 6

    def test_bh_more_powerful_than_bonferroni_with_fdr_control(self):
        res = simlib.bh_experiment(trials=8, seed=9, m=200, k=10, B=400)
        assert res["fdr"] <= 0.08
        assert res["power_bh"] > res["power_bonferroni"]


class TestCostModel:
    CACHES = {"l1d": 48 * 1024, "l2": 2 * 1024 ** 2, "l3": 32 * 1024 ** 2}

    def test_theil_sen_ignores_outliers(self):
        x = np.arange(1, 21, dtype=float)
        y = 5 + 2 * x
        y[3] = 1000
        a, b = costmodel.theil_sen(x, y)
        assert b == pytest.approx(2.0) and a == pytest.approx(5.0)

    def test_single_size_cannot_be_fitted(self):
        assert costmodel.fit([(1000, 5.0), (1000, 5.1)], 8, self.CACHES) is None

    def test_extrapolation_label(self):
        model = costmodel.fit([(n, 100 + n) for n in (1000, 2000, 4000)], 8, self.CACHES)
        assert not model.extrapolated(8000)
        assert model.extrapolated(8001)

    def test_shift_attribution(self):
        base = costmodel.fit([(n, 1000 + 2 * n) for n in range(1000, 9000, 1000)], 8,
                             self.CACHES)
        slower_per_element = costmodel.fit([(n, 1000 + 3 * n) for n in range(1000, 9000, 1000)],
                                           8, self.CACHES)
        s = costmodel.shift(base, slower_per_element, 5000)
        assert s["beta"] == pytest.approx(0.5) and abs(s["alpha"]) < 1e-9


def _fake_sessions(rng, sizes, label, runs=8, per_elem=1.0, overhead=2000.0):
    sessions = []
    for si in range(1):
        s = store.Session(f"{label}{si}", label, 0.0, "{}", None, None, None, None, {})
        for r in range(runs):
            run = store.Run(f"{label}{si}r{r}", s.session_id, 1e9, None, None, 0, 0.0)
            noise = math.exp(rng.normal(0, 0.02))
            for n in sizes:
                t = (overhead + per_elem * n) * noise
                samples = list(t * np.exp(rng.normal(0, 0.03, 8)))
                run.calls.append(store.CallRow("site", "np.sum", "float64", n, True, "numpy",
                                               8, sum(samples), sum(map(math.log, samples)),
                                               samples, {}))
            s.runs.append(run)
        sessions.append(s)
    return sessions


class TestSizeNormalization:
    def test_twice_larger_inputs_are_not_a_regression(self, monkeypatch):
        """Plan test: baseline n in [1e5, 1e6], current at 2x n, identical code."""
        monkeypatch.setattr(hw, "cache_sizes", lambda: {"l1d": 1e15, "l2": 2e15, "l3": 3e15})
        rng = np.random.default_rng(10)
        base_sizes = [int(x) for x in np.geomspace(1e5, 1e6, 6)]
        base = _fake_sessions(rng, base_sizes, "b")
        cur = _fake_sessions(rng, [2 * n for n in base_sizes], "c")
        report = analysis.compare_sessions(base, cur, {}, theta=0.05, seed=1,
                                           with_diagnosis=False)
        site = next(s for s in report.sites if s.key == "site")
        assert site.verdict.verdict != verdict.REGRESSION
        assert "size-normalized" in site.size_note
        assert abs(site.comparison.estimate) < 0.03

    def test_real_per_element_slowdown_is_still_caught(self, monkeypatch):
        monkeypatch.setattr(hw, "cache_sizes", lambda: {"l1d": 1e15, "l2": 2e15, "l3": 3e15})
        rng = np.random.default_rng(11)
        sizes = [int(x) for x in np.geomspace(1e5, 1e6, 6)]
        base = _fake_sessions(rng, sizes, "b")
        cur = _fake_sessions(rng, [2 * n for n in sizes], "c", per_elem=1.5)
        report = analysis.compare_sessions(base, cur, {}, theta=0.05, seed=2,
                                           with_diagnosis=False)
        site = next(s for s in report.sites if s.key == "site")
        assert site.verdict.verdict == verdict.REGRESSION
        assert site.alpha_beta["beta"] == pytest.approx(0.5, abs=0.1)


def test_legacy_engine_port_matches_v01_decision():
    rng = np.random.default_rng(12)
    b = rng.normal(100, 1, 200)
    assert legacy.evaluate(b, b * 1.5, rng=rng)[0] == "regression"
    assert legacy.evaluate(b, b, rng=rng)[0] == "within_noise"
