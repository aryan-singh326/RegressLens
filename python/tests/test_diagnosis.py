"""Diagnosis framework (WP7): detectors, naive-Bayes combination, and
fault-injection fixtures. The 30-trial-per-cause run (including the
bandwidth hog) is validation/diagnosis_faults.py."""
import math
import os

import pytest

from conftest import FIXTURES
from regresslens import analysis, diagnosis, store
from regresslens.launcher import Launcher

FAULTS = os.path.join(FIXTURES, "faults.py")


def row(n=1000, dtype="float64", contiguous=True, kernel="avx2", count=10, sel=None):
    extra = {"sel_sum": sel * count, "sel_n": count} if sel is not None else {}
    return store.CallRow("site", "filter", dtype, n, contiguous, kernel, count, 100.0 * count,
                         count * math.log(100.0), [100.0] * min(count, 64), extra)


def run(compute=1e5, mem=2e7):
    return store.Run("r", "s", 1e9, compute, mem, 0, 0.0)


def view(rows, runs=None):
    return diagnosis.SiteView(rows, runs or [run() for _ in range(5)], "pipe.py:3 f")


def top(base, cur):
    hyps = diagnosis.diagnose(view(*base) if isinstance(base, tuple) else view(base),
                              view(*cur) if isinstance(cur, tuple) else view(cur), "site")
    return hyps[0].name if hyps else None


class TestDetectors:
    def test_null_gives_no_diagnosis(self):
        assert top([row()], [row()]) is None

    def test_contiguity_loss(self):
        hyps = diagnosis.diagnose(view([row()]),
                                  view([row(contiguous=False, kernel="numpy_fallback")]),
                                  "site")
        assert hyps[0].name == "contiguity_loss"
        assert "ascontiguousarray" in hyps[0].remediation
        assert 0.5 < hyps[0].posterior <= 1.0

    def test_dtype_promotion(self):
        assert top([row(dtype="float32")], [row(dtype="float64")]) == "dtype_promotion"

    def test_kernel_selection_with_selectivity_drift(self):
        hyps = diagnosis.diagnose(view([row(kernel="avx2", sel=0.5)]),
                                  view([row(kernel="scalar", sel=0.95)]), "site")
        assert hyps[0].name == "kernel_selection" and "AVX2 band" in hyps[0].evidence

    def test_numpy_fallback_not_explained_by_contiguity(self):
        assert top([row(kernel="avx2")], [row(kernel="numpy")]) == "numpy_fallback"

    def test_size_regime(self):
        assert top([row(n=10_000)], [row(n=100_000)]) == "size_regime"

    def test_environment_drift(self):
        base = ([row()], [run(1e5 * (1 + 0.001 * i), 2e7) for i in range(6)])
        cur = ([row()], [run(1e5 * (1 + 0.001 * i), 2.6e7) for i in range(6)])
        assert top(base, cur) == "environment_drift"

    def test_independent_evidence_multiplies(self):
        """Naive Bayes: two causes from two detectors are both reported,
        ranked by likelihood ratio."""
        hyps = diagnosis.diagnose(
            view([row(dtype="float32", n=10_000)]),
            view([row(dtype="float64", n=100_000)]), "site")
        assert {h.name for h in hyps} == {"dtype_promotion", "size_regime"}
        assert hyps == sorted(hyps, key=lambda h: -h.likelihood_ratio)


def _sessions(fault, runs=3):
    lnch = Launcher(["python", FAULTS], label=f"fault:{fault}", capture=True, quiet=True,
                    extra_env={"RGLNS_FAULT": fault})
    for _ in range(runs):
        r = lnch.run()
        assert r.exit_code == 0, r.stderr
    return lnch.session_id


def _top_diagnosis(base_id, cur_id):
    """The strongest code-cause hypothesis. environment_drift is left out:
    it reports the real state of the machine (contention from other jobs
    moves the canaries), which these tests do not control. Its detector
    is tested above and with an injected bandwidth hog in
    validation/diagnosis_faults.py."""
    conn = store.connect()
    base, cur = store.load_sessions(conn, [base_id]), store.load_sessions(conn, [cur_id])
    info = store.call_site_info(conn, {c.call_site_id for s in base + cur for r in s.runs
                                       for c in r.calls})
    report = analysis.compare_sessions(base, cur, info, seed=1)
    best = None
    for site in report.sites:
        for h in site.hypotheses:
            if h.name == "environment_drift":
                continue
            if best is None or h.likelihood_ratio > best.likelihood_ratio:
                best = h
    return best.name if best else None


@pytest.mark.parametrize("fault,expected", [
    ("strided", "contiguity_loss"),
    ("float64", "dtype_promotion"),
    ("bigger", "size_regime"),
    ("none", None),
])
def test_fault_fixture_top_diagnosis(fault, expected):
    base = _sessions("none")
    assert _top_diagnosis(base, _sessions(fault)) == expected


def test_selectivity_fault_changes_kernel(avx2):
    base = _sessions("none")
    assert _top_diagnosis(base, _sessions("selectivity")) == "kernel_selection"
