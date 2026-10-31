"""Anytime-valid interleaved comparison (WP4)."""
import math
import os
import sys

import numpy as np
import pytest

from conftest import commit, make_git_repo, run_rglns
from regresslens.stats import sequential, verdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "validation"))
import simlib  # noqa: E402


def test_halfwidth_matches_plan_formula():
    n, s2, t2, a = 10, 0.03 ** 2, math.log1p(0.05) ** 2, 0.05
    expected = math.sqrt(2 * s2 * (s2 + n * t2) / (n * n * t2)
                         * math.log(math.sqrt(1 + n * t2 / s2) / a))
    assert sequential.halfwidth(n, s2, t2, a) == pytest.approx(expected)


def test_small_sample_variance_inflation_converges_to_1_15():
    assert sequential.variance_inflation(4) > 2.5
    assert sequential.variance_inflation(400) == pytest.approx(1.15)


def test_stop_rules():
    cs = sequential.ConfidenceSequence(theta=0.05, min_blocks=4)
    for d in [0.20, 0.21, 0.19, 0.20]:
        cs.add(d)
    assert cs.decision() == verdict.REGRESSION
    cs = sequential.ConfidenceSequence(theta=0.05, min_blocks=4)
    for d in [-0.002, 0.001, 0.0, 0.002, -0.001, 0.0]:
        cs.add(d)
    assert cs.decision() == verdict.EQUIVALENT
    cs = sequential.ConfidenceSequence(theta=0.05, min_blocks=4)
    for d in [-0.3, -0.31, -0.29, -0.3]:
        cs.add(d)
    assert cs.decision() == verdict.IMPROVEMENT
    cs = sequential.ConfidenceSequence(theta=0.05, min_blocks=4, max_blocks=5)
    for d in [0.1, -0.1, 0.05, -0.06, 0.02]:
        cs.add(d)
    assert cs.decision() == verdict.INCONCLUSIVE


@pytest.mark.parametrize("sigma", [0.02, 0.05])
def test_null_never_declares_at_default_theta(sigma):
    assert simlib.sequential_null(400, sigma, theta=0.05, seed=1) <= 0.055


@pytest.mark.parametrize("sigma", [0.02, 0.05])
def test_always_valid_even_at_theta_zero(sigma):
    """The strict version: any interval excluding 0 under the null counts.
    Full 5000-trial runs: validation/compare_sequential.py."""
    assert simlib.sequential_null(600, sigma, theta=0.0, seed=2) <= 0.07


def test_naive_repeated_bootstrap_inflates_false_positives():
    assert simlib.sequential_null(150, 0.05, theta=0.0, seed=3, naive=True) > 0.25


def test_efficiency():
    rng = np.random.default_rng(4)
    blocks = [simlib.sequential_trial(rng, 0.10, 0.03)[1] for _ in range(300)]
    assert np.median(blocks) <= 8
    decisions = [simlib.sequential_trial(rng, 0.03, 0.03)[0] for _ in range(300)]
    assert np.mean([d == verdict.REGRESSION for d in decisions]) <= 0.05


BENCH = "import numpy as np\nfrom signals import smooth\nx = np.random.default_rng(0).uniform(-1, 1, 300_000)\nfor _ in range(12):\n    smooth(x)\n"
SIGNALS = "import numpy as np\ndef smooth(x):\n    return np.convolve(x, np.ones({w}) / {w}, mode='valid')\n"


@pytest.mark.slow
def test_real_git_fixture_flags_injected_slowdown(tmp_path):
    repo = make_git_repo(str(tmp_path / "repo"), {"bench.py": BENCH,
                                                   "signals.py": SIGNALS.format(w=5)})
    commit(repo, {"signals.py": SIGNALS.format(w=60)}, "wider window")
    md = tmp_path / "out.md"
    proc = run_rglns("compare", "HEAD~1", "HEAD", "--cmd", "python bench.py", "--seed", "1",
                     "--markdown", str(md), cwd=repo)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "REGRESSION" in proc.stdout and "signals.py" in proc.stdout
    assert "<!-- rglns-compare -->" in md.read_text()


@pytest.mark.slow
def test_a_a_comparison_is_not_a_regression(tmp_path):
    repo = make_git_repo(str(tmp_path / "repo"), {"bench.py": BENCH,
                                                   "signals.py": SIGNALS.format(w=5)})
    proc = run_rglns("compare", "HEAD", "HEAD", "--cmd", "python bench.py", "--seed", "2",
                     "--max-blocks", "30", "--no-observe", cwd=repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "REGRESSION" not in proc.stdout and "IMPROVEMENT" not in proc.stdout
