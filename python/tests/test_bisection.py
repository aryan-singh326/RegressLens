"""Probabilistic bisection (WP5)."""
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import commit, make_git_repo, run_rglns
from regresslens.stats.bisection import Bisector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "validation"))
import simlib  # noqa: E402


def test_posterior_concentrates_on_the_step():
    rng = np.random.default_rng(0)
    bis = Bisector(10)
    mu = lambda i: 0.2 if i >= 6 else 0.0
    for i in (0, 10):
        bis.add(i, rng.normal(mu(i), 0.02, 20))
    for _ in range(12):
        i = bis.next_commit()
        bis.add(i, rng.normal(mu(i), 0.02, 3))
    k, p = bis.best()
    assert k == 6 and p > 0.95
    assert bis.credible_set(0.95) == [6]
    assert bis.m_bad - bis.m_good == pytest.approx(0.2, abs=0.02)


def test_next_commit_skips_unbuildable():
    bis = Bisector(9)
    bis.add(0, [0.0, 0.01, -0.01])
    bis.add(9, [1.0, 1.01, 0.99])
    first = bis.next_commit()
    bis.exclude(first)
    assert bis.next_commit() != first


@pytest.mark.parametrize("N,effect,sigma", [(16, 0.10, 0.05), (128, 0.30, 0.10),
                                            (1024, 0.10, 0.02)])
def test_accuracy_and_calibration(N, effect, sigma):
    rng = np.random.default_rng(N)
    res = [simlib.probabilistic_bisect(rng, N, int(rng.integers(1, N + 1)), effect, sigma)
           for _ in range(80)]
    assert np.mean([r["correct"] for r in res]) >= 0.9
    assert np.mean([r["in_credible"] for r in res]) >= 0.9  # >= 94% at 2000 trials


def test_informative_prior_saves_runs_and_misleading_prior_still_converges():
    rng = np.random.default_rng(1)
    N, eff, sigma = 128, 0.10, 0.05
    def runs(prior_p):
        out = []
        for _ in range(60):
            k = int(rng.integers(1, N + 1))
            prior = simlib.prior_vector(rng, N, k, prior_p) if prior_p is not None else None
            out.append(simlib.probabilistic_bisect(rng, N, k, eff, sigma, prior=prior))
        return out
    uniform, informative, misleading = runs(None), runs(0.8), runs(0.0)
    assert np.mean([r["runs"] for r in informative]) < np.mean([r["runs"] for r in uniform])
    assert np.mean([r["correct"] for r in misleading]) >= 0.85


def test_gradual_drift_is_recognized():
    rng = np.random.default_rng(2)
    N = 20
    bis = Bisector(N)
    for i in range(N + 1):
        bis.add(i, rng.normal(0.01 * i, 0.005, 6))
    assert bis.drift_log_bayes_factor() > math.log(10)
    step = Bisector(N)
    for i in range(N + 1):
        step.add(i, rng.normal(0.2 if i >= 11 else 0.0, 0.005, 6))
    assert step.drift_log_bayes_factor() < 0


PIPELINE = '''import numpy as np
def run(data):
    col = data[:, 0].copy()
    total = 0.0
    for _ in range(30):
        total += col.sum()
    return total
'''
BENCH = '''import numpy as np
from pkg.pipeline import run
data = np.random.default_rng(0).uniform(size=(400_000, 8))
for _ in range(4):
    run(data)
'''


@pytest.mark.slow
def test_integration_finds_commit_27_of_40(tmp_path):
    """Plan test: a 40-commit repo where commit 27 makes an array
    non-contiguous; bisect must name commit 27 within a time budget."""
    repo = make_git_repo(str(tmp_path / "repo"), {"pkg/__init__.py": "",
                                                   "pkg/pipeline.py": PIPELINE,
                                                   "bench.py": BENCH, "notes.md": "x\n"},
                         "commit 0")
    for i in range(1, 41):
        if i == 27:
            commit(repo, {"pkg/pipeline.py": PIPELINE.replace("data[:, 0].copy()",
                                                              "data[:, 0]")},
                   "commit 27: drop the copy")
        else:
            commit(repo, {"notes.md": f"note {i}\n"}, f"commit {i}")
    import time
    t0 = time.time()
    out = tmp_path / "bisect.json"
    proc = run_rglns("bisect", "HEAD~40", "HEAD", "--cmd", "python bench.py", "--json",
                     str(out), cwd=repo, timeout=900)
    elapsed = time.time() - t0
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "commit 27: drop the copy" in proc.stdout
    assert elapsed < 600
    # The attribution prior flags exactly the commits touching the
    # regressed call site's files: only commit 27.
    import json
    result = json.loads(out.read_text())
    assert "pkg/pipeline.py" in result["prior_files"]
    culprit = subprocess.run(["git", "rev-parse", "HEAD~13"], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.strip()
    assert result["prior_flagged"] == [culprit]


def test_endpoint_failure_names_the_endpoint_and_shows_why(tmp_path):
    repo = make_git_repo(str(tmp_path / "repo"), {"bench.py": "print('ok')\n"}, "commit 0")
    commit(repo, {"notes.md": "x\n"}, "commit 1")
    commit(repo, {"bench.py": "raise RuntimeError('broken at the bad end')\n"}, "commit 2")
    proc = run_rglns("bisect", "HEAD~2", "HEAD", "--cmd", "python bench.py", cwd=repo,
                     env={"RGLNS_CANARY": "0"})
    assert proc.returncode != 0
    assert "at the bad endpoint" in proc.stderr and "good endpoint" not in proc.stderr
    assert "RuntimeError: broken at the bad end" in proc.stderr
