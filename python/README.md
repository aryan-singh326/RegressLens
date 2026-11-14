# RegressLens

Performance regression testing for scientific Python and NumPy code. RegressLens
tells you **whether** your code got slower, **where**, **which commit** did it,
and **why**, with statistics that stay honest on noisy laptops and shared CI
runners.

```
pip install regresslens              # pure Python, any platform
pip install "regresslens[accel]"     # + optional native kernels
```

Python 3.9+, NumPy 1.22+. No code changes: RegressLens runs your existing script
and observes its NumPy calls.

## Why another benchmarking tool

Timing a script before and after a change is unreliable. CPU frequency, other
processes and cloud neighbours move run times by more than the regressions you
care about, and a slower machine looks exactly like slower code. RegressLens is
built around that:

* **Interleaved, anytime-valid comparisons.** `rglns compare` alternates the two
  revisions in ABBA/BAAB blocks, so drift hits both equally, and tracks the
  effect with an anytime-valid confidence sequence, so it can check after
  every block and stop as soon as the answer is clear. In null simulations it
  declared a false difference in at most 3.5% of runs at α = 5%; checking a
  fixed-size bootstrap interval after every block does so in 54%.
* **Drift canaries.** Every run also times two fixed reference workloads (compute
  and memory). If they moved, the verdict is *environment drift*, not
  *regression*.
* **Call-site attribution and diagnosis.** It records which NumPy call sites got
  slower and checks known causes: an input that became non-contiguous, dtype
  promotion, a kernel switch, a different size regime, with a suggested fix
  where there is one.
* **Bisection that uses all of this.** `rglns bisect` finds the culprit commit
  from noisy timings, giving extra prior weight to commits that touched the
  files on the regressed call site's stack.

## Example

The second commit below makes `load()` return a column of a wider table, which is
a strided view of the same values. The output is real, shortened where marked
`...` and wrapped to fit:

```
$ rglns compare HEAD~1 HEAD --cmd "python3 bench.py"
rglns compare: HEAD~1 (A) vs HEAD (B)
  verdict: REGRESSION
  effect of B vs A: +37.79%, anytime-valid 95% confidence sequence [+19.43%, +58.96%]
  4 interleaved blocks (16 measured runs), threshold 5%, 5 s
  canary shift B vs A: compute +12.7%, memory +0.4%; drift-bounded interval [+5.9%, +58.3%]
Per-call-site drill-down (exploratory): ...
  regression         +529.3% [+266.1%, +696.8%]         bench.py:13  [np.sum]
      diagnosis (naive-Bayes combination of independent detectors):
         94%  contiguity_loss: 100% of calls at bench.py:13 received a non-contiguous
              array (baseline 0%). ...
              -> Make the array contiguous before this call (np.ascontiguousarray),
                 ideally where the strided view is created. ...
```

## Commands

| Command | What it does |
|---|---|
| `rglns run -- python job.py` | Runs unmodified code and records per-call-site timings. `--observe-operators` also covers `a + b` and `x[mask]` in your own modules. |
| `rglns profile -- python job.py` | Runtime breakdown by call site. |
| `rglns compare A B --cmd "python bench.py"` | Interleaved comparison of two git revisions (in worktrees) with an anytime-valid confidence sequence. Verdict: regression, equivalent, improvement or inconclusive. |
| `rglns baseline --name v1 -- python job.py`<br>`rglns check --baseline v1 -- python job.py` | Records baseline sessions, then compares a new session against them: run-level bootstrap intervals, drift-bounded correction, Benjamini-Hochberg across call sites, size normalization and a ranked diagnosis. |
| `rglns bisect GOOD BAD --cmd "python bench.py"` | Probabilistic bisection over the commits between GOOD and BAD; reports the culprit with a 95% credible set. |
| `rglns watch --label nightly` | Change-point detection over a recurring pipeline's sessions (BOCPD). |
| `rglns verify` | Checks the native kernels against exact results (see below). |

`rglns sessions` lists recorded sessions, and `rglns telemetry` controls opt-in
telemetry, which is off unless you turn it on ([docs/telemetry.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/telemetry.md)).

## In CI

The GitHub Action runs `rglns compare` between the PR's base and head inside one
job, so no hosted baseline storage is needed. It keeps a single PR comment up to
date and fails the check only on a *regression* verdict.

```yaml
on: pull_request
permissions:
  contents: read
  pull-requests: write      # for the PR comment
jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with: {fetch-depth: 0}
      - uses: actions/setup-python@v6
        with: {python-version: "3.12"}
      - uses: aryan-singh326/RegressLens@v0.2.0
        with:
          command: python bench.py
          setup: pip install -e .
```

Inputs such as `theta` (threshold, default 5%), `max-blocks` and `base-ref` are
listed in [action.yml](https://github.com/aryan-singh326/RegressLens/blob/main/action.yml).

## Native kernels

`regresslens[accel]` adds compiled kernels for `rl.array` and for
`--accelerate` (which routes eligible `np.sum`/`np.mean` calls), and a C fast
path for observation. Everything works without them.

```python
import numpy as np
import regresslens as rl

prices = rl.array(np.load("prices.npy"))
signal = prices.rolling_mean(20).project_affine(1 / 3.0, -2.0)
print(signal.filter_gt(0.5).mean())
```

* **Accuracy:** the policy is to be never less accurate than NumPy against an
  exact reference, except within one unit in the last place. `rglns verify`
  tests it on a generated corpus, and CI checks 7,000 new cases every night.
  Sums are compensated, rolling sums and means re-anchor when their own error
  bound grows, and NaN and ±Inf follow NumPy's semantics
  ([docs/accuracy.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/accuracy.md)).
* **CPUs:** a portable scalar kernel set everywhere, plus AVX2+FMA kernels on
  x86-64, chosen at import time from runtime CPU detection. The same wheel is
  safe on any CPU, and CI runs it under an emulated CPU without AVX.
* **Wheels:** Linux x86_64 and aarch64 (manylinux_2_28), macOS x86_64 and arm64,
  Windows amd64, CPython 3.9-3.14. Elsewhere, pip builds from source, which
  needs a C++17 compiler.

## Documentation

* [docs/statistics.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/statistics.md): how check, compare, bisect and watch work
* [docs/validation.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/validation.md): simulation and real-machine validation results
* [docs/observation.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/observation.md): `rglns run`, what each Python version can observe, overhead
* [docs/accuracy.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/accuracy.md): the accuracy policy and `rglns verify`
* [docs/kernels.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/kernels.md): kernel sets, CPU dispatch, selector thresholds
* [docs/data-model.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/data-model.md): the local trace database
* [docs/telemetry.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/telemetry.md): opt-in telemetry and its schema
* [regressbench/](https://github.com/aryan-singh326/RegressLens/blob/main/regressbench/README.md): real NumPy-ecosystem regressions with known culprit commits
* [docs/novelty.md](https://github.com/aryan-singh326/RegressLens/blob/main/docs/novelty.md): what is and is not new here, with references
* [ROADMAP.md](https://github.com/aryan-singh326/RegressLens/blob/main/ROADMAP.md): what v0.1 planned for v0.2, and what happened to each item

## Development

```
python -m pip install ./cpp "./python[dev]"     # native kernels + package (editable: -e)
python -m pytest python/tests -m "not slow"
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
```

## History

0.1.0 (August 2026) was a single package: `rl.array` with scalar, AVX2 and
multithreaded AVX2 kernels for four operations on x86-64 Linux, and
`rglns baseline/check/profile` with per-call bootstrap intervals and Bonferroni
correction. 0.2.0 splits it into a pure-Python package and an optional native
one, observes unmodified code, replaces the statistics, and adds `compare`,
`bisect`, `watch`, `verify` and the GitHub Action (see [ROADMAP.md](https://github.com/aryan-singh326/RegressLens/blob/main/ROADMAP.md)).

v0.1's benchmark, kept for the record and not re-measured for 0.2: a mixed
pipeline (`diff_log -> rolling_mean -> zscore -> filter_gt -> mean`) run 100
times on two machines, with the output checked on every run.

| | Correctness | RegressLens median | NumPy median | Bootstrap 95% CI |
|---|---|---|---|---|
| GitHub Codespaces | 100/100 exact matches | 13.5ms | 21.8ms | [-41.2%, -31.0%] |
| Personal PC (WSL2) | 100/100 exact matches | 12.4ms | 17.0ms | [-30.2%, -20.0%] |

## License

Apache-2.0.
