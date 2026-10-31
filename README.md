# RegressLens

Performance regression intelligence for scientific Python / NumPy pipelines:
find out **whether** your code got slower, **where**, **which commit** did it, and
**why**, with statistics that stay honest on noisy machines.

```
pip install regresslens              # pure Python, every platform
pip install "regresslens[accel]"     # + native scalar/AVX2 kernels (binary wheels)
```

This README describes 0.2.0, which is not on PyPI yet: PyPI has 0.1.0, the
earlier design (see [History](#history)). Until 0.2.0 is published, install
from a checkout with `python -m pip install ./cpp ./python`.

RegressLens works without the native kernels; `[accel]` adds faster, more
accurate kernels for `rl.array` and `--accelerate`, and a C fast path for
observation. Wheels: Linux x86_64/aarch64 (manylinux_2_28), macOS x86_64/arm64,
Windows amd64, CPython 3.9-3.13. AVX2 kernels are chosen at import time from
runtime CPU detection, so the same wheel is safe on every CPU.

## Commands

| | |
|---|---|
| `rglns run -- python job.py` | Run unmodified code with NumPy call observation; per-call-site stats are stored, a coverage report goes to stderr. `--observe-operators` also observes `a + b` and `x[mask]` in your modules; `--accelerate` routes eligible `np.sum`/`np.mean` calls to the native kernels. |
| `rglns profile -- python job.py` | Runtime breakdown by call site. |
| `rglns baseline --name v1 -- python job.py`<br>`rglns check --baseline v1 -- python job.py` | Compare a new session against recorded baseline sessions: run-level bootstrap, drift-bounded correction, Benjamini-Hochberg across call sites, size normalization, verdicts *regression / equivalent / inconclusive / environment drift*, and a ranked diagnosis. |
| `rglns compare main HEAD --cmd "python bench.py"` | Interleaved ABBA/BAAB measurement of two revisions (git worktrees) with an anytime-valid confidence sequence: stops as soon as the answer is clear. |
| `rglns bisect GOOD BAD --cmd "python bench.py"` | Probabilistic bisection with a prior from runtime call-site attribution; reports the culprit with a 95% credible set. |
| `rglns watch --label nightly` | Change-point detection (BOCPD) over a recurring pipeline's sessions. |
| `rglns verify` | Checks the native kernels' accuracy against exact results. |

GitHub Action (fails the check only on a `regression` verdict, one PR comment
updated per push). By default it installs RegressLens from the action's own
source, so pin the action to a release tag once one exists:

```yaml
- uses: actions/checkout@v4
  with: {fetch-depth: 0}
- uses: actions/setup-python@v5
  with: {python-version: "3.12"}
- uses: aryan-singh326/RegressLens@main
  with:
    command: python bench.py
    setup: pip install -e .
```

## Explicit acceleration

```python
import numpy as np
import regresslens as rl

prices = rl.array(np.load("prices.npy"))
signal = prices.rolling_mean(20).project_affine(1 / 3.0, -2.0)
print(signal.filter_gt(0.5).mean())
```

`rl.array` operations (sum/mean, project_affine, filter_gt, rolling
sum/mean/var) use the native kernels for 1-D contiguous float32/float64 input
and NumPy otherwise; results follow NumPy's NaN/Inf semantics and are at least
as accurate as NumPy (docs/accuracy.md).

## Documentation

* [docs/observation.md](docs/observation.md): `rglns run`, what each Python version can observe, overhead
* [docs/statistics.md](docs/statistics.md): check, compare, bisect and watch, in detail
* [docs/validation.md](docs/validation.md): Monte Carlo and real-machine validation results
* [docs/accuracy.md](docs/accuracy.md): the accuracy policy and `rglns verify`
* [docs/kernels.md](docs/kernels.md): kernel sets, CPU dispatch, characterization
* [docs/data-model.md](docs/data-model.md): the trace database (schema v2)
* [docs/telemetry.md](docs/telemetry.md): opt-in telemetry and its schema
* [regressbench/](regressbench/README.md): real regressions with known culprits
* [docs/novelty.md](docs/novelty.md): what is and is not new here
* [ROADMAP.md](ROADMAP.md): what v0.1 planned for v0.2, and what happened to each item

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
one, observes unmodified code, and replaces the statistics (see
[ROADMAP.md](ROADMAP.md)).

v0.1's benchmark, kept for the record and not re-measured for 0.2: a mixed
pipeline (`diff_log -> rolling_mean -> zscore -> filter_gt -> mean`) run 100
times on two machines, with the output checked on every run.

| | Correctness | RegressLens median | NumPy median | Bootstrap 95% CI |
|---|---|---|---|---|
| GitHub Codespaces | 100/100 exact matches | 13.5ms | 21.8ms | [-41.2%, -31.0%] |
| Personal PC (WSL2) | 100/100 exact matches | 12.4ms | 17.0ms | [-30.2%, -20.0%] |

Two notes from that phase still apply: the adaptive kernel-selection research
question was deferred, and the selector stays behind one interface
(`cpp/include/regresslens/kernel_selector.hpp`) so it can be replaced later;
and validation caught a real SQLite performance bug (the default
rollback-journal mode added ~0.5 ms of fsync per traced call), fixed with WAL
mode.

Apache-2.0.
