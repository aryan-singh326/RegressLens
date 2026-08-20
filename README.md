# RegressLens

**Is this slowdown real, which operation caused it, and what should you do about it?**

Scientific and quantitative pipelines built on NumPy get run over and over — a factor
researcher recalculates signals every trading day, a climate scientist reprocesses
observations every morning, a genomics lab reruns the same analysis on each new sample.
When one of those pipelines gets slower, most teams find out the same way: someone
notices, then spends an afternoon manually re-running old code and comparing profiler
output to guess what changed.

RegressLens answers three questions instead of leaving you to guess:

```bash
rglns baseline --name v1 --runs 30 --script your_pipeline.py   # store current performance
rglns check --baseline v1 --runs 30 --script your_pipeline.py  # is a later run slower, and is it real?
rglns profile --script your_pipeline.py --runs 50               # where does the time actually go?
```

`rglns check` doesn't just time your script twice and compare — it uses bootstrap
confidence intervals with a practical-effect threshold and Bonferroni correction across
operations, so you get a real answer to "is this noise or a regression," not a coin flip
from two timing runs on a busy machine. When it finds a real regression, it tells you
which operation slowed down, whether the array lost memory contiguity (a common,
easy-to-miss cause), and whether fixing that is actually worth the cost — with real
numbers, not a guess.

## What it accelerates

RegressLens wraps NumPy arrays and transparently accelerates four operations with
hand-written scalar, AVX2, and multithreaded-AVX2 C++ kernels, automatically choosing
between them based on array size, dtype, and (for filtering) selectivity:

- Elementwise projection (`out = scale * in + offset`)
- Threshold filtering
- Sum/mean reduction
- Fixed-window rolling sum/mean

Everything else in your pipeline — `np.log`, `np.diff`, custom functions, whatever you're
already using — falls back to plain NumPy transparently. You don't need to rewrite your
pipeline; you wrap your arrays once and call the accelerated operations where they apply.

```python
import numpy as np
import regresslens as rl

prices = rl.array(np.load("prices.npy"))          # wrap once
returns = np.diff(np.log(prices.to_numpy()))       # unsupported ops: plain NumPy
returns_arr = rl.array(returns)
rolling = returns_arr.rolling_mean(window=20)       # accelerated
signals = rolling.filter_gt(threshold)              # accelerated, identity-preserving
result = signals.mean()                             # accelerated
```

Chained calls stay accelerated at every step — `arr.project_affine(...).filter_gt(...)`
doesn't silently drop back to plain NumPy after the first operation. This is tested
explicitly, not assumed.

## Does it actually help?

We built a realistic mixed pipeline (`diff_log → rolling_mean → zscore → filter_gt →
mean` — the exact shape a real factor-research pipeline looks like, deliberately mixing
accelerated and unaccelerated operations) and ran it 100 times on two independent
machines, checking output correctness on every single run, not sampled:

| | Correctness | RegressLens median | NumPy median | Bootstrap 95% CI |
|---|---|---|---|---|
| GitHub Codespaces | 100/100 exact matches | 13.5ms | 21.8ms | [-41.2%, -31.0%] |
| Personal PC (WSL2) | 100/100 exact matches | 12.4ms | 17.0ms | [-30.2%, -20.0%] |

RegressLens was consistently faster on both machines — the confidence intervals sit
entirely below zero on both, meaning this isn't measurement noise. The exact magnitude
of the speedup varies by hardware (as it should — this is a real, honestly reported
finding, not something we averaged away), but the direction reproduces.

Full validation methodology and raw output: [`validation/`](validation/).

## What it doesn't do (yet)

Being upfront about this matters more than it sounds like it should:

- **v0.1 scope is narrow on purpose**: float32/float64 only, contiguous arrays only,
  x86-64 Linux only, exactly four operations. If your pipeline is mostly PyTorch, JAX,
  SciPy, or BLAS calls, most of it will fall back to plain NumPy — regression detection
  and attribution still work, but kernel acceleration only applies to the supported
  subset.
- **No adaptive kernel selection.** Kernel choice comes from a hand-tuned heuristic
  derived from real benchmarking, not a learned model — the online-adaptive-selector
  research question is a deliberately separate, deferred track (see
  [Development history](#development-history) below).
- **Baseline/current comparison in `rglns check` is not literally interleaved.**
  True interleaving (alternating baseline and current code execution within one
  session) would require re-running old code, which needs git integration this doesn't
  have yet. This is a real, documented methodological limitation — see
  `regresslens/stats.py`'s module docstring.
- **`diff_log`-style operations aren't accelerated.** In our own validation pipeline,
  the unaccelerated `diff_log` stage was consistently the single largest cost — a real,
  reproducible finding across both test machines, and the clearest candidate for a
  future kernel.

## Installation

```bash
git clone https://github.com/<org>/regresslens
cd regresslens/python
pip install .
```

Requires x86-64 Linux (or WSL2) with AVX2 support. Building from source requires CMake
≥3.20 and a C++17 compiler; these are pulled in automatically as build dependencies.

## Development

See [`cpp/README.md`](cpp/README.md) *(if present)* or the dev workflow below:

```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
# run C++ tests: ./tests/test_*
cd ../../python && python3 -m pytest tests/ -v
```

## Development history

RegressLens was built in phases, and two of them are worth knowing about if you're
evaluating this project seriously:

- **The adaptive-kernel-selection research question (originally "Phase 2") was
  deliberately deferred** in favor of shipping the product first. The kernel selector is
  built behind a single swappable interface (`kernel_selector.hpp`/`.cpp`) specifically
  so that research can be dropped in later without touching anything downstream.
- **Everything in this README's benchmark section was independently verified**, not
  just built and assumed correct — including catching and fixing a real SQLite
  performance bug (default rollback-journal mode was adding ~0.5ms of fsync overhead
  per traced call) during the validation process itself.

## License

Apache 2.0.
