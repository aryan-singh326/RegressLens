"""Fault-injection fixture for the diagnosis framework (WP7).

RGLNS_FAULT selects one cause, the same code path otherwise:
  none         baseline behavior
  strided      the input becomes a strided view (contiguity loss)
  float64      the input is upcast with astype(float64) (dtype promotion)
  bigger       10x larger n (size regime shift)
  selectivity  filter selectivity moves from ~0.5 to ~0.95 (kernel selection)
The bandwidth-hog cause needs no code change: validation/diagnosis_faults.py
runs a memory hog next to the unchanged program.
"""
import os

import numpy as np

import regresslens as rl

FAULT = os.environ.get("RGLNS_FAULT", "none")
n = 200_000 * (10 if FAULT == "bigger" else 1)
rng = np.random.default_rng(1)
base = rng.uniform(0.0, 1.0, n).astype(np.float32)

if FAULT == "strided":
    wide = np.zeros((n, 4), dtype=np.float32)
    wide[:, 0] = base
    data = wide[:, 0]
elif FAULT == "float64":
    data = base.astype(np.float64)
else:
    data = base

threshold = 0.05 if FAULT == "selectivity" else 0.5


def pipeline(x):
    arr = rl.array(x)
    kept = arr.filter_gt(threshold)
    return kept.sum() + arr.rolling_mean(20).sum()


for _ in range(5):
    pipeline(data)
