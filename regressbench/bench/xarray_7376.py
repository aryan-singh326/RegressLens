"""RegressBench xarray-7376-groupby-map-multiindex: https://github.com/pydata/xarray/issues/7376"""
import os, sys
sys.path.insert(0, os.getcwd())  # import the package from the checked-out tree
import numpy as np
import pandas as pd
import xarray as xr

# Reproducer from the issue, with two changes:
#   * m = 3_000 instead of 100_000, so a run takes seconds (0.008 s before
#     the culprit, ~6 s after);
#   * the issue's `ds.assign(x=...)` wrapper is dropped: right after the
#     culprit commit it raised an unrelated ValueError ("cannot remove
#     coordinate(s) ...", fixed later), which would make those commits
#     look unbuildable. The slow part is groupby(...).map itself.
m = 3_000
df = pd.DataFrame({
    "i1": [1] * m + [2] * m + [3] * m + [4] * m,
    "i2": list(range(m)) * 4,
    "d3": np.random.default_rng(0).integers(0, 2, 4 * m).astype(bool),
})
ds = df.to_xarray().set_coords(["i1", "i2"]).set_index(index=["i1", "i2"])

def per_grp(da):
    return da

ds["d3"].groupby("i1").map(per_grp)
