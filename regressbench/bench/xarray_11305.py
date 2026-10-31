"""RegressBench xarray-11305-to-dataframe-multiindex: https://github.com/pydata/xarray/issues/11305"""
import os, sys
sys.path.insert(0, os.getcwd())  # import the package from the checked-out tree
import numpy as np
import xarray as xr

# Reproducer from the issue with the middle dimension reduced 10x
# (100 x 2000 x 300 int64 is 480 MB).
da = xr.DataArray(np.arange(100 * 200 * 300).reshape(100, 200, 300), name="foo")
da.to_dataframe()
