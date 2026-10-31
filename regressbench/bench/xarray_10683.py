"""RegressBench xarray-10683-interp-vectorized: https://github.com/pydata/xarray/issues/10683"""
import os, sys
sys.path.insert(0, os.getcwd())  # import the package from the checked-out tree
import numpy as np
import xarray as xr

# Reproducer from the issue (one repeat instead of ten).
rng = np.random.default_rng(seed=42)
n_t, n_r, n_z = 122, 65, 65
da = xr.DataArray(rng.random((n_t, n_r, n_z)), dims=["t", "r", "z"],
                  coords=dict(t=np.linspace(0.01, 0.855, n_t), r=np.linspace(0.04, 4.0, n_r),
                              z=np.linspace(-1.2, 1.2, n_z)))
interp_z = xr.DataArray(rng.random(n_t), dims=["t"], coords=dict(t=da.t.values))
da.interp(z=interp_z)
