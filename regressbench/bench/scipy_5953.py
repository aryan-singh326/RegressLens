"""RegressBench scipy-5953-ckdtree-linf: https://github.com/scipy/scipy/issues/5953"""
import os, sys
sys.path.insert(0, os.getcwd())  # import the package from the checked-out tree
import numpy as np
from scipy.spatial import cKDTree

# Reproducer from the issue with 200k instead of 6M points.
arr = np.random.RandomState(0).uniform(size=(200_000, 3))
cKDTree(arr).query(arr, 1, p=np.inf)
