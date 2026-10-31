"""RegressBench pandas-38038-groupby-rolling: https://github.com/pandas-dev/pandas/issues/38038"""
import os, sys
sys.path.insert(0, os.getcwd())  # import the package from the checked-out tree
import numpy as np
import pandas as pd

# Reproducer from the issue with 1M instead of 10M rows.
pd.DataFrame({"a": np.random.default_rng(0).standard_normal(1_000_000), "b": 1}) \
    .groupby("b").rolling(3).mean()
