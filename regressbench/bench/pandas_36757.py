"""RegressBench pandas-36757-groupby-fillna: https://github.com/pandas-dev/pandas/issues/36757"""
import os, sys
sys.path.insert(0, os.getcwd())  # import the package from the checked-out tree
import numpy as np
import pandas as pd

# Reproducer from the issue.
N = 2000
df = pd.DataFrame({"A": [1] * N, "B": [np.nan, 1.0] * (N // 2)})
df = df.sort_values("A").set_index("A")
df["B"] = df.groupby("A")["B"].fillna(method="ffill")
