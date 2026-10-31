"""RegressBench pandas-57366-to-datetime-float-unit: https://github.com/pandas-dev/pandas/issues/57366"""
import pandas as pd

# The asv benchmark inference.ToDatetimeFromIntsFloats.time_sec_float64.
ts_sec_float = pd.Series(range(1521080307, 1521685107), dtype="int64").astype("float64")
for _ in range(3):
    pd.to_datetime(ts_sec_float, unit="s")
