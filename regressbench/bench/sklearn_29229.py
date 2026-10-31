"""RegressBench sklearn-29229-columntransformer-njobs: https://github.com/scikit-learn/scikit-learn/issues/29229"""
import random

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline

# Reproducer from the issue (n_col = 20, 20k rows).
def list_sum(col):
    return col.map(lambda x: sum(x))

random.seed(0)
n_col = 20
df = pd.DataFrame({f"{i}": [[random.random() for _ in range(random.randint(1, 5))]
                            for _ in range(20_000)] for i in range(n_col)})
pipeline = Pipeline([("transformer", ColumnTransformer(
    [(f"{i}", FunctionTransformer(list_sum), [f"{i}"]) for i in range(n_col)], n_jobs=2))])
with joblib.parallel_backend(backend="loky", mmap_mode="r+"):
    pipeline.fit_transform(df)
