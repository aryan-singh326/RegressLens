"""RegressBench numpy-26807-log-complex64: https://github.com/numpy/numpy/issues/26807"""
import numpy as np

# Reproducer from the issue.
n = 1_000_000
rng = np.random.default_rng(0)
z = rng.normal(loc=[0, 0], scale=2, size=(n, 2)).view(np.complex128)[:, 0]
zs = z.astype(np.complex64)
for _ in range(5):
    np.log(zs)
