"""The eight call forms the observation backends are specified against
(WP2). tests/test_observe.py runs this under `rglns run` and checks which
forms each backend observes on each Python version."""
import threading

import numpy as np


def forms():
    x = np.linspace(0.0, 1.0, 1000)
    m = x.reshape(10, 100)
    a = x.copy()
    b = x[::-1].copy()
    mask = x > 0.5
    np.sum(x)                                  # 1 module function
    x.sum()                                    # 2 ndarray method
    np.convolve(x, np.ones(5), mode="valid")   # 3 Python-level NumPy function
    np.add(a, b)                               # 4 ufunc object
    a + b                                      # 5 operator
    x[mask]                                    # 6 boolean indexing
    np.mean(m, axis=1)                         # 7 function with kwargs
    worker = threading.Thread(target=lambda: np.sum(b))  # 8 call in a thread
    worker.start()
    worker.join()
    print("checksum", round(float(np.sum(x) + x.sum()), 6))


if __name__ == "__main__":
    forms()
