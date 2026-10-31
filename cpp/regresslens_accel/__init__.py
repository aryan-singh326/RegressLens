"""Native kernels for RegressLens (the ``regresslens[accel]`` extra).

Do not import this package directly; ``regresslens`` loads it lazily
and falls back to NumPy implementations when it is missing. The
compiled module ``regresslens_accel._native`` picks the scalar or AVX2
kernel set at import time from runtime CPU detection, so the same
wheel is safe on every CPU of its architecture.
"""

__version__ = "0.2.0"
