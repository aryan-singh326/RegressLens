"""RegressLens: performance regression intelligence for recurring
scientific Python/NumPy pipelines.

v0.1 scope: float32/float64, contiguous arrays, x86-64 Linux.
Currently wired up: sum/mean reduction. Projection, filter, and
rolling follow the same pattern (see array.py) and are not yet
implemented in the Python layer as of this checkpoint.
"""
from .array import array, UnsupportedDTypeError

__all__ = ["array", "UnsupportedDTypeError"]
__version__ = "0.1.0.dev0"
