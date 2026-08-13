"""Tests for regresslens.array — the Python/NumPy integration layer.

Run with: python3 -m pytest test_array.py -v
"""
import numpy as np
import pytest

import regresslens as rl


class TestArrayConstruction:
    def test_wraps_float64(self):
        arr = rl.array(np.array([1.0, 2.0, 3.0]))
        assert arr.dtype == np.float64

    def test_wraps_float32(self):
        arr = rl.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert arr.dtype == np.float32

    def test_rejects_unsupported_dtype(self):
        with pytest.raises(rl.UnsupportedDTypeError):
            rl.array(np.array([1, 2, 3], dtype=np.int64))

    def test_rejects_unsupported_dtype_from_list(self):
        # int64 is what plain Python ints coerce to by default.
        with pytest.raises(rl.UnsupportedDTypeError):
            rl.array([1, 2, 3])


class TestSum:
    def test_sum_matches_numpy_f64(self):
        rng = np.random.default_rng(1)
        data = rng.uniform(-1000, 1000, 50_000).astype(np.float64)
        arr = rl.array(data)
        assert arr.sum() == pytest.approx(np.sum(data), rel=1e-9)

    def test_sum_matches_numpy_f32(self):
        rng = np.random.default_rng(2)
        data = rng.uniform(-1000, 1000, 50_000).astype(np.float32)
        arr = rl.array(data)
        assert arr.sum() == pytest.approx(float(np.sum(data, dtype=np.float64)),
                                           rel=1e-3)

    def test_mean_matches_numpy(self):
        rng = np.random.default_rng(3)
        data = rng.uniform(-1000, 1000, 10_000).astype(np.float64)
        arr = rl.array(data)
        assert arr.mean() == pytest.approx(np.mean(data), rel=1e-9)

    def test_mean_of_empty_raises(self):
        arr = rl.array(np.array([], dtype=np.float64))
        with pytest.raises(ValueError):
            arr.mean()

    def test_sum_of_empty_is_zero(self):
        arr = rl.array(np.array([], dtype=np.float64))
        assert arr.sum() == 0.0


class TestNumpyDispatch:
    """These are the tests that actually matter: do NumPy's own
    entry points route through RegressLens's kernels, or does
    wrapping in rl.array just add overhead for nothing?"""

    def test_np_sum_is_intercepted(self):
        rng = np.random.default_rng(4)
        data = rng.uniform(-1000, 1000, 50_000).astype(np.float64)
        arr = rl.array(data)
        assert np.sum(arr) == pytest.approx(np.sum(data), rel=1e-9)

    def test_np_sum_with_axis_falls_back_not_errors(self):
        # v0.1 only supports whole-array reduction. This must not
        # crash — it must fall back to correct NumPy behavior.
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr = rl.array(data)
        result = np.sum(arr, axis=0)
        np.testing.assert_array_equal(result, np.sum(data, axis=0))

    def test_unsupported_ufunc_falls_back_correctly(self):
        data = np.array([1.0, 4.0, 9.0, 16.0])
        arr = rl.array(data)
        result = np.sqrt(arr)
        np.testing.assert_array_almost_equal(result, np.sqrt(data))

    def test_binary_ufunc_between_two_wrapped_arrays(self):
        a = rl.array(np.array([1.0, 2.0, 3.0]))
        b = rl.array(np.array([10.0, 20.0, 30.0]))
        result = np.add(a, b)
        np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])

    def test_array_function_dispatch_only_fires_for_wrapped_types(self):
        # Sanity check on the dispatch mechanism itself: calling
        # np.sum on a PLAIN ndarray must behave exactly as normal
        # NumPy, completely unaffected by regresslens being imported.
        data = np.array([1.0, 2.0, 3.0])
        assert np.sum(data) == 6.0


class TestNonContiguousFallback:
    def test_sliced_array_falls_back_without_crashing(self):
        rng = np.random.default_rng(5)
        data = rng.uniform(-10, 10, 200).astype(np.float64)
        sliced = data[::2]
        assert not sliced.flags["C_CONTIGUOUS"]
        arr = rl.array(sliced)
        # Must not raise, must produce the correct answer via fallback.
        assert arr.sum() == pytest.approx(np.sum(sliced), rel=1e-9)


class TestEscapeHatch:
    def test_to_numpy_returns_underlying_array(self):
        data = np.array([1.0, 2.0, 3.0])
        arr = rl.array(data)
        assert arr.to_numpy() is data

    def test_np_asarray_works_via_array_protocol(self):
        data = np.array([1.0, 2.0, 3.0])
        arr = rl.array(data)
        result = np.asarray(arr)
        np.testing.assert_array_equal(result, data)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
