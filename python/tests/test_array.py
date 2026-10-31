"""rl.array: results, fallbacks, identity preservation, recording."""
import numpy as np
import pytest

import regresslens as rl
from regresslens import _accel, observe, store


def _rows(op=None):
    observe.flush()
    conn = store.connect()
    try:
        sql = "SELECT operator, dtype, n, contiguous, kernel, count, extra_json FROM calls"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [r for r in rows if op is None or r[0] == op]


class TestConstruction:
    def test_wraps_float64_and_float32(self):
        assert rl.array(np.array([1.0, 2.0])).dtype == np.float64
        assert rl.array(np.array([1.0], dtype=np.float32)).dtype == np.float32

    @pytest.mark.parametrize("data", [np.array([1, 2, 3]), [1, 2, 3]])
    def test_rejects_unsupported_dtype(self, data):
        with pytest.raises(rl.UnsupportedDTypeError):
            rl.array(data)


class TestReductions:
    def test_sum_matches_numpy_and_dtype(self):
        data = np.random.default_rng(1).uniform(-1000, 1000, 50_000)
        out = rl.array(data).sum()
        assert out == pytest.approx(np.sum(data), rel=1e-12)
        assert isinstance(out, np.float64)

    def test_float32_sum_is_float32_and_accurate(self):
        data = np.random.default_rng(2).uniform(-1000, 1000, 50_000).astype(np.float32)
        out = rl.array(data).sum()
        assert isinstance(out, np.float32)
        assert out == pytest.approx(float(np.sum(data, dtype=np.float64)), rel=1e-6)

    def test_mean(self):
        data = np.random.default_rng(3).uniform(-1000, 1000, 10_000)
        assert rl.array(data).mean() == pytest.approx(np.mean(data), rel=1e-12)

    def test_mean_of_empty_raises_and_sum_of_empty_is_zero(self):
        empty = rl.array(np.array([], dtype=np.float64))
        assert empty.sum() == 0.0
        with pytest.raises(ValueError):
            empty.mean()

    def test_compensated_sum_beats_naive_cancellation(self, native):
        data = np.array([1e16, 1.0, -1e16])
        assert rl.array(data).sum() == 1.0

    def test_nan_and_inf_follow_ieee(self):
        assert np.isnan(rl.array(np.array([1.0, np.nan, 2.0])).sum())
        assert rl.array(np.array([1.0, np.inf])).sum() == np.inf
        assert np.isnan(rl.array(np.array([np.inf, 1.0, -np.inf])).sum())

    def test_two_dimensional_input_falls_back_instead_of_crashing(self):
        data = np.arange(12.0).reshape(3, 4)
        assert rl.array(data).sum() == 66.0
        assert rl.array(data).mean() == pytest.approx(5.5)


class TestNumpyDispatch:
    def test_np_sum_is_intercepted(self):
        data = np.random.default_rng(4).uniform(-1, 1, 1000)
        assert np.sum(rl.array(data)) == pytest.approx(np.sum(data), rel=1e-12)
        assert any(r[0] == "reduction" for r in _rows())

    def test_np_sum_with_axis_falls_back(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(np.sum(rl.array(data), axis=0), [4.0, 6.0])

    def test_ufuncs_fall_back(self):
        a = rl.array(np.array([1.0, 4.0, 9.0]))
        np.testing.assert_allclose(np.sqrt(a), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(np.add(a, a), [2.0, 8.0, 18.0])

    def test_plain_ndarray_unaffected(self):
        assert np.sum(np.array([1.0, 2.0, 3.0])) == 6.0

    def test_escape_hatches(self):
        data = np.array([1.0, 2.0, 3.0])
        assert rl.array(data).to_numpy() is data
        np.testing.assert_array_equal(np.asarray(rl.array(data)), data)


class TestElementwiseAndFilter:
    def test_projection(self):
        data = np.random.default_rng(10).uniform(-100, 100, 20_000)
        out = rl.array(data).project_affine(2.5, -1.5)
        assert isinstance(out, rl.array)
        np.testing.assert_allclose(out.to_numpy(), 2.5 * data - 1.5, rtol=1e-15)

    def test_filter(self):
        data = np.random.default_rng(12).uniform(0, 1, 20_000)
        out = rl.array(data).filter_gt(0.5)
        np.testing.assert_array_equal(out.to_numpy(), data[data > 0.5])

    def test_filter_threshold_is_exclusive_and_nan_never_passes(self):
        assert len(rl.array(np.array([5.0, 5.0])).filter_gt(5.0)) == 0
        data = np.array([np.nan, 1.0, np.inf, -np.inf, 3.0])
        np.testing.assert_array_equal(rl.array(data).filter_gt(0.0).to_numpy(),
                                      [1.0, np.inf, 3.0])

    @pytest.mark.parametrize("op", ["project_affine", "filter_gt"])
    def test_non_contiguous_falls_back(self, op):
        data = np.random.default_rng(13).uniform(0, 1, 400)[::2]
        arr = rl.array(data)
        out = arr.project_affine(3.0, 1.0) if op == "project_affine" else arr.filter_gt(0.5)
        expected = 3.0 * data + 1.0 if op == "project_affine" else data[data > 0.5]
        np.testing.assert_allclose(out.to_numpy(), expected)
        kernels = {r[4] for r in _rows()}
        assert kernels & {"numpy_fallback", "numpy"}


class TestRolling:
    def test_rolling_sum_matches_convolve(self):
        data = np.random.default_rng(14).uniform(-10, 10, 5000)
        out = rl.array(data).rolling_sum(20)
        np.testing.assert_allclose(out.to_numpy(), np.convolve(data, np.ones(20), "valid"),
                                   rtol=1e-12, atol=1e-12)

    def test_rolling_mean_and_var(self):
        data = np.array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(rl.array(data).rolling_mean(2).to_numpy(), [3.0, 5.0, 7.0])
        np.testing.assert_allclose(rl.array(data).rolling_var(2).to_numpy(), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(rl.array(data).rolling_var(3, ddof=1).to_numpy(),
                                   [4.0, 4.0])

    def test_window_larger_than_array_is_empty_and_window_zero_raises(self):
        assert len(rl.array(np.array([1.0, 2.0])).rolling_sum(10)) == 0
        with pytest.raises(ValueError):
            rl.array(np.array([1.0, 2.0])).rolling_sum(0)

    def test_single_nan_is_confined_to_its_windows(self):
        """The Phase 4 defect: one NaN poisoned every later window
        (35 of 37 here) where np.convolve confines it to 4."""
        x = np.arange(1, 41, dtype=np.float64)
        x[5] = np.nan
        out = rl.array(x).rolling_sum(4).to_numpy()
        ref = np.convolve(x, np.ones(4), mode="valid")
        np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
        assert int(np.isnan(out).sum()) == 4

    def test_price_like_variance_is_accurate(self):
        """Adversarial case from the plan: mean 1e6, std 1e-2."""
        x = np.random.default_rng(5).normal(1e6, 1e-2, 20_000)
        out = rl.array(x).rolling_var(20).to_numpy()
        ref = np.var(np.lib.stride_tricks.sliding_window_view(x, 20), axis=-1)
        assert np.max(np.abs(out - ref) / ref) <= 1e-9

    def test_non_contiguous_rolling(self):
        data = np.random.default_rng(15).uniform(-10, 10, 200)[::2]
        np.testing.assert_allclose(rl.array(data).rolling_sum(5).to_numpy(),
                                   np.convolve(data, np.ones(5), "valid"), rtol=1e-12)


class TestIdentityPreservation:
    def test_chain_stays_accelerated(self):
        data = np.random.default_rng(99).uniform(-10, 10, 5000)
        step1 = rl.array(data).project_affine(2.0, 1.0)
        step2 = step1.filter_gt(5.0)
        step3 = step2.rolling_mean(10)
        assert all(isinstance(s, rl.array) for s in (step1, step2, step3))
        np1 = 2.0 * data + 1.0
        np2 = np1[np1 > 5.0]
        np3 = np.convolve(np2, np.ones(10) / 10, mode="valid")
        assert step3.sum() == pytest.approx(np.sum(np3), rel=1e-9)


class TestRecording:
    def test_calls_are_aggregated_not_written_per_call(self):
        data = np.ones(1000)
        arr = rl.array(data)
        for _ in range(50):
            arr.sum()
        conn = store.connect()
        try:
            assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0
        finally:
            conn.close()
        rows = _rows("reduction")
        assert len(rows) == 1 and rows[0][5] == 50

    def test_filter_records_selectivity_and_rolling_window(self):
        rl.array(np.arange(10.0)).filter_gt(4.5)
        rl.array(np.arange(100.0)).rolling_sum(25)
        import json
        by_op = {r[0]: json.loads(r[6]) for r in _rows()}
        assert by_op["filter"]["sel_sum"] == pytest.approx(0.5)
        assert by_op["rolling"]["window"] == 25

    def test_kernel_name_without_native_module(self, monkeypatch):
        monkeypatch.setattr(_accel, "native", lambda: None)
        assert rl.array(np.ones(10)).sum() == 10.0
        assert {r[4] for r in _rows()} == {"numpy"}


class TestSelectivityFeedback:
    def test_history_is_recorded_per_call_site(self, native):
        data = np.random.default_rng(3).uniform(0, 1, 7777)
        arr = rl.array(data)
        for _ in range(6):
            arr.filter_gt(0.98)  # ~2% selectivity, always from this one line
        observe.flush()
        conn = store.connect()
        try:
            sites = {r[0] for r in conn.execute(
                "SELECT call_site_id FROM calls WHERE operator = 'filter'")}
            assert len(sites) == 1
            total, count = store.site_selectivity_history(conn, sites.pop())
        finally:
            conn.close()
        assert count == 6 and total / count < 0.05

    def test_skewed_history_selects_scalar_kernel(self, avx2):
        data = np.random.default_rng(3).uniform(0, 1, 7777)
        _, _, kernel, _ = avx2.filter_gt(data, 0.98, "auto", 0, 0.02)
        assert kernel == "scalar"
        _, _, kernel, _ = avx2.filter_gt(data, 0.5, "auto", 0, 0.5)
        assert kernel == "avx2"
