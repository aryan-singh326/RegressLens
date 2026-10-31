"""Property tests (Hypothesis): NaN and +/-Inf injected at random positions
into every operation and dtype must give NumPy's result (equal_nan=True),
on every kernel path: native scalar, native AVX2, and the NumPy fallback."""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import regresslens as rl
from regresslens import _accel

DTYPES = [np.float32, np.float64]
SPECIALS = [np.nan, np.inf, -np.inf]


@st.composite
def arrays_with_specials(draw, min_size=1, max_size=300):
    dtype = draw(st.sampled_from(DTYPES))
    n = draw(st.integers(min_size, max_size))
    seed = draw(st.integers(0, 2 ** 32 - 1))
    x = np.random.default_rng(seed).uniform(-1000, 1000, n).astype(dtype)
    k = draw(st.integers(0, min(n, 6)))
    positions = draw(st.lists(st.integers(0, n - 1), min_size=k, max_size=k))
    kinds = draw(st.lists(st.sampled_from(SPECIALS), min_size=k, max_size=k))
    for p, v in zip(positions, kinds):
        x[p] = v
    return x


def kernels():
    native = _accel.native()
    out = ["numpy"]
    if native is not None:
        out.append("scalar")
        if native.info()["avx2_active"]:
            out.append("avx2")
    return out


@pytest.fixture(params=kernels())
def kernel(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(_accel, "native", lambda: None)
        yield request.param
    else:
        with _accel.using_kernel(request.param):
            yield request.param


SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.function_scoped_fixture])


def same(got, want, rtol):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.array_equal(np.isposinf(got), np.isposinf(want))
    assert np.array_equal(np.isneginf(got), np.isneginf(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=rtol, atol=rtol * 1000)


def rtol_for(x):
    return 1e-4 if x.dtype == np.float32 else 1e-9


@SETTINGS
@given(x=arrays_with_specials())
def test_sum_and_mean(kernel, x):
    with np.errstate(all="ignore"):
        same(rl.array(x).sum(), np.sum(x.astype(np.float64)), rtol_for(x))
        same(rl.array(x).mean(), np.mean(x.astype(np.float64)), rtol_for(x))


@SETTINGS
@given(x=arrays_with_specials(), scale=st.floats(-10, 10), offset=st.floats(-100, 100))
def test_projection(kernel, x, scale, offset):
    with np.errstate(all="ignore"):
        want = x * x.dtype.type(scale) + x.dtype.type(offset)
    same(rl.array(x).project_affine(scale, offset).to_numpy(), want, rtol_for(x))


@SETTINGS
@given(x=arrays_with_specials(), threshold=st.floats(-1000, 1000))
def test_filter(kernel, x, threshold):
    got = rl.array(x).filter_gt(threshold).to_numpy()
    want = x[x > x.dtype.type(threshold)]
    assert np.array_equal(got, want, equal_nan=True)


@SETTINGS
@given(x=arrays_with_specials(), window=st.integers(1, 40))
def test_rolling_sum_and_mean(kernel, x, window):
    if window > len(x):
        assert len(rl.array(x).rolling_sum(window)) == 0
        return
    with np.errstate(all="ignore"):
        want = np.convolve(x.astype(np.float64), np.ones(window), mode="valid")
    same(rl.array(x).rolling_sum(window).to_numpy(), want, rtol_for(x))
    same(rl.array(x).rolling_mean(window).to_numpy(), want / window, rtol_for(x))


@SETTINGS
@given(x=arrays_with_specials(), window=st.integers(1, 40), ddof=st.integers(0, 1))
def test_rolling_var(kernel, x, window, ddof):
    if window > len(x) or ddof >= window:
        return
    with np.errstate(all="ignore"):
        want = np.var(np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), window),
                      axis=-1, ddof=ddof)
    same(rl.array(x).rolling_var(window, ddof=ddof).to_numpy(), want,
         1e-4 if x.dtype == np.float32 else 1e-7)
