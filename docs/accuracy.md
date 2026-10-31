# Accuracy policy

Native results are not bit-identical to NumPy, so the guarantee is stated
against the **exact** result and tested:

> For every operation, dtype and native kernel, on every case of the test
> corpus, the largest error over the outputs, in ulps of the exact value, is at
> most max(NumPy's largest error on the same case, 1 ulp).

In words: never less accurate than NumPy, except within one unit of rounding
of the exact answer. Exact references (`regresslens/exact.py`) use integer
arithmetic on each float's mantissa and exponent, so sums, window sums and
window variances are computed with no rounding at all.

`rglns verify` checks the policy on a generated corpus: six distributions
(uniform, normal, price-like random walk, mean 1e6 with std 1e-2, ±1e8
cancellation, 16 orders of magnitude), random sizes and windows, both dtypes,
every available kernel, and NaN/±Inf injected into a quarter of the cases.
For each (operation, dtype, kernel) it reports the maximum observed error, the
case count and the seed. Failing cases are shrunk and saved to
`~/.regresslens/verify_failures/` as `.npz` + `.json` for replay. Results:
[validation.md](validation.md#wp8-rglns-verify-accuracy-policy). On 14,000
cases every native kernel (sum, mean, rolling sum/mean/variance, AVX2
projection) stays within 0.5 ulp of exact, i.e. correctly rounded on every
case tested; the filter is exact. NumPy's `np.convolve`-based rolling mean is
off by up to 5e11 ulp on the same inputs. The scalar projection matches NumPy
bit for bit (same two roundings), so its error is NumPy's.

## How the kernels get there

* **Sums and means:** compensated (Knuth TwoSum) accumulation in double for
  both dtypes, in scalar, AVX2 and multithreaded kernels. Means divide the
  double-double sum with a single rounding.
* **Projection:** the scalar kernel computes `scale * x + offset` exactly as
  NumPy does (two roundings, `-ffp-contract=off`). AVX2 uses FMA (one rounding)
  for every element, including the tail.
* **Rolling sum/mean:** an error-free sliding accumulator, so the error does
  not grow with n (the 0.1 float32 kernel reached 5.6% relative error).
* **Rolling variance:** deviations from a shift K and their squares as exact
  double-doubles (TwoSum, Dekker TwoProduct), and a double-double final
  formula. K re-anchors every window and whenever cancellation would exceed
  1e10. Windows of equal values return exactly 0. The plan's adversarial case
  (mean 1e6, std 1e-2) has relative error ~1e-14 against `np.var` (requirement
  1e-9).

## NaN and Inf

Results follow IEEE / NumPy semantics on every path:

* sums: NaN anywhere gives NaN; +Inf and -Inf together give NaN; otherwise the
  infinity;
* filter: `NaN > t` is false, as in `x[x > t]`;
* rolling windows: a window is NaN only if it contains a NaN (or both
  infinities), and is Inf only if it contains that infinity. One pass detects
  non-finite input and routes it to a NaN-segmented algorithm that keeps
  per-window NaN/+Inf/-Inf counts. Finite inputs large enough to risk
  accumulator overflow go to NumPy instead.

Hypothesis property tests (`tests/test_nan_safety.py`) inject NaN and ±Inf at
random positions into every operation and dtype, on the scalar, AVX2 and NumPy
paths, and compare with NumPy using `equal_nan=True`.
