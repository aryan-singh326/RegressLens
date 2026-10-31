#pragma once
#include <cfloat>
#include <cmath>
#include <limits>

#include "regresslens/column.hpp"
#include "regresslens/compensated.hpp"

// Fixed-window rolling kernels, scalar (portable) versions.
//
// Output length is n - window + 1 (no partial windows), matching
// np.convolve(x, np.ones(window), mode="valid").
//
// ACCURACY. A plain sliding accumulator (add the incoming element,
// subtract the outgoing one) carries every earlier window's rounding
// error forward, so its error grows with n instead of with the window.
// Measured in Phase 1: 5.6% relative error for float32 at
// n=1,000,003, window=500. These kernels keep the accumulator as an
// error-free (s, c) pair, so every output is within about one ulp of
// the exact window sum, which is more accurate than np.convolve's
// per-window dot product.
//
// NaN/Inf. A sliding accumulator also never recovers from a
// non-finite input: once Inf enters, subtracting it later yields NaN
// forever, so one NaN used to poison every later window (np.convolve
// confines it to the windows that contain it). Callers must scan the
// input first and send non-finite input to the *_nonfinite_scalar
// variants, which track NaN/+Inf/-Inf counts per window and reproduce
// IEEE summation semantics exactly.

namespace regresslens {

template <typename T>
ScanResult scan_scalar(const T* in, size_t n) {
    ScanResult r{false, 0.0};
    for (size_t i = 0; i < n; ++i) {
        double a = std::fabs(static_cast<double>(in[i]));
        if (!(a <= DBL_MAX)) {  // NaN compares false; Inf exceeds DBL_MAX
            r.nonfinite = true;
        } else if (a > r.absmax) {
            r.absmax = a;
        }
    }
    return r;
}

namespace rolling_detail {

// Exact difference a - b as (d, e) with d + e == a - b.
inline void two_diff(double a, double b, double& d, double& e) {
    d = a - b;
    double bp = d - a;
    e = (a - (d - bp)) - (b + bp);
}

// Window value from the compensated accumulator, rounded once (the
// mean divides the double-double sum, not its rounded value).
template <bool Mean>
inline double window_value(const CompSum& acc, double w) {
    return Mean ? dd_div(DD{acc.s, acc.c}, w) : acc.s + acc.c;
}

template <typename T, bool Mean>
void rolling_finite(const T* in, T* out, size_t n, size_t window) {
    const double w = static_cast<double>(window);
    CompSum acc;
    for (size_t i = 0; i < window; ++i) acc.add(static_cast<double>(in[i]));
    out[0] = static_cast<T>(window_value<Mean>(acc, w));

    const size_t out_count = n - window + 1;
    for (size_t i = 1; i < out_count; ++i) {
        double d, e;
        two_diff(static_cast<double>(in[i + window - 1]),
                 static_cast<double>(in[i - 1]), d, e);
        acc.add(d);
        acc.c += e;
        out[i] = static_cast<T>(window_value<Mean>(acc, w));
    }
}

struct NonfiniteCounts {
    size_t nan = 0, pinf = 0, ninf = 0;
};

template <typename T>
inline void account(NonfiniteCounts& nf, CompSum& acc, T x, bool adding) {
    if (std::isnan(x)) {
        adding ? ++nf.nan : --nf.nan;
    } else if (std::isinf(x)) {
        size_t& slot = x > 0 ? nf.pinf : nf.ninf;
        adding ? ++slot : --slot;
    } else {
        acc.add(adding ? static_cast<double>(x) : -static_cast<double>(x));
    }
}

template <typename T, bool Mean>
inline T emit_nonfinite(const NonfiniteCounts& nf, const CompSum& acc, double w) {
    if (nf.nan > 0 || (nf.pinf > 0 && nf.ninf > 0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (nf.pinf > 0) return std::numeric_limits<T>::infinity();
    if (nf.ninf > 0) return -std::numeric_limits<T>::infinity();
    return static_cast<T>(window_value<Mean>(acc, w));
}

template <typename T, bool Mean>
void rolling_nonfinite(const T* in, T* out, size_t n, size_t window) {
    const double w = static_cast<double>(window);
    NonfiniteCounts nf;
    CompSum acc;
    for (size_t i = 0; i < window; ++i) account(nf, acc, in[i], true);
    out[0] = emit_nonfinite<T, Mean>(nf, acc, w);

    const size_t out_count = n - window + 1;
    for (size_t i = 1; i < out_count; ++i) {
        account(nf, acc, in[i + window - 1], true);
        account(nf, acc, in[i - 1], false);
        out[i] = emit_nonfinite<T, Mean>(nf, acc, w);
    }
}

}  // namespace rolling_detail

// --- Table entry points (finite input, 1 <= window <= n) ---

template <typename T>
void rolling_sum_finite_scalar(const T* in, T* out, size_t n, size_t window) {
    rolling_detail::rolling_finite<T, false>(in, out, n, window);
}

template <typename T>
void rolling_mean_finite_scalar(const T* in, T* out, size_t n, size_t window) {
    rolling_detail::rolling_finite<T, true>(in, out, n, window);
}

template <typename T>
void rolling_sum_nonfinite_scalar(const T* in, T* out, size_t n, size_t window) {
    rolling_detail::rolling_nonfinite<T, false>(in, out, n, window);
}

template <typename T>
void rolling_mean_nonfinite_scalar(const T* in, T* out, size_t n, size_t window) {
    rolling_detail::rolling_nonfinite<T, true>(in, out, n, window);
}

// --- Rolling variance ---
//
// var = (S2 - S1^2 / w) / (w - ddof) with S1 = sum(x - K) and
// S2 = sum((x - K)^2) over the window. A raw E[x^2] - E[x]^2 prefix
// formulation (K = 0) loses every significant digit on price-like data
// (mean 1e6, std 1e-2). Here:
//   * deviations x - K are exact double-doubles (TwoSum) and squares are
//     exact double-doubles (Dekker TwoProduct), accumulated with
//     compensation, so S1 and S2 carry ~106 bits;
//   * K is the mean of a recent window, re-anchored every `window`
//     outputs (an O(window) rebuild, O(1) amortized), so the cancelling
//     term (mean - K)^2 stays small even when the series drifts;
//   * a guard re-anchors immediately when S1^2 / w exceeds the variance
//     term by more than kMaxCancellation. On data spanning many orders
//     of magnitude (1e-8 next to 1e8) an inherited K can be so far from
//     the current window that even ~106 bits cancel away;
//   * the final subtraction and divisions are double-double, so each
//     output is rounded once;
//   * windows whose values are all equal return exactly 0 (a sliding
//     count of unequal neighbours), as the exact variance is 0.
// Added and removed terms use the same K, so they cancel exactly.

struct DDSum {
    CompSum hi;
    double lo = 0.0;
    void add(DD x) {
        hi.add(x.hi);
        lo += x.lo;
    }
    void sub(DD x) {
        hi.add(-x.hi);
        lo -= x.lo;
    }
    DD value() const { return two_sum(hi.s, hi.c + lo); }
};

inline DD dd_square(DD y) {
    DD p = two_prod(y.hi, y.hi);
    p.lo += 2.0 * y.hi * y.lo + y.lo * y.lo;
    return p;
}

// Double-double carries ~106 bits; allowing at most 1e10 (~33 bits) of
// cancellation leaves well over the 53 bits a correctly rounded double
// needs.
constexpr double kMaxCancellation = 1e10;

// Variance of the window; `ok` is false when the shifted sums cancel too
// much to trust (the caller re-anchors K and recomputes).
inline double dd_variance(const DDSum& s1, const DDSum& s2, double w, double denom,
                          bool& ok) {
    DD a = s1.value();
    DD q = dd_div_dd(dd_square(a), w);  // S1^2 / w
    DD b = s2.value();
    DD d = two_sum(b.hi, -q.hi);        // S2 - S1^2 / w
    d.lo += b.lo - q.lo;
    d = two_sum(d.hi, d.lo);
    ok = !(std::fabs(q.hi) > kMaxCancellation * std::fabs(d.hi));
    double var = dd_div(d, denom);
    return var < 0.0 ? 0.0 : var;
}

template <typename T>
void rolling_var_scalar_impl(const T* in, T* out, size_t n, size_t window,
                             size_t ddof) {
    const size_t out_count = n - window + 1;
    const double w = static_cast<double>(window);
    const double denom = static_cast<double>(window) - static_cast<double>(ddof);
    double K = 0.0;
    DDSum s1, s2;
    size_t bad = 0;      // non-finite values currently in the window
    size_t unequal = 0;  // pairs (j-1, j) inside the window with x[j-1] != x[j]

    auto neq = [&](size_t j) { return !(in[j - 1] == in[j]); };  // NaN pairs: unequal
    auto add_term = [&](DDSum& a, DDSum& b, double x, bool adding) {
        DD y = two_sum(x, -K);
        DD y2 = dd_square(y);
        if (adding) {
            a.add(y);
            b.add(y2);
        } else {
            a.sub(y);
            b.sub(y2);
        }
    };
    auto rebuild = [&](size_t start) {
        CompSum mean_acc;
        size_t finite = 0;
        bad = 0;
        for (size_t j = start; j < start + window; ++j) {
            double x = static_cast<double>(in[j]);
            if (std::isfinite(x)) {
                mean_acc.add(x);
                ++finite;
            } else {
                ++bad;
            }
        }
        K = finite ? mean_acc.value() / static_cast<double>(finite) : 0.0;
        s1 = DDSum{};
        s2 = DDSum{};
        for (size_t j = start; j < start + window; ++j) {
            double x = static_cast<double>(in[j]);
            if (std::isfinite(x)) add_term(s1, s2, x, true);
        }
    };
    auto emit = [&](size_t i) {
        if (bad > 0) {
            out[i] = std::numeric_limits<T>::quiet_NaN();
        } else if (unequal == 0) {
            out[i] = T(0);
        } else {
            bool ok;
            double var = dd_variance(s1, s2, w, denom, ok);
            if (!ok) {
                rebuild(i);  // K = this window's mean: cancellation ~1
                var = dd_variance(s1, s2, w, denom, ok);
            }
            out[i] = static_cast<T>(var);
        }
    };

    for (size_t j = 1; j < window; ++j) unequal += neq(j) ? 1 : 0;
    rebuild(0);
    emit(0);
    for (size_t i = 1; i < out_count; ++i) {
        // Slide the neighbour count: pair (i-1, i) leaves the window,
        // pair (i+w-2, i+w-1) enters it.
        if (window > 1) {
            unequal -= neq(i) ? 1 : 0;
            unequal += neq(i + window - 1) ? 1 : 0;
        }
        if (i % window == 0) {
            rebuild(i);
        } else {
            double x_in = static_cast<double>(in[i + window - 1]);
            double x_out = static_cast<double>(in[i - 1]);
            if (std::isfinite(x_in)) {
                add_term(s1, s2, x_in, true);
            } else {
                ++bad;
            }
            if (std::isfinite(x_out)) {
                add_term(s1, s2, x_out, false);
            } else {
                --bad;
            }
        }
        emit(i);
    }
}

// --- Convenience API (used by the C++ tests) ---
// Handles window == 0 / n < window by returning 0 outputs, scans the
// input, and routes non-finite input to the NaN-segmented algorithm.

template <typename T>
size_t rolling_sum_scalar(const T* in, T* out, size_t n, size_t window) {
    if (window == 0 || n < window) return 0;
    if (scan_scalar<T>(in, n).nonfinite) {
        rolling_sum_nonfinite_scalar<T>(in, out, n, window);
    } else {
        rolling_sum_finite_scalar<T>(in, out, n, window);
    }
    return n - window + 1;
}

template <typename T>
size_t rolling_mean_scalar(const T* in, T* out, size_t n, size_t window) {
    if (window == 0 || n < window) return 0;
    if (scan_scalar<T>(in, n).nonfinite) {
        rolling_mean_nonfinite_scalar<T>(in, out, n, window);
    } else {
        rolling_mean_finite_scalar<T>(in, out, n, window);
    }
    return n - window + 1;
}

template <typename T>
size_t rolling_var_scalar(const T* in, T* out, size_t n, size_t window,
                          size_t ddof = 0) {
    if (window == 0 || n < window) return 0;
    rolling_var_scalar_impl<T>(in, out, n, window, ddof);
    return n - window + 1;
}

}  // namespace regresslens
