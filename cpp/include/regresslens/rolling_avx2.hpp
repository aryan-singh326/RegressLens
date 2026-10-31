#pragma once
#include <cfloat>

#include "regresslens/avx2_common.hpp"
#include "regresslens/projection_avx2.hpp"  // avx2_detail::fma_sd

// Include only from translation units compiled with -mavx2 -mfma.
//
// DESIGN NOTE: a rolling sum is a recurrence (window i+1 = window i +
// incoming - outgoing), so it cannot be vectorized across consecutive
// outputs. Instead the output range is split into 4 contiguous segments
// and each AVX2 lane runs the scalar kernel's compensated sliding window
// over its own segment: 4 independent dependency chains instead of one.
// The scalar kernel is latency-bound on its single TwoSum chain, so this
// is where the speedup comes from.
//
// ACCURACY is identical to the scalar kernel: each lane keeps an
// error-free (s, c) accumulator, so every window sum is within about one
// ulp of exact, and the mean divides the double-double sum with one
// rounding. The pre-0.2 AVX2 design (exclusive prefix sums, then
// P[i+w] - P[i]) cancelled two large prefix values per window, and even
// with double-double prefixes the low-order terms' own rounding
// resurfaced as a few ulps on wide-dynamic-range data. This design has
// no prefix arrays, so it also needs no temporary allocation.
//
// Preconditions (enforced by the dispatcher): every input finite,
// 1 <= window <= n. Non-finite input goes to the scalar NaN-segmented
// kernel instead.

namespace regresslens {
namespace avx2_detail {

constexpr int kLanes = 4;

template <typename T>
static inline __m256d gather4(const T* in, const long long* idx) {
    return _mm256_set_pd(static_cast<double>(in[idx[3]]), static_cast<double>(in[idx[2]]),
                         static_cast<double>(in[idx[1]]), static_cast<double>(in[idx[0]]));
}

// Mean of a double-double (s + c) with one final rounding; FMA gives the
// exact remainder of q * w.
static inline __m256d dd_div_w(__m256d s, __m256d c, __m256d vw) {
    __m256d hi = _mm256_add_pd(s, c);  // normalize with TwoSum
    __m256d bp = _mm256_sub_pd(hi, s);
    __m256d lo = _mm256_add_pd(_mm256_sub_pd(s, _mm256_sub_pd(hi, bp)), _mm256_sub_pd(c, bp));
    __m256d q1 = _mm256_div_pd(hi, vw);
    __m256d p = _mm256_mul_pd(q1, vw);
    __m256d perr = _mm256_fmsub_pd(q1, vw, p);  // q1*w - p, exactly
    __m256d r = _mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(hi, p), perr), lo);
    return _mm256_add_pd(q1, _mm256_div_pd(r, vw));
}

template <typename T>
static inline void store_lanes(T* out, const long long* pos, const bool* active, __m256d v) {
    alignas(32) double lanes[kLanes];
    _mm256_store_pd(lanes, v);
    for (int l = 0; l < kLanes; ++l) {
        if (active[l]) out[pos[l]] = static_cast<T>(lanes[l]);
    }
}

template <typename T, bool Mean>
static inline size_t rolling_avx2_impl(const T* in, T* out, size_t n, size_t window) {
    if (window == 0 || n < window) return 0;
    const long long out_count = static_cast<long long>(n - window + 1);
    const long long w = static_cast<long long>(window);
    const long long seg = (out_count + kLanes - 1) / kLanes;
    const __m256d vw = _mm256_set1_pd(static_cast<double>(window));

    // Lane l covers outputs [begin[l], end[l]); trailing lanes may be empty.
    long long begin[kLanes], end[kLanes];
    for (int l = 0; l < kLanes; ++l) {
        begin[l] = l * seg < out_count ? l * seg : out_count;
        end[l] = (l + 1) * seg < out_count ? (l + 1) * seg : out_count;
    }

    // First window of every lane, compensated (scalar; O(4 * window)).
    alignas(32) double s0[kLanes], c0[kLanes];
    for (int l = 0; l < kLanes; ++l) {
        SumParts acc{0.0, 0.0};
        if (begin[l] < end[l]) {
            for (long long j = begin[l]; j < begin[l] + w; ++j) {
                two_sum(acc, static_cast<double>(in[j]));
            }
        }
        s0[l] = acc.s;
        c0[l] = acc.c;
    }
    __m256d s = _mm256_load_pd(s0), c = _mm256_load_pd(c0);

    long long pos[kLanes], in_idx[kLanes], out_idx[kLanes];
    bool active[kLanes];
    for (int l = 0; l < kLanes; ++l) {
        pos[l] = begin[l];
        active[l] = begin[l] < end[l];
    }
    store_lanes(out, pos, active, Mean ? dd_div_w(s, c, vw) : _mm256_add_pd(s, c));

    for (long long k = 1; k < seg; ++k) {
        for (int l = 0; l < kLanes; ++l) {
            active[l] = begin[l] + k < end[l];
            // Inactive lanes read a valid dummy element and are not stored.
            long long p = active[l] ? begin[l] + k : begin[l] < end[l] ? begin[l] : 0;
            pos[l] = p;
            in_idx[l] = active[l] ? p + w - 1 : p;
            out_idx[l] = active[l] ? p - 1 : p;
        }
        __m256d a = gather4(in, in_idx);   // incoming
        __m256d b = gather4(in, out_idx);  // outgoing
        // Exact difference a - b = d + e.
        __m256d d = _mm256_sub_pd(a, b);
        __m256d bp = _mm256_sub_pd(d, a);
        __m256d e = _mm256_sub_pd(_mm256_sub_pd(a, _mm256_sub_pd(d, bp)), _mm256_add_pd(b, bp));
        // s += d exactly (TwoSum), errors into c.
        __m256d t = _mm256_add_pd(s, d);
        __m256d bp2 = _mm256_sub_pd(t, s);
        __m256d err = _mm256_add_pd(_mm256_sub_pd(s, _mm256_sub_pd(t, bp2)), _mm256_sub_pd(d, bp2));
        c = _mm256_add_pd(c, _mm256_add_pd(err, e));
        s = t;
        store_lanes(out, pos, active, Mean ? dd_div_w(s, c, vw) : _mm256_add_pd(s, c));
    }
    return static_cast<size_t>(out_count);
}

}  // namespace avx2_detail

static inline size_t rolling_sum_avx2(const double* in, double* out, size_t n,
                                      size_t window) {
    return avx2_detail::rolling_avx2_impl<double, false>(in, out, n, window);
}
static inline size_t rolling_sum_avx2(const float* in, float* out, size_t n,
                                      size_t window) {
    return avx2_detail::rolling_avx2_impl<float, false>(in, out, n, window);
}
static inline size_t rolling_mean_avx2(const double* in, double* out, size_t n,
                                       size_t window) {
    return avx2_detail::rolling_avx2_impl<double, true>(in, out, n, window);
}
static inline size_t rolling_mean_avx2(const float* in, float* out, size_t n,
                                       size_t window) {
    return avx2_detail::rolling_avx2_impl<float, true>(in, out, n, window);
}

// One-pass scan: any NaN/Inf present, and max |x| over finite values.
static inline ScanResult scan_avx2(const double* in, size_t n) {
    using namespace avx2_detail;
    const __m256d vmax_finite = _mm256_set1_pd(DBL_MAX);
    __m256d vabsmax = _mm256_setzero_pd();
    int bad = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a = abs_pd(_mm256_loadu_pd(in + i));
        // NLE_UQ: true when a > DBL_MAX (Inf) or a is NaN.
        bad |= _mm256_movemask_pd(_mm256_cmp_pd(a, vmax_finite, _CMP_NLE_UQ));
        // Non-finite lanes are masked to 0 so they never become absmax.
        a = _mm256_and_pd(a, _mm256_cmp_pd(a, vmax_finite, _CMP_LE_OQ));
        vabsmax = _mm256_max_pd(vabsmax, a);
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, vabsmax);
    ScanResult r{bad != 0, 0.0};
    for (double v : lanes) r.absmax = v > r.absmax ? v : r.absmax;
    for (; i < n; ++i) {
        double a = in[i] < 0 ? -in[i] : in[i];
        if (!(a <= DBL_MAX)) {
            r.nonfinite = true;
        } else if (a > r.absmax) {
            r.absmax = a;
        }
    }
    return r;
}

static inline ScanResult scan_avx2(const float* in, size_t n) {
    const __m256 vmax_finite = _mm256_set1_ps(FLT_MAX);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 vabsmax = _mm256_setzero_ps();
    int bad = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_andnot_ps(sign, _mm256_loadu_ps(in + i));
        bad |= _mm256_movemask_ps(_mm256_cmp_ps(a, vmax_finite, _CMP_NLE_UQ));
        a = _mm256_and_ps(a, _mm256_cmp_ps(a, vmax_finite, _CMP_LE_OQ));
        vabsmax = _mm256_max_ps(vabsmax, a);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, vabsmax);
    ScanResult r{bad != 0, 0.0};
    for (float v : lanes) r.absmax = v > r.absmax ? v : r.absmax;
    for (; i < n; ++i) {
        float a = in[i] < 0 ? -in[i] : in[i];
        if (!(a <= FLT_MAX)) {
            r.nonfinite = true;
        } else if (a > r.absmax) {
            r.absmax = a;
        }
    }
    return r;
}

}  // namespace regresslens
