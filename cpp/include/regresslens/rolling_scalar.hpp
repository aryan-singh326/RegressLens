#pragma once
#include "regresslens/column.hpp"

namespace regresslens {

// Fixed-window rolling sum: out[i] = sum(in[i .. i+window-1]) for
// i in [0, n-window]. Output size is n - window + 1 (returns 0 if
// n < window — no partial windows in v0.1, matching the "no NaN
// handling" scope: a partial window would need a sentinel value).
//
// Uses an O(n) sliding accumulator (add the incoming element,
// subtract the outgoing one) rather than recomputing each window's
// sum from scratch (which would be O(n*window)). This is a real
// algorithmic choice with a real precision tradeoff: the sliding
// accumulator's rounding error can drift over long sequences in a
// way that recomputing from scratch does not, because each output
// depends on the accumulated rounding history of every window
// before it, not just its own window's inputs.
//
// MEASURED AND FIXED: the naive float32-accumulator version of this
// kernel drifted up to 5.6% relative error at n=1,000,003/window=500
// against a trustworthy double-precision reference — unacceptable
// for a tool whose entire premise is trustworthy measurement. Fixed
// by accumulating internally in double for float32 inputs (see
// below), which brought AVX2's equivalent design down to ~6e-8
// error at the same scale. Applying the same fix here for symmetry
// and because 5.6% error is not a documentable quirk, it's a bug.
template <typename T>
struct RollingAccumulatorType {
    using type = T;
};
template <>
struct RollingAccumulatorType<float> {
    using type = double;  // float32 sliding accumulator drifts too
                           // much at scale; accumulate wider.
};
template <typename T>
size_t rolling_sum_scalar(const T* in, T* out, size_t n, size_t window) {
    if (window == 0 || n < window) return 0;

    using AccT = typename RollingAccumulatorType<T>::type;
    AccT acc = AccT(0);
    for (size_t i = 0; i < window; ++i) acc += (AccT)in[i];
    out[0] = (T)acc;

    size_t out_count = n - window + 1;
    for (size_t i = 1; i < out_count; ++i) {
        acc += (AccT)in[i + window - 1] - (AccT)in[i - 1];
        out[i] = (T)acc;
    }
    return out_count;
}

template <typename T>
size_t rolling_mean_scalar(const T* in, T* out, size_t n, size_t window) {
    size_t count = rolling_sum_scalar<T>(in, out, n, window);
    T inv_window = T(1) / T(window);
    for (size_t i = 0; i < count; ++i) out[i] *= inv_window;
    return count;
}

}  // namespace regresslens
