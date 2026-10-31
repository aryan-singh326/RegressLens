#pragma once
#include <immintrin.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "regresslens/kernels.hpp"

// Helpers shared by the AVX2 kernel headers. Include ONLY from
// translation units compiled with -mavx2 -mfma (or /arch:AVX2).
//
// Every function here (and in the other *_avx2.hpp headers) is
// `static inline`: internal linkage, so the linker can never fold an
// AVX2-compiled copy into the scalar translation unit. For the same
// reason the AVX2 headers avoid std containers and algorithms. Their
// out-of-line instantiations would be shared COMDATs (an AVX2-compiled
// std::vector<double>, say, could replace the baseline one the
// bindings use). The memory these kernels need comes from RawBuffer.

namespace regresslens {
namespace avx2_detail {

static inline bool is_finite_bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return ((u >> 52) & 0x7ffu) != 0x7ffu;
}

// Scalar TwoSum on a SumParts accumulator (see compensated.hpp).
static inline void two_sum(SumParts& a, double x) {
    double t = a.s + x;
    double bp = t - a.s;
    a.c += (a.s - (t - bp)) + (x - bp);
    a.s = t;
}

static inline void merge_parts(SumParts& a, const SumParts& b) {
    two_sum(a, b.s);
    a.c += b.c;
}

static inline double finish(const SumParts& p) {
    return is_finite_bits(p.s) ? p.s + p.c : p.s;
}

// Four-lane TwoSum: s + c tracks the exact running sum per lane.
static inline void two_sum_pd(__m256d& s, __m256d& c, __m256d x) {
    __m256d t = _mm256_add_pd(s, x);
    __m256d bp = _mm256_sub_pd(t, s);
    __m256d e = _mm256_add_pd(_mm256_sub_pd(s, _mm256_sub_pd(t, bp)),
                              _mm256_sub_pd(x, bp));
    c = _mm256_add_pd(c, e);
    s = t;
}

// Folds four (s, c) lane accumulators into one SumParts, in a fixed
// lane order so results are deterministic.
static inline void fold_lanes(SumParts& acc, __m256d s, __m256d c) {
    alignas(32) double ss[4], cc[4];
    _mm256_store_pd(ss, s);
    _mm256_store_pd(cc, c);
    for (int k = 0; k < 4; ++k) {
        two_sum(acc, ss[k]);
        acc.c += cc[k];
    }
}

static inline __m256d abs_pd(__m256d v) {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
}

// Minimal owning buffer (malloc/free are C functions, never inline
// COMDATs). Move-only by construction: copying is deleted.
template <typename T>
struct RawBuffer {
    T* data = nullptr;
    explicit RawBuffer(std::size_t count)
        : data(static_cast<T*>(std::malloc((count ? count : 1) * sizeof(T)))) {}
    ~RawBuffer() { std::free(data); }
    RawBuffer(const RawBuffer&) = delete;
    RawBuffer& operator=(const RawBuffer&) = delete;
    bool ok() const { return data != nullptr; }
};

}  // namespace avx2_detail
}  // namespace regresslens
