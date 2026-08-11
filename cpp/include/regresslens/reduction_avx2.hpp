#pragma once
#include <immintrin.h>

#include "regresslens/column.hpp"

// This header must ONLY be included from translation units compiled
// with -mavx2 (see bench/CMakeLists.txt and src targets). Do not
// include it from files that are meant to build on non-AVX2 hosts —
// that's what reduction_scalar.hpp is for.

namespace regresslens {

// AVX2 float64 sum reduction. 4-wide (256-bit / 8 bytes = 4 doubles),
// unrolled 4x per the project brief's kernel spec (so 16 elements
// processed per loop iteration), with a scalar tail for the
// remainder that doesn't fill a full unrolled block.
inline double reduce_sum_avx2(const double* data, size_t n) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    size_t i = 0;
    const size_t block = 16;  // 4 lanes * 4-way unroll
    for (; i + block <= n; i += block) {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(data + i));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(data + i + 4));
        acc2 = _mm256_add_pd(acc2, _mm256_loadu_pd(data + i + 8));
        acc3 = _mm256_add_pd(acc3, _mm256_loadu_pd(data + i + 12));
    }

    __m256d acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1),
                                 _mm256_add_pd(acc2, acc3));

    // Horizontal sum of the 4 lanes.
    double lanes[4];
    _mm256_storeu_pd(lanes, acc);
    double total = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    // Scalar tail for the remainder (n % 16 elements).
    for (; i < n; ++i) {
        total += data[i];
    }
    return total;
}

inline float reduce_sum_avx2(const float* data, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    const size_t block = 32;  // 8 lanes * 4-way unroll
    for (; i + block <= n; i += block) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(data + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(data + i + 24));
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                _mm256_add_ps(acc2, acc3));

    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    float total = 0.0f;
    for (float v : lanes) total += v;

    for (; i < n; ++i) {
        total += data[i];
    }
    return total;
}

}  // namespace regresslens
