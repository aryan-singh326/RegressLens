#pragma once
#include <immintrin.h>

#include "regresslens/column.hpp"

// Must only be included from translation units compiled with
// -mavx2 -mfma. Not included from portable/scalar-only files.

namespace regresslens {

// AVX2+FMA projection: out[i] = scale * in[i] + offset.
// 4-wide for float64, 8-wide for float32, unrolled 4x per the
// project's kernel spec (matches the pattern used in reduction_avx2).
inline void project_affine_avx2(const double* in, double* out, size_t n,
                                 double scale, double offset) {
    __m256d vscale = _mm256_set1_pd(scale);
    __m256d voffset = _mm256_set1_pd(offset);

    size_t i = 0;
    const size_t block = 16;  // 4 lanes * 4-way unroll
    for (; i + block <= n; i += block) {
        __m256d x0 = _mm256_loadu_pd(in + i);
        __m256d x1 = _mm256_loadu_pd(in + i + 4);
        __m256d x2 = _mm256_loadu_pd(in + i + 8);
        __m256d x3 = _mm256_loadu_pd(in + i + 12);
        _mm256_storeu_pd(out + i, _mm256_fmadd_pd(x0, vscale, voffset));
        _mm256_storeu_pd(out + i + 4, _mm256_fmadd_pd(x1, vscale, voffset));
        _mm256_storeu_pd(out + i + 8, _mm256_fmadd_pd(x2, vscale, voffset));
        _mm256_storeu_pd(out + i + 12, _mm256_fmadd_pd(x3, vscale, voffset));
    }
    // Scalar tail for the remainder.
    for (; i < n; ++i) {
        out[i] = scale * in[i] + offset;
    }
}

inline void project_affine_avx2(const float* in, float* out, size_t n,
                                 float scale, float offset) {
    __m256 vscale = _mm256_set1_ps(scale);
    __m256 voffset = _mm256_set1_ps(offset);

    size_t i = 0;
    const size_t block = 32;  // 8 lanes * 4-way unroll
    for (; i + block <= n; i += block) {
        __m256 x0 = _mm256_loadu_ps(in + i);
        __m256 x1 = _mm256_loadu_ps(in + i + 8);
        __m256 x2 = _mm256_loadu_ps(in + i + 16);
        __m256 x3 = _mm256_loadu_ps(in + i + 24);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(x0, vscale, voffset));
        _mm256_storeu_ps(out + i + 8, _mm256_fmadd_ps(x1, vscale, voffset));
        _mm256_storeu_ps(out + i + 16, _mm256_fmadd_ps(x2, vscale, voffset));
        _mm256_storeu_ps(out + i + 24, _mm256_fmadd_ps(x3, vscale, voffset));
    }
    for (; i < n; ++i) {
        out[i] = scale * in[i] + offset;
    }
}

}  // namespace regresslens
