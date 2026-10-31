#pragma once
#include "regresslens/avx2_common.hpp"

// AVX2+FMA projection: out[i] = scale * in[i] + offset. Include only
// from translation units compiled with -mavx2 -mfma.
//
// Every element, including the scalar tail, is computed with a single
// fused multiply-add, so each output is correctly rounded (at least
// as accurate as NumPy's separate multiply and add). The tail uses an
// explicit FMA intrinsic rather than relying on the compiler's
// floating-point contraction setting.

namespace regresslens {

namespace avx2_detail {
static inline double fma_sd(double a, double b, double c) {
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
}
static inline float fma_ss(float a, float b, float c) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
}  // namespace avx2_detail

static inline void project_affine_avx2(const double* in, double* out, size_t n,
                                       double scale, double offset) {
    const __m256d vscale = _mm256_set1_pd(scale);
    const __m256d voffset = _mm256_set1_pd(offset);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {  // 4 lanes x 4-way unroll
        __m256d x0 = _mm256_loadu_pd(in + i);
        __m256d x1 = _mm256_loadu_pd(in + i + 4);
        __m256d x2 = _mm256_loadu_pd(in + i + 8);
        __m256d x3 = _mm256_loadu_pd(in + i + 12);
        _mm256_storeu_pd(out + i, _mm256_fmadd_pd(x0, vscale, voffset));
        _mm256_storeu_pd(out + i + 4, _mm256_fmadd_pd(x1, vscale, voffset));
        _mm256_storeu_pd(out + i + 8, _mm256_fmadd_pd(x2, vscale, voffset));
        _mm256_storeu_pd(out + i + 12, _mm256_fmadd_pd(x3, vscale, voffset));
    }
    for (; i < n; ++i) out[i] = avx2_detail::fma_sd(in[i], scale, offset);
}

static inline void project_affine_avx2(const float* in, float* out, size_t n,
                                       float scale, float offset) {
    const __m256 vscale = _mm256_set1_ps(scale);
    const __m256 voffset = _mm256_set1_ps(offset);

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {  // 8 lanes x 4-way unroll
        __m256 x0 = _mm256_loadu_ps(in + i);
        __m256 x1 = _mm256_loadu_ps(in + i + 8);
        __m256 x2 = _mm256_loadu_ps(in + i + 16);
        __m256 x3 = _mm256_loadu_ps(in + i + 24);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(x0, vscale, voffset));
        _mm256_storeu_ps(out + i + 8, _mm256_fmadd_ps(x1, vscale, voffset));
        _mm256_storeu_ps(out + i + 16, _mm256_fmadd_ps(x2, vscale, voffset));
        _mm256_storeu_ps(out + i + 24, _mm256_fmadd_ps(x3, vscale, voffset));
    }
    for (; i < n; ++i) out[i] = avx2_detail::fma_ss(in[i], scale, offset);
}

}  // namespace regresslens
