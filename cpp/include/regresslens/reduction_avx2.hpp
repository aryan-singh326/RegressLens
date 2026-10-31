#pragma once
#include "regresslens/avx2_common.hpp"

// AVX2 compensated sum reduction. Include only from translation units
// compiled with -mavx2 -mfma (see avx2_common.hpp for why every
// function here is static inline).
//
// Four 4-lane TwoSum accumulators (16 independent chains) process 16
// doubles per iteration. float32 input is widened to double lanes
// before accumulation, so both dtypes get the same accuracy as the
// scalar kernel (within about one ulp of the exact sum).

namespace regresslens {

static inline SumParts reduce_sum_avx2_parts(const double* data, size_t n) {
    using namespace avx2_detail;
    __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
    __m256d c0 = s0, c1 = s0, c2 = s0, c3 = s0;

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        two_sum_pd(s0, c0, _mm256_loadu_pd(data + i));
        two_sum_pd(s1, c1, _mm256_loadu_pd(data + i + 4));
        two_sum_pd(s2, c2, _mm256_loadu_pd(data + i + 8));
        two_sum_pd(s3, c3, _mm256_loadu_pd(data + i + 12));
    }
    SumParts acc{0.0, 0.0};
    fold_lanes(acc, s0, c0);
    fold_lanes(acc, s1, c1);
    fold_lanes(acc, s2, c2);
    fold_lanes(acc, s3, c3);
    for (; i < n; ++i) two_sum(acc, data[i]);
    return acc;
}

static inline SumParts reduce_sum_avx2_parts(const float* data, size_t n) {
    using namespace avx2_detail;
    __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
    __m256d c0 = s0, c1 = s0, c2 = s0, c3 = s0;

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a = _mm256_loadu_ps(data + i);
        __m256 b = _mm256_loadu_ps(data + i + 8);
        two_sum_pd(s0, c0, _mm256_cvtps_pd(_mm256_castps256_ps128(a)));
        two_sum_pd(s1, c1, _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1)));
        two_sum_pd(s2, c2, _mm256_cvtps_pd(_mm256_castps256_ps128(b)));
        two_sum_pd(s3, c3, _mm256_cvtps_pd(_mm256_extractf128_ps(b, 1)));
    }
    SumParts acc{0.0, 0.0};
    fold_lanes(acc, s0, c0);
    fold_lanes(acc, s1, c1);
    fold_lanes(acc, s2, c2);
    fold_lanes(acc, s3, c3);
    for (; i < n; ++i) two_sum(acc, static_cast<double>(data[i]));
    return acc;
}

static inline double reduce_sum_avx2(const double* data, size_t n) {
    return avx2_detail::finish(reduce_sum_avx2_parts(data, n));
}

static inline double reduce_sum_avx2(const float* data, size_t n) {
    return avx2_detail::finish(reduce_sum_avx2_parts(data, n));
}

}  // namespace regresslens
