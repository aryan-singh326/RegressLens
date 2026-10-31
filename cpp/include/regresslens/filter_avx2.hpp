#pragma once
#include <cstdint>

#include "regresslens/avx2_common.hpp"

// Include only from translation units compiled with -mavx2.
//
// DESIGN NOTE: AVX2 has no native "compress" instruction (that
// arrived with AVX-512 VBMI2). So this kernel vectorizes the
// comparison step (the part that causes branch misprediction in a
// naive scalar loop when selectivity is unpredictable), then uses a
// precomputed lookup table to know which lanes passed, avoiding a
// per-element branch during compaction. The compaction copy loop
// itself is not further vectorized.
//
// NaN semantics match NumPy: _CMP_GT_OQ is an ordered comparison, so
// NaN > t is false for every t, and a NaN threshold selects nothing.

namespace regresslens {
namespace avx2_detail {

struct LaneTable4 {
    std::uint8_t idx[16][4];
    std::uint8_t count[16];
};

struct LaneTable8 {
    std::uint8_t idx[256][8];
    std::uint8_t count[256];
};

static inline LaneTable4 build_lane_table4() {
    LaneTable4 t{};
    for (int m = 0; m < 16; ++m) {
        for (int b = 0; b < 4; ++b) {
            if (m & (1 << b)) t.idx[m][t.count[m]++] = static_cast<std::uint8_t>(b);
        }
    }
    return t;
}

static inline LaneTable8 build_lane_table8() {
    LaneTable8 t{};
    for (int m = 0; m < 256; ++m) {
        for (int b = 0; b < 8; ++b) {
            if (m & (1 << b)) t.idx[m][t.count[m]++] = static_cast<std::uint8_t>(b);
        }
    }
    return t;
}

// Function-local statics initialize thread-safely (C++11), which
// matters because the multithreaded filter can hit first use from
// several workers at once.
static inline const LaneTable4& lane_table4() {
    static const LaneTable4 table = build_lane_table4();
    return table;
}

static inline const LaneTable8& lane_table8() {
    static const LaneTable8 table = build_lane_table8();
    return table;
}

}  // namespace avx2_detail

static inline size_t filter_gt_avx2(const double* in, double* out, size_t n,
                                    double threshold) {
    const __m256d vthresh = _mm256_set1_pd(threshold);
    const auto& table = avx2_detail::lane_table4();
    size_t count = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(in + i);
        int mask = _mm256_movemask_pd(_mm256_cmp_pd(v, vthresh, _CMP_GT_OQ));
        const std::uint8_t k_count = table.count[mask];
        for (std::uint8_t k = 0; k < k_count; ++k) {
            out[count++] = in[i + table.idx[mask][k]];
        }
    }
    for (; i < n; ++i) {
        if (in[i] > threshold) out[count++] = in[i];
    }
    return count;
}

static inline size_t filter_gt_avx2(const float* in, float* out, size_t n,
                                    float threshold) {
    const __m256 vthresh = _mm256_set1_ps(threshold);
    const auto& table = avx2_detail::lane_table8();
    size_t count = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(v, vthresh, _CMP_GT_OQ));
        const std::uint8_t k_count = table.count[mask];
        for (std::uint8_t k = 0; k < k_count; ++k) {
            out[count++] = in[i + table.idx[mask][k]];
        }
    }
    for (; i < n; ++i) {
        if (in[i] > threshold) out[count++] = in[i];
    }
    return count;
}

}  // namespace regresslens
