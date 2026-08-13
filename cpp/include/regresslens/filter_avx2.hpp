#pragma once
#include <cstdint>
#include <immintrin.h>

#include "regresslens/column.hpp"

// Must only be included from translation units compiled with -mavx2.
//
// DESIGN NOTE: AVX2 has no native "compress" instruction (that
// arrived with AVX-512 VBMI2, which is out of scope for v0.1's
// baseline). So this kernel vectorizes the comparison step (the part
// that causes branch misprediction in a naive scalar loop when
// selectivity is unpredictable), then uses a precomputed lookup
// table to know which lanes passed, avoiding a per-element branch
// during compaction. The compaction copy loop itself is not further
// vectorized. This is a known, intentional scope limit for v0.1 —
// not an oversight — and should be revisited if characterization
// data shows filter's AVX2 speedup is disappointing relative to
// projection/reduction.

namespace regresslens {
namespace detail {

struct LaneIndices4 {
    uint8_t idx[4];
    uint8_t count;
};

inline LaneIndices4 make_lane_indices4(int mask) {
    LaneIndices4 r{{0, 0, 0, 0}, 0};
    for (int b = 0; b < 4; ++b) {
        if (mask & (1 << b)) r.idx[r.count++] = static_cast<uint8_t>(b);
    }
    return r;
}

inline const LaneIndices4* lane_table4() {
    static LaneIndices4 table[16];
    static bool initialized = false;
    if (!initialized) {
        for (int m = 0; m < 16; ++m) table[m] = make_lane_indices4(m);
        initialized = true;
    }
    return table;
}

struct LaneIndices8 {
    uint8_t idx[8];
    uint8_t count;
};

inline LaneIndices8 make_lane_indices8(int mask) {
    LaneIndices8 r{{0, 0, 0, 0, 0, 0, 0, 0}, 0};
    for (int b = 0; b < 8; ++b) {
        if (mask & (1 << b)) r.idx[r.count++] = static_cast<uint8_t>(b);
    }
    return r;
}

inline const LaneIndices8* lane_table8() {
    static LaneIndices8 table[256];
    static bool initialized = false;
    if (!initialized) {
        for (int m = 0; m < 256; ++m) table[m] = make_lane_indices8(m);
        initialized = true;
    }
    return table;
}

}  // namespace detail

inline size_t filter_gt_avx2(const double* in, double* out, size_t n,
                              double threshold) {
    __m256d vthresh = _mm256_set1_pd(threshold);
    const auto* table = detail::lane_table4();
    size_t count = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(in + i);
        __m256d cmp = _mm256_cmp_pd(v, vthresh, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(cmp);
        double lanes[4];
        _mm256_storeu_pd(lanes, v);
        const auto& li = table[mask];
        for (uint8_t k = 0; k < li.count; ++k) {
            out[count++] = lanes[li.idx[k]];
        }
    }
    for (; i < n; ++i) {
        if (in[i] > threshold) out[count++] = in[i];
    }
    return count;
}

inline size_t filter_gt_avx2(const float* in, float* out, size_t n,
                              float threshold) {
    __m256 vthresh = _mm256_set1_ps(threshold);
    const auto* table = detail::lane_table8();
    size_t count = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        __m256 cmp = _mm256_cmp_ps(v, vthresh, _CMP_GT_OQ);
        int mask = _mm256_movemask_ps(cmp);
        float lanes[8];
        _mm256_storeu_ps(lanes, v);
        const auto& li = table[mask];
        for (uint8_t k = 0; k < li.count; ++k) {
            out[count++] = lanes[li.idx[k]];
        }
    }
    for (; i < n; ++i) {
        if (in[i] > threshold) out[count++] = in[i];
    }
    return count;
}

}  // namespace regresslens
