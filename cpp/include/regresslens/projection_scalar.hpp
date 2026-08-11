#pragma once
#include "regresslens/column.hpp"

namespace regresslens {

// Scalar projection: out[i] = scale * in[i] + offset. This is the
// representative elementwise kernel for the "projection" operation
// category (diff_log, zscore, and similar ops all reduce to an
// affine or near-affine elementwise transform at the kernel level).
// Portable baseline — no -march=native.
template <typename T>
void project_affine_scalar(const T* in, T* out, size_t n, T scale, T offset) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = scale * in[i] + offset;
    }
}

}  // namespace regresslens
