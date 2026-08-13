#pragma once
#include "regresslens/column.hpp"

namespace regresslens {

// Threshold filter: writes elements of `in` that are strictly
// greater than `threshold` into `out`, in order, and returns how
// many were written. Unlike reduction (scalar output) and projection
// (fixed-size output), filter's output size is DATA-DEPENDENT — the
// caller must allocate `out` with capacity >= n and use the
// returned count to know how much of it is valid.
//
// This data-dependence is also why filter's performance depends on
// selectivity (fraction of elements passing), not just N — see the
// characterization table category for this operation.
template <typename T>
size_t filter_gt_scalar(const T* in, T* out, size_t n, T threshold) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (in[i] > threshold) {
            out[count++] = in[i];
        }
    }
    return count;
}

}  // namespace regresslens
