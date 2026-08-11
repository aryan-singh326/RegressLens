#pragma once
#include "regresslens/column.hpp"

namespace regresslens {

// Scalar sum reduction. This is the baseline kernel: always correct,
// compiled without -march=native, must work on any x86-64 machine
// regardless of what SIMD extensions it has. This is also the
// reference implementation the AVX2 kernel gets checked against.
template <typename T>
T reduce_sum_scalar(const T* data, size_t n) {
    T acc = T(0);
    for (size_t i = 0; i < n; ++i) {
        acc += data[i];
    }
    return acc;
}

inline double reduce_sum_scalar(const ColumnBuffer& col) {
    if (!col.is_contiguous()) {
        // v0.1 kernels only handle contiguous input; the Python
        // integration layer (Phase 3) is responsible for routing
        // non-contiguous arrays to the contiguity-loss diagnosis
        // path instead of calling this directly.
        throw std::invalid_argument(
            "reduce_sum_scalar: non-contiguous ColumnBuffer");
    }
    switch (col.dtype) {
        case DType::Float32:
            return reduce_sum_scalar<float>(col.as_f32(), col.size);
        case DType::Float64:
            return reduce_sum_scalar<double>(col.as_f64(), col.size);
    }
    throw std::logic_error("unreachable: unknown DType");
}

}  // namespace regresslens
