#pragma once
#include <stdexcept>

#include "regresslens/column.hpp"
#include "regresslens/compensated.hpp"

namespace regresslens {

// Scalar compensated sum reduction. Portable baseline: compiled with
// no architecture flags, must run on any CPU. Accumulates in double
// for both float32 and float64 input (float32 values are exactly
// representable in double) and carries a TwoSum error term, so the
// result is within about one ulp of the exact sum.
//
// Four independent accumulators break the add-latency dependency
// chain. Each chain is still summed strictly in order, so this is not
// a reassociation the compiler could have done on its own.
template <typename T>
SumParts reduce_sum_scalar_parts(const T* data, size_t n) {
    CompSum a0, a1, a2, a3;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        a0.add(static_cast<double>(data[i]));
        a1.add(static_cast<double>(data[i + 1]));
        a2.add(static_cast<double>(data[i + 2]));
        a3.add(static_cast<double>(data[i + 3]));
    }
    for (; i < n; ++i) a0.add(static_cast<double>(data[i]));
    a0.merge(a1);
    a2.merge(a3);
    a0.merge(a2);
    return a0.parts();
}

template <typename T>
double reduce_sum_scalar(const T* data, size_t n) {
    return finish_sum(reduce_sum_scalar_parts<T>(data, n));
}

inline double reduce_sum_scalar(const ColumnBuffer& col) {
    if (!col.is_contiguous()) {
        // Kernels only handle contiguous input; the Python layer
        // routes non-contiguous arrays to the NumPy fallback.
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
