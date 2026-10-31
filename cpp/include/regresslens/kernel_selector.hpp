#pragma once
#include <cstddef>

#include "regresslens/column.hpp"

// This header defines the ONE interface the rest of the code depends on
// for kernel selection. Nothing downstream of it should know whether the
// answer comes from the characterization-derived heuristic implemented
// today (src/kernel_selector.cpp) or from a future online adaptive
// selector. Swapping the implementation behind select_kernel() is meant
// to be the only change such a switch requires; needing to touch code
// outside that file would mean the boundary leaked.

namespace regresslens {

enum class Operation {
    Reduction,
    Projection,
    Filter,
    Rolling,     // rolling sum and rolling mean
    RollingVar,  // scalar-only kernel
};

enum class KernelChoice {
    Scalar,
    Avx2,
    MtAvx2,
};

// Everything a selection policy could plausibly need to make a
// decision. Deliberately a plain struct, not a ColumnBuffer, because
// selection happens BEFORE a kernel runs and some fields here
// (selectivity, window) don't exist on ColumnBuffer at all.
struct SelectionContext {
    Operation op;
    DType dtype;
    size_t n;
    bool contiguous;
    unsigned available_threads;

    // Only meaningful for Filter. The Python layer passes the mean
    // selectivity observed at this call site in earlier calls and
    // runs (trace history), or 0.5 when there is none yet.
    double estimated_selectivity = 0.5;

    // Only meaningful for Rolling / RollingVar.
    size_t window = 0;
};

// The interface. One function. Everything else in this header is
// just the vocabulary it uses.
KernelChoice select_kernel(const SelectionContext& ctx);

}  // namespace regresslens
