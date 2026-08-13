#pragma once
#include <cstddef>

#include "regresslens/column.hpp"

// This header defines the ONE interface Phase 3+ code is allowed to
// depend on for kernel selection. Nothing downstream of this should
// know or care whether the answer comes from a hand-written
// heuristic (what's implemented today) or an online adaptive
// selector (Phase 2's eventual output, deliberately deferred).
//
// Swapping the implementation behind KernelSelector is meant to be
// the ONLY change required when Phase 2's research concludes. If a
// future change requires touching code outside this file to adopt a
// new selection policy, that's a sign the boundary leaked somewhere
// and should be fixed, not worked around.

namespace regresslens {

enum class Operation {
    Reduction,
    Projection,
    Filter,
    Rolling,
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

    // Only meaningful for Filter. Estimated or historical
    // selectivity — v0.1's placeholder heuristic uses a fixed
    // assumption when this isn't known in advance (see
    // kernel_selector.cpp); a smarter policy could use runtime
    // history here instead.
    double estimated_selectivity = 0.5;

    // Only meaningful for Rolling.
    size_t window = 0;
};

// The interface. One function. Everything else in this header is
// just the vocabulary it uses.
KernelChoice select_kernel(const SelectionContext& ctx);

}  // namespace regresslens
