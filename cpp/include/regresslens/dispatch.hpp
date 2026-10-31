#pragma once
#include <string>

#include "regresslens/kernel_selector.hpp"
#include "regresslens/kernels.hpp"

// Runtime kernel-set selection, done once at module init.
//
// RGLNS_FORCE_KERNEL=scalar|avx2 overrides the selector for every
// operation (used by the test suite to exercise each kernel set on
// the same machine). Forcing avx2 on a CPU without AVX2+FMA, or
// setting an unknown value, is reported through runtime().error so
// the extension fails loudly at import instead of crashing later.

namespace regresslens {

enum class ForcedKernel { None, Scalar, Avx2 };

struct Runtime {
    bool avx2_compiled;     // AVX2 translation unit built into this binary
    bool cpu_avx2;          // CPU + OS support AVX2 and FMA
    const KernelSet* avx2;  // non-null only when both of the above hold
    ForcedKernel forced;
    std::string error;      // non-empty: refuse to load
};

const Runtime& runtime();

// Kernel set for a choice; Scalar choices always map to the scalar set.
const KernelSet& kernel_set_for(KernelChoice choice);

// Resolves the kernel for one call. `requested` is "auto" (selector,
// subject to RGLNS_FORCE_KERNEL) or an explicit "scalar" | "avx2" |
// "mt_avx2". Throws std::invalid_argument for unknown or unavailable
// explicit requests. The result is always runnable on this CPU.
KernelChoice resolve_kernel(const SelectionContext& ctx, const std::string& requested,
                            bool op_has_mt);

const char* kernel_choice_name(KernelChoice k);

}  // namespace regresslens
