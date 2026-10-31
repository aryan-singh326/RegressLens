// Compiled without architecture flags.
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS  // std::getenv
#endif
#include "regresslens/dispatch.hpp"

#include <cstdlib>
#include <stdexcept>

#include "regresslens/cpu_features.hpp"

#ifndef REGRESSLENS_HAVE_AVX2_TU
#define REGRESSLENS_HAVE_AVX2_TU 0
#endif

namespace regresslens {

namespace {

Runtime init_runtime() {
    Runtime rt{};
    rt.avx2_compiled = REGRESSLENS_HAVE_AVX2_TU != 0;
    rt.cpu_avx2 = cpu_has_avx2_fma();
#if REGRESSLENS_HAVE_AVX2_TU
    rt.avx2 = (rt.avx2_compiled && rt.cpu_avx2) ? &avx2_kernel_set() : nullptr;
#else
    rt.avx2 = nullptr;
#endif
    rt.forced = ForcedKernel::None;

    const char* env = std::getenv("RGLNS_FORCE_KERNEL");
    const std::string forced = env ? env : "";
    if (forced.empty() || forced == "auto") {
        rt.forced = ForcedKernel::None;
    } else if (forced == "scalar") {
        rt.forced = ForcedKernel::Scalar;
    } else if (forced == "avx2") {
        rt.forced = ForcedKernel::Avx2;
        if (rt.avx2 == nullptr) {
            rt.error = rt.avx2_compiled
                           ? "RGLNS_FORCE_KERNEL=avx2 but this CPU lacks AVX2/FMA "
                             "(or the OS does not enable YMM state)"
                           : "RGLNS_FORCE_KERNEL=avx2 but this build has no AVX2 "
                             "kernels (non-x86 platform)";
        }
    } else {
        rt.error = "RGLNS_FORCE_KERNEL must be 'scalar' or 'avx2', got '" + forced + "'";
    }
    return rt;
}

}  // namespace

const Runtime& runtime() {
    static const Runtime rt = init_runtime();
    return rt;
}

const KernelSet& kernel_set_for(KernelChoice choice) {
    const Runtime& rt = runtime();
    if (choice == KernelChoice::Scalar || rt.avx2 == nullptr) {
        return scalar_kernel_set();
    }
    return *rt.avx2;
}

const char* kernel_choice_name(KernelChoice k) {
    switch (k) {
        case KernelChoice::Scalar: return "scalar";
        case KernelChoice::Avx2: return "avx2";
        case KernelChoice::MtAvx2: return "mt_avx2";
    }
    return "unknown";
}

KernelChoice resolve_kernel(const SelectionContext& ctx, const std::string& requested,
                            bool op_has_mt) {
    const Runtime& rt = runtime();
    const bool avx2_ok = rt.avx2 != nullptr;

    if (requested == "scalar") return KernelChoice::Scalar;
    if (requested == "avx2" || requested == "mt_avx2") {
        if (!avx2_ok) {
            throw std::invalid_argument("kernel '" + requested +
                                        "' is not available on this CPU/build");
        }
        if (requested == "mt_avx2") {
            if (!op_has_mt) {
                throw std::invalid_argument("this operation has no mt_avx2 kernel");
            }
            return KernelChoice::MtAvx2;
        }
        return KernelChoice::Avx2;
    }
    if (requested != "auto") {
        throw std::invalid_argument("kernel must be 'auto', 'scalar', 'avx2' or "
                                    "'mt_avx2', got '" + requested + "'");
    }

    if (rt.forced == ForcedKernel::Scalar) return KernelChoice::Scalar;
    if (rt.forced == ForcedKernel::Avx2) return KernelChoice::Avx2;

    KernelChoice choice = select_kernel(ctx);
    if (!avx2_ok) return KernelChoice::Scalar;
    if (choice == KernelChoice::MtAvx2 && !op_has_mt) return KernelChoice::Avx2;
    return choice;
}

}  // namespace regresslens
