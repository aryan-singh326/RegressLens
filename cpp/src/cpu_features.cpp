// Runtime CPU feature detection. Compiled without architecture flags.
#include "regresslens/cpu_features.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#include <intrin.h>
#endif

namespace regresslens {

bool cpu_has_avx2_fma() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    // MSVC has no __builtin_cpu_supports; query CPUID directly.
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuid(regs, 1);
    const bool fma = (regs[2] & (1 << 12)) != 0;
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!(fma && osxsave && avx)) return false;
    // The OS must save XMM and YMM state on context switch.
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;  // AVX2
#elif defined(__x86_64__) || defined(__i386__)
    // libgcc/compiler-rt check the XGETBV OS-support bits before
    // reporting AVX-family features.
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace regresslens
