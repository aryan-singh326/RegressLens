#pragma once

namespace regresslens {

// True when the CPU supports AVX2 and FMA AND the operating system
// saves the YMM register state (XGETBV), i.e. when the AVX2 kernel
// set is safe to call. Always false on non-x86 builds.
bool cpu_has_avx2_fma();

}  // namespace regresslens
