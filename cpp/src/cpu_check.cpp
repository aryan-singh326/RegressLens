// Toolchain sanity check: confirms the build works and reports which
// kernel sets this machine will use at runtime.
#include <cstdio>

#include "regresslens/dispatch.hpp"

int main() {
    const regresslens::Runtime& rt = regresslens::runtime();
    std::printf("Toolchain OK.\n");
    std::printf("AVX2 kernels compiled in: %s\n", rt.avx2_compiled ? "yes" : "no");
    std::printf("CPU supports AVX2+FMA:    %s\n", rt.cpu_avx2 ? "yes" : "no");
    std::printf("Active kernel sets:       scalar%s\n", rt.avx2 ? ", avx2" : "");
    if (!rt.error.empty()) std::printf("Dispatch error: %s\n", rt.error.c_str());
    return rt.error.empty() ? 0 : 1;
}
