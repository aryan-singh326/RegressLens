// Step 2 sanity check. This file has no long-term purpose beyond
// proving the build works and telling you what the current machine
// actually supports before you write AVX2 intrinsics against it.
#include <cstdio>

int main() {
    __builtin_cpu_init();
    bool has_avx2 = __builtin_cpu_supports("avx2");
    bool has_avx512 = __builtin_cpu_supports("avx512f");

    std::printf("Toolchain OK.\n");
    std::printf("AVX2 available:    %s\n", has_avx2 ? "yes" : "no");
    std::printf("AVX-512 available: %s\n", has_avx512 ? "yes" : "no");

    if (!has_avx2) {
        std::printf(
            "WARNING: this machine has no AVX2. v0.1 targets x86-64 "
            "Linux with AVX2 — benchmarking here will not produce "
            "valid characterization data.\n");
    }
    return 0;
}
