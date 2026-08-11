// Tests AVX2 against the SCALAR kernel's output, not against a
// separate reference. This isolates "is my AVX2 code correct" from
// "is my correctness-checking logic correct" — that second question
// was already answered in test_correctness_harness.cpp and
// test_reduction_scalar.cpp. If this test fails, the bug is in the
// AVX2 intrinsics, not in the surrounding test infrastructure.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/reduction_avx2.hpp"
#include "regresslens/reduction_scalar.hpp"

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

template <typename T>
static void check_matches_scalar(const char* label, const std::vector<T>& data,
                                  T rel_tol) {
    T scalar_result = regresslens::reduce_sum_scalar<T>(data.data(), data.size());
    T avx2_result = regresslens::reduce_sum_avx2(data.data(), data.size());

    T diff = fabsl((long double)scalar_result - (long double)avx2_result);
    T scale = std::max(fabsl((long double)scalar_result), (long double)1);
    T rel_diff = diff / scale;

    if (rel_diff > rel_tol) {
        std::fprintf(stderr,
                      "%s MISMATCH: scalar=%.17g avx2=%.17g rel_diff=%.3e "
                      "(n=%zu)\n",
                      label, (double)scalar_result, (double)avx2_result,
                      (double)rel_diff, data.size());
    }
    CHECK(rel_diff <= rel_tol);
    std::printf("%s passed (n=%zu, rel_diff=%.3e)\n", label, data.size(),
                (double)rel_diff);
}

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist_d(-1000.0, 1000.0);
    std::uniform_real_distribution<float> dist_f(-1000.0f, 1000.0f);

    // Sizes deliberately include non-multiples of the unroll block
    // (16 for f64, 32 for f32) to exercise the scalar tail path, plus
    // 0 and 1 as edge cases.
    for (size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(17),
                      size_t(1000), size_t(100'000), size_t(1'000'003)}) {
        std::vector<double> data_d(n);
        for (auto& v : data_d) v = dist_d(rng);
        check_matches_scalar<double>("f64", data_d, 1e-9);

        std::vector<float> data_f(n);
        for (auto& v : data_f) v = dist_f(rng);
        // Looser tolerance for float32: AVX2 uses a different
        // summation order (4-way tree-ish accumulation) than the
        // scalar sequential loop, so rounding error accumulates
        // differently, not just less.
        check_matches_scalar<float>("f32", data_f, 1e-3f);
    }

    std::printf("All AVX2 reduction tests passed.\n");
    return 0;
}
