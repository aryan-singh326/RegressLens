#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "regresslens/reduction_mt_avx2.hpp"
#include "regresslens/reduction_scalar.hpp"

#include "test_util.hpp"

template <typename T>
static void check_matches_scalar(const char* label, size_t n, size_t num_threads) {
    std::mt19937 rng(static_cast<unsigned>(n + num_threads * 7));
    std::uniform_real_distribution<T> dist(T(-1000), T(1000));
    std::vector<T> data(n);
    for (auto& v : data) v = dist(rng);

    double scalar_result = regresslens::reduce_sum_scalar<T>(data.data(), n);
    double mt_result = regresslens::reduce_sum_mt_avx2<T>(data.data(), n, num_threads);
    double rel = std::fabs(scalar_result - mt_result) /
                 std::max(std::fabs(scalar_result), 1.0);
    if (rel > 1e-14) {
        std::fprintf(stderr,
                     "%s MISMATCH: n=%zu threads=%zu scalar=%.17g mt=%.17g rel=%.3e\n",
                     label, n, num_threads, scalar_result, mt_result, rel);
        CHECK(false);
    }
    std::printf("%s passed (n=%zu, threads=%zu, rel_diff=%.3e)\n", label, n,
                num_threads, rel);
}

int main() {
    SKIP_UNLESS_AVX2();
    // Thread counts that don't divide n, more threads than cores, more
    // threads than elements, and more than kMaxKernelThreads (clamped).
    for (size_t n : {size_t(0), size_t(1), size_t(7), size_t(1000), size_t(100'000)}) {
        for (size_t threads : {size_t(1), size_t(2), size_t(3), size_t(8), size_t(64),
                               size_t(200)}) {
            check_matches_scalar<double>("f64", n, threads);
            check_matches_scalar<float>("f32", n, threads);
        }
    }
    std::printf("All multithreaded reduction tests passed.\n");
    return 0;
}
