#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/reduction_mt_avx2.hpp"
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
static void check_matches_scalar(const char* label, size_t n, size_t num_threads,
                                  T rel_tol) {
    std::mt19937 rng(static_cast<unsigned>(n + num_threads * 7));
    std::uniform_real_distribution<T> dist(T(-1000), T(1000));
    std::vector<T> data(n);
    for (auto& v : data) v = dist(rng);

    T scalar_result = regresslens::reduce_sum_scalar<T>(data.data(), n);
    T mt_result = regresslens::reduce_sum_mt_avx2<T>(data.data(), n, num_threads);

    T diff = std::fabs(scalar_result - mt_result);
    T scale = std::max(std::fabs(scalar_result), T(1));
    T rel_diff = diff / scale;

    if (rel_diff > rel_tol) {
        std::fprintf(stderr,
                      "%s MISMATCH: n=%zu threads=%zu scalar=%.9g mt=%.9g "
                      "rel_diff=%.3e\n",
                      label, n, num_threads, (double)scalar_result,
                      (double)mt_result, (double)rel_diff);
        CHECK(false);
    }
    std::printf("%s passed (n=%zu, threads=%zu, rel_diff=%.3e)\n", label, n,
                num_threads, (double)rel_diff);
}

int main() {
    // Sweep thread counts including ones that don't divide n evenly
    // (tests the remainder-distribution logic), more threads than
    // cores available (must still be correct, just not faster), and
    // more threads than elements (degenerate case).
    for (size_t n : {size_t(0), size_t(1), size_t(7), size_t(1000), size_t(100'000)}) {
        for (size_t threads : {size_t(1), size_t(2), size_t(3), size_t(8), size_t(64)}) {
            check_matches_scalar<double>("f64", n, threads, 1e-9);
            check_matches_scalar<float>("f32", n, threads, 1e-3f);
        }
    }
    std::printf("All multithreaded reduction tests passed.\n");
    return 0;
}
