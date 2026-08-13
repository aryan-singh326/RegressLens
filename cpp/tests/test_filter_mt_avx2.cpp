// Order preservation is the thing most likely to break in a
// multithreaded filter, so this checks element-by-element equality
// against scalar output, not just counts or set membership.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/filter_mt_avx2.hpp"
#include "regresslens/filter_scalar.hpp"

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
                                  double selectivity) {
    std::mt19937 rng(static_cast<unsigned>(n * 31 + num_threads * 7 + selectivity * 1000));
    T threshold = static_cast<T>(1.0 - selectivity);
    std::uniform_real_distribution<T> dist(T(0), T(1));

    std::vector<T> in(n);
    for (auto& v : in) v = dist(rng);

    std::vector<T> scalar_out(n), mt_out(n);
    size_t scalar_count =
        regresslens::filter_gt_scalar<T>(in.data(), scalar_out.data(), n, threshold);
    size_t mt_count = regresslens::filter_gt_mt_avx2<T>(in.data(), mt_out.data(), n,
                                                          threshold, num_threads);

    if (scalar_count != mt_count) {
        std::fprintf(stderr,
                      "%s COUNT MISMATCH: n=%zu threads=%zu sel=%.2f scalar=%zu mt=%zu\n",
                      label, n, num_threads, selectivity, scalar_count, mt_count);
        CHECK(false);
    }
    for (size_t i = 0; i < scalar_count; ++i) {
        if (scalar_out[i] != mt_out[i]) {
            std::fprintf(stderr,
                          "%s ORDER/VALUE MISMATCH at i=%zu: n=%zu threads=%zu "
                          "scalar=%.9g mt=%.9g\n",
                          label, i, n, num_threads, (double)scalar_out[i],
                          (double)mt_out[i]);
            CHECK(false);
        }
    }
    std::printf("%s passed (n=%zu, threads=%zu, sel=%.2f, matched=%zu)\n", label, n,
                num_threads, selectivity, scalar_count);
}

int main() {
    for (size_t n : {size_t(0), size_t(1), size_t(50), size_t(1000), size_t(100'000)}) {
        for (size_t threads : {size_t(1), size_t(2), size_t(3), size_t(8), size_t(64)}) {
            for (double sel : {0.0, 0.3, 0.5, 0.7, 1.0}) {
                check_matches_scalar<double>("f64", n, threads, sel);
                check_matches_scalar<float>("f32", n, threads, sel);
            }
        }
    }
    std::printf("All multithreaded filter tests passed.\n");
    return 0;
}
