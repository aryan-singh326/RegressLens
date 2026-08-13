// Filter needs testing across two dimensions scalar/AVX2 comparisons
// for reduction and projection didn't: size AND selectivity. A bug
// in the lookup-table compaction could easily be sized-dependent
// (e.g. only breaks on the last partial block) or mask-dependent
// (e.g. only breaks for specific popcounts), so both are swept here.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/filter_avx2.hpp"
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
static void check_matches_scalar(const char* label, size_t n, double selectivity) {
    std::mt19937 rng(static_cast<unsigned>(n * 1000 + selectivity * 100));
    // Threshold chosen so P(value > threshold) ~= selectivity, given
    // values uniform in [0, 1).
    T threshold = static_cast<T>(1.0 - selectivity);
    std::uniform_real_distribution<T> dist(T(0), T(1));

    std::vector<T> in(n);
    for (auto& v : in) v = dist(rng);

    std::vector<T> scalar_out(n), avx2_out(n);
    size_t scalar_count =
        regresslens::filter_gt_scalar<T>(in.data(), scalar_out.data(), n, threshold);
    size_t avx2_count =
        regresslens::filter_gt_avx2(in.data(), avx2_out.data(), n, threshold);

    if (scalar_count != avx2_count) {
        std::fprintf(stderr,
                      "%s COUNT MISMATCH: scalar=%zu avx2=%zu (n=%zu, sel=%.2f)\n",
                      label, scalar_count, avx2_count, n, selectivity);
        CHECK(false);
    }
    for (size_t i = 0; i < scalar_count; ++i) {
        if (scalar_out[i] != avx2_out[i]) {
            std::fprintf(stderr,
                          "%s VALUE MISMATCH at i=%zu: scalar=%.9g avx2=%.9g "
                          "(n=%zu, sel=%.2f)\n",
                          label, i, (double)scalar_out[i], (double)avx2_out[i],
                          n, selectivity);
            CHECK(false);
        }
    }
    std::printf("%s passed (n=%zu, selectivity=%.2f, matched=%zu)\n", label, n,
                selectivity, scalar_count);
}

static void test_edge_cases() {
    // Mask 0 (nothing passes) and mask 15/255 (everything passes)
    // are the extremes of the lookup table — make sure they're
    // covered explicitly, not just probabilistically via random data.
    double in_d[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double out_d[8];
    CHECK(regresslens::filter_gt_avx2(in_d, out_d, 8, 100.0) == 0);
    CHECK(regresslens::filter_gt_avx2(in_d, out_d, 8, -100.0) == 8);

    float in_f[16];
    for (int i = 0; i < 16; ++i) in_f[i] = (float)i;
    float out_f[16];
    CHECK(regresslens::filter_gt_avx2(in_f, out_f, 16, 100.0f) == 0);
    CHECK(regresslens::filter_gt_avx2(in_f, out_f, 16, -100.0f) == 16);
    std::printf("test_edge_cases passed\n");
}

int main() {
    test_edge_cases();

    for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(5),
                      size_t(7), size_t(8), size_t(9), size_t(1000),
                      size_t(100'000), size_t(1'000'003)}) {
        for (double sel : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}) {
            check_matches_scalar<double>("f64", n, sel);
            check_matches_scalar<float>("f32", n, sel);
        }
    }

    std::printf("All AVX2 filter tests passed.\n");
    return 0;
}
