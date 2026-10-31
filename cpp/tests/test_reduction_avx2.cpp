// Tests the AVX2 compensated sum against the scalar compensated sum.
// Both are accurate to about one ulp of the exact sum, so they must
// agree to near machine precision (not the 1e-3 the old uncompensated
// float32 kernels needed).
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "regresslens/reduction_avx2.hpp"
#include "regresslens/reduction_scalar.hpp"

#include "test_util.hpp"

template <typename T>
static void check_matches_scalar(const char* label, const std::vector<T>& data) {
    double scalar_result = regresslens::reduce_sum_scalar<T>(data.data(), data.size());
    double avx2_result = regresslens::reduce_sum_avx2(data.data(), data.size());
    double rel = std::fabs(scalar_result - avx2_result) /
                 std::max(std::fabs(scalar_result), 1.0);
    if (rel > 1e-14) {
        std::fprintf(stderr, "%s MISMATCH: scalar=%.17g avx2=%.17g rel=%.3e n=%zu\n",
                     label, scalar_result, avx2_result, rel, data.size());
    }
    CHECK(rel <= 1e-14);
    std::printf("%s passed (n=%zu, rel_diff=%.3e)\n", label, data.size(), rel);
}

static void test_cancellation_in_every_lane() {
    // 1e16 + 1 - 1e16 placed at every offset modulo the 16-element
    // block, so each lane and the scalar tail see the cancellation.
    for (size_t pad = 0; pad < 40; ++pad) {
        std::vector<double> d(pad, 0.0);
        d.push_back(1e16);
        d.insert(d.end(), 15, 0.0);
        d.push_back(1.0);
        d.insert(d.end(), 15, 0.0);
        d.push_back(-1e16);
        CHECK(regresslens::reduce_sum_avx2(d.data(), d.size()) == 1.0);
    }
    std::printf("test_cancellation_in_every_lane passed\n");
}

static void test_nonfinite_semantics() {
    const double inf = std::numeric_limits<double>::infinity();
    for (size_t pos : {size_t(0), size_t(5), size_t(17), size_t(40)}) {
        std::vector<double> d(41, 1.0);
        d[pos] = inf;
        CHECK(regresslens::reduce_sum_avx2(d.data(), d.size()) == inf);
        d[(pos + 3) % 41] = -inf;
        CHECK(std::isnan(regresslens::reduce_sum_avx2(d.data(), d.size())));
        std::vector<float> f(41, 1.0f);
        f[pos] = std::numeric_limits<float>::quiet_NaN();
        CHECK(std::isnan(regresslens::reduce_sum_avx2(f.data(), f.size())));
    }
    std::printf("test_nonfinite_semantics passed\n");
}

int main() {
    SKIP_UNLESS_AVX2();
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist_d(-1000.0, 1000.0);
    std::uniform_real_distribution<float> dist_f(-1000.0f, 1000.0f);

    // Sizes include non-multiples of the 16-element block to exercise
    // the scalar tail, plus 0 and 1.
    for (size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(17),
                     size_t(1000), size_t(100'000), size_t(1'000'003)}) {
        std::vector<double> data_d(n);
        for (auto& v : data_d) v = dist_d(rng);
        check_matches_scalar<double>("f64", data_d);

        std::vector<float> data_f(n);
        for (auto& v : data_f) v = dist_f(rng);
        check_matches_scalar<float>("f32", data_f);
    }
    test_cancellation_in_every_lane();
    test_nonfinite_semantics();
    std::printf("All AVX2 reduction tests passed.\n");
    return 0;
}
