// Compares AVX2 (prefix-sum + vectorized diff) against scalar
// (sliding accumulator). These are two DIFFERENT algorithms with
// different rounding behavior, not just different instruction sets —
// so this test is also implicitly checking whether the prefix-sum
// approach's cancellation error (subtracting two large nearly-equal
// prefix sums to get a small window sum) is acceptable at scale.
// If float32 fails here, that's real information about a precision
// limit of this AVX2 design, not a test bug.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/rolling_avx2.hpp"
#include "regresslens/rolling_scalar.hpp"

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// Ground truth: sums each window from scratch in double precision.
// This is what AVX2 gets held to a tight tolerance against. Scalar
// is NOT used as ground truth here — test_rolling_scalar.cpp's
// test_f32_drift_at_scale already showed the sliding accumulator
// itself drifts by an amount that grows with both N and window size,
// so comparing AVX2 against scalar with a fixed tolerance was the
// wrong test design (an earlier version of this file did that and
// kept failing as window size increased, because it was really
// measuring scalar's drift, not AVX2's correctness).
template <typename T>
static std::vector<double> naive_double_reference(const T* in, size_t n, size_t window) {
    std::vector<double> out(n - window + 1);
    for (size_t i = 0; i < out.size(); ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < window; ++j) acc += (double)in[i + j];
        out[i] = acc;
    }
    return out;
}

template <typename T>
static void check_against_reference(const char* label, size_t n, size_t window,
                                     T rel_tol) {
    std::mt19937 rng(static_cast<unsigned>(n * 100 + window));
    std::uniform_real_distribution<T> dist(T(-100), T(100));
    std::vector<T> in(n);
    for (auto& v : in) v = dist(rng);

    std::vector<T> scalar_out(n), avx2_out(n);
    size_t scalar_count =
        regresslens::rolling_sum_scalar<T>(in.data(), scalar_out.data(), n, window);
    size_t avx2_count = regresslens::rolling_sum_avx2(in.data(), avx2_out.data(), n, window);
    CHECK(scalar_count == avx2_count);

    auto reference = naive_double_reference<T>(in.data(), n, window);

    double avx2_max_rel_diff = 0, scalar_max_rel_diff = 0;
    for (size_t i = 0; i < avx2_count; ++i) {
        double ascale = std::max(std::fabs(reference[i]), 1.0);
        avx2_max_rel_diff = std::max(avx2_max_rel_diff,
                                      std::fabs((double)avx2_out[i] - reference[i]) / ascale);
        scalar_max_rel_diff = std::max(
            scalar_max_rel_diff, std::fabs((double)scalar_out[i] - reference[i]) / ascale);
    }

    // AVX2 is the real correctness gate: tight tolerance against
    // ground truth.
    if (avx2_max_rel_diff > rel_tol) {
        std::fprintf(stderr,
                      "%s AVX2 EXCEEDS TOLERANCE vs ground truth: n=%zu "
                      "window=%zu max_rel_diff=%.3e (tol=%.3e)\n",
                      label, n, window, avx2_max_rel_diff, (double)rel_tol);
        CHECK(false);
    }
    // Scalar's drift is informational, not gated — it's documented,
    // expected behavior (see rolling_scalar.hpp), not a bug this
    // test should fail on. Printed so the pattern stays visible.
    std::printf("%s passed (n=%zu, window=%zu, avx2_rel_diff=%.3e, "
                "scalar_rel_diff=%.3e [informational])\n",
                label, n, window, avx2_max_rel_diff, scalar_max_rel_diff);
}

static void test_edge_cases() {
    double in[6] = {1, 2, 3, 4, 5, 6};
    double out[6];
    CHECK(regresslens::rolling_sum_avx2(in, out, 6, 3) == 4);
    CHECK(out[0] == 6.0);
    CHECK(out[3] == 15.0);

    // window > n
    CHECK(regresslens::rolling_sum_avx2(in, out, 6, 10) == 0);
    // window == n
    CHECK(regresslens::rolling_sum_avx2(in, out, 6, 6) == 1);
    CHECK(out[0] == 21.0);
    std::printf("test_edge_cases passed\n");
}

int main() {
    test_edge_cases();

    for (size_t n : {size_t(20), size_t(1000), size_t(100'000), size_t(1'000'003)}) {
        for (size_t window : {size_t(1), size_t(5), size_t(20), size_t(500)}) {
            if (window > n) continue;
            // Both tolerances are now tight because both are checked
            // against ground truth, not against each other. AVX2's
            // double-precision-internal design earns the tight f32
            // tolerance here.
            check_against_reference<double>("f64", n, window, 1e-9);
            check_against_reference<float>("f32", n, window, 1e-4f);
        }
    }

    std::printf("All AVX2 rolling tests passed.\n");
    return 0;
}
