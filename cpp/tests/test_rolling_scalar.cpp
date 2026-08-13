#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/rolling_scalar.hpp"

using regresslens::rolling_mean_scalar;
using regresslens::rolling_sum_scalar;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// Deliberately O(n*window) and deliberately NOT the incremental
// sliding-window technique — this is the ground truth the real
// kernel gets checked against, so it must not share any code path
// with rolling_sum_scalar.
template <typename T>
static std::vector<T> naive_rolling_sum(const T* in, size_t n, size_t window) {
    if (window == 0 || n < window) return {};
    std::vector<T> out(n - window + 1);
    for (size_t i = 0; i < out.size(); ++i) {
        T acc = T(0);
        for (size_t j = 0; j < window; ++j) acc += in[i + j];
        out[i] = acc;
    }
    return out;
}

static void test_known_values() {
    double in[6] = {1, 2, 3, 4, 5, 6};
    double out[6];
    size_t count = rolling_sum_scalar<double>(in, out, 6, 3);
    CHECK(count == 4);
    CHECK(out[0] == 6.0);   // 1+2+3
    CHECK(out[1] == 9.0);   // 2+3+4
    CHECK(out[2] == 12.0);  // 3+4+5
    CHECK(out[3] == 15.0);  // 4+5+6
    std::printf("test_known_values passed\n");
}

static void test_rolling_mean() {
    double in[4] = {2, 4, 6, 8};
    double out[4];
    size_t count = rolling_mean_scalar<double>(in, out, 4, 2);
    CHECK(count == 3);
    CHECK(out[0] == 3.0);  // (2+4)/2
    CHECK(out[1] == 5.0);  // (4+6)/2
    CHECK(out[2] == 7.0);  // (6+8)/2
    std::printf("test_rolling_mean passed\n");
}

static void test_window_equals_n() {
    double in[3] = {1, 2, 3};
    double out[3];
    size_t count = rolling_sum_scalar<double>(in, out, 3, 3);
    CHECK(count == 1);
    CHECK(out[0] == 6.0);
    std::printf("test_window_equals_n passed\n");
}

static void test_window_larger_than_n() {
    double in[3] = {1, 2, 3};
    double out[3];
    size_t count = rolling_sum_scalar<double>(in, out, 3, 5);
    CHECK(count == 0);
    std::printf("test_window_larger_than_n passed\n");
}

static void test_window_one_is_identity() {
    double in[4] = {1.5, -2.5, 3.5, 0.0};
    double out[4];
    size_t count = rolling_sum_scalar<double>(in, out, 4, 1);
    CHECK(count == 4);
    for (int i = 0; i < 4; ++i) CHECK(out[i] == in[i]);
    std::printf("test_window_one_is_identity passed\n");
}

static void test_against_naive_reference_random() {
    std::mt19937 rng(21);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (size_t n : {size_t(10), size_t(100), size_t(10'000)}) {
        for (size_t window : {size_t(1), size_t(2), size_t(5), size_t(50)}) {
            if (window > n) continue;
            std::vector<double> in(n);
            for (auto& v : in) v = dist(rng);

            std::vector<double> out(n);
            size_t count = rolling_sum_scalar<double>(in.data(), out.data(), n, window);
            auto reference = naive_rolling_sum<double>(in.data(), n, window);

            CHECK(count == reference.size());
            for (size_t i = 0; i < count; ++i) {
                double diff = std::fabs(out[i] - reference[i]);
                double scale = std::max(std::fabs(reference[i]), 1.0);
                // Sliding accumulator vs from-scratch summation can
                // diverge slightly due to different rounding paths —
                // this is real, documented behavior (see header
                // comment), not a bug. Tolerance reflects that.
                if ((diff / scale) > 1e-9) {
                    std::fprintf(stderr,
                                  "MISMATCH n=%zu window=%zu i=%zu got=%.17g "
                                  "want=%.17g\n",
                                  n, window, i, out[i], reference[i]);
                    CHECK(false);
                }
            }
        }
    }
    std::printf("test_against_naive_reference_random passed\n");
}

// Confirms the double-accumulator fix actually worked. This was
// informational-only before the fix (scalar drift was expected and
// undocumented as a hard bound); now that scalar accumulates in
// double internally for float32, this is a real correctness gate.
static void test_f32_drift_at_scale() {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    size_t n = 1'000'003;
    size_t window = 500;  // the worst case that measured 5.6% before the fix
    std::vector<float> in(n);
    for (auto& v : in) v = dist(rng);

    std::vector<float> out(n);
    size_t count = rolling_sum_scalar<float>(in.data(), out.data(), n, window);

    double max_rel_diff = 0;
    for (size_t i = 0; i < count; ++i) {
        double ref = 0.0;
        for (size_t j = 0; j < window; ++j) ref += (double)in[i + j];
        double diff = std::fabs((double)out[i] - ref);
        double scale = std::max(std::fabs(ref), 1.0);
        max_rel_diff = std::max(max_rel_diff, diff / scale);
    }
    std::printf(
        "test_f32_drift_at_scale: n=%zu window=%zu max_rel_diff_vs_double_ref=%.3e\n",
        n, window, max_rel_diff);
    // Before the double-accumulator fix, this reached 5.6e-2 at
    // this exact size/window. Post-fix it should be near float32
    // machine epsilon, not just "better."
    CHECK(max_rel_diff < 1e-4);
}

int main() {
    test_known_values();
    test_f32_drift_at_scale();
    test_rolling_mean();
    test_window_equals_n();
    test_window_larger_than_n();
    test_window_one_is_identity();
    test_against_naive_reference_random();
    std::printf("All scalar rolling tests passed.\n");
    return 0;
}
