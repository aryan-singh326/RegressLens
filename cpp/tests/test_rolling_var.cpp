// Rolling variance against a two-pass reference (mean first, then
// squared deviations: the same algorithm np.var uses), including the
// adversarial price-like case that breaks E[x^2] - E[x]^2.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "regresslens/rolling_scalar.hpp"

#include "test_util.hpp"

using regresslens::rolling_var_scalar;

template <typename T>
static std::vector<double> two_pass_reference(const std::vector<T>& in, size_t window,
                                              size_t ddof) {
    std::vector<double> out(in.size() - window + 1);
    for (size_t i = 0; i < out.size(); ++i) {
        double mean = exact_sum(in.data() + i, window) / static_cast<double>(window);
        std::vector<double> sq(window);
        for (size_t j = 0; j < window; ++j) {
            double d = static_cast<double>(in[i + j]) - mean;
            sq[j] = d * d;
        }
        out[i] = exact_sum(sq.data(), window) / static_cast<double>(window - ddof);
    }
    return out;
}

template <typename T>
static double max_rel_error(const std::vector<T>& got, const std::vector<double>& want) {
    double worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        double denom = std::max(std::fabs(want[i]), 1e-300);
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]) / denom);
    }
    return worst;
}

static void test_known_values() {
    std::vector<double> in = {1, 2, 3, 4};
    std::vector<double> out(3);
    CHECK(rolling_var_scalar<double>(in.data(), out.data(), 4, 2) == 3);
    for (double v : out) CHECK(v == 0.25);
    CHECK(rolling_var_scalar<double>(in.data(), out.data(), 4, 2, 1) == 3);
    for (double v : out) CHECK(v == 0.5);
    CHECK(rolling_var_scalar<double>(in.data(), out.data(), 4, 5) == 0);
    std::printf("test_known_values passed\n");
}

// The plan's adversarial case: mean 1e6, std 1e-2. Relative error
// versus the two-pass reference must stay <= 1e-9.
static void test_price_like_adversarial() {
    std::mt19937 rng(5);
    std::normal_distribution<double> dist(1e6, 1e-2);
    for (size_t window : {size_t(5), size_t(20), size_t(250)}) {
        std::vector<double> in(100'000);
        for (auto& v : in) v = dist(rng);
        std::vector<double> out(in.size() - window + 1);
        rolling_var_scalar<double>(in.data(), out.data(), in.size(), window);
        double err = max_rel_error(out, two_pass_reference(in, window, 0));
        std::printf("  adversarial window=%zu max_rel_err=%.3e\n", window, err);
        CHECK(err <= 1e-9);
    }
    std::printf("test_price_like_adversarial passed\n");
}

// A drifting random walk is what defeats a single fixed shift; the
// periodic re-anchoring keeps it accurate.
static void test_drifting_random_walk() {
    std::mt19937 rng(9);
    std::normal_distribution<double> step(0.0, 1.0);
    std::vector<double> in(200'000);
    double level = 1e5;
    for (auto& v : in) {
        level += step(rng);
        v = level;
    }
    for (size_t window : {size_t(3), size_t(50)}) {
        std::vector<double> out(in.size() - window + 1);
        rolling_var_scalar<double>(in.data(), out.data(), in.size(), window, 1);
        double err = max_rel_error(out, two_pass_reference(in, window, 1));
        std::printf("  random walk window=%zu max_rel_err=%.3e\n", window, err);
        CHECK(err <= 1e-9);
    }
    std::printf("test_drifting_random_walk passed\n");
}

static void test_float32() {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> in(10'000);
    for (auto& v : in) v = dist(rng);
    std::vector<float> out(in.size() - 30 + 1);
    rolling_var_scalar<float>(in.data(), out.data(), in.size(), 30);
    double err = max_rel_error(out, two_pass_reference(in, 30, 0));
    CHECK(err <= 2e-7);  // one float32 rounding of a near-exact double
    std::printf("test_float32 passed (max_rel_err=%.3e)\n", err);
}

static void test_nan_confined_to_its_windows() {
    std::vector<double> in(40);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<double>(i % 7);
    in[10] = std::numeric_limits<double>::quiet_NaN();
    in[30] = std::numeric_limits<double>::infinity();
    const size_t window = 4;
    std::vector<double> out(in.size() - window + 1);
    rolling_var_scalar<double>(in.data(), out.data(), in.size(), window);
    for (size_t i = 0; i < out.size(); ++i) {
        bool contains_bad = (i <= 10 && 10 < i + window) || (i <= 30 && 30 < i + window);
        CHECK(std::isnan(out[i]) == contains_bad);
    }
    std::printf("test_nan_confined_to_its_windows passed\n");
}

int main() {
    test_known_values();
    test_price_like_adversarial();
    test_drifting_random_walk();
    test_float32();
    test_nan_confined_to_its_windows();
    std::printf("All rolling variance tests passed.\n");
    return 0;
}
