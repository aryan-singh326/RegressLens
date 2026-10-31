// NaN/Inf safety for the rolling kernels (Phase 4 defect: one NaN
// used to poison every later window). Reference semantics are those
// of summing each window independently in IEEE arithmetic, which is
// what np.convolve does: NaN anywhere -> NaN, +Inf and -Inf together
// -> NaN, otherwise the infinity, otherwise the finite sum.
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "regresslens/rolling_scalar.hpp"

#include "test_util.hpp"

template <typename T>
static std::vector<double> naive_windows(const std::vector<T>& in, size_t window) {
    std::vector<double> out(in.size() - window + 1);
    for (size_t i = 0; i < out.size(); ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < window; ++j) acc += static_cast<double>(in[i + j]);
        out[i] = acc;
    }
    return out;
}

template <typename T>
static void check_same(const std::vector<T>& got, const std::vector<double>& want,
                       const char* label) {
    CHECK(got.size() >= want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        double g = static_cast<double>(got[i]);
        if (std::isnan(want[i])) {
            CHECK(std::isnan(g));
        } else if (std::isinf(want[i])) {
            CHECK(g == want[i]);
        } else {
            double tol = 1e-5 * std::max(1.0, std::fabs(want[i]));
            if (!(std::fabs(g - want[i]) <= tol)) {
                std::fprintf(stderr, "%s mismatch at %zu: got %.17g want %.17g\n", label,
                             i, g, want[i]);
                CHECK(false);
            }
        }
    }
}

template <typename T>
static void random_injection_trials(const char* label) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> val(-100.0, 100.0);
    std::uniform_int_distribution<int> kind(0, 2);
    const T specials[3] = {std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::infinity(),
                           -std::numeric_limits<T>::infinity()};
    for (int trial = 0; trial < 300; ++trial) {
        size_t n = 1 + rng() % 400;
        size_t window = 1 + rng() % n;
        std::vector<T> in(n);
        for (auto& v : in) v = static_cast<T>(val(rng));
        size_t injections = rng() % 6;
        for (size_t k = 0; k < injections; ++k) in[rng() % n] = specials[kind(rng)];

        std::vector<T> out(n);
        size_t count = regresslens::rolling_sum_scalar<T>(in.data(), out.data(), n, window);
        CHECK(count == n - window + 1);
        auto want = naive_windows(in, window);
        check_same(out, want, label);

        regresslens::rolling_mean_scalar<T>(in.data(), out.data(), n, window);
        for (auto& w : want) w /= static_cast<double>(window);
        check_same(out, want, label);
    }
    std::printf("%s random injection trials passed\n", label);
}

static void test_single_nan_is_confined() {
    std::vector<double> in(40);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<double>(i + 1);
    in[5] = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out(in.size());
    size_t count = regresslens::rolling_sum_scalar<double>(in.data(), out.data(), 40, 4);
    size_t nan_windows = 0;
    for (size_t i = 0; i < count; ++i) nan_windows += std::isnan(out[i]) ? 1 : 0;
    // Exactly the 4 windows containing index 5 (np.convolve gives 4;
    // the 0.1.0 kernel gave 35 of 37).
    CHECK(nan_windows == 4);
    CHECK(out[10] == 11 + 12 + 13 + 14);
    std::printf("test_single_nan_is_confined passed\n");
}

static void test_scan() {
    std::vector<double> a = {1.0, -7.5, 3.0};
    auto r = regresslens::scan_scalar<double>(a.data(), a.size());
    CHECK(!r.nonfinite && r.absmax == 7.5);
    a[1] = std::numeric_limits<double>::quiet_NaN();
    r = regresslens::scan_scalar<double>(a.data(), a.size());
    CHECK(r.nonfinite && r.absmax == 3.0);
    std::vector<float> f = {1.0f, -std::numeric_limits<float>::infinity()};
    CHECK(regresslens::scan_scalar<float>(f.data(), f.size()).nonfinite);
    std::printf("test_scan passed\n");
}

int main() {
    test_single_nan_is_confined();
    test_scan();
    random_injection_trials<double>("f64");
    random_injection_trials<float>("f32");
    std::printf("All non-finite rolling tests passed.\n");
    return 0;
}
