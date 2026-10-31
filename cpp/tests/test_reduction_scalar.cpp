// Validates the scalar compensated sum against an exact reference
// (Shewchuk summation) and exactly representable cancellation cases,
// plus IEEE NaN/Inf semantics.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "regresslens/reduction_scalar.hpp"

#include "test_util.hpp"

using regresslens::reduce_sum_scalar;

static void test_small_known_values() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    CHECK(reduce_sum_scalar<double>(data.data(), data.size()) == 15.0);
    std::printf("test_small_known_values passed\n");
}

static void test_empty() {
    CHECK(reduce_sum_scalar<double>(nullptr, 0) == 0.0);
    CHECK(reduce_sum_scalar<float>(nullptr, 0) == 0.0);
    std::printf("test_empty passed\n");
}

// Plain summation returns 0 here (1 is absorbed by 1e16); the
// compensated kernel must recover it exactly, in every chain position.
static void test_cancellation_is_exact() {
    for (size_t pad = 0; pad < 8; ++pad) {
        std::vector<double> data(pad, 0.0);
        data.push_back(1e16);
        data.push_back(1.0);
        data.push_back(-1e16);
        CHECK(reduce_sum_scalar<double>(data.data(), data.size()) == 1.0);
    }
    std::printf("test_cancellation_is_exact passed\n");
}

static void test_large_random_f64() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> data(1'000'000);
    for (auto& v : data) v = dist(rng);

    double got = reduce_sum_scalar<double>(data.data(), data.size());
    double want = exact_sum(data.data(), data.size());
    double rel = std::fabs(got - want) / std::max(std::fabs(want), 1.0);
    CHECK(rel < 1e-14);
    std::printf("test_large_random_f64 passed (rel_diff=%.3e)\n", rel);
}

// float32 input accumulates in double, so the result is essentially
// exact (the old float accumulator reached 1e-3 relative error here).
static void test_large_random_f32() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> data(1'000'000);
    for (auto& v : data) v = dist(rng);

    double got = reduce_sum_scalar<float>(data.data(), data.size());
    double want = exact_sum(data.data(), data.size());
    double rel = std::fabs(got - want) / std::max(std::fabs(want), 1.0);
    CHECK(rel < 1e-14);
    std::printf("test_large_random_f32 passed (rel_diff=%.3e)\n", rel);
}

static void test_nonfinite_semantics() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> a = {1.0, inf, 2.0};
    CHECK(reduce_sum_scalar<double>(a.data(), a.size()) == inf);
    std::vector<double> b = {1.0, -inf, 2.0, 3.0, 4.0};
    CHECK(reduce_sum_scalar<double>(b.data(), b.size()) == -inf);
    std::vector<double> c = {inf, 1.0, -inf};
    CHECK(std::isnan(reduce_sum_scalar<double>(c.data(), c.size())));
    std::vector<double> d = {1.0, 2.0, nan, 3.0, 4.0, 5.0};
    CHECK(std::isnan(reduce_sum_scalar<double>(d.data(), d.size())));
    std::vector<float> e = {1.0f, std::numeric_limits<float>::infinity()};
    CHECK(reduce_sum_scalar<float>(e.data(), e.size()) == inf);
    std::printf("test_nonfinite_semantics passed\n");
}

int main() {
    test_small_known_values();
    test_empty();
    test_cancellation_is_exact();
    test_large_random_f64();
    test_large_random_f32();
    test_nonfinite_semantics();
    std::printf("All scalar reduction tests passed.\n");
    return 0;
}
