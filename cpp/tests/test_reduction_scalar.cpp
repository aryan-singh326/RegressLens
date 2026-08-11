// Validates reduce_sum_scalar against a higher-precision reference.
// This matters even for the "reference" kernel itself: float32
// accumulation in a naive loop loses precision at large N, so we
// check against long double accumulation, not against itself.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/reduction_scalar.hpp"

using regresslens::reduce_sum_scalar;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

template <typename T>
static long double reference_sum(const std::vector<T>& data) {
    long double acc = 0.0L;
    for (T v : data) acc += (long double)v;
    return acc;
}

static void test_small_known_values() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double got = reduce_sum_scalar<double>(data.data(), data.size());
    CHECK(got == 15.0);
    std::printf("test_small_known_values passed\n");
}

static void test_empty() {
    std::vector<double> data;
    double got = reduce_sum_scalar<double>(data.data(), 0);
    CHECK(got == 0.0);
    std::printf("test_empty passed\n");
}

static void test_large_random_f64() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> data(1'000'000);
    for (auto& v : data) v = dist(rng);

    double got = reduce_sum_scalar<double>(data.data(), data.size());
    long double want = reference_sum(data);
    long double rel_diff = fabsl((long double)got - want) /
                            std::max(fabsl(want), 1.0L);

    CHECK(rel_diff < 1e-9L);
    std::printf("test_large_random_f64 passed (rel_diff=%.3Le)\n", rel_diff);
}

static void test_large_random_f32() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> data(1'000'000);
    for (auto& v : data) v = dist(rng);

    float got = reduce_sum_scalar<float>(data.data(), data.size());
    long double want = reference_sum(data);
    long double rel_diff = fabsl((long double)got - want) /
                            std::max(fabsl(want), 1.0L);

    // Looser tolerance for float32: naive sequential summation
    // accumulates real rounding error at 1M elements. This is
    // measuring the actual precision behavior, not an arbitrary
    // number — if this fails, it's telling you something true about
    // float32 summation, not a bug in the test.
    CHECK(rel_diff < 1e-3L);
    std::printf("test_large_random_f32 passed (rel_diff=%.3Le)\n", rel_diff);
}

int main() {
    test_small_known_values();
    test_empty();
    test_large_random_f64();
    test_large_random_f32();
    std::printf("All scalar reduction tests passed.\n");
    return 0;
}
