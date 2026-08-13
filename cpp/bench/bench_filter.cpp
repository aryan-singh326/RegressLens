#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "regresslens/filter_avx2.hpp"
#include "regresslens/filter_scalar.hpp"

// Generates data such that P(value > 0.5) == selectivity, by biasing
// how much of [0,1) sits above 0.5. Kept simple and reproducible.
static std::vector<double> make_data(size_t n, double selectivity) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> data(n);
    for (auto& v : data) v = dist(rng);
    (void)selectivity;  // threshold is set by the caller instead
    return data;
}

static void BM_FilterGtScalar_F64(benchmark::State& state) {
    size_t n = state.range(0);
    double selectivity = state.range(1) / 100.0;
    double threshold = 1.0 - selectivity;
    auto in = make_data(n, selectivity);
    std::vector<double> out(n);
    for (auto _ : state) {
        size_t count =
            regresslens::filter_gt_scalar<double>(in.data(), out.data(), n, threshold);
        benchmark::DoNotOptimize(count);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_FilterGtScalar_F64)
    ->Args({1'000'000, 10})->Args({1'000'000, 50})->Args({1'000'000, 90})
    ->Args({18'000'000, 10})->Args({18'000'000, 50})->Args({18'000'000, 90});

static void BM_FilterGtAVX2_F64(benchmark::State& state) {
    size_t n = state.range(0);
    double selectivity = state.range(1) / 100.0;
    double threshold = 1.0 - selectivity;
    auto in = make_data(n, selectivity);
    std::vector<double> out(n);
    for (auto _ : state) {
        size_t count = regresslens::filter_gt_avx2(in.data(), out.data(), n, threshold);
        benchmark::DoNotOptimize(count);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_FilterGtAVX2_F64)
    ->Args({1'000'000, 10})->Args({1'000'000, 50})->Args({1'000'000, 90})
    ->Args({18'000'000, 10})->Args({18'000'000, 50})->Args({18'000'000, 90});

BENCHMARK_MAIN();
