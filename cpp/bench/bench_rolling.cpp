#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "regresslens/rolling_avx2.hpp"
#include "regresslens/rolling_scalar.hpp"

static std::vector<double> make_data(size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::vector<double> data(n);
    for (auto& v : data) v = dist(rng);
    return data;
}

static void BM_RollingSumScalar_F64(benchmark::State& state) {
    size_t n = state.range(0);
    size_t window = state.range(1);
    auto in = make_data(n);
    std::vector<double> out(n);
    for (auto _ : state) {
        size_t count = regresslens::rolling_sum_scalar<double>(in.data(), out.data(), n, window);
        benchmark::DoNotOptimize(count);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_RollingSumScalar_F64)
    ->Args({1'000'000, 5})->Args({1'000'000, 20})->Args({1'000'000, 500})
    ->Args({18'000'000, 20});

static void BM_RollingSumAVX2_F64(benchmark::State& state) {
    size_t n = state.range(0);
    size_t window = state.range(1);
    auto in = make_data(n);
    std::vector<double> out(n);
    for (auto _ : state) {
        size_t count = regresslens::rolling_sum_avx2(in.data(), out.data(), n, window);
        benchmark::DoNotOptimize(count);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_RollingSumAVX2_F64)
    ->Args({1'000'000, 5})->Args({1'000'000, 20})->Args({1'000'000, 500})
    ->Args({18'000'000, 20});

BENCHMARK_MAIN();
