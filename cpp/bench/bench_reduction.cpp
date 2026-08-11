#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "regresslens/reduction_scalar.hpp"

static std::vector<double> make_data(size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> data(n);
    for (auto& v : data) v = dist(rng);
    return data;
}

static void BM_ReduceSumScalar_F64(benchmark::State& state) {
    size_t n = state.range(0);
    auto data = make_data(n);
    for (auto _ : state) {
        double result = regresslens::reduce_sum_scalar<double>(data.data(), n);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double));
}
// Array sizes from 10k to 100M rows per the Phase 1 harness spec.
BENCHMARK(BM_ReduceSumScalar_F64)
    ->Arg(10'000)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->Arg(18'000'000)
    ->Arg(100'000'000);

BENCHMARK_MAIN();
