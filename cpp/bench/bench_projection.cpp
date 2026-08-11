#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "regresslens/projection_avx2.hpp"
#include "regresslens/projection_scalar.hpp"

static std::vector<double> make_data(size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> data(n);
    for (auto& v : data) v = dist(rng);
    return data;
}

static void BM_ProjectAffineScalar_F64(benchmark::State& state) {
    size_t n = state.range(0);
    auto in = make_data(n);
    std::vector<double> out(n);
    for (auto _ : state) {
        regresslens::project_affine_scalar<double>(in.data(), out.data(), n,
                                                     2.0, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2);
}
BENCHMARK(BM_ProjectAffineScalar_F64)
    ->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(18'000'000);

static void BM_ProjectAffineAVX2_F64(benchmark::State& state) {
    size_t n = state.range(0);
    auto in = make_data(n);
    std::vector<double> out(n);
    for (auto _ : state) {
        regresslens::project_affine_avx2(in.data(), out.data(), n, 2.0, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2);
}
BENCHMARK(BM_ProjectAffineAVX2_F64)
    ->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(18'000'000);

BENCHMARK_MAIN();
