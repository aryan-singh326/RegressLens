// Google Benchmark microbenchmarks for every kernel set.
//
// Like characterize.cpp, this file is compiled WITHOUT architecture
// flags and calls kernels only through the runtime kernel tables, so
// the "scalar" rows measure the portable build that ships (the old
// per-kernel bench files were compiled with -mavx2 -mfma, which
// auto-vectorized their scalar baselines).
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "regresslens/dispatch.hpp"
#include "regresslens/kernels.hpp"

using namespace regresslens;

namespace {

const KernelSet* set_for(int64_t which) {
    return which == 0 ? &scalar_kernel_set() : runtime().avx2;
}

std::vector<double> make_data(size_t n, double lo, double hi) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(lo, hi);
    std::vector<double> data(n);
    for (auto& v : data) v = dist(rng);
    return data;
}

void BM_Sum(benchmark::State& state) {
    const KernelSet* ks = set_for(state.range(0));
    if (!ks) return state.SkipWithError("AVX2 kernel set unavailable");
    const size_t n = static_cast<size_t>(state.range(1));
    auto data = make_data(n, -1000, 1000);
    for (auto _ : state) benchmark::DoNotOptimize(ks->f64.sum(data.data(), n));
    state.SetBytesProcessed(state.iterations() * n * sizeof(double));
}

void BM_Projection(benchmark::State& state) {
    const KernelSet* ks = set_for(state.range(0));
    if (!ks) return state.SkipWithError("AVX2 kernel set unavailable");
    const size_t n = static_cast<size_t>(state.range(1));
    auto in = make_data(n, -1000, 1000);
    std::vector<double> out(n);
    for (auto _ : state) {
        ks->f64.project_affine(in.data(), out.data(), n, 2.0, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2);
}

void BM_Filter(benchmark::State& state) {
    const KernelSet* ks = set_for(state.range(0));
    if (!ks) return state.SkipWithError("AVX2 kernel set unavailable");
    const size_t n = static_cast<size_t>(state.range(1));
    const double selectivity = state.range(2) / 100.0;
    auto in = make_data(n, 0, 1);
    std::vector<double> out(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            ks->f64.filter_gt(in.data(), out.data(), n, 1.0 - selectivity));
    }
    state.SetItemsProcessed(state.iterations() * n);
}

void BM_RollingSum(benchmark::State& state) {
    const KernelSet* ks = set_for(state.range(0));
    if (!ks) return state.SkipWithError("AVX2 kernel set unavailable");
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t window = static_cast<size_t>(state.range(2));
    auto in = make_data(n, -100, 100);
    std::vector<double> out(n);
    for (auto _ : state) {
        ks->f64.rolling_sum(in.data(), out.data(), n, window);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

}  // namespace

// range(0): 0 = scalar set, 1 = AVX2 set.
BENCHMARK(BM_Sum)->ArgsProduct({{0, 1}, {10'000, 1'000'000, 18'000'000}});
BENCHMARK(BM_Projection)->ArgsProduct({{0, 1}, {10'000, 1'000'000, 18'000'000}});
BENCHMARK(BM_Filter)->ArgsProduct({{0, 1}, {1'000'000}, {10, 50, 90}});
BENCHMARK(BM_RollingSum)->ArgsProduct({{0, 1}, {1'000'000}, {5, 20, 500}});

BENCHMARK_MAIN();
