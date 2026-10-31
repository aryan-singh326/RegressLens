// Kernel characterization driver.
//
// Sweeps every shipped kernel set across array sizes (and selectivity
// or window where relevant) and writes a CSV to stdout. Feed the CSV
// to derive_thresholds.py to re-derive the kernel selector's
// thresholds (src/kernel_selector.cpp).
//
// This binary is compiled WITHOUT architecture flags and reaches the
// kernels only through the runtime kernel tables, so "scalar" here is
// the portable scalar build that ships. The pre-0.2 driver was
// compiled with -mavx2 -mfma for the whole executable, which let the
// compiler auto-vectorize the "scalar" loops with AVX2 and confounded
// the projection result ("AVX2 doesn't beat scalar").
//
// Run on quiet, dedicated hardware. Shared or cloud VMs give
// directionally useful data at best.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "regresslens/cpu_features.hpp"
#include "regresslens/dispatch.hpp"
#include "regresslens/kernels.hpp"

using namespace regresslens;
using Clock = std::chrono::steady_clock;

namespace {

struct Timing {
    double median, p10, p90;
};

template <typename Fn>
Timing time_kernel(Fn&& fn, int reps) {
    fn();  // warm caches, page in buffers, spin up the core
    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    auto at = [&](double q) { return samples[static_cast<size_t>(q * (samples.size() - 1))]; };
    return Timing{at(0.5), at(0.1), at(0.9)};
}

template <typename T>
const OpTable<T>& table_for(const KernelSet& set);
template <>
const OpTable<float>& table_for<float>(const KernelSet& set) { return set.f32; }
template <>
const OpTable<double>& table_for<double>(const KernelSet& set) { return set.f64; }

int reps_for(size_t n) { return n >= 10'000'000 ? 9 : n >= 1'000'000 ? 21 : 51; }

volatile double g_sink;  // defeats dead-code elimination

void emit(const char* op, const char* kernel, const char* dtype, size_t n,
          const std::string& param, const Timing& t, int reps) {
    std::printf("%s,%s,%s,%zu,%s,%.1f,%.1f,%.1f,%d\n", op, kernel, dtype, n,
                param.c_str(), t.median, t.p10, t.p90, reps);
    std::fflush(stdout);
}

template <typename T>
void sweep(const char* dtype, const std::vector<const KernelSet*>& sets, unsigned threads) {
    const std::vector<size_t> sizes = {10'000, 100'000, 1'000'000, 4'000'000, 8'000'000,
                                       18'000'000};
    for (size_t n : sizes) {
        const int reps = reps_for(n);
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
        std::vector<T> data(n), out(n);
        for (auto& v : data) v = static_cast<T>(dist(rng));

        for (const KernelSet* ks : sets) {
            const OpTable<T>& t = table_for<T>(*ks);
            emit("reduction", ks->name, dtype, n, "",
                 time_kernel([&] { g_sink = t.sum(data.data(), n).s; }, reps), reps);
            if (t.sum_mt) {
                emit("reduction", "mt_avx2", dtype, n, "threads=" + std::to_string(threads),
                     time_kernel([&] { g_sink = t.sum_mt(data.data(), n, threads).s; }, reps),
                     reps);
            }
            emit("projection", ks->name, dtype, n, "",
                 time_kernel([&] { t.project_affine(data.data(), out.data(), n, T(2), T(1)); },
                             reps),
                 reps);
            for (size_t window : {size_t(5), size_t(20), size_t(500)}) {
                emit("rolling", ks->name, dtype, n, "window=" + std::to_string(window),
                     time_kernel([&] { t.rolling_sum(data.data(), out.data(), n, window); },
                                 reps),
                     reps);
            }
        }

        // Filter: performance depends on selectivity (branch
        // predictability and output volume), not just n.
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::vector<T> u(n);
        for (auto& v : u) v = static_cast<T>(unit(rng));
        for (double sel : {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
            const T threshold = static_cast<T>(1.0 - sel);
            char param[32];
            std::snprintf(param, sizeof param, "sel=%.2f", sel);
            for (const KernelSet* ks : sets) {
                const OpTable<T>& t = table_for<T>(*ks);
                emit("filter", ks->name, dtype, n, param,
                     time_kernel([&] { g_sink = (double)t.filter_gt(u.data(), out.data(), n, threshold); },
                                 reps),
                     reps);
                if (t.filter_gt_mt) {
                    emit("filter", "mt_avx2", dtype, n,
                         std::string(param) + "+threads=" + std::to_string(threads),
                         time_kernel([&] {
                             g_sink = (double)t.filter_gt_mt(u.data(), out.data(), n,
                                                             threshold, threads);
                         }, reps),
                         reps);
                }
            }
        }
    }
}

}  // namespace

int main() {
    const Runtime& rt = runtime();
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;
    std::printf("# regresslens characterize\n");
    std::printf("# cpu_avx2_fma=%d avx2_compiled=%d hardware_threads=%u\n", rt.cpu_avx2,
                rt.avx2_compiled, threads);
    std::printf("operation,kernel,dtype,n,param,median_ns,p10_ns,p90_ns,reps\n");

    std::vector<const KernelSet*> sets = {&scalar_kernel_set()};
    if (rt.avx2) sets.push_back(rt.avx2);
    sweep<double>("f64", sets, threads);
    sweep<float>("f32", sets, threads);
    return 0;
}
