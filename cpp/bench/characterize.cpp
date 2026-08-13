// Unified Phase 1 characterization driver.
//
// This is NOT a Google Benchmark target — it's a standalone binary
// that sweeps every implemented kernel across the array sizes and
// (where relevant) selectivities/windows called for in the project
// brief's characterization table, and writes a CSV. Run this on the
// dedicated-tenancy hardware, not in a shared/cloud VM — see the
// project's own hardware fingerprinting requirement. The numbers
// from this sandbox are not valid characterization data; only the
// mechanism (does it run, does it produce sane output) has been
// verified here.
//
// Output columns: operation,kernel,dtype,n,window_or_selectivity,
//                 median_ns,warm_or_cold,threads
//
// This is a starting point, not the final harness — it does not yet
// do bootstrap CI, warm/cold separation, or hardware fingerprinting
// (all called for in the Phase 1 spec). Those are the next things
// to add once this runs cleanly on real hardware and you've
// confirmed the sweep dimensions are the right ones.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "regresslens/filter_avx2.hpp"
#include "regresslens/filter_mt_avx2.hpp"
#include "regresslens/filter_scalar.hpp"
#include "regresslens/projection_avx2.hpp"
#include "regresslens/projection_scalar.hpp"
#include "regresslens/reduction_avx2.hpp"
#include "regresslens/reduction_mt_avx2.hpp"
#include "regresslens/reduction_scalar.hpp"
#include "regresslens/rolling_avx2.hpp"
#include "regresslens/rolling_scalar.hpp"

using Clock = std::chrono::steady_clock;

static double median_of(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t mid = v.size() / 2;
    return v.size() % 2 == 0 ? (v[mid - 1] + v[mid]) / 2.0 : v[mid];
}

// Runs `fn` `reps` times, returns the median wall-clock time in ns.
// Not a substitute for the bootstrap-CI methodology the actual
// product uses (see project brief) — this is characterization
// tooling for building the heuristic, not the shipped regression
// detector.
template <typename Fn>
static double time_median_ns(Fn&& fn, int reps = 15) {
    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    return median_of(samples);
}

static std::vector<double> make_data(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> data(n);
    for (auto& v : data) v = dist(rng);
    return data;
}

int main() {
    std::printf("operation,kernel,dtype,n,param,median_ns\n");

    const std::vector<size_t> sizes = {10'000, 100'000, 1'000'000, 18'000'000};
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;

    // --- Reduction ---
    for (size_t n : sizes) {
        auto data = make_data(n);
        std::vector<double> out(n);

        double t_scalar = time_median_ns([&]() {
            volatile double r = regresslens::reduce_sum_scalar<double>(data.data(), n);
            (void)r;
        });
        std::printf("reduction,scalar,f64,%zu,,%f\n", n, t_scalar);

        double t_avx2 = time_median_ns([&]() {
            volatile double r = regresslens::reduce_sum_avx2(data.data(), n);
            (void)r;
        });
        std::printf("reduction,avx2,f64,%zu,,%f\n", n, t_avx2);

        double t_mt = time_median_ns([&]() {
            volatile double r =
                regresslens::reduce_sum_mt_avx2<double>(data.data(), n, hw_threads);
            (void)r;
        });
        std::printf("reduction,mt_avx2,f64,%zu,threads=%u,%f\n", n, hw_threads, t_mt);
    }

    // --- Projection ---
    for (size_t n : sizes) {
        auto data = make_data(n);
        std::vector<double> out(n);

        double t_scalar = time_median_ns([&]() {
            regresslens::project_affine_scalar<double>(data.data(), out.data(), n,
                                                         2.0, 1.0);
        });
        std::printf("projection,scalar,f64,%zu,,%f\n", n, t_scalar);

        double t_avx2 = time_median_ns([&]() {
            regresslens::project_affine_avx2(data.data(), out.data(), n, 2.0, 1.0);
        });
        std::printf("projection,avx2,f64,%zu,,%f\n", n, t_avx2);
    }

    // --- Filter (across selectivities) ---
    for (size_t n : sizes) {
        for (double sel : {0.1, 0.5, 0.9}) {
            std::mt19937 rng(7);
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            std::vector<double> in(n);
            for (auto& v : in) v = dist(rng);
            double threshold = 1.0 - sel;
            std::vector<double> out(n);

            char param[32];
            std::snprintf(param, sizeof(param), "sel=%.1f", sel);

            double t_scalar = time_median_ns([&]() {
                volatile size_t c =
                    regresslens::filter_gt_scalar<double>(in.data(), out.data(), n, threshold);
                (void)c;
            });
            std::printf("filter,scalar,f64,%zu,%s,%f\n", n, param, t_scalar);

            double t_avx2 = time_median_ns([&]() {
                volatile size_t c = regresslens::filter_gt_avx2(in.data(), out.data(), n, threshold);
                (void)c;
            });
            std::printf("filter,avx2,f64,%zu,%s,%f\n", n, param, t_avx2);

            double t_mt = time_median_ns([&]() {
                volatile size_t c = regresslens::filter_gt_mt_avx2<double>(
                    in.data(), out.data(), n, threshold, hw_threads);
                (void)c;
            });
            std::printf("filter,mt_avx2,f64,%zu,%s+threads=%u,%f\n", n, param,
                         hw_threads, t_mt);
        }
    }

    // --- Rolling (across windows) ---
    for (size_t n : sizes) {
        for (size_t window : {size_t(5), size_t(20), size_t(500)}) {
            if (window > n) continue;
            auto in = make_data(n);
            std::vector<double> out(n);
            char param[32];
            std::snprintf(param, sizeof(param), "window=%zu", window);

            double t_scalar = time_median_ns([&]() {
                volatile size_t c =
                    regresslens::rolling_sum_scalar<double>(in.data(), out.data(), n, window);
                (void)c;
            });
            std::printf("rolling,scalar,f64,%zu,%s,%f\n", n, param, t_scalar);

            double t_avx2 = time_median_ns([&]() {
                volatile size_t c = regresslens::rolling_sum_avx2(in.data(), out.data(), n, window);
                (void)c;
            });
            std::printf("rolling,avx2,f64,%zu,%s,%f\n", n, param, t_avx2);
        }
    }

    return 0;
}
