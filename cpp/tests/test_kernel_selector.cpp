#include <cstdio>
#include <cstdlib>

#include "regresslens/kernel_selector.hpp"

using regresslens::KernelChoice;
using regresslens::Operation;
using regresslens::SelectionContext;
using regresslens::select_kernel;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

static void test_projection_defaults_to_scalar() {
    // Bandwidth-bound finding: AVX2 not confirmed to help, at any
    // size or thread count.
    SelectionContext ctx{Operation::Projection, regresslens::DType::Float64,
                          100'000'000, true, 8};
    CHECK(select_kernel(ctx) == KernelChoice::Scalar);
    std::printf("test_projection_defaults_to_scalar passed\n");
}

static void test_rolling_defaults_to_scalar() {
    // Sequential-dependency finding: AVX2 lost outright.
    SelectionContext ctx{Operation::Rolling, regresslens::DType::Float64,
                          100'000'000, true, 8};
    ctx.window = 20;
    CHECK(select_kernel(ctx) == KernelChoice::Scalar);
    std::printf("test_rolling_defaults_to_scalar passed\n");
}

static void test_reduction_uses_avx2_below_mt_threshold() {
    SelectionContext ctx{Operation::Reduction, regresslens::DType::Float64,
                          1000, true, 8};
    CHECK(select_kernel(ctx) == KernelChoice::Avx2);
    std::printf("test_reduction_uses_avx2_below_mt_threshold passed\n");
}

static void test_reduction_uses_mt_above_threshold_with_threads() {
    SelectionContext ctx{Operation::Reduction, regresslens::DType::Float64,
                          10'000'000, true, 8};
    CHECK(select_kernel(ctx) == KernelChoice::MtAvx2);
    std::printf("test_reduction_uses_mt_above_threshold_with_threads passed\n");
}

static void test_reduction_stays_avx2_if_only_one_thread_available() {
    // Matches the sandbox finding that MT overhead dominates even
    // just from spawning a single thread — no reason to pay thread
    // overhead if there's nothing to parallelize across.
    SelectionContext ctx{Operation::Reduction, regresslens::DType::Float64,
                          10'000'000, true, 1};
    CHECK(select_kernel(ctx) == KernelChoice::Avx2);
    std::printf("test_reduction_stays_avx2_if_only_one_thread_available passed\n");
}

static void test_filter_uses_avx2_by_default() {
    SelectionContext ctx{Operation::Filter, regresslens::DType::Float64, 1000,
                          true, 8};
    ctx.estimated_selectivity = 0.5;
    CHECK(select_kernel(ctx) == KernelChoice::Avx2);
    std::printf("test_filter_uses_avx2_by_default passed\n");
}

static void test_filter_uses_mt_above_threshold_with_threads() {
    SelectionContext ctx{Operation::Filter, regresslens::DType::Float64,
                          10'000'000, true, 8};
    CHECK(select_kernel(ctx) == KernelChoice::MtAvx2);
    std::printf("test_filter_uses_mt_above_threshold_with_threads passed\n");
}

static void test_filter_uses_scalar_at_skewed_low_selectivity() {
    SelectionContext ctx{Operation::Filter, regresslens::DType::Float64, 1000,
                          true, 8};
    ctx.estimated_selectivity = 0.05;  // ~5% match rate, outside the 0.3-0.7 band
    CHECK(select_kernel(ctx) == KernelChoice::Scalar);
    std::printf("test_filter_uses_scalar_at_skewed_low_selectivity passed\n");
}

static void test_filter_uses_scalar_at_skewed_high_selectivity() {
    SelectionContext ctx{Operation::Filter, regresslens::DType::Float64, 1000,
                          true, 8};
    ctx.estimated_selectivity = 0.95;  // ~95% match rate, outside the band
    CHECK(select_kernel(ctx) == KernelChoice::Scalar);
    std::printf("test_filter_uses_scalar_at_skewed_high_selectivity passed\n");
}

static void test_filter_band_boundaries_are_inclusive() {
    SelectionContext ctx{Operation::Filter, regresslens::DType::Float64, 1000,
                          true, 8};
    ctx.estimated_selectivity = 0.3;
    CHECK(select_kernel(ctx) == KernelChoice::Avx2);
    ctx.estimated_selectivity = 0.7;
    CHECK(select_kernel(ctx) == KernelChoice::Avx2);
    std::printf("test_filter_band_boundaries_are_inclusive passed\n");
}

static void test_filter_stays_scalar_even_at_high_n_when_selectivity_skewed() {
    // MT-AVX2 must not override the selectivity decision — a
    // skewed-selectivity filter should stay scalar regardless of
    // size, since the underlying AVX2 kernel isn't favored here at
    // all, MT or otherwise.
    SelectionContext ctx{Operation::Filter, regresslens::DType::Float64,
                          10'000'000, true, 8};
    ctx.estimated_selectivity = 0.02;
    CHECK(select_kernel(ctx) == KernelChoice::Scalar);
    std::printf("test_filter_stays_scalar_even_at_high_n_when_selectivity_skewed passed\n");
}

int main() {
    test_projection_defaults_to_scalar();
    test_rolling_defaults_to_scalar();
    test_reduction_uses_avx2_below_mt_threshold();
    test_reduction_uses_mt_above_threshold_with_threads();
    test_reduction_stays_avx2_if_only_one_thread_available();
    test_filter_uses_avx2_by_default();
    test_filter_uses_mt_above_threshold_with_threads();
    test_filter_uses_scalar_at_skewed_low_selectivity();
    test_filter_uses_scalar_at_skewed_high_selectivity();
    test_filter_band_boundaries_are_inclusive();
    test_filter_stays_scalar_even_at_high_n_when_selectivity_skewed();
    std::printf("All kernel selector tests passed.\n");
    return 0;
}
