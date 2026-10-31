// The selector's decisions, as derived from characterization data
// (see the comment block in src/kernel_selector.cpp).
#include <cstdio>

#include "regresslens/kernel_selector.hpp"

#include "test_util.hpp"

using regresslens::DType;
using regresslens::KernelChoice;
using regresslens::Operation;
using regresslens::SelectionContext;
using regresslens::select_kernel;

static SelectionContext ctx(Operation op, size_t n, unsigned threads = 8,
                            double selectivity = 0.5) {
    SelectionContext c{op, DType::Float64, n, true, threads};
    c.estimated_selectivity = selectivity;
    c.window = 20;
    return c;
}

static void test_reduction() {
    CHECK(select_kernel(ctx(Operation::Reduction, 1000)) == KernelChoice::Avx2);
    CHECK(select_kernel(ctx(Operation::Reduction, 1'000'000)) == KernelChoice::Avx2);
    CHECK(select_kernel(ctx(Operation::Reduction, 4'000'000)) == KernelChoice::MtAvx2);
    // No point paying for threads when there is only one.
    CHECK(select_kernel(ctx(Operation::Reduction, 50'000'000, 1)) == KernelChoice::Avx2);
    std::printf("test_reduction passed\n");
}

static void test_projection_stays_scalar() {
    for (size_t n : {size_t(1000), size_t(100'000), size_t(100'000'000)}) {
        CHECK(select_kernel(ctx(Operation::Projection, n)) == KernelChoice::Scalar);
    }
    std::printf("test_projection_stays_scalar passed\n");
}

static void test_filter_selectivity_band() {
    CHECK(select_kernel(ctx(Operation::Filter, 1000, 8, 0.5)) == KernelChoice::Avx2);
    CHECK(select_kernel(ctx(Operation::Filter, 1000, 8, 0.3)) == KernelChoice::Avx2);
    CHECK(select_kernel(ctx(Operation::Filter, 1000, 8, 0.7)) == KernelChoice::Avx2);
    CHECK(select_kernel(ctx(Operation::Filter, 1000, 8, 0.05)) == KernelChoice::Scalar);
    CHECK(select_kernel(ctx(Operation::Filter, 1000, 8, 0.95)) == KernelChoice::Scalar);
    // Multithreading never overrides the selectivity decision.
    CHECK(select_kernel(ctx(Operation::Filter, 50'000'000, 8, 0.02)) == KernelChoice::Scalar);
    CHECK(select_kernel(ctx(Operation::Filter, 1'000'000, 8, 0.5)) == KernelChoice::MtAvx2);
    CHECK(select_kernel(ctx(Operation::Filter, 1'000'000, 1, 0.5)) == KernelChoice::Avx2);
    std::printf("test_filter_selectivity_band passed\n");
}

static void test_rolling() {
    CHECK(select_kernel(ctx(Operation::Rolling, 10'000)) == KernelChoice::Scalar);
    CHECK(select_kernel(ctx(Operation::Rolling, 100'000)) == KernelChoice::Avx2);
    CHECK(select_kernel(ctx(Operation::RollingVar, 100'000'000)) == KernelChoice::Scalar);
    std::printf("test_rolling passed\n");
}

int main() {
    test_reduction();
    test_projection_stays_scalar();
    test_filter_selectivity_band();
    test_rolling();
    std::printf("All kernel selector tests passed.\n");
    return 0;
}
