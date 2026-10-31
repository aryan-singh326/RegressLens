// Kernel selection heuristic, derived from characterization data.
//
// Source: bench/results/characterize-core-ultra7-255u-windows-msvc.csv,
// produced by bench/characterize (compiled WITHOUT architecture flags, so
// "scalar" is the portable kernel that ships) and reduced by
// bench/derive_thresholds.py. A kernel is chosen only where it was at
// least 5% faster at that size and at every larger measured size.
// Inside that margin the simpler kernel wins, because a 2% edge on one
// machine does not survive moving to another.
//
// Re-derive these constants on new hardware with:
//   characterize > results/<host>.csv && python derive_thresholds.py results/<host>.csv
//
// Findings (Intel Core Ultra 7 255U, MSVC 14.51):
//   * Reduction: AVX2 (4x4-lane compensated TwoSum) 3.2-4.1x faster
//     than the scalar compensated kernel at every size. Multithreading
//     loses below ~1M elements (thread start-up) and wins from 4M
//     (1.2x at 4M, 1.7-5.5x at 8M+).
//   * Projection: AVX2+FMA is 0-46% faster, but not consistently 5%
//     faster at DRAM-bound sizes (f64 at 8M+: 1.00x), and one run
//     measured it slower at 10k. Scalar stays the default. The Phase 1
//     finding "AVX2 doesn't beat scalar" came from a scalar baseline
//     auto-vectorized with -mavx2 -mfma, so it no longer applies. The
//     honest summary is "memory-bound; small, inconsistent gain".
//   * Filter: AVX2 wins in 100% of size/dtype combinations at
//     selectivity 0.3-0.7 and loses at 0.9+ (more output bytes per
//     input, the compaction loop dominates). Multithreading inside the
//     band wins from 1M.
//   * Rolling (sum/mean): the AVX2 kernel runs four compensated sliding
//     windows in parallel lanes and is 1.4-2.4x faster from 100k up; at
//     10k float32 it is ~10% slower (lane setup).
//   * Rolling variance: scalar only.
//
// This file is the only one that should change when thresholds are
// re-derived or an adaptive selector replaces the heuristic (see the
// interface notes in kernel_selector.hpp).
#include "regresslens/kernel_selector.hpp"

namespace regresslens {

namespace {

constexpr size_t kReductionMtMinN = 4'000'000;
constexpr size_t kFilterMtMinN = 1'000'000;
constexpr size_t kRollingAvx2MinN = 100'000;
constexpr double kFilterAvx2SelectivityLow = 0.3;
constexpr double kFilterAvx2SelectivityHigh = 0.7;

}  // namespace

KernelChoice select_kernel(const SelectionContext& ctx) {
    switch (ctx.op) {
        case Operation::Reduction:
            if (ctx.n >= kReductionMtMinN && ctx.available_threads > 1) {
                return KernelChoice::MtAvx2;
            }
            return KernelChoice::Avx2;

        case Operation::Projection:
            return KernelChoice::Scalar;

        case Operation::Filter: {
            const bool in_band = ctx.estimated_selectivity >= kFilterAvx2SelectivityLow &&
                                 ctx.estimated_selectivity <= kFilterAvx2SelectivityHigh;
            if (!in_band) return KernelChoice::Scalar;
            if (ctx.n >= kFilterMtMinN && ctx.available_threads > 1) {
                return KernelChoice::MtAvx2;
            }
            return KernelChoice::Avx2;
        }

        case Operation::Rolling:
            return ctx.n >= kRollingAvx2MinN ? KernelChoice::Avx2 : KernelChoice::Scalar;

        case Operation::RollingVar:
            return KernelChoice::Scalar;  // only a scalar kernel exists
    }
    return KernelChoice::Scalar;  // unreachable, but a safe default
}

}  // namespace regresslens
