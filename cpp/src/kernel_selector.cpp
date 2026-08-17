// PLACEHOLDER HEURISTIC — provisional, not from rigorous
// characterization.
//
// These thresholds come from the rough, single-core, low-clock
// sandbox findings gathered while building the Phase 1 kernels, NOT
// from the dedicated-hardware characterization the project brief
// specifies. They are directionally informed, not measured:
//   - Reduction: AVX2 consistently beat scalar in sandbox testing.
//   - Projection: memory-bandwidth-bound; AVX2 did NOT reliably
//     beat scalar (sometimes slightly slower). Default to scalar.
//   - Filter: AVX2 won specifically near ~50% selectivity (~30%
//     faster in sandbox benchmarks); roughly tied at 10%/90%
//     selectivity, scalar sometimes slightly ahead there. Branches
//     on ctx.estimated_selectivity accordingly (see below) — this
//     is now a real, if still sandbox-derived, data-driven decision,
//     not a flat default. Callers that don't have a real selectivity
//     estimate pass 0.5 (the neutral assumption), which lands in the
//     AVX2-favorable band, matching the previous flat-default
//     behavior for anyone not yet supplying real history.
//   - Rolling: scalar beat the AVX2 prefix-sum design outright in
//     sandbox testing, due to sequential-pass + allocation overhead
//     not being recovered by the vectorized second phase. Default
//     to scalar.
//   - MT-AVX2 (any op): thread creation/join overhead dominated at
//     small N even with a single thread in sandbox testing. Gated
//     behind a conservative large-N threshold and requires more
//     than 1 available thread.
//
// REPLACE THESE THRESHOLDS once real characterization data exists
// (see cpp/bench/characterize.cpp — run on dedicated-tenancy
// hardware, not this sandbox). This file is the ONLY file that
// should need to change when that happens, or when Phase 2's
// research produces an online adaptive selector to use instead of a
// static heuristic — see kernel_selector.hpp for the interface
// boundary this is meant to respect.
#include "regresslens/kernel_selector.hpp"

namespace regresslens {

namespace {

// Provisional. Not measured on real hardware.
constexpr size_t kMtMinN = 5'000'000;

// Provisional, from sandbox benchmarking only. AVX2's win over
// scalar for filter was clearest in the 0.3-0.7 selectivity range;
// outside that band the two were close enough that scalar's simpler
// branch-predictor-friendly behavior at skewed selectivity makes it
// the safer default. Revisit with real hardware data.
constexpr double kFilterAvx2SelectivityLow = 0.3;
constexpr double kFilterAvx2SelectivityHigh = 0.7;

}  // namespace

KernelChoice select_kernel(const SelectionContext& ctx) {
    switch (ctx.op) {
        case Operation::Reduction:
            if (ctx.n >= kMtMinN && ctx.available_threads > 1) {
                return KernelChoice::MtAvx2;
            }
            return KernelChoice::Avx2;

        case Operation::Projection:
            // Memory-bandwidth-bound; AVX2 not confirmed to help.
            return KernelChoice::Scalar;

        case Operation::Filter: {
            bool selectivity_favors_avx2 =
                ctx.estimated_selectivity >= kFilterAvx2SelectivityLow &&
                ctx.estimated_selectivity <= kFilterAvx2SelectivityHigh;
            if (!selectivity_favors_avx2) {
                return KernelChoice::Scalar;
            }
            if (ctx.n >= kMtMinN && ctx.available_threads > 1) {
                return KernelChoice::MtAvx2;
            }
            return KernelChoice::Avx2;
        }

        case Operation::Rolling:
            // AVX2 prefix-sum design lost to scalar in sandbox
            // testing at every size/window tested. No MT variant
            // exists yet (sequential-dependency concerns apply to
            // MT even more than to single-threaded AVX2).
            return KernelChoice::Scalar;
    }
    return KernelChoice::Scalar;  // unreachable, but a safe default
}

}  // namespace regresslens
