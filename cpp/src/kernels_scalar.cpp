// Scalar kernel table. Compiled with NO architecture flags: this is
// the portable baseline that must run on any CPU of the target
// architecture (any x86-64, any aarch64). Do not add -march/-mavx*
// here or to any target-wide option that reaches this file. That was
// the source of the SIGILL-on-import defect in 0.1.0.dev0, and it also
// let the compiler auto-vectorize "scalar" loops with AVX2, which
// confounded the Phase 1 scalar-vs-AVX2 characterization.
#include "regresslens/filter_scalar.hpp"
#include "regresslens/kernels.hpp"
#include "regresslens/projection_scalar.hpp"
#include "regresslens/reduction_scalar.hpp"
#include "regresslens/rolling_scalar.hpp"

namespace regresslens {

namespace {

template <typename T>
SumParts sum_entry(const T* in, size_t n) {
    return reduce_sum_scalar_parts<T>(in, n);
}

template <typename T>
OpTable<T> make_scalar_table() {
    OpTable<T> t{};
    t.sum = &sum_entry<T>;
    t.project_affine = &project_affine_scalar<T>;
    t.filter_gt = &filter_gt_scalar<T>;
    t.rolling_sum = &rolling_sum_finite_scalar<T>;
    t.rolling_mean = &rolling_mean_finite_scalar<T>;
    t.scan = &scan_scalar<T>;
    t.sum_mt = nullptr;
    t.filter_gt_mt = nullptr;
    return t;
}

}  // namespace

const KernelSet& scalar_kernel_set() {
    static const KernelSet set{"scalar", make_scalar_table<float>(),
                               make_scalar_table<double>()};
    return set;
}

void rolling_sum_nonfinite(const float* in, float* out, size_t n, size_t window) {
    rolling_sum_nonfinite_scalar<float>(in, out, n, window);
}
void rolling_sum_nonfinite(const double* in, double* out, size_t n, size_t window) {
    rolling_sum_nonfinite_scalar<double>(in, out, n, window);
}
void rolling_mean_nonfinite(const float* in, float* out, size_t n, size_t window) {
    rolling_mean_nonfinite_scalar<float>(in, out, n, window);
}
void rolling_mean_nonfinite(const double* in, double* out, size_t n, size_t window) {
    rolling_mean_nonfinite_scalar<double>(in, out, n, window);
}
void rolling_var(const float* in, float* out, size_t n, size_t window, size_t ddof) {
    rolling_var_scalar_impl<float>(in, out, n, window, ddof);
}
void rolling_var(const double* in, double* out, size_t n, size_t window, size_t ddof) {
    rolling_var_scalar_impl<double>(in, out, n, window, ddof);
}

}  // namespace regresslens
