#pragma once
#include <cstddef>

// ISA-neutral kernel tables. This header contains only plain data
// types and declarations, so it is safe to include from any
// translation unit regardless of its architecture flags.
//
// Each ISA is compiled in its own translation unit:
//   src/kernels_scalar.cpp  no architecture flags (portable baseline)
//   src/kernels_avx2.cpp    -mavx2 -mfma or /arch:AVX2, x86-64 only
// and exposes one KernelSet of function pointers. src/dispatch.cpp
// picks the set at module init from runtime CPU detection, so the
// extension never executes an AVX2 instruction on a CPU without it.

namespace regresslens {

// A compensated sum: the true sum is s + c (see compensated.hpp).
struct SumParts {
    double s;
    double c;
};

// One-pass input scan used to route rolling kernels: NaN/Inf inputs
// go to the NaN-segmented algorithm, and finite inputs whose
// magnitude could overflow an accumulator go to the NumPy fallback.
struct ScanResult {
    bool nonfinite;  // any NaN or +/-Inf present
    double absmax;   // max |x| over the finite elements
};

template <typename T>
struct OpTable {
    // Compensated sum, accumulated in double for both dtypes.
    SumParts (*sum)(const T* in, size_t n);
    void (*project_affine)(const T* in, T* out, size_t n, T scale, T offset);
    size_t (*filter_gt)(const T* in, T* out, size_t n, T threshold);
    // Rolling kernels write n - window + 1 outputs. Preconditions:
    // 1 <= window <= n and every input finite (check with scan first).
    void (*rolling_sum)(const T* in, T* out, size_t n, size_t window);
    void (*rolling_mean)(const T* in, T* out, size_t n, size_t window);
    ScanResult (*scan)(const T* in, size_t n);
    // Multithreaded variants; nullptr when the ISA has none.
    SumParts (*sum_mt)(const T* in, size_t n, unsigned threads);
    size_t (*filter_gt_mt)(const T* in, T* out, size_t n, T threshold,
                           unsigned threads);
};

struct KernelSet {
    const char* name;
    OpTable<float> f32;
    OpTable<double> f64;
};

// Always available.
const KernelSet& scalar_kernel_set();

// Defined only when the AVX2 translation unit is compiled in
// (REGRESSLENS_HAVE_AVX2_TU). Never call it directly; ask
// dispatch.hpp's runtime() for the set, which checks the CPU first.
const KernelSet& avx2_kernel_set();

// Scalar-only algorithms used by every ISA. They are not hot enough
// (NaN-bearing input, rolling variance) to justify AVX2 versions.
void rolling_sum_nonfinite(const float* in, float* out, size_t n, size_t window);
void rolling_sum_nonfinite(const double* in, double* out, size_t n, size_t window);
void rolling_mean_nonfinite(const float* in, float* out, size_t n, size_t window);
void rolling_mean_nonfinite(const double* in, double* out, size_t n, size_t window);
// Handles NaN/Inf internally (any non-finite value in a window makes
// that window's variance NaN, matching np.var).
void rolling_var(const float* in, float* out, size_t n, size_t window, size_t ddof);
void rolling_var(const double* in, double* out, size_t n, size_t window, size_t ddof);

}  // namespace regresslens
