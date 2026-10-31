// AVX2 kernel table. This is the ONLY translation unit in the
// extension compiled with -mavx2 -mfma (/arch:AVX2 on MSVC), and it
// is only compiled on x86-64. Nothing in it runs unless
// dispatch.cpp's runtime CPU check finds AVX2 and FMA.
//
// Linker hazard (read before editing): an inline function or template
// that is instantiated here AND in a baseline TU is emitted twice as
// a COMDAT and folded into one copy. If the linker keeps this file's
// copy, baseline code calls VEX-encoded instructions and crashes on
// pre-AVX CPUs. So this file and the *_avx2.hpp headers use only
// static inline functions and C library calls. They use no std
// containers or algorithms, and none of the scalar headers. The qemu
// Nehalem CI job (.github/workflows/ci.yml) is the regression test.
#include "regresslens/filter_avx2.hpp"
#include "regresslens/filter_mt_avx2.hpp"
#include "regresslens/kernels.hpp"
#include "regresslens/projection_avx2.hpp"
#include "regresslens/reduction_avx2.hpp"
#include "regresslens/reduction_mt_avx2.hpp"
#include "regresslens/rolling_avx2.hpp"

namespace regresslens {

namespace {

template <typename T>
SumParts sum_entry(const T* in, size_t n) {
    return reduce_sum_avx2_parts(in, n);
}

template <typename T>
SumParts sum_mt_entry(const T* in, size_t n, unsigned threads) {
    return reduce_sum_mt_avx2_parts<T>(in, n, threads);
}

template <typename T>
void project_entry(const T* in, T* out, size_t n, T scale, T offset) {
    project_affine_avx2(in, out, n, scale, offset);
}

template <typename T>
size_t filter_entry(const T* in, T* out, size_t n, T threshold) {
    return filter_gt_avx2(in, out, n, threshold);
}

template <typename T>
size_t filter_mt_entry(const T* in, T* out, size_t n, T threshold, unsigned threads) {
    return filter_gt_mt_avx2<T>(in, out, n, threshold, threads);
}

template <typename T>
void rolling_sum_entry(const T* in, T* out, size_t n, size_t window) {
    rolling_sum_avx2(in, out, n, window);
}

template <typename T>
void rolling_mean_entry(const T* in, T* out, size_t n, size_t window) {
    rolling_mean_avx2(in, out, n, window);
}

template <typename T>
ScanResult scan_entry(const T* in, size_t n) {
    return scan_avx2(in, n);
}

template <typename T>
OpTable<T> make_avx2_table() {
    OpTable<T> t{};
    t.sum = &sum_entry<T>;
    t.project_affine = &project_entry<T>;
    t.filter_gt = &filter_entry<T>;
    t.rolling_sum = &rolling_sum_entry<T>;
    t.rolling_mean = &rolling_mean_entry<T>;
    t.scan = &scan_entry<T>;
    t.sum_mt = &sum_mt_entry<T>;
    t.filter_gt_mt = &filter_mt_entry<T>;
    return t;
}

}  // namespace

const KernelSet& avx2_kernel_set() {
    static const KernelSet set{"avx2", make_avx2_table<float>(),
                               make_avx2_table<double>()};
    return set;
}

}  // namespace regresslens
