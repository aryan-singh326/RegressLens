#pragma once
#include <thread>

#include "regresslens/reduction_avx2.hpp"

// Include only from translation units compiled with -mavx2 -mfma.

namespace regresslens {

// Upper bound on worker threads per call. Fixed so the per-thread
// partial sums live in a stack array (no std::vector in AVX2 code,
// see avx2_common.hpp).
constexpr size_t kMaxKernelThreads = 64;

// Multithreaded compensated sum: num_threads contiguous chunks, each
// reduced by the AVX2 kernel into an (s, c) pair, then merged in chunk
// order so the result is deterministic for a given thread count.
// Threads are created per call rather than pooled; the kernel
// selector only picks this variant at sizes where that overhead is
// negligible.
template <typename T>
static inline SumParts reduce_sum_mt_avx2_parts(const T* data, size_t n,
                                                size_t num_threads) {
    if (n == 0) return SumParts{0.0, 0.0};
    if (num_threads == 0) num_threads = 1;
    if (num_threads > kMaxKernelThreads) num_threads = kMaxKernelThreads;
    if (num_threads > n) num_threads = n;

    SumParts partial[kMaxKernelThreads];
    std::thread workers[kMaxKernelThreads];

    const size_t chunk = n / num_threads;
    const size_t remainder = n % num_threads;
    size_t offset = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        // Spread the remainder over the first chunks so sizes differ
        // by at most one element.
        const size_t len = chunk + (t < remainder ? 1 : 0);
        workers[t] = std::thread([data, offset, len, t, &partial]() {
            partial[t] = reduce_sum_avx2_parts(data + offset, len);
        });
        offset += len;
    }
    for (size_t t = 0; t < num_threads; ++t) workers[t].join();

    SumParts acc{0.0, 0.0};
    for (size_t t = 0; t < num_threads; ++t) avx2_detail::merge_parts(acc, partial[t]);
    return acc;
}

template <typename T>
static inline double reduce_sum_mt_avx2(const T* data, size_t n, size_t num_threads) {
    return avx2_detail::finish(reduce_sum_mt_avx2_parts<T>(data, n, num_threads));
}

}  // namespace regresslens
