#pragma once
#include <cstring>
#include <thread>

#include "regresslens/filter_avx2.hpp"
#include "regresslens/reduction_mt_avx2.hpp"  // kMaxKernelThreads

// Include only from translation units compiled with -mavx2.

namespace regresslens {

// Multithreaded AVX2 filter. Each worker filters one contiguous chunk
// directly into `out` at that chunk's own offset. This is safe because
// a chunk's output can never be longer than the chunk, and `out` has
// capacity n. The segments are then compacted leftward in chunk order
// with memmove (each destination starts at or before its source).
// Output order therefore matches the input order, and no per-thread
// scratch buffers are needed.
template <typename T>
static inline size_t filter_gt_mt_avx2(const T* in, T* out, size_t n, T threshold,
                                       size_t num_threads) {
    if (n == 0) return 0;
    if (num_threads == 0) num_threads = 1;
    if (num_threads > kMaxKernelThreads) num_threads = kMaxKernelThreads;
    if (num_threads > n) num_threads = n;

    size_t offsets[kMaxKernelThreads];
    size_t counts[kMaxKernelThreads];
    std::thread workers[kMaxKernelThreads];

    const size_t chunk = n / num_threads;
    const size_t remainder = n % num_threads;
    size_t offset = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        const size_t len = chunk + (t < remainder ? 1 : 0);
        offsets[t] = offset;
        workers[t] = std::thread([in, out, offset, len, threshold, t, &counts]() {
            counts[t] = filter_gt_avx2(in + offset, out + offset, len, threshold);
        });
        offset += len;
    }
    for (size_t t = 0; t < num_threads; ++t) workers[t].join();

    size_t total = counts[0];  // chunk 0 is already in place
    for (size_t t = 1; t < num_threads; ++t) {
        if (counts[t] > 0 && total != offsets[t]) {
            std::memmove(out + total, out + offsets[t], counts[t] * sizeof(T));
        }
        total += counts[t];
    }
    return total;
}

}  // namespace regresslens
