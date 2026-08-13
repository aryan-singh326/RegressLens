#pragma once
#include <algorithm>
#include <thread>
#include <vector>

#include "regresslens/reduction_avx2.hpp"

// Must be included from translation units compiled with -mavx2.

namespace regresslens {

// Multithreaded AVX2 sum reduction: partitions the array into
// num_threads contiguous chunks, runs the AVX2 reduction kernel on
// each chunk in its own thread, then combines the partial sums.
// Uses a fixed thread count per call rather than a persistent thread
// pool — per the project brief's kernel spec, a persistent pool
// belongs in the runtime integration layer (Phase 3), not the
// Phase 1 kernel itself. This keeps the kernel's own interface
// simple and testable in isolation.
//
// Reduction is "embarrassingly parallel" for this purpose: summing
// disjoint chunks and adding the partial sums together gives the
// same mathematical result as summing everything sequentially,
// modulo floating-point reassociation — see the correctness test for
// how that's handled.
template <typename T>
T reduce_sum_mt_avx2(const T* data, size_t n, size_t num_threads) {
    if (n == 0) return T(0);
    if (num_threads == 0) num_threads = 1;
    num_threads = std::min(num_threads, n);

    std::vector<T> partial_sums(num_threads, T(0));
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk = n / num_threads;
    size_t remainder = n % num_threads;

    size_t offset = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        // Distribute the remainder across the first `remainder`
        // threads so chunk sizes differ by at most 1 element,
        // rather than dumping all the leftover into the last thread.
        size_t this_chunk = chunk + (t < remainder ? 1 : 0);
        threads.emplace_back([&, offset, this_chunk, t]() {
            partial_sums[t] = reduce_sum_avx2(data + offset, this_chunk);
        });
        offset += this_chunk;
    }
    for (auto& th : threads) th.join();

    T total = T(0);
    for (T p : partial_sums) total += p;
    return total;
}

}  // namespace regresslens
