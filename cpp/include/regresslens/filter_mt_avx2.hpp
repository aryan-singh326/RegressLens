#pragma once
#include <algorithm>
#include <thread>
#include <vector>

#include "regresslens/filter_avx2.hpp"

// Must be included from translation units compiled with -mavx2.

namespace regresslens {

// Multithreaded AVX2 filter: partitions into num_threads contiguous
// chunks, runs the AVX2 filter kernel on each chunk independently
// into its OWN local buffer (each chunk's output size is data-
// dependent and unknown until that chunk finishes), then
// concatenates results in chunk order into `out`. Order preservation
// matters — element i's relative position to element j (i<j) in the
// original array must be preserved in the filtered output, so
// chunks must be concatenated in index order, not completion order.
//
// This costs one extra copy (from each thread's local buffer into
// the final `out` array) compared to reduction's MT design, because
// filter's per-thread output size isn't known in advance the way
// reduction's is (a single scalar). That extra copy is a real,
// measurable cost this kernel pays that reduction doesn't — worth
// capturing in characterization data, not just assuming away.
template <typename T>
size_t filter_gt_mt_avx2(const T* in, T* out, size_t n, T threshold,
                          size_t num_threads) {
    if (n == 0) return 0;
    if (num_threads == 0) num_threads = 1;
    num_threads = std::min(num_threads, n);

    std::vector<std::vector<T>> local_outs(num_threads);
    std::vector<size_t> local_counts(num_threads, 0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk = n / num_threads;
    size_t remainder = n % num_threads;

    size_t offset = 0;
    std::vector<size_t> offsets(num_threads);
    std::vector<size_t> chunk_sizes(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        size_t this_chunk = chunk + (t < remainder ? 1 : 0);
        offsets[t] = offset;
        chunk_sizes[t] = this_chunk;
        offset += this_chunk;
    }

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            local_outs[t].resize(chunk_sizes[t]);  // worst case: all pass
            local_counts[t] = filter_gt_avx2(in + offsets[t], local_outs[t].data(),
                                              chunk_sizes[t], threshold);
        });
    }
    for (auto& th : threads) th.join();

    size_t total = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        std::copy(local_outs[t].begin(), local_outs[t].begin() + local_counts[t],
                  out + total);
        total += local_counts[t];
    }
    return total;
}

}  // namespace regresslens
