#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <vector>

namespace regresslens {

// A kernel under test: takes a pointer to input data and element
// count, returns a scalar result. This matches the shape of a
// reduction operation. Other operation categories (projection,
// filter, rolling) will need their own harness signatures later —
// don't generalize this prematurely before you have a second
// operation to learn the right abstraction from.
template <typename T>
using ReductionKernel = std::function<T(const T*, size_t)>;

// Runs `kernel` against `reference` on the given input and reports
// whether they agree within tolerance. Per the project brief: "a
// kernel that produces wrong results is not measured" — this
// function is the gate benchmarking must pass through first.
//
// rel_tol is a relative tolerance because float32 accumulation error
// grows with N; an absolute tolerance would either be too loose for
// small arrays or too strict for large ones.
template <typename T>
bool check_reduction_correctness(const char* kernel_name,
                                  ReductionKernel<T> kernel,
                                  ReductionKernel<T> reference,
                                  const T* data, size_t n,
                                  T rel_tol = std::numeric_limits<T>::epsilon() * 100) {
    T got = kernel(data, n);
    T want = reference(data, n);

    T diff = std::fabs(got - want);
    T scale = std::max(std::fabs(want), T(1));
    bool ok = (diff / scale) <= rel_tol;

    if (!ok) {
        std::fprintf(stderr,
                      "CORRECTNESS FAILURE [%s]: got=%.17g want=%.17g "
                      "rel_diff=%.3e (tol=%.3e) n=%zu\n",
                      kernel_name, (double)got, (double)want,
                      (double)(diff / scale), (double)rel_tol, n);
    }
    return ok;
}

// Elementwise kernel signature: writes n outputs into `out` given n
// inputs from `in`. Used for projection, filter, and rolling ops —
// anything that doesn't collapse to a scalar.
template <typename T>
using ElementwiseKernel = std::function<void(const T*, T*, size_t)>;

// Compares two output arrays element-by-element. Unlike reduction
// checking, there's no single scale to normalize against — each
// element gets its own relative tolerance check, and we report the
// FIRST mismatch (not all of them) since a single bug usually
// produces thousands of near-identical failures and printing all of
// them just buries the useful signal.
template <typename T>
bool check_elementwise_correctness(const char* kernel_name,
                                    ElementwiseKernel<T> kernel,
                                    ElementwiseKernel<T> reference,
                                    const T* data, size_t n,
                                    T rel_tol = std::numeric_limits<T>::epsilon() * 100) {
    std::vector<T> got(n), want(n);
    kernel(data, got.data(), n);
    reference(data, want.data(), n);

    for (size_t i = 0; i < n; ++i) {
        T diff = std::fabs(got[i] - want[i]);
        T scale = std::max(std::fabs(want[i]), T(1));
        if ((diff / scale) > rel_tol) {
            std::fprintf(stderr,
                          "CORRECTNESS FAILURE [%s]: first mismatch at "
                          "i=%zu got=%.17g want=%.17g rel_diff=%.3e "
                          "(tol=%.3e) n=%zu\n",
                          kernel_name, i, (double)got[i], (double)want[i],
                          (double)(diff / scale), (double)rel_tol, n);
            return false;
        }
    }
    return true;
}

}  // namespace regresslens
