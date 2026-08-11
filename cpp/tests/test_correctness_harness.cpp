// Tests the harness itself, not a real kernel yet — we don't have
// real kernels until step 5. This proves check_reduction_correctness
// correctly accepts a matching kernel and correctly rejects a broken
// one, before we trust it to gate real benchmarking.
#include <cstdio>
#include <cstdlib>

#include "regresslens/correctness.hpp"

using regresslens::check_reduction_correctness;
using regresslens::ReductionKernel;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

static double naive_sum(const double* data, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) acc += data[i];
    return acc;
}

static double broken_sum(const double* data, size_t n) {
    // Deliberately wrong: skips the last element.
    double acc = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) acc += data[i];
    return acc;
}

int main() {
    double data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    ReductionKernel<double> reference = naive_sum;
    ReductionKernel<double> good = naive_sum;
    ReductionKernel<double> broken = broken_sum;

    bool good_passes =
        check_reduction_correctness<double>("naive_sum", good, reference, data, 5);
    CHECK(good_passes);
    std::printf("harness correctly accepted a matching kernel\n");

    bool broken_fails =
        check_reduction_correctness<double>("broken_sum", broken, reference, data, 5);
    CHECK(!broken_fails);
    std::printf("harness correctly rejected a broken kernel\n");

    std::printf("All correctness harness tests passed.\n");
    return 0;
}
