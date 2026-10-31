#pragma once
// Plain-check test helpers. Deliberately NOT assert(): Release builds
// define NDEBUG, which compiles asserts out, and a test that silently
// checks nothing is worse than no test. CHECK() always fires.
#include <cstdio>
#include <cstdlib>

#include "regresslens/cpu_features.hpp"

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// Exit code CTest maps to "skipped" (SKIP_RETURN_CODE 77).
constexpr int kSkipReturnCode = 77;

// For tests compiled with AVX2 flags: bail out before running any
// AVX2 kernel on a CPU without AVX2/FMA. cpu_has_avx2_fma() itself is
// compiled without architecture flags.
#define SKIP_UNLESS_AVX2()                                                \
    do {                                                                  \
        if (!regresslens::cpu_has_avx2_fma()) {                           \
            std::printf("SKIPPED: CPU lacks AVX2/FMA\n");                 \
            return kSkipReturnCode;                                       \
        }                                                                 \
    } while (0)

#include <cmath>
#include <utility>
#include <vector>

// Exact-to-one-rounding reference sum (Shewchuk's algorithm, the one
// behind Python's math.fsum). Needed because long double is just
// double on MSVC, too imprecise to check a compensated kernel.
template <typename T>
inline double exact_sum(const T* data, size_t n) {
    std::vector<double> partials;
    for (size_t k = 0; k < n; ++k) {
        double x = static_cast<double>(data[k]);
        size_t i = 0;
        for (size_t j = 0; j < partials.size(); ++j) {
            double y = partials[j];
            if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
            double hi = x + y;
            double lo = y - (hi - x);
            if (lo != 0.0) partials[i++] = lo;
            x = hi;
        }
        partials.resize(i);
        partials.push_back(x);
    }
    double acc = 0.0;
    for (size_t j = partials.size(); j-- > 0;) acc += partials[j];
    return acc;
}
