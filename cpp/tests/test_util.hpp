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
#include <limits>
#include <random>
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

inline double ulp_of(double x) {
    x = std::fabs(x);
    return std::nextafter(x, std::numeric_limits<double>::infinity()) - x;
}

// Blocks of values up to 1e8 alternating with blocks of values below
// 1e-8. A sliding accumulator that keeps the rounding residue of values
// that have left the window gets the tiny windows wrong by many ulps
// (the nightly `rglns verify` found 4-256 ulp), although each of them
// summed alone is exact.
inline std::vector<double> big_then_tiny_blocks(unsigned seed, size_t block = 200,
                                                size_t blocks = 20) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> big(-1e8, 1e8), tiny(-1e-8, 1e-8);
    std::vector<double> in;
    for (size_t b = 0; b < blocks; ++b) {
        for (size_t i = 0; i < block; ++i) in.push_back(big(rng));
        for (size_t i = 0; i < block; ++i) in.push_back(tiny(rng));
    }
    return in;
}

// Every finite output within one ulp of the exact window sum (the
// accuracy policy's bound) and NaN exactly where the window has a NaN.
template <typename F>
inline void check_window_sums(const std::vector<double>& in, size_t window, F kernel) {
    std::vector<double> out(in.size());
    size_t count = kernel(in.data(), out.data(), in.size(), window);
    CHECK(count == in.size() - window + 1);
    for (size_t i = 0; i < count; ++i) {
        bool has_nan = false;
        for (size_t j = i; j < i + window; ++j) has_nan = has_nan || std::isnan(in[j]);
        if (has_nan) {
            CHECK(std::isnan(out[i]));
            continue;
        }
        double want = exact_sum(in.data() + i, window);
        if (!(std::fabs(out[i] - want) <= ulp_of(want))) {
            std::fprintf(stderr, "window=%zu i=%zu got=%.17g want=%.17g\n", window, i, out[i],
                         want);
            CHECK(false);
        }
    }
}
