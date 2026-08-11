#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/projection_avx2.hpp"
#include "regresslens/projection_scalar.hpp"

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

template <typename T>
static void check_matches_scalar(const char* label, size_t n, T rel_tol) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<T> dist(T(-1000), T(1000));
    std::vector<T> in(n);
    for (auto& v : in) v = dist(rng);

    std::vector<T> scalar_out(n), avx2_out(n);
    regresslens::project_affine_scalar<T>(in.data(), scalar_out.data(), n,
                                           T(2.5), T(-1.5));
    regresslens::project_affine_avx2(in.data(), avx2_out.data(), n, T(2.5),
                                      T(-1.5));

    for (size_t i = 0; i < n; ++i) {
        T diff = std::fabs(scalar_out[i] - avx2_out[i]);
        T scale = std::max(std::fabs(scalar_out[i]), T(1));
        if ((diff / scale) > rel_tol) {
            std::fprintf(stderr,
                          "%s MISMATCH at i=%zu: scalar=%.9g avx2=%.9g "
                          "(n=%zu)\n",
                          label, i, (double)scalar_out[i], (double)avx2_out[i],
                          n);
            CHECK(false);
        }
    }
    std::printf("%s passed (n=%zu)\n", label, n);
}

static void test_in_place_avx2() {
    // AVX2 kernel must also support out == in, same as scalar.
    std::vector<double> data(100);
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (auto& v : data) v = dist(rng);

    std::vector<double> expected(100);
    regresslens::project_affine_scalar<double>(data.data(), expected.data(),
                                                 100, 3.0, 2.0);
    regresslens::project_affine_avx2(data.data(), data.data(), 100, 3.0, 2.0);

    for (size_t i = 0; i < 100; ++i) {
        CHECK(std::fabs(data[i] - expected[i]) < 1e-9);
    }
    std::printf("test_in_place_avx2 passed\n");
}

int main() {
    // Sizes around the unroll block boundaries (16 for f64, 32 for
    // f32) plus edge cases, same discipline as the reduction AVX2
    // test.
    for (size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(17),
                      size_t(1000), size_t(100'000), size_t(1'000'003)}) {
        check_matches_scalar<double>("f64", n, 1e-12);
        check_matches_scalar<float>("f32", n, 1e-5f);
    }
    test_in_place_avx2();
    std::printf("All AVX2 projection tests passed.\n");
    return 0;
}
