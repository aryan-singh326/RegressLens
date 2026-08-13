#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/filter_scalar.hpp"

using regresslens::filter_gt_scalar;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

static void test_known_values() {
    double in[6] = {1.0, 5.0, 2.0, 8.0, -3.0, 4.0};
    double out[6];
    size_t count = filter_gt_scalar<double>(in, out, 6, 3.0);
    CHECK(count == 3);
    CHECK(out[0] == 5.0);
    CHECK(out[1] == 8.0);
    CHECK(out[2] == 4.0);
    std::printf("test_known_values passed\n");
}

static void test_none_pass() {
    double in[4] = {1.0, 2.0, 3.0, 4.0};
    double out[4];
    size_t count = filter_gt_scalar<double>(in, out, 4, 100.0);
    CHECK(count == 0);
    std::printf("test_none_pass passed\n");
}

static void test_all_pass() {
    double in[4] = {1.0, 2.0, 3.0, 4.0};
    double out[4];
    size_t count = filter_gt_scalar<double>(in, out, 4, -100.0);
    CHECK(count == 4);
    for (int i = 0; i < 4; ++i) CHECK(out[i] == in[i]);
    std::printf("test_all_pass passed\n");
}

static void test_threshold_boundary_is_exclusive() {
    // "greater than", not "greater than or equal" — an element
    // exactly equal to the threshold must NOT pass.
    double in[3] = {5.0, 5.0, 5.0};
    double out[3];
    size_t count = filter_gt_scalar<double>(in, out, 3, 5.0);
    CHECK(count == 0);
    std::printf("test_threshold_boundary_is_exclusive passed\n");
}

static void test_empty() {
    double* in = nullptr;
    double* out = nullptr;
    size_t count = filter_gt_scalar<double>(in, out, 0, 1.0);
    CHECK(count == 0);
    std::printf("test_empty passed\n");
}

int main() {
    test_known_values();
    test_none_pass();
    test_all_pass();
    test_threshold_boundary_is_exclusive();
    test_empty();
    std::printf("All scalar filter tests passed.\n");
    return 0;
}
