#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "regresslens/projection_scalar.hpp"

using regresslens::project_affine_scalar;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

static void test_known_values() {
    double in[4] = {1.0, 2.0, 3.0, 4.0};
    double out[4];
    project_affine_scalar<double>(in, out, 4, 2.0, 10.0);
    CHECK(out[0] == 12.0);
    CHECK(out[1] == 14.0);
    CHECK(out[2] == 16.0);
    CHECK(out[3] == 18.0);
    std::printf("test_known_values passed\n");
}

static void test_identity_transform() {
    // scale=1, offset=0 should be a no-op copy.
    std::vector<double> in = {5.5, -3.2, 0.0, 1e6};
    std::vector<double> out(in.size());
    project_affine_scalar<double>(in.data(), out.data(), in.size(), 1.0, 0.0);
    for (size_t i = 0; i < in.size(); ++i) CHECK(out[i] == in[i]);
    std::printf("test_identity_transform passed\n");
}

static void test_empty() {
    double* in = nullptr;
    double* out = nullptr;
    project_affine_scalar<double>(in, out, 0, 2.0, 1.0);  // must not crash
    std::printf("test_empty passed\n");
}

static void test_in_place() {
    // Projection must support out == in (common pattern: transform
    // an array in place rather than allocating a new one).
    std::vector<double> data = {1.0, 2.0, 3.0};
    project_affine_scalar<double>(data.data(), data.data(), data.size(), 3.0, 1.0);
    CHECK(data[0] == 4.0);
    CHECK(data[1] == 7.0);
    CHECK(data[2] == 10.0);
    std::printf("test_in_place passed\n");
}

int main() {
    test_known_values();
    test_identity_transform();
    test_empty();
    test_in_place();
    std::printf("All scalar projection tests passed.\n");
    return 0;
}
