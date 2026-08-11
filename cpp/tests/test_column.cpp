// Plain-check tests. No framework dependency by design — this file
// should stay trivial to read and trivial to build. If it grows past
// a few hundred lines, switch to a real test framework then.
//
// Deliberately NOT using assert(): CMAKE_BUILD_TYPE=Release defines
// -DNDEBUG, which compiles asserts out entirely. A test that silently
// does nothing in Release builds is worse than no test — it prints
// "passed" while checking nothing. CHECK() always fires.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "regresslens/column.hpp"

using regresslens::ColumnBuffer;
using regresslens::DType;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,     \
                         __FILE__, __LINE__);                             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

static void test_contiguous_f64() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    ColumnBuffer col{data.data(), data.size(), DType::Float64, 1};

    CHECK(col.is_contiguous());
    CHECK(col.size == 4);
    CHECK(col.byte_size() == 32);
    CHECK(col.as_f64()[0] == 1.0);
    CHECK(col.as_f64()[3] == 4.0);
    std::printf("test_contiguous_f64 passed\n");
}

static void test_non_contiguous_stride() {
    // Simulates a NumPy slice like arr[::2] — every other element.
    std::vector<float> data = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f, 99.0f};
    ColumnBuffer col{data.data(), 3, DType::Float32, 2};

    CHECK(!col.is_contiguous());
    CHECK(col.size == 3);
    // byte_size() reports logical element count, not the span of
    // underlying memory touched — that distinction matters for the
    // remediation cost estimate later (copy cost is based on size,
    // not stride span).
    CHECK(col.byte_size() == 12);
    std::printf("test_non_contiguous_stride passed\n");
}

static void test_dtype_size() {
    CHECK(regresslens::dtype_size(DType::Float32) == 4);
    CHECK(regresslens::dtype_size(DType::Float64) == 8);
    std::printf("test_dtype_size passed\n");
}

static void test_empty_buffer() {
    ColumnBuffer col{nullptr, 0, DType::Float64, 1};
    CHECK(col.size == 0);
    CHECK(col.byte_size() == 0);
    CHECK(col.is_contiguous());  // stride of an empty view is still 1
    std::printf("test_empty_buffer passed\n");
}

int main() {
    test_contiguous_f64();
    test_non_contiguous_stride();
    test_dtype_size();
    test_empty_buffer();
    std::printf("All column tests passed.\n");
    return 0;
}
