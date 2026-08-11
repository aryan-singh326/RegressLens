#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace regresslens {

enum class DType : uint8_t {
    Float32,
    Float64,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
    }
    throw std::logic_error("unreachable: unknown DType");
}

// A non-owning view over a 1D numeric buffer. Mirrors what a NumPy
// array exposes via its buffer protocol: a pointer, an element
// count, a dtype, and a stride. RegressLens does NOT own the memory
// here — the Python wrapper owns the underlying NumPy allocation.
//
// v0.1 scope: no null bitmap, no multi-dimensional strides. A single
// stride value describes the whole buffer because v0.1 only supports
// contiguous OR simple constant-stride 1D views.
struct ColumnBuffer {
    void* data;
    size_t size;        // element count, NOT bytes
    DType dtype;
    ptrdiff_t stride;   // in elements. 1 == contiguous.

    bool is_contiguous() const { return stride == 1; }

    size_t byte_size() const { return size * dtype_size(dtype); }

    // Typed accessors. Caller is responsible for checking dtype
    // matches — this is a deliberately thin view type, not a
    // type-safe wrapper. Kernels dispatch on dtype once, then use
    // these directly in the hot loop.
    float* as_f32() { return static_cast<float*>(data); }
    const float* as_f32() const { return static_cast<const float*>(data); }
    double* as_f64() { return static_cast<double*>(data); }
    const double* as_f64() const { return static_cast<const double*>(data); }
};

}  // namespace regresslens
