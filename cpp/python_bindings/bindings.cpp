// Python bridge for RegressLens native kernels. Starts with ONE
// operation (reduction) end-to-end, same discipline as Phase 1:
// prove the mechanism works before replicating it three more times
// for projection/filter/rolling.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <thread>

#include "regresslens/kernel_selector.hpp"
#include "regresslens/reduction_avx2.hpp"
#include "regresslens/reduction_mt_avx2.hpp"
#include "regresslens/reduction_scalar.hpp"

namespace py = pybind11;
using namespace regresslens;

namespace {

const char* kernel_choice_name(KernelChoice k) {
    switch (k) {
        case KernelChoice::Scalar: return "scalar";
        case KernelChoice::Avx2: return "avx2";
        case KernelChoice::MtAvx2: return "mt_avx2";
    }
    return "unknown";
}

// Dispatches to the selected kernel for a given typed buffer. Kept
// as its own template so float32/float64 share dispatch logic
// instead of duplicating the switch statement per dtype.
template <typename T>
double dispatch_reduce_sum(const T* data, size_t n, KernelChoice choice,
                            unsigned threads) {
    switch (choice) {
        case KernelChoice::Scalar:
            return static_cast<double>(reduce_sum_scalar<T>(data, n));
        case KernelChoice::Avx2:
            return static_cast<double>(reduce_sum_avx2(data, n));
        case KernelChoice::MtAvx2:
            return static_cast<double>(reduce_sum_mt_avx2<T>(data, n, threads));
    }
    throw std::logic_error("unreachable: unknown KernelChoice");
}

// The one function exposed for reduction. Returns (value, kernel_name)
// rather than just the value, because Phase 3's trace persistence
// needs to know which kernel ran for every call — this is that
// observability hook, exposed from the start rather than retrofitted.
py::tuple reduce_sum_native(py::array arr, int threads_override) {
    // v0.1 scope: contiguous only. Non-contiguous input is the
    // contiguity-loss diagnosis path's job (Phase 3 Python layer),
    // not this native function's — it raises here, and the Python
    // wrapper is responsible for deciding what to do about that
    // (currently: fall back to NumPy, see array.py).
    py::buffer_info info = arr.request();
    bool contiguous = (info.strides.size() == 1 &&
                        info.strides[0] == static_cast<py::ssize_t>(info.itemsize));
    if (!contiguous) {
        throw std::invalid_argument(
            "reduce_sum_native: non-contiguous array. The Python layer "
            "should route this to NumPy fallback, not call this function "
            "directly.");
    }

    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;
    unsigned threads = threads_override > 0 ? static_cast<unsigned>(threads_override)
                                             : hw_threads;

    size_t n = static_cast<size_t>(info.shape.empty() ? 0 : info.shape[0]);

    std::string dtype_str = py::str(arr.dtype());
    DType dtype;
    if (dtype_str.find("float64") != std::string::npos) {
        dtype = DType::Float64;
    } else if (dtype_str.find("float32") != std::string::npos) {
        dtype = DType::Float32;
    } else {
        throw std::invalid_argument(
            "reduce_sum_native: unsupported dtype '" + dtype_str +
            "'. v0.1 supports float32 and float64 only.");
    }

    SelectionContext ctx{Operation::Reduction, dtype, n, contiguous, threads};
    KernelChoice choice = select_kernel(ctx);

    double result;
    if (dtype == DType::Float64) {
        result = dispatch_reduce_sum<double>(static_cast<const double*>(info.ptr), n,
                                              choice, threads);
    } else {
        result = dispatch_reduce_sum<float>(static_cast<const float*>(info.ptr), n,
                                             choice, threads);
    }

    return py::make_tuple(result, kernel_choice_name(choice));
}

}  // namespace

PYBIND11_MODULE(_regresslens_native, m) {
    m.doc() = "RegressLens native kernel bindings (internal — use the "
              "regresslens Python package, not this module directly)";

    m.def("reduce_sum_native", &reduce_sum_native, py::arg("array"),
          py::arg("threads_override") = -1,
          "Sum-reduce a contiguous float32/float64 NumPy array using the "
          "kernel selector. Returns (value, kernel_name_used). Raises on "
          "non-contiguous input — callers should fall back to NumPy in "
          "that case, not treat this as a bug to work around here.");
}
