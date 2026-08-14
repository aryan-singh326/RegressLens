// Python bridge for RegressLens native kernels. Starts with ONE
// operation (reduction) end-to-end, same discipline as Phase 1:
// prove the mechanism works before replicating it three more times
// for projection/filter/rolling.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <stdexcept>
#include <thread>

#include "regresslens/filter_avx2.hpp"
#include "regresslens/filter_mt_avx2.hpp"
#include "regresslens/filter_scalar.hpp"
#include "regresslens/kernel_selector.hpp"
#include "regresslens/projection_avx2.hpp"
#include "regresslens/projection_scalar.hpp"
#include "regresslens/reduction_avx2.hpp"
#include "regresslens/reduction_mt_avx2.hpp"
#include "regresslens/reduction_scalar.hpp"
#include "regresslens/rolling_avx2.hpp"
#include "regresslens/rolling_scalar.hpp"

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

struct ArrayInfo {
    py::buffer_info info;
    bool contiguous;
    DType dtype;
    size_t n;
};

// Shared by all four operation bindings — extracts and validates the
// dtype/contiguity/size info every one of them needs, so that logic
// exists in exactly one place rather than four slightly-different
// copies.
ArrayInfo get_array_info(py::array& arr, const char* fn_name) {
    py::buffer_info info = arr.request();
    bool contiguous = (info.strides.size() == 1 &&
                        info.strides[0] == static_cast<py::ssize_t>(info.itemsize));
    if (!contiguous) {
        throw std::invalid_argument(
            std::string(fn_name) +
            ": non-contiguous array. The Python layer should route this "
            "to NumPy fallback, not call this function directly.");
    }

    std::string dtype_str = py::str(arr.dtype());
    DType dtype;
    if (dtype_str.find("float64") != std::string::npos) {
        dtype = DType::Float64;
    } else if (dtype_str.find("float32") != std::string::npos) {
        dtype = DType::Float32;
    } else {
        throw std::invalid_argument(std::string(fn_name) + ": unsupported dtype '" +
                                     dtype_str +
                                     "'. v0.1 supports float32 and float64 only.");
    }

    size_t n = static_cast<size_t>(info.shape.empty() ? 0 : info.shape[0]);
    return ArrayInfo{std::move(info), contiguous, dtype, n};
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

using Clock = std::chrono::steady_clock;
inline double elapsed_ns(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

// The one function exposed for reduction. Returns (value, kernel_name,
// runtime_ns) rather than just the value, because Phase 3's trace
// persistence needs to know which kernel ran and how long it took
// for every call — this is that observability hook, exposed from the
// start rather than retrofitted. Timing wraps ONLY the kernel
// dispatch, not array validation/dtype detection/selection — those
// are the "interception overhead" the project brief tracks
// separately (see cpp/bench/characterize.cpp), not part of the
// kernel's own runtime.
py::tuple reduce_sum_native(py::array arr, int threads_override) {
    ArrayInfo ai = get_array_info(arr, "reduce_sum_native");

    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;
    unsigned threads = threads_override > 0 ? static_cast<unsigned>(threads_override)
                                             : hw_threads;

    SelectionContext ctx{Operation::Reduction, ai.dtype, ai.n, ai.contiguous, threads};
    KernelChoice choice = select_kernel(ctx);

    auto t0 = Clock::now();
    double result;
    if (ai.dtype == DType::Float64) {
        result = dispatch_reduce_sum<double>(static_cast<const double*>(ai.info.ptr),
                                              ai.n, choice, threads);
    } else {
        result = dispatch_reduce_sum<float>(static_cast<const float*>(ai.info.ptr),
                                             ai.n, choice, threads);
    }
    auto t1 = Clock::now();

    return py::make_tuple(result, kernel_choice_name(choice), elapsed_ns(t0, t1));
}

// --- Projection ---

template <typename T>
void dispatch_project_affine(const T* in, T* out, size_t n, T scale, T offset,
                              KernelChoice choice) {
    switch (choice) {
        case KernelChoice::Scalar:
            project_affine_scalar<T>(in, out, n, scale, offset);
            return;
        case KernelChoice::Avx2:
            project_affine_avx2(in, out, n, scale, offset);
            return;
        case KernelChoice::MtAvx2:
            // No MT variant exists for projection — it's memory-
            // bandwidth-bound, so the placeholder heuristic never
            // selects MtAvx2 for this op (see kernel_selector.cpp).
            // Falling through to AVX2 here is defensive, not
            // expected to ever execute.
            project_affine_avx2(in, out, n, scale, offset);
            return;
    }
}

py::tuple project_affine_native(py::array arr, double scale, double offset) {
    ArrayInfo ai = get_array_info(arr, "project_affine_native");
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;

    SelectionContext ctx{Operation::Projection, ai.dtype, ai.n, ai.contiguous,
                          hw_threads};
    KernelChoice choice = select_kernel(ctx);

    if (ai.dtype == DType::Float64) {
        py::array_t<double> out(ai.n);
        auto t0 = Clock::now();
        dispatch_project_affine<double>(static_cast<const double*>(ai.info.ptr),
                                         out.mutable_data(), ai.n, scale, offset,
                                         choice);
        auto t1 = Clock::now();
        return py::make_tuple(out, kernel_choice_name(choice), elapsed_ns(t0, t1));
    } else {
        py::array_t<float> out(ai.n);
        auto t0 = Clock::now();
        dispatch_project_affine<float>(static_cast<const float*>(ai.info.ptr),
                                        out.mutable_data(), ai.n,
                                        static_cast<float>(scale),
                                        static_cast<float>(offset), choice);
        auto t1 = Clock::now();
        return py::make_tuple(out, kernel_choice_name(choice), elapsed_ns(t0, t1));
    }
}

// --- Filter ---

template <typename T>
size_t dispatch_filter_gt(const T* in, T* out, size_t n, T threshold,
                           KernelChoice choice, unsigned threads) {
    switch (choice) {
        case KernelChoice::Scalar:
            return filter_gt_scalar<T>(in, out, n, threshold);
        case KernelChoice::Avx2:
            return filter_gt_avx2(in, out, n, threshold);
        case KernelChoice::MtAvx2:
            return filter_gt_mt_avx2<T>(in, out, n, threshold, threads);
    }
    throw std::logic_error("unreachable: unknown KernelChoice");
}

py::tuple filter_gt_native(py::array arr, double threshold, int threads_override) {
    ArrayInfo ai = get_array_info(arr, "filter_gt_native");
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;
    unsigned threads = threads_override > 0 ? static_cast<unsigned>(threads_override)
                                             : hw_threads;

    SelectionContext ctx{Operation::Filter, ai.dtype, ai.n, ai.contiguous, threads};
    // v0.1 has no runtime selectivity estimate yet — the placeholder
    // heuristic doesn't currently branch on it (see
    // kernel_selector.cpp), but the field is populated honestly
    // as "unknown/assumed" rather than silently defaulted, so this
    // is the one place that assumption is visible.
    ctx.estimated_selectivity = 0.5;
    KernelChoice choice = select_kernel(ctx);

    if (ai.dtype == DType::Float64) {
        py::array_t<double> out(ai.n);  // worst case: everything passes
        auto t0 = Clock::now();
        size_t count = dispatch_filter_gt<double>(
            static_cast<const double*>(ai.info.ptr), out.mutable_data(), ai.n,
            threshold, choice, threads);
        auto t1 = Clock::now();
        return py::make_tuple(out, count, kernel_choice_name(choice),
                               elapsed_ns(t0, t1));
    } else {
        py::array_t<float> out(ai.n);
        auto t0 = Clock::now();
        size_t count = dispatch_filter_gt<float>(
            static_cast<const float*>(ai.info.ptr), out.mutable_data(), ai.n,
            static_cast<float>(threshold), choice, threads);
        auto t1 = Clock::now();
        return py::make_tuple(out, count, kernel_choice_name(choice),
                               elapsed_ns(t0, t1));
    }
}

// --- Rolling ---

template <typename T>
size_t dispatch_rolling_sum(const T* in, T* out, size_t n, size_t window,
                             KernelChoice choice) {
    switch (choice) {
        case KernelChoice::Scalar:
            return rolling_sum_scalar<T>(in, out, n, window);
        case KernelChoice::Avx2:
        case KernelChoice::MtAvx2:
            // No MT variant exists for rolling. AVX2 requested but
            // characterization showed scalar wins outright for this
            // op — the placeholder heuristic never selects Avx2 or
            // MtAvx2 here (see kernel_selector.cpp). This branch is
            // defensive dead code, kept correct rather than removed,
            // in case the heuristic changes before this comment does.
            return rolling_sum_avx2(in, out, n, window);
    }
    throw std::logic_error("unreachable: unknown KernelChoice");
}

py::tuple rolling_sum_native(py::array arr, size_t window) {
    ArrayInfo ai = get_array_info(arr, "rolling_sum_native");
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;

    SelectionContext ctx{Operation::Rolling, ai.dtype, ai.n, ai.contiguous,
                          hw_threads};
    ctx.window = window;
    KernelChoice choice = select_kernel(ctx);

    size_t out_count = (window == 0 || ai.n < window) ? 0 : ai.n - window + 1;

    if (ai.dtype == DType::Float64) {
        py::array_t<double> out(out_count);
        auto t0 = Clock::now();
        size_t count = dispatch_rolling_sum<double>(
            static_cast<const double*>(ai.info.ptr), out.mutable_data(), ai.n,
            window, choice);
        auto t1 = Clock::now();
        (void)count;  // == out_count by construction; kept for symmetry/clarity
        return py::make_tuple(out, kernel_choice_name(choice), elapsed_ns(t0, t1));
    } else {
        py::array_t<float> out(out_count);
        auto t0 = Clock::now();
        size_t count = dispatch_rolling_sum<float>(
            static_cast<const float*>(ai.info.ptr), out.mutable_data(), ai.n,
            window, choice);
        auto t1 = Clock::now();
        (void)count;
        return py::make_tuple(out, kernel_choice_name(choice), elapsed_ns(t0, t1));
    }
}

}  // namespace

PYBIND11_MODULE(_regresslens_native, m) {
    m.doc() = "RegressLens native kernel bindings (internal — use the "
              "regresslens Python package, not this module directly)";

    m.def("reduce_sum_native", &reduce_sum_native, py::arg("array"),
          py::arg("threads_override") = -1,
          "Sum-reduce a contiguous float32/float64 NumPy array using the "
          "kernel selector. Returns (value, kernel_name_used, "
          "runtime_ns). Raises on non-contiguous input — callers should "
          "fall back to NumPy in that case, not treat this as a bug to "
          "work around here.");

    m.def("project_affine_native", &project_affine_native, py::arg("array"),
          py::arg("scale"), py::arg("offset"),
          "Elementwise out = scale*in + offset on a contiguous "
          "float32/float64 NumPy array. Returns (result_array, "
          "kernel_name_used, runtime_ns). Raises on non-contiguous "
          "input.");

    m.def("filter_gt_native", &filter_gt_native, py::arg("array"),
          py::arg("threshold"), py::arg("threads_override") = -1,
          "Threshold filter (strictly greater than) on a contiguous "
          "float32/float64 NumPy array. Returns (full_capacity_array, "
          "valid_count, kernel_name_used, runtime_ns) — caller must "
          "slice to [:valid_count]; the returned array is allocated at "
          "worst-case size (n) since filter's output size is "
          "data-dependent and unknown before the kernel runs. Raises on "
          "non-contiguous input.");

    m.def("rolling_sum_native", &rolling_sum_native, py::arg("array"),
          py::arg("window"),
          "Fixed-window rolling sum on a contiguous float32/float64 "
          "NumPy array. Returns (result_array, kernel_name_used, "
          "runtime_ns), where result_array has length "
          "max(0, n - window + 1). Raises on non-contiguous input.");
}
