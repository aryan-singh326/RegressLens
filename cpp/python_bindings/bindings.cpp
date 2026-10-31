// Python bridge for the RegressLens native kernels
// (module regresslens_accel._native, shipped in the regresslens-accel
// wheel and loaded lazily by regresslens._accel).
//
// Compiled WITHOUT architecture flags. Every kernel call goes through
// a KernelSet chosen at runtime (see dispatch.hpp), so importing this
// module is safe on any CPU of the target architecture.
//
// Every operation returns the kernel actually used and the kernel's
// runtime in ns alongside the result. The timer wraps only the kernel
// itself: validation, selection and output allocation are the
// interception overhead, measured separately. The GIL is released
// while a kernel runs.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>

#include "regresslens/compensated.hpp"
#include "regresslens/cpu_features.hpp"
#include "regresslens/dispatch.hpp"
#include "regresslens/kernel_selector.hpp"
#include "regresslens/kernels.hpp"

namespace py = pybind11;
using namespace regresslens;

namespace regresslens {
int add_monitor_functions(PyObject* module);  // monitor.cpp
}

namespace {

// Bumped whenever the Python-visible signature of this module changes.
// regresslens checks it and falls back to NumPy on a mismatch rather
// than calling an incompatible extension.
constexpr int kAbiVersion = 2;

using Clock = std::chrono::steady_clock;
inline double elapsed_ns(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

struct Input {
    const void* ptr;
    size_t n;
    DType dtype;
};

// Accepts only 1-D, C-contiguous float32/float64 arrays. The Python
// layer routes everything else to its NumPy fallback; reaching this
// error means that routing has a bug.
Input get_input(const py::array& arr, const char* fn) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(fn) + ": expected a 1-D array");
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::invalid_argument(std::string(fn) +
                                    ": non-contiguous array (use the NumPy fallback)");
    }
    DType dtype;
    if (arr.dtype().is(py::dtype::of<double>())) {
        dtype = DType::Float64;
    } else if (arr.dtype().is(py::dtype::of<float>())) {
        dtype = DType::Float32;
    } else {
        throw std::invalid_argument(std::string(fn) + ": unsupported dtype " +
                                    std::string(py::str(arr.dtype())) +
                                    " (float32/float64 only)");
    }
    return Input{arr.data(), static_cast<size_t>(arr.shape(0)), dtype};
}

unsigned resolve_threads(int threads) {
    if (threads > 0) return static_cast<unsigned>(threads);
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1u : hw;
}

template <typename T>
const OpTable<T>& table_for(const KernelSet& set);
template <>
const OpTable<float>& table_for<float>(const KernelSet& set) { return set.f32; }
template <>
const OpTable<double>& table_for<double>(const KernelSet& set) { return set.f64; }

// --- Reduction ---

template <typename T>
py::tuple reduce_typed(const Input& in, KernelChoice choice, unsigned threads, bool mean) {
    const OpTable<T>& t = table_for<T>(kernel_set_for(choice));
    const T* p = static_cast<const T*>(in.ptr);
    SumParts parts{};
    auto t0 = Clock::now();
    {
        py::gil_scoped_release release;
        parts = (choice == KernelChoice::MtAvx2) ? t.sum_mt(p, in.n, threads)
                                                 : t.sum(p, in.n);
    }
    auto t1 = Clock::now();
    // The mean divides the compensated (double-double) sum, so it is
    // rounded once, not twice.
    const double value = mean ? mean_from_parts(parts, static_cast<double>(in.n))
                              : finish_sum(parts);
    return py::make_tuple(value, kernel_choice_name(choice), elapsed_ns(t0, t1));
}

py::tuple reduce(const py::array& arr, const std::string& kernel, int threads, bool mean) {
    Input in = get_input(arr, mean ? "reduce_mean" : "reduce_sum");
    if (mean && in.n == 0) throw std::invalid_argument("reduce_mean: empty array");
    unsigned th = resolve_threads(threads);
    SelectionContext ctx{Operation::Reduction, in.dtype, in.n, true, th};
    KernelChoice choice = resolve_kernel(ctx, kernel, /*op_has_mt=*/true);
    return in.dtype == DType::Float64 ? reduce_typed<double>(in, choice, th, mean)
                                      : reduce_typed<float>(in, choice, th, mean);
}

// --- Projection ---

template <typename T>
py::tuple project_typed(const Input& in, double scale, double offset,
                        KernelChoice choice) {
    const OpTable<T>& t = table_for<T>(kernel_set_for(choice));
    py::array_t<T> out(static_cast<py::ssize_t>(in.n));
    T* o = out.mutable_data();
    auto t0 = Clock::now();
    {
        py::gil_scoped_release release;
        t.project_affine(static_cast<const T*>(in.ptr), o, in.n, static_cast<T>(scale),
                         static_cast<T>(offset));
    }
    auto t1 = Clock::now();
    return py::make_tuple(out, kernel_choice_name(choice), elapsed_ns(t0, t1));
}

py::tuple project_affine(const py::array& arr, double scale, double offset,
                         const std::string& kernel) {
    Input in = get_input(arr, "project_affine");
    SelectionContext ctx{Operation::Projection, in.dtype, in.n, true,
                         resolve_threads(0)};
    KernelChoice choice = resolve_kernel(ctx, kernel, /*op_has_mt=*/false);
    return in.dtype == DType::Float64 ? project_typed<double>(in, scale, offset, choice)
                                      : project_typed<float>(in, scale, offset, choice);
}

// --- Filter ---

template <typename T>
py::tuple filter_typed(const Input& in, double threshold, KernelChoice choice,
                       unsigned threads) {
    const OpTable<T>& t = table_for<T>(kernel_set_for(choice));
    // Output size is data-dependent: allocate the worst case (every
    // element passes); the caller slices to [:count].
    py::array_t<T> out(static_cast<py::ssize_t>(in.n));
    T* o = out.mutable_data();
    const T* p = static_cast<const T*>(in.ptr);
    const T th = static_cast<T>(threshold);
    size_t count = 0;
    auto t0 = Clock::now();
    {
        py::gil_scoped_release release;
        count = (choice == KernelChoice::MtAvx2) ? t.filter_gt_mt(p, o, in.n, th, threads)
                                                 : t.filter_gt(p, o, in.n, th);
    }
    auto t1 = Clock::now();
    return py::make_tuple(out, count, kernel_choice_name(choice), elapsed_ns(t0, t1));
}

py::tuple filter_gt(const py::array& arr, double threshold, const std::string& kernel,
                    int threads, double estimated_selectivity) {
    Input in = get_input(arr, "filter_gt");
    unsigned th = resolve_threads(threads);
    SelectionContext ctx{Operation::Filter, in.dtype, in.n, true, th};
    // Outside [0, 1] (the Python layer passes -1) means "no history".
    ctx.estimated_selectivity =
        (estimated_selectivity >= 0.0 && estimated_selectivity <= 1.0)
            ? estimated_selectivity
            : 0.5;
    KernelChoice choice = resolve_kernel(ctx, kernel, /*op_has_mt=*/true);
    return in.dtype == DType::Float64 ? filter_typed<double>(in, threshold, choice, th)
                                      : filter_typed<float>(in, threshold, choice, th);
}

// --- Rolling ---

enum class RollingOp { Sum, Mean, Var };

// Finite inputs above this magnitude could overflow a window
// accumulator. They return None so the Python layer uses NumPy, whose
// overflow behavior then applies unchanged.
template <typename T>
bool overflow_risk(const ScanResult& scan, size_t window, RollingOp op) {
    if (sizeof(T) == sizeof(float)) return false;  // accumulated in double
    if (op == RollingOp::Var) return scan.absmax > 1e150;
    return scan.absmax > DBL_MAX / (4.0 * static_cast<double>(window));
}

template <typename T>
py::tuple rolling_typed(const Input& in, size_t window, size_t ddof, RollingOp op,
                        KernelChoice choice) {
    const OpTable<T>& t = table_for<T>(kernel_set_for(choice));
    const T* p = static_cast<const T*>(in.ptr);
    const size_t out_count = in.n - window + 1;
    py::array_t<T> out(static_cast<py::ssize_t>(out_count));
    T* o = out.mutable_data();

    const char* kernel_name = kernel_choice_name(choice);
    bool fallback = false;
    auto t0 = Clock::now();
    {
        py::gil_scoped_release release;
        // The one-pass scan is part of the kernel's cost, so it is
        // inside the timed region.
        ScanResult scan = t.scan(p, in.n);
        if (overflow_risk<T>(scan, window, op)) {
            fallback = true;
        } else if (op == RollingOp::Var) {
            rolling_var(p, o, in.n, window, ddof);
            kernel_name = "scalar";
        } else if (scan.nonfinite) {
            if (op == RollingOp::Sum) {
                rolling_sum_nonfinite(p, o, in.n, window);
            } else {
                rolling_mean_nonfinite(p, o, in.n, window);
            }
            kernel_name = "scalar_nf";
        } else if (op == RollingOp::Sum) {
            t.rolling_sum(p, o, in.n, window);
        } else {
            t.rolling_mean(p, o, in.n, window);
        }
    }
    auto t1 = Clock::now();
    if (fallback) {
        return py::make_tuple(py::none(), "numpy_fallback", elapsed_ns(t0, t1));
    }
    return py::make_tuple(out, kernel_name, elapsed_ns(t0, t1));
}

py::tuple rolling(const py::array& arr, size_t window, size_t ddof, RollingOp op,
                  const std::string& kernel, const char* fn) {
    Input in = get_input(arr, fn);
    if (window == 0) throw std::invalid_argument(std::string(fn) + ": window must be >= 1");
    if (in.n < window) {
        // No complete window: empty result, no kernel ran.
        py::array out = in.dtype == DType::Float64
                            ? py::array(py::array_t<double>(0))
                            : py::array(py::array_t<float>(0));
        return py::make_tuple(out, "none", 0.0);
    }
    SelectionContext ctx{op == RollingOp::Var ? Operation::RollingVar : Operation::Rolling,
                         in.dtype, in.n, true, resolve_threads(0)};
    ctx.window = window;
    KernelChoice choice = op == RollingOp::Var
                              ? KernelChoice::Scalar
                              : resolve_kernel(ctx, kernel, /*op_has_mt=*/false);
    return in.dtype == DType::Float64
               ? rolling_typed<double>(in, window, ddof, op, choice)
               : rolling_typed<float>(in, window, ddof, op, choice);
}

py::dict info() {
    const Runtime& rt = runtime();
    py::dict d;
    d["abi_version"] = kAbiVersion;
    d["avx2_compiled"] = rt.avx2_compiled;
    d["cpu_avx2_fma"] = rt.cpu_avx2;
    d["avx2_active"] = rt.avx2 != nullptr;
    d["forced_kernel"] = rt.forced == ForcedKernel::Scalar ? "scalar"
                         : rt.forced == ForcedKernel::Avx2 ? "avx2"
                                                           : "";
    d["isa"] = rt.avx2 != nullptr ? "avx2" : "scalar";
    return d;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "RegressLens native kernels (internal: use the regresslens package)";

    const Runtime& rt = runtime();
    if (!rt.error.empty()) {
        throw py::import_error("regresslens_accel: " + rt.error);
    }

    m.attr("ABI_VERSION") = kAbiVersion;
    // sys.monitoring fast path for `rglns run` (raw METH_FASTCALL functions).
    if (regresslens::add_monitor_functions(m.ptr()) < 0) throw py::error_already_set();

    m.def("info", &info, "Build and runtime kernel-dispatch information.");

    m.def(
        "reduce_sum",
        [](const py::array& a, const std::string& kernel, int threads) {
            return reduce(a, kernel, threads, false);
        },
        py::arg("array"), py::arg("kernel") = "auto", py::arg("threads") = 0,
        "Compensated sum of a 1-D contiguous float32/float64 array, accumulated in "
        "double. Returns (value, kernel, runtime_ns).");

    m.def(
        "reduce_mean",
        [](const py::array& a, const std::string& kernel, int threads) {
            return reduce(a, kernel, threads, true);
        },
        py::arg("array"), py::arg("kernel") = "auto", py::arg("threads") = 0,
        "Mean (compensated sum divided with a single rounding). Returns "
        "(value, kernel, runtime_ns).");

    m.def("project_affine", &project_affine, py::arg("array"), py::arg("scale"),
          py::arg("offset"), py::arg("kernel") = "auto",
          "out = scale * x + offset. Returns (out, kernel, runtime_ns).");

    m.def("filter_gt", &filter_gt, py::arg("array"), py::arg("threshold"),
          py::arg("kernel") = "auto", py::arg("threads") = 0,
          py::arg("estimated_selectivity") = -1.0,
          "Elements strictly greater than threshold, in order. Returns "
          "(capacity_n_array, count, kernel, runtime_ns); slice to [:count].");

    m.def(
        "rolling_sum",
        [](const py::array& a, size_t window, const std::string& kernel) {
            return rolling(a, window, 0, RollingOp::Sum, kernel, "rolling_sum");
        },
        py::arg("array"), py::arg("window"), py::arg("kernel") = "auto",
        "Rolling window sum (n - window + 1 outputs). Returns (out | None, kernel, "
        "runtime_ns); None means use the NumPy fallback (overflow risk).");

    m.def(
        "rolling_mean",
        [](const py::array& a, size_t window, const std::string& kernel) {
            return rolling(a, window, 0, RollingOp::Mean, kernel, "rolling_mean");
        },
        py::arg("array"), py::arg("window"), py::arg("kernel") = "auto",
        "Rolling window mean. Same return convention as rolling_sum.");

    m.def(
        "rolling_var",
        [](const py::array& a, size_t window, size_t ddof) {
            return rolling(a, window, ddof, RollingOp::Var, "scalar", "rolling_var");
        },
        py::arg("array"), py::arg("window"), py::arg("ddof") = 0,
        "Rolling window variance (np.var semantics with ddof). Same return "
        "convention as rolling_sum.");

    m.def("cpu_has_avx2_fma", &cpu_has_avx2_fma);
}
