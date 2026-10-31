// Fast sys.monitoring callbacks for `rglns run` (Python 3.12+).
//
// The pure-Python backend (regresslens/observe/backend_monitoring.py)
// spends ~1.5 us of bookkeeping per observed call, several times the
// cost of a tiny NumPy call. When regresslens-accel is installed, the
// same backend registers these C callbacks instead:
//
//   mon_call(code, offset, callable, arg0)    CALL
//   mon_c_return(code, offset, callable, arg0) C_RETURN
//   mon_c_raise(code, offset, callable, arg0)  C_RAISE
//   mon_py_return(code, offset, retval)        PY_RETURN (Python-level NumPy functions)
//
// Only the per-call hot path is in C: location lookup, array
// description, timing and aggregation (count, sum, sum of logs, 64-sample
// reservoir). Everything that happens once per location (classifying the
// callable, resolving the call site) stays in Python: the first CALL at a
// new location calls the registered Python `slow_path`, which returns
// either None (return DISABLE: CPython never fires at that location again)
// or how to handle the location.
//
// Arrays are inspected through NumPy's PyArrayObject header (data, nd,
// dimensions, strides, base, descr, flags), whose layout has been stable
// since NumPy 1.7 and is unchanged in 2.x, so no NumPy headers are
// needed at build time. All callbacks run with the GIL held (sys.monitoring
// guarantees it), which protects the shared tables; the call stacks are
// per thread because calls on different threads interleave.
//
// Compiled without architecture flags, like the rest of the bindings.
// pybind11 includes Python.h the safe way: with _DEBUG hidden, so a
// Debug build of this extension still links against a release Python.
#include <pybind11/pybind11.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ArrayHead {  // NumPy's PyArrayObject_fields prefix
    PyObject_HEAD
    char* data;
    int nd;
    Py_ssize_t* dimensions;
    Py_ssize_t* strides;
    PyObject* base;
    PyObject* descr;
    int flags;
};
constexpr int kNpyCContiguous = 0x0001;
constexpr int kReservoir = 64;

enum Kind { kFixed = 0, kMethod = 1, kPyFunc = 2 };

struct Location {
    PyObject* code;      // strong ref
    int offset;
    int site;            // Python-side site index
    int kind;
    PyObject* fixed;     // strong ref: the callable this location calls (kFixed/kPyFunc)
    PyObject* pycode;    // strong ref: callee code object (kPyFunc)
    std::string method;  // method name (kMethod)
    int op;              // Python-side operator index
};

struct Entry {
    PyObject* code;
    int offset;
    int loc;
    int dtype;
    long long n;
    int contig;
    PyObject* pycode;
    std::int64_t t0;
};

struct BucketKey {
    int loc, dtype, contig;
    long long n;
    bool operator==(const BucketKey& o) const {
        return loc == o.loc && dtype == o.dtype && contig == o.contig && n == o.n;
    }
};
struct BucketHash {
    size_t operator()(const BucketKey& k) const {
        std::uint64_t h = static_cast<std::uint64_t>(k.loc) * 0x9E3779B97F4A7C15ull;
        h ^= static_cast<std::uint64_t>(k.n) + 0x7F4A7C15ull + (h << 6) + (h >> 2);
        h ^= static_cast<std::uint64_t>(k.dtype * 3 + k.contig) + (h << 6) + (h >> 2);
        return static_cast<size_t>(h);
    }
};

struct Bucket {
    long long count = 0;
    double sum = 0.0;
    double sum_log = 0.0;
    std::vector<double> samples;
};

struct LocKey {
    PyObject* code;
    int offset;
    bool operator==(const LocKey& o) const { return code == o.code && offset == o.offset; }
};
struct LocHash {
    size_t operator()(const LocKey& k) const {
        return std::hash<const void*>()(k.code) ^ (static_cast<size_t>(k.offset) * 0x9E3779B1u);
    }
};

struct State {
    PyObject* slow_path = nullptr;  // Python callable
    PyObject* disable = nullptr;    // sys.monitoring.DISABLE
    PyTypeObject* ndarray = nullptr;
    std::unordered_map<LocKey, int, LocHash> loc_index;  // -1: disabled
    std::vector<Location> locs;
    std::unordered_map<PyObject*, int> dtype_index;     // descr -> index (strong refs in dtypes)
    std::vector<PyObject*> dtypes;
    std::unordered_map<BucketKey, Bucket, BucketHash> buckets;
    long long calls = 0;
    std::int64_t outer_ns = 0;
    std::uint64_t rng = 0x2545F4914F6CDD1Dull;
};
State g;

struct ThreadStack {
    std::vector<Entry> stack;
};
thread_local ThreadStack t_stack;

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline std::uint64_t next_rng() {  // xorshift64*
    g.rng ^= g.rng >> 12;
    g.rng ^= g.rng << 25;
    g.rng ^= g.rng >> 27;
    return g.rng * 0x2545F4914F6CDD1Dull;
}

int dtype_of(PyObject* descr) {
    auto it = g.dtype_index.find(descr);
    if (it != g.dtype_index.end()) return it->second;
    Py_INCREF(descr);
    int idx = static_cast<int>(g.dtypes.size());
    g.dtypes.push_back(descr);
    g.dtype_index.emplace(descr, idx);
    return idx;
}

void describe(PyObject* obj, int& dtype, long long& n, int& contig) {
    dtype = -1;
    n = -1;
    contig = -1;
    if (g.ndarray == nullptr || obj == nullptr || !PyObject_TypeCheck(obj, g.ndarray)) return;
    const ArrayHead* a = reinterpret_cast<const ArrayHead*>(obj);
    long long size = 1;
    for (int i = 0; i < a->nd; ++i) size *= static_cast<long long>(a->dimensions[i]);
    n = size;
    contig = (a->flags & kNpyCContiguous) ? 1 : 0;
    dtype = dtype_of(a->descr);
}

// Calls the Python slow path for a new (or polymorphic) location.
// Returns the location index, or -1 (disable / do not observe).
int resolve_location(PyObject* code, int offset, PyObject* callable, bool remember) {
    PyObject* res = PyObject_CallFunction(g.slow_path, "OiO", code, offset, callable);
    if (res == nullptr) {
        PyErr_Clear();  // never break the user's program
        if (remember) g.loc_index.emplace(LocKey{code, offset}, -1);
        return -1;
    }
    int idx = -1;
    if (res != Py_None && PyTuple_Check(res) && PyTuple_GET_SIZE(res) == 6) {
        Location loc;
        loc.code = code;
        Py_INCREF(code);
        loc.offset = offset;
        loc.site = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(res, 0)));
        loc.kind = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(res, 1)));
        loc.op = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(res, 5)));
        PyObject* fixed = PyTuple_GET_ITEM(res, 2);
        PyObject* pycode = PyTuple_GET_ITEM(res, 3);
        PyObject* name = PyTuple_GET_ITEM(res, 4);
        loc.fixed = fixed == Py_None ? nullptr : (Py_INCREF(fixed), fixed);
        loc.pycode = pycode == Py_None ? nullptr : (Py_INCREF(pycode), pycode);
        if (name != Py_None) {
            const char* s = PyUnicode_AsUTF8(name);
            loc.method = s ? s : "";
        }
        if (PyErr_Occurred()) PyErr_Clear();
        idx = static_cast<int>(g.locs.size());
        g.locs.push_back(std::move(loc));
    }
    Py_DECREF(res);
    if (remember) g.loc_index[LocKey{code, offset}] = idx;
    return idx;
}

PyObject* mon_call(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4) Py_RETURN_NONE;
    PyObject* code = args[0];
    int offset = static_cast<int>(PyLong_AsLong(args[1]));
    PyObject* callable = args[2];
    PyObject* arg0 = args[3];

    int idx;
    auto it = g.loc_index.find(LocKey{code, offset});
    if (it == g.loc_index.end()) {
        idx = resolve_location(code, offset, callable, true);
        if (idx < 0) {
            Py_INCREF(g.disable);
            return g.disable;  // CPython stops firing CALL here
        }
    } else {
        idx = it->second;
        if (idx < 0) Py_RETURN_NONE;
    }
    const Location* loc = &g.locs[idx];
    PyObject* array_obj = arg0;
    if (loc->kind == kMethod) {
        if (!PyCFunction_Check(callable) ||
            std::strcmp(reinterpret_cast<PyCFunctionObject*>(callable)->m_ml->ml_name,
                        loc->method.c_str()) != 0) {
            Py_RETURN_NONE;  // same location, a different method: not observed
        }
        array_obj = PyCFunction_GET_SELF(callable);
    } else if (callable != loc->fixed) {
        // Polymorphic location: classify this callable (not cached).
        idx = resolve_location(code, offset, callable, false);
        if (idx < 0) Py_RETURN_NONE;
        loc = &g.locs[idx];
    }
    Entry e;
    e.code = code;
    e.offset = offset;
    e.loc = idx;
    describe(array_obj, e.dtype, e.n, e.contig);
    e.pycode = loc->kind == kPyFunc ? loc->pycode : nullptr;
    e.t0 = now_ns();
    t_stack.stack.push_back(e);
    Py_RETURN_NONE;
}

void record(const Entry& e, std::int64_t t1, bool outermost) {
    double ns = static_cast<double>(t1 - e.t0);
    if (ns < 1.0) ns = 1.0;
    Bucket& b = g.buckets[BucketKey{e.loc, e.dtype, e.contig, e.n}];
    b.count += 1;
    b.sum += ns;
    b.sum_log += std::log(ns);
    if (static_cast<int>(b.samples.size()) < kReservoir) {
        b.samples.push_back(ns);
    } else {
        std::uint64_t j = next_rng() % static_cast<std::uint64_t>(b.count);
        if (j < static_cast<std::uint64_t>(kReservoir)) b.samples[j] = ns;
    }
    g.calls += 1;
    if (outermost) g.outer_ns += static_cast<std::int64_t>(ns);
}

void finish(PyObject* code, int offset, bool do_record) {
    std::int64_t t1 = now_ns();
    auto& st = t_stack.stack;
    if (st.empty()) return;
    size_t i = st.size();
    // The matching entry is normally on top; a stranded entry (a
    // Python-level callee that raised, so no PY_RETURN came) may sit
    // above it.
    while (i > 0 && !(st[i - 1].code == code && st[i - 1].offset == offset &&
                      st[i - 1].pycode == nullptr)) {
        if (st.size() - i >= 8) return;
        --i;
    }
    if (i == 0) return;
    Entry e = st[i - 1];
    st.resize(i - 1);
    if (do_record) record(e, t1, st.empty());
}

PyObject* mon_c_return(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs >= 2) finish(args[0], static_cast<int>(PyLong_AsLong(args[1])), true);
    Py_RETURN_NONE;
}

PyObject* mon_c_raise(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs >= 2) finish(args[0], static_cast<int>(PyLong_AsLong(args[1])), false);
    Py_RETURN_NONE;
}

PyObject* mon_py_return(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    std::int64_t t1 = now_ns();
    auto& st = t_stack.stack;
    if (nargs >= 1 && !st.empty() && st.back().pycode == args[0]) {
        Entry e = st.back();
        st.pop_back();
        record(e, t1, st.empty());
    }
    Py_RETURN_NONE;
}

// mon_init(slow_path, disable): (re)initializes the tables.
PyObject* mon_init(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "mon_init(slow_path, disable)");
        return nullptr;
    }
    Py_XDECREF(g.slow_path);
    Py_XDECREF(g.disable);
    g.slow_path = args[0];
    g.disable = args[1];
    Py_INCREF(g.slow_path);
    Py_INCREF(g.disable);
    g.loc_index.clear();
    g.buckets.clear();
    g.calls = 0;
    g.outer_ns = 0;
    t_stack.stack.clear();
    Py_RETURN_NONE;
}

// mon_set_ndarray(numpy.ndarray): enables array description.
PyObject* mon_set_ndarray(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1 || !PyType_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "mon_set_ndarray(type)");
        return nullptr;
    }
    Py_INCREF(args[0]);
    g.ndarray = reinterpret_cast<PyTypeObject*>(args[0]);
    Py_RETURN_NONE;
}

// mon_drain() -> (rows, dtypes, calls, outer_ns); clears the buckets.
// rows: (site, op, dtype_index|-1, n|-1, contig|-1, count, sum, sum_log, [samples])
PyObject* mon_drain(PyObject*, PyObject* const*, Py_ssize_t) {
    PyObject* rows = PyList_New(0);
    if (!rows) return nullptr;
    for (const auto& kv : g.buckets) {
        const Location& loc = g.locs[kv.first.loc];
        const Bucket& b = kv.second;
        PyObject* samples = PyList_New(static_cast<Py_ssize_t>(b.samples.size()));
        for (size_t i = 0; i < b.samples.size(); ++i) {
            PyList_SET_ITEM(samples, static_cast<Py_ssize_t>(i), PyFloat_FromDouble(b.samples[i]));
        }
        PyObject* row = Py_BuildValue("(iiiLiLddN)", loc.site, loc.op, kv.first.dtype,
                                      kv.first.n, kv.first.contig, b.count, b.sum, b.sum_log,
                                      samples);
        if (!row || PyList_Append(rows, row) < 0) {
            Py_XDECREF(row);
            Py_DECREF(rows);
            return nullptr;
        }
        Py_DECREF(row);
    }
    PyObject* dtypes = PyList_New(static_cast<Py_ssize_t>(g.dtypes.size()));
    for (size_t i = 0; i < g.dtypes.size(); ++i) {
        Py_INCREF(g.dtypes[i]);
        PyList_SET_ITEM(dtypes, static_cast<Py_ssize_t>(i), g.dtypes[i]);
    }
    PyObject* out = Py_BuildValue("(NNLL)", rows, dtypes, g.calls,
                                  static_cast<long long>(g.outer_ns));
    g.buckets.clear();
    g.calls = 0;
    g.outer_ns = 0;
    return out;
}

PyMethodDef kMonitorMethods[] = {
    {"mon_call", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(mon_call)),
     METH_FASTCALL, "sys.monitoring CALL callback (rglns run fast path)"},
    {"mon_c_return",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(mon_c_return)),
     METH_FASTCALL, "sys.monitoring C_RETURN callback"},
    {"mon_c_raise", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(mon_c_raise)),
     METH_FASTCALL, "sys.monitoring C_RAISE callback"},
    {"mon_py_return",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(mon_py_return)),
     METH_FASTCALL, "sys.monitoring PY_RETURN callback"},
    {"mon_init", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(mon_init)),
     METH_FASTCALL, "mon_init(slow_path, disable)"},
    {"mon_set_ndarray",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(mon_set_ndarray)),
     METH_FASTCALL, "mon_set_ndarray(numpy.ndarray)"},
    {"mon_drain", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(mon_drain)),
     METH_FASTCALL, "mon_drain() -> (rows, dtypes, calls, outer_ns)"},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

namespace regresslens {

int add_monitor_functions(PyObject* module) {
    return PyModule_AddFunctions(module, kMonitorMethods);
}

}  // namespace regresslens
