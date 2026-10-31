# Native kernels and CPU dispatch

`regresslens-accel` (built from `cpp/`) contains one extension module with two
kernel sets:

| translation unit | flags | contents |
|---|---|---|
| `src/kernels_scalar.cpp` | none (portable baseline) | compensated sum, projection, filter, rolling sum/mean (finite and NaN-segmented), rolling variance, input scan |
| `src/kernels_avx2.cpp` | `-mavx2 -mfma` / `/arch:AVX2`, x86-64 only | AVX2 sum (4x4-lane TwoSum), MT-AVX2 sum, FMA projection, AVX2 / MT-AVX2 filter, 4-lane compensated rolling, AVX2 scan |

`src/dispatch.cpp` picks the AVX2 set at import only when the CPU reports AVX2
and FMA and the OS saves YMM state: `__builtin_cpu_supports` on GCC/Clang,
CPUID + XGETBV on MSVC. The bindings and every other file compile without
architecture flags. `RGLNS_FORCE_KERNEL=scalar|avx2` forces one set; forcing
avx2 where it is unavailable fails the import with a clear message instead of
crashing later.

## Defects in 0.1.0.dev0, verified before any change

| defect | evidence (gcc, Linux) |
|---|---|
| NaN poisoning: the sliding and prefix-sum rolling kernels never recover from a NaN or Inf | one NaN in 40 values: 35 of 37 windows NaN; `np.convolve`: 4 |
| SIGILL: "scalar" code compiled with `-mavx2 -mfma` | `qemu-x86_64 -cpu Nehalem`: NumPy alone runs; `import regresslens` dies with exit 132 at import |
| confounded characterization: scalar loops auto-vectorized | the scalar projection template compiled with the extension's flags emits packed ymm FMA; with `-O3` alone, none |

## The COMDAT hazard

Splitting files is not enough. Suppose an inline function or template is
instantiated in both the AVX2 file and a baseline file, for example an
`std::vector<double>` method. The linker then keeps one copy, and if it keeps
the AVX2-compiled one, baseline code runs AVX instructions. So the AVX2 headers
use only `static inline` functions and C library calls, with no std containers
or algorithms (temporary memory is a malloc-backed `RawBuffer`).
`tools/check_isa_isolation.py` enforces this in CI: zero VEX instructions in
baseline objects, and zero weak code symbols shared with the AVX2 object. With
GCC 13+ the AVX2 object has exactly one weak symbol, libstdc++'s empty
`std::thread::_M_thread_deps_never_run()` (from the threaded kernels), which no
baseline object defines and which compiles to a bare `ret`. The `no-avx` CI job
runs the scalar tests and the Python smoke tests under `qemu -cpu Nehalem`.

## Accuracy-driven kernel design

See [accuracy.md](accuracy.md). In short, every sum is an error-free (s, c)
pair. Means and variances divide double-double values with a single rounding.
Rolling variance shifts by a recent window mean, re-anchors automatically when
cancellation would exceed 1e10, and returns exactly 0 for constant windows.
The AVX2 rolling kernel runs four compensated sliding windows in parallel
lanes. The old prefix-sum design cancelled large prefix values and was both
slower and less accurate.

## Characterization and selector thresholds

`bench/characterize` calls the shipped kernel tables (no architecture flags),
so "scalar" is now really scalar. Results are in
[`cpp/bench/results/`](../cpp/bench/results/); `bench/derive_thresholds.py`
turns them into decisions. A kernel is chosen where it is at least 5% faster at
that size and at every larger size:

| operation | decision | evidence (Core Ultra 7 255U, MSVC) |
|---|---|---|
| reduction | AVX2 always; MT from 4M | AVX2 3.2-4.1x vs scalar; MT 0.5x at 1M, 1.2x at 4M, 1.7-5.5x at 8M+ |
| projection | scalar | AVX2+FMA 0-46% faster, but not consistently 5% at DRAM sizes (f64 at 8M+: 1.00x) |
| filter | AVX2 at selectivity 0.3-0.7; MT from 1M in that band | AVX2 wins 100% of size/dtype combinations at 0.3-0.7, 0% at 0.9+ |
| rolling sum/mean | AVX2 from 100k | 1.4-2.4x from 100k; ~10% slower at 10k (float32) |
| rolling variance | scalar | only kernel |

The Phase 1 finding "projection: AVX2 doesn't beat scalar" was confounded:
its "scalar" baseline was auto-vectorized AVX2. With a true scalar baseline,
AVX2 projection is faster in cache and memory-bound in DRAM, but still not
reliably enough to switch.

Re-derive on new hardware: `characterize > results/<host>.csv && python
derive_thresholds.py results/<host>.csv`, then update `src/kernel_selector.cpp`
(the only file the selector policy lives in).

## Fifth-kernel gate (`diff`)

The plan builds a native `diff` kernel only if `np.diff` is more than 30% of
`diff_log`. `tools/kernel_gate.py` measures it on 1M float64 values:

| platform | raw diff share | compute-only share |
|---|---|---|
| Windows 11, NumPy 2.5 | 30-33% over runs | 8-9% |
| Linux (container), NumPy 2.4 | ~11.4% | lower |

On Windows the raw share passes the gate only because ~70% of `np.diff`'s time
is page-faulting a freshly allocated 8 MB output, which a native kernel would
pay just the same. The part a kernel could speed up, the subtraction itself, is
about 9% of `diff_log`. So no `diff` kernel was built; the gate tool reports
both shares and decides on the compute share.
