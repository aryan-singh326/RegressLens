# Observing unmodified NumPy code: `rglns run`

```
rglns run [--label X] [--runs N] [--accelerate] [--observe-operators] -- python script.py
rglns run -- python -m mypackage.job
```

## How it works

The launcher prepends `regresslens/_boot` to `PYTHONPATH` and sets
`RGLNS_ACTIVE=1`. Python imports `_boot/sitecustomize.py` at start-up, in the
measured process and in every Python process it starts, and that file:

1. installs an observation backend (below);
2. chains to the `sitecustomize` it shadows (conda and many virtualenvs ship
   one), so that one still runs;
3. registers an exit handler that prints a coverage report to **stderr**
   (stdout stays byte-identical, which is tested) and writes the aggregated
   calls in one transaction.

Nothing is written per call. Per call site and argument shape (operator,
dtype, n, contiguity), the process keeps count, sum of ns, sum of log ns and
a 64-sample reservoir.

None of this imports NumPy. A program that sets `OMP_NUM_THREADS` (or anything
else) before its own `import numpy` behaves the same under `rglns run`, which
is tested. Calls made while NumPy is still importing are ignored, because its
partially initialized module cannot be inspected safely. If an observer
callback hits an internal error, observation stops with one line on stderr and
the program continues; the exception never reaches the program.

`python -S`, `python -I` and embedded interpreters skip `sitecustomize`, so
those are not observed.

## Backends

**Python 3.12+: `sys.monitoring` (PEP 669).** The CALL callback classifies the
callable. At a location that is not user code or not NumPy-related it returns
`DISABLE`, and CPython never fires CALL there again. Timing comes from C_RETURN
/ C_RAISE for C-level callables, and from PY_RETURN, enabled only on those code
objects, for the few NumPy functions written in Python. With
`regresslens[accel]` installed, the per-call callbacks are C functions
(`cpp/python_bindings/monitor.cpp`). Only once-per-location work calls back
into Python.

**Python 3.9-3.11: `sys.setprofile` + `threading.setprofile`.** This backend is
slower, because the hook runs on every call in the program, and it cannot see
ufunc objects.

## What each backend observes

Confirmed on CPython 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14, with current NumPy
and with NumPy 1.22 and 1.24 (`tests/fixtures/eight_forms.py`,
`tests/test_observe.py`). Before 1.25, NumPy implements `np.sum`,
`np.concatenate` and the other dispatched functions as Python code in a
pseudo-file, `<__array_function__ internals>`. Observation treats it as NumPy;
treating it as user code (as any other `<...>` file) would attribute every such
call to NumPy's own dispatcher instead of the line that made it.

| call form | 3.12+ `sys.monitoring` | 3.9-3.11 `setprofile` | with `--observe-operators` |
|---|---|---|---|
| `np.sum(x)` | yes | yes | yes |
| `x.sum()` | yes | yes | yes |
| `np.convolve(...)` | yes | yes | yes |
| `np.add(a, b)` (ufunc object) | yes | **no** | as base backend |
| `a + b` | no | no | **yes** |
| `x[mask]` | no | no | **yes** |
| `np.mean(x, axis=1)` | yes | yes | yes |
| call in a thread | yes | yes | yes |

The coverage report at exit names what was not observable, and reports the
share of wall time spent inside observed calls. It never implies coverage it
does not have:

```
rglns: observed 80 NumPy calls at 4 call sites [sys.monitoring (C fast path)]
rglns: 12.7% of wall time (0.008 s of 0.066 s) was inside observed calls
rglns: not observed: arithmetic operators (a + b), indexing (x[mask]), non-NumPy libraries
```

## `--observe-operators` (Tier D)

An import hook rewrites the AST of the user's own modules:
`a + b` becomes `__rglns_binop__(SITE, "add", a, b)` and `x[k]` becomes
`__rglns_getitem__(SITE, x, k)`. Both evaluate exactly `operator.add` and
`operator.getitem`, and time only when an ndarray is involved. Rewritten code
never touches the .pyc cache. The main script is run through
`python -m regresslens.observe.tierd`. Expect noticeably more overhead, since
every operator in user code becomes a call.

## `--accelerate`

Replaces only `numpy.sum` and `numpy.mean` (ndarray is an immutable C type, so
`x.sum()` cannot be patched). Calls are accelerated only from user code, only
for exactly-ndarray, 1-D, C-contiguous float32/float64 input, and only with
default kwargs. Everything else is delegated to the original function. Result
types match NumPy's. Values can differ in the last bits because the native sum
is compensated, which is at least as accurate (docs/accuracy.md). Every path is
differential-tested against plain NumPy (`test_accelerate_differential`).

## Call-site identity

`call_site_id = sha1(path relative to repo root | enclosing qualname | k)`,
where k is the ordinal of the NumPy call within the enclosing function. NumPy
calls are classified syntactically, from the file's own imports plus ndarray
method names. Line numbers are display-only, so inserting blank lines keeps
IDs, while renaming the enclosing function changes them
(`tests/test_callsite.py`). Up to 8 normalized user stack frames are stored per
call site, and `rglns bisect` uses them for its attribution prior.

## Overhead

Measured with `validation/overhead.py`; results in
[validation.md](validation.md#wp2-observation-overhead). On the validation
pipeline (1M rows) the overhead is within noise, about 1%, against a target
of 5%. On 100k-call microloops of tiny calls, the C fast path adds about
105-175 ns per observed call: +35% on `np.add` of 16 elements, +31% on
`x.sum()` and +18% on `np.sum(x)`. The plan's 25% target is met only for
`np.sum`. CPython's own cost of delivering CALL and C_RETURN to a callback,
measured with C callbacks that do nothing, is already 70-86 ns, 23% of the
other two calls. That leaves about 7 ns per call for the observer itself,
less than two clock reads cost, so no `sys.monitoring` observer that times
each call can meet the target on calls this small. The work in our callbacks
(lookup, array description, two clock reads, aggregation) is 30-65 ns.

The pure-Python path costs roughly 1.5 us per call. An early version spent
1.4 us of that hashing code objects, which CPython does not cache, so every
per-location cache is now keyed by `id(code)`.
