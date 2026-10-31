"""rglns verify: test the accuracy guarantee on a generated corpus.

ACCURACY POLICY (docs/accuracy.md). Results are not bit-identical to
NumPy, so the guarantee is relative to the exact result: for every
operation, dtype and native kernel, on every corpus case,

    max_err(RegressLens) <= max(max_err(NumPy), 1 ulp)

where max_err is the largest error over the case's outputs, in ulps of
the exact value (regresslens/exact.py). In words: never less accurate
than NumPy, except within one unit of rounding of the exact result.

NaN/Inf: for every case with injected NaN/+Inf/-Inf, the positions and
kinds of non-finite outputs must equal NumPy's exactly.

Failing cases are shrunk (smaller inputs that still fail) and saved to
~/.regresslens/verify_failures/ as .npz with their parameters, so they
can be replayed.
"""
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .. import _accel, exact
from .common import eprint

OPS = ("sum", "mean", "projection", "filter", "rolling_sum", "rolling_mean", "rolling_var")
DISTRIBUTIONS = ("uniform", "normal", "price", "offset", "cancel", "wide")


def _data(rng, dist, n, dtype):
    if dist == "uniform":
        x = rng.uniform(-1000, 1000, n)
    elif dist == "normal":
        x = rng.normal(0, 1, n)
    elif dist == "price":
        x = 100 * np.exp(np.cumsum(rng.normal(2e-4, 1e-2, n)))
    elif dist == "offset":
        x = rng.normal(1e6, 1e-2, n)
    elif dist == "cancel":
        x = np.where(rng.random(n) < 0.5, 1e8, -1e8) + rng.normal(0, 1, n)
    else:  # wide dynamic range
        x = np.sign(rng.normal(size=n)) * 10.0 ** rng.uniform(-8, 8, n)
    return x.astype(dtype)


def _inject(rng, x):
    x = x.copy()
    k = int(rng.integers(1, max(2, len(x) // 20) + 1))
    pos = rng.integers(0, len(x), k)
    x[pos] = rng.choice(np.array([np.nan, np.inf, -np.inf], dtype=x.dtype), k)
    return x


@dataclass
class Case:
    op: str
    dtype: str
    dist: str
    n: int
    window: int
    seed: int
    nonfinite: bool
    params: dict = field(default_factory=dict)

    def data(self):
        rng = np.random.default_rng(self.seed)
        x = _data(rng, self.dist, self.n, np.dtype(self.dtype))
        return _inject(rng, x) if self.nonfinite else x


def make_cases(count, seed, ops, dtypes, max_n):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(count):
        op = ops[i % len(ops)]
        dtype = dtypes[(i // len(ops)) % len(dtypes)]
        dist = DISTRIBUTIONS[int(rng.integers(len(DISTRIBUTIONS)))]
        cap = min(max_n, 3000) if op in ("projection", "rolling_var") else max_n
        n = int(np.exp(rng.uniform(0, math.log(cap))))
        window = int(rng.integers(1, min(n, 200) + 1))
        nonfinite = bool(rng.random() < 0.25) and n >= 2
        params = {}
        if op == "projection":
            params = {"scale": float(rng.normal(0, 3)), "offset": float(rng.normal(0, 100))}
        elif op == "filter":
            params = {"threshold": float(rng.normal(0, 500))}
        elif op == "rolling_var":
            params = {"ddof": int(rng.integers(0, 2)) if window > 1 else 0}
        cases.append(Case(op, dtype, dist, n, window, int(rng.integers(2 ** 31)), nonfinite,
                          params))
    return cases


def _numpy_result(case, x):
    dt = np.dtype(case.dtype).type
    p = case.params
    with np.errstate(all="ignore"):
        if case.op == "sum":
            return np.sum(x)
        if case.op == "mean":
            return np.mean(x)
        if case.op == "projection":
            return x * dt(p["scale"]) + dt(p["offset"])
        if case.op == "filter":
            return x[x > dt(p["threshold"])]
        if case.op in ("rolling_sum", "rolling_mean"):
            r = np.convolve(x, np.ones(case.window, dtype=x.dtype), mode="valid")
            return r if case.op == "rolling_sum" else r / dt(case.window)
        view = np.lib.stride_tricks.sliding_window_view(x, case.window)
        return np.var(view, axis=-1, ddof=p["ddof"]).astype(x.dtype)


def _native_result(native, case, x, kernel):
    p = case.params
    if case.op in ("sum", "mean"):
        threads = 4 if kernel == "mt_avx2" else 0
        fn = native.reduce_sum if case.op == "sum" else native.reduce_mean
        return x.dtype.type(fn(x, kernel, threads)[0])
    if case.op == "projection":
        return native.project_affine(x, p["scale"], p["offset"], kernel)[0]
    if case.op == "filter":
        threads = 4 if kernel == "mt_avx2" else 0
        full, count, _, _ = native.filter_gt(x, p["threshold"], kernel, threads)
        return full[:count]
    if case.op == "rolling_var":
        return native.rolling_var(x, case.window, p["ddof"])[0]
    return getattr(native, case.op)(x, case.window, kernel)[0]


def _exact(case, x):
    dt = np.dtype(case.dtype).type
    p = case.params
    finite = x[np.isfinite(x)]
    if case.op == "sum":
        return [exact.exact_sum(finite)] if len(finite) == len(x) else [None]
    if case.op == "mean":
        return [exact.exact_sum(x) / len(x)] if len(finite) == len(x) else [None]
    if case.op == "projection":
        return exact.affine(x, dt(p["scale"]), dt(p["offset"]))
    if case.op == "rolling_sum":
        return exact.rolling_sums(x, case.window)
    if case.op == "rolling_mean":
        return [None if s is None else s / case.window
                for s in exact.rolling_sums(x, case.window)]
    if case.op == "rolling_var":
        return exact.rolling_vars(x, case.window, p["ddof"])
    return None


def _kinds(a):
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    return np.where(np.isnan(a), 1, np.where(a == np.inf, 2, np.where(a == -np.inf, 3, 0)))


def _max_err(values, ref, dtype):
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    worst = 0.0
    for v, r in zip(values, ref):
        if r is not None and math.isfinite(v):
            worst = max(worst, exact.ulp_error(v, r, dtype))
    return worst


def check_case(native, case, kernel, x=None):
    """(ok, ours_err, numpy_err, message) for one case and kernel."""
    x = case.data() if x is None else x
    if case.op.startswith("rolling") and len(x) < case.window:
        return True, 0.0, 0.0, ""
    ours = _native_result(native, case, x, kernel)
    ref_np = _numpy_result(case, x)
    if ours is None:  # native declined (overflow risk) -> NumPy path
        return True, 0.0, 0.0, ""
    if case.op == "filter":
        same = np.array_equal(np.asarray(ours), np.asarray(ref_np))
        return same, 0.0, 0.0, "" if same else "filter output differs from NumPy"
    if not np.array_equal(_kinds(ours), _kinds(ref_np)):
        return False, math.inf, 0.0, "NaN/Inf pattern differs from NumPy"
    ref = _exact(case, x)
    dtype = np.dtype(case.dtype).type
    e_ours = _max_err(ours, ref, dtype)
    e_np = _max_err(ref_np, ref, dtype)
    ok = e_ours <= max(e_np, 1.0)
    return ok, e_ours, e_np, "" if ok else f"{e_ours:.2f} ulp vs NumPy {e_np:.2f} ulp"


def kernels_for(native, op):
    info = native.info()
    ks = ["scalar"]
    if info["avx2_active"] and op != "rolling_var":
        ks.append("avx2")
        if op in ("sum", "mean", "filter"):
            ks.append("mt_avx2")
    return ks


def shrink(native, case, kernel, x):
    """Greedy shrink: keep halving the input (and the window with it)
    while the case still fails."""
    best = x
    progress = True
    while progress and len(best) > 2:
        progress = False
        half = len(best) // 2
        for part in (best[:half], best[half:]):
            trial = Case(**{**case.__dict__, "n": len(part),
                            "window": min(case.window, len(part))})
            if trial.op.startswith("rolling") and len(part) < 1:
                continue
            ok, *_ = check_case(native, trial, kernel, part)
            if not ok:
                best, case = part, trial
                progress = True
                break
    return case, best


def save_failure(directory, case, kernel, x, message):
    os.makedirs(directory, exist_ok=True)
    stem = f"{case.op}-{case.dtype}-{kernel}-{case.seed}"
    path = os.path.join(directory, stem + ".npz")
    np.savez(path, x=x)
    with open(os.path.join(directory, stem + ".json"), "w") as f:
        json.dump({**case.__dict__, "kernel": kernel, "message": message,
                   "n": int(len(x))}, f, indent=2)
    return path


def run_verify(args):
    native = _accel.native()
    if native is None:
        eprint("regresslens-accel is not installed: RegressLens is running on NumPy itself, "
               "so there is nothing to verify (install regresslens[accel]).")
        return 0
    ops = tuple(args.ops.split(",")) if args.ops else OPS
    dtypes = tuple(args.dtypes.split(",")) if args.dtypes else ("float32", "float64")
    cases = make_cases(args.cases, args.seed, ops, dtypes, args.max_n)
    failures_dir = args.failures_dir or os.path.expanduser(
        os.path.join("~", ".regresslens", "verify_failures"))
    stats = {}
    failures = []
    nan_cases = nan_bad = 0
    t0 = time.time()
    for case in cases:
        x = case.data()
        for kernel in kernels_for(native, case.op):
            ok, e_ours, e_np, msg = check_case(native, case, kernel, x)
            key = (case.op, case.dtype, kernel)
            s = stats.setdefault(key, {"cases": 0, "max_ours": 0.0, "max_numpy": 0.0,
                                       "failures": 0})
            s["cases"] += 1
            if math.isfinite(e_ours):
                s["max_ours"] = max(s["max_ours"], e_ours)
            s["max_numpy"] = max(s["max_numpy"], e_np)
            if case.nonfinite:
                nan_cases += 1
                nan_bad += 0 if ok or "pattern" not in msg else 1
            if not ok:
                s["failures"] += 1
                small_case, small_x = shrink(native, case, kernel, x)
                path = save_failure(failures_dir, small_case, kernel, small_x, msg)
                failures.append((key, case.seed, msg, path))
    print(f"rglns verify: {len(cases)} cases (seed {args.seed}), {time.time() - t0:.1f} s; "
          f"accuracy policy: max error <= max(NumPy's, 1 ulp) vs the exact result")
    print(f"  {'operation':<13} {'dtype':<8} {'kernel':<8} {'cases':>6} {'max err':>11} "
          f"{'NumPy max':>11}  result")
    for (op, dtype, kernel), s in sorted(stats.items()):
        res = "ok" if not s["failures"] else f"FAIL ({s['failures']})"
        print(f"  {op:<13} {dtype:<8} {kernel:<8} {s['cases']:>6} {s['max_ours']:>9.2f}ul "
              f"{s['max_numpy']:>9.2f}ul  {res}")
    print(f"  NaN/Inf semantics: {nan_cases} case-kernel checks with injected NaN/Inf, "
          f"{nan_bad} mismatches")
    for key, seed, msg, path in failures[:20]:
        print(f"  FAILURE {'/'.join(key)} seed {seed}: {msg}  (shrunk case: {path})")
    args.verdict_counts = {"fail": len(failures)}
    return 1 if failures else 0
