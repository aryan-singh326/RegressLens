"""Zero-code-change observation (`rglns run`, WP2)."""
import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from conftest import FIXTURES
from regresslens import launcher, store

EIGHT_FORMS = os.path.join(FIXTURES, "eight_forms.py")

ALL_FORMS = {"np.sum@forms", "ndarray.sum@forms", "np.convolve@forms", "np.add@forms",
             "np.mean@forms", "np.sum@forms.<locals>.<lambda>"}
OPERATOR_FORMS = {"op.add@forms", "index.mask@forms"}
# Observed sets per backend, confirmed on CPython 3.9-3.14 (docs/observation.md).
EXPECTED = {
    "monitoring": ALL_FORMS,
    "profile": ALL_FORMS - {"np.add@forms"},  # setprofile never reports ufunc objects
}
HAS_MONITORING = sys.version_info >= (3, 12)


def observed(session_id):
    conn = store.connect()
    try:
        rows = conn.execute(
            "SELECT c.operator, s.qualname FROM calls c JOIN call_sites s USING (call_site_id) "
            "JOIN runs r USING (run_id) WHERE r.session_id = ?", (session_id,)).fetchall()
    finally:
        conn.close()
    interesting = {"np.sum", "ndarray.sum", "np.convolve", "np.add", "np.mean", "op.add",
                   "index.mask"}
    return {f"{op}@{q}" for op, q in rows if op in interesting}


def run(argv, **kw):
    kw.setdefault("label", "test")
    kw.setdefault("capture", True)
    lnch = launcher.Launcher(argv, **kw)
    result = lnch.run()
    return lnch, result


BACKENDS = (["monitoring"] if HAS_MONITORING else []) + ["profile"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_eight_forms_observed_and_unobserved(backend):
    env = {"RGLNS_BACKEND": "profile"} if backend == "profile" else {}
    lnch, result = run(["python", EIGHT_FORMS], extra_env=env)
    assert result.exit_code == 0, result.stderr
    seen = observed(lnch.session_id)
    assert seen == EXPECTED[backend]
    assert not seen & OPERATOR_FORMS
    # The coverage report names what this backend cannot see.
    assert "not observed:" in result.stderr and "arithmetic operators" in result.stderr
    if backend == "profile":
        assert "ufunc objects" in result.stderr


@pytest.mark.parametrize("backend", BACKENDS)
def test_tier_d_observes_operators_and_indexing(backend):
    env = {"RGLNS_BACKEND": "profile"} if backend == "profile" else {}
    lnch, result = run(["python", EIGHT_FORMS], extra_env=env, observe_operators=True)
    assert result.exit_code == 0, result.stderr
    assert observed(lnch.session_id) == EXPECTED[backend] | OPERATOR_FORMS


RESULTS_SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np
    rng = np.random.default_rng(11)
    x = rng.normal(size=10_000).astype(np.float32)
    y = rng.normal(size=10_000)
    out = {
        "a": np.convolve(y, np.ones(7) / 7, mode="valid"),
        "b": np.cumsum(x) * 2 + y[: x.size],
        "c": np.sort(y)[::-3],
        "d": np.array([np.sum(y), y.sum(), np.mean(x), x.mean(), np.std(y)]),
    }
    for k, v in out.items():
        print(k, v.dtype, v.shape, repr(v[:3].tolist()))
    np.savez(sys.argv[1], **out)
""")


@pytest.mark.parametrize("flags", [[], ["--observe-operators"]])
def test_instrumented_runs_are_byte_identical(tmp_path, flags):
    script = tmp_path / "results.py"
    script.write_text(RESULTS_SCRIPT)
    plain = subprocess.run([sys.executable, str(script), str(tmp_path / "plain.npz")],
                           capture_output=True)
    observed_run = subprocess.run(
        [sys.executable, "-m", "regresslens.cli", "run", *flags, "--", sys.executable,
         str(script), str(tmp_path / "obs.npz")], capture_output=True)
    assert plain.returncode == 0 and observed_run.returncode == 0, observed_run.stderr
    assert plain.stdout == observed_run.stdout  # raw bytes: stdout passes through untouched
    assert b"rglns: observed" in observed_run.stderr
    a, b = np.load(tmp_path / "plain.npz"), np.load(tmp_path / "obs.npz")
    for key in a.files:
        assert a[key].tobytes() == b[key].tobytes(), key


def test_child_python_processes_are_observed(tmp_path):
    child = tmp_path / "child.py"
    child.write_text("import numpy as np\nnp.convolve(np.ones(100), np.ones(3))\n")
    parent = tmp_path / "parent.py"
    parent.write_text(f"import subprocess, sys\nimport numpy as np\nnp.sum(np.ones(5))\n"
                      f"subprocess.run([sys.executable, {str(child)!r}], check=True)\n")
    lnch, result = run(["python", str(parent)])
    assert result.exit_code == 0, result.stderr
    conn = store.connect()
    ops = {r[0] for r in conn.execute(
        "SELECT operator FROM calls JOIN runs USING (run_id) WHERE session_id = ?",
        (lnch.session_id,))}
    assert {"np.sum", "np.convolve"} <= ops


def test_existing_sitecustomize_is_chained(tmp_path):
    site = tmp_path / "their_site"
    site.mkdir()
    marker = tmp_path / "marker.txt"
    (site / "sitecustomize.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\n")
    script = tmp_path / "s.py"
    script.write_text("import numpy as np\nnp.sum(np.ones(3))\n")
    lnch, result = run(["python", str(script)],
                       extra_env={"PYTHONPATH": str(site) + os.pathsep + os.environ["PYTHONPATH"]})
    assert result.exit_code == 0, result.stderr
    assert marker.read_text() == "ran"
    assert "np.sum@<module>" in {f"{o}@{q}" for o, q in store.connect().execute(
        "SELECT c.operator, s.qualname FROM calls c JOIN call_sites s USING (call_site_id)")}


def test_coverage_report_goes_to_stderr_only(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("import numpy as np\nprint('hello')\nnp.sum(np.ones(3))\n")
    _, result = run(["python", str(script)])
    assert result.stdout == "hello\n"
    assert "rglns: observed" in result.stderr


@pytest.mark.parametrize("opts", [{}, {"accelerate": True}, {"observe_operators": True}])
def test_numpy_is_not_imported_before_the_program_imports_it(tmp_path, opts):
    """Programs that set OMP_NUM_THREADS etc. before `import numpy` must
    behave the same under rglns, so the bootstrap never imports NumPy."""
    script = tmp_path / "s.py"
    script.write_text("import sys\nprint('numpy' in sys.modules)\n"
                      "import numpy as np\nnp.sum(np.ones(3))\n")
    _, result = run(["python", str(script)], canaries=True, **opts)
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "False\n"
    assert "rglns: observed" in result.stderr


@pytest.mark.skipif(not HAS_MONITORING, reason="C fast path is for sys.monitoring")
def test_c_fast_path_and_pure_python_path_agree(native):
    fast, r1 = run(["python", EIGHT_FORMS])
    pure, r2 = run(["python", EIGHT_FORMS], extra_env={"RGLNS_PURE_OBSERVER": "1"})
    assert "C fast path" in r1.stderr and "C fast path" not in r2.stderr
    assert observed(fast.session_id) == observed(pure.session_id) == ALL_FORMS
    conn = store.connect()
    def shape(sid):
        return sorted(conn.execute(
            "SELECT c.operator, c.dtype, c.n, c.contiguous, c.count FROM calls c "
            "JOIN runs USING (run_id) WHERE session_id = ?", (sid,)).fetchall())
    assert shape(fast.session_id) == shape(pure.session_id)


ACCEL_SCRIPT = textwrap.dedent("""
    import json, sys
    import numpy as np
    rng = np.random.default_rng(5)
    cases = {
        "f64": rng.normal(size=10_001),
        "f32": rng.normal(size=10_001).astype(np.float32),
        "strided": rng.normal(size=20_000)[::2],
        "two_d": rng.normal(size=(50, 40)),
        "ints": np.arange(1000),
        "nan": np.array([1.0, np.nan, 3.0]),
        "inf": np.array([1.0, np.inf, 3.0]),
        "cancel": np.array([1e16, 1.0, -1e16]),
    }
    out = {}
    for name, a in cases.items():
        for fn in (np.sum, np.mean):
            r = fn(a)
            out[f"{name}.{fn.__name__}"] = [repr(type(r).__name__), float(r)]
        out[name + ".axis"] = np.sum(a, axis=0).tolist() if a.ndim == 2 else None
        out[name + ".dtype"] = float(np.sum(a, dtype=np.float64))
    json.dump(out, open(sys.argv[1], "w"))
""")


def test_accelerate_differential(tmp_path, native):
    script = tmp_path / "acc.py"
    script.write_text(ACCEL_SCRIPT)
    plain = subprocess.run([sys.executable, str(script), str(tmp_path / "plain.json")])
    lnch, res = run(["python", str(script), str(tmp_path / "acc.json")], accelerate=True)
    assert plain.returncode == 0 and res.exit_code == 0, res.stderr
    a = json.load(open(tmp_path / "plain.json"))
    b = json.load(open(tmp_path / "acc.json"))
    assert a.keys() == b.keys()
    for key in a:
        if key.startswith("cancel."):
            continue  # NumPy is wrong here; checked against the exact answer below
        if isinstance(a[key], list) and len(a[key]) == 2 and isinstance(a[key][0], str):
            assert a[key][0] == b[key][0], key            # same NumPy scalar type
            x, y = a[key][1], b[key][1]
            if np.isnan(x):
                assert np.isnan(y), key
            else:
                assert y == pytest.approx(x, rel=2e-7 if "f32" in key else 1e-12), key
        else:                                             # delegated paths: identical
            assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key])), key
    # Compensated summation is exact here; NumPy's pairwise sum returns 0.
    assert a["cancel.sum"][1] == 0.0 and b["cancel.sum"][1] == 1.0
    assert b["cancel.mean"][1] == pytest.approx(1 / 3, rel=1e-15)
    conn = store.connect()
    kernels = {r[0] for r in conn.execute(
        "SELECT kernel FROM calls JOIN runs USING (run_id) WHERE session_id = ?",
        (lnch.session_id,))}
    assert kernels & {"scalar", "avx2", "mt_avx2"} and "numpy" in kernels


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MONITORING, reason="documented for sys.monitoring")
def test_overhead_on_validation_pipeline_is_small(tmp_path):
    """<= 5% on the validation pipeline. Best of 5 processes per side, with
    plain and observed runs alternating: measuring all plain runs first
    would count any drift in machine state as overhead."""
    script = tmp_path / "pipe.py"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {os.path.join(root, 'validation')!r})
        import pipeline
        prices = pipeline.generate_prices(n=1_000_000, seed=1)
        best = float("inf")
        for _ in range(15):
            t0 = time.perf_counter()
            pipeline.run_numpy_baseline(prices)
            best = min(best, time.perf_counter() - t0)
        print(best)
    """))
    plain, observed_t = [], []
    for _ in range(5):
        plain.append(float(subprocess.run([sys.executable, str(script)], capture_output=True,
                                          text=True, check=True).stdout))
        observed_t.append(float(run(["python", str(script)])[1].stdout))
    assert min(observed_t) <= min(plain) * 1.05 + 1e-4, (plain, observed_t)
