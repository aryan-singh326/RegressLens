"""Distribution split and CPU dispatch (WP1)."""
import json
import os
import subprocess
import sys

import pytest

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def py(code, **env):
    full = dict(os.environ)
    full.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=full, cwd=PYTHON_DIR)


def test_pure_package_works_without_native_module():
    proc = py("import json, numpy as np, regresslens as rl\n"
              "from regresslens import observe, store\n"
              "x = np.arange(1, 41, dtype=float); x[5] = np.nan\n"
              "r = rl.array(x).rolling_sum(4).to_numpy()\n"
              "observe.flush()\n"
              "k = {row[0] for row in store.connect().execute('SELECT kernel FROM calls')}\n"
              "print(json.dumps([rl.accel_info()['native'], int(np.isnan(r).sum()), sorted(k)]))",
              RGLNS_DISABLE_ACCEL="1")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [False, 4, ["numpy"]]


def test_forced_scalar_kernel(native):
    proc = py("import json, numpy as np\nfrom regresslens_accel import _native as n\n"
              "print(json.dumps([n.info()['forced_kernel'], n.reduce_sum(np.ones(10**6))[1],"
              " n.rolling_sum(np.ones(10**6), 5)[1]]))", RGLNS_FORCE_KERNEL="scalar")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == ["scalar", "scalar", "scalar"]


def test_invalid_forced_kernel_fails_loudly(native):
    proc = py("import regresslens as rl, numpy as np; rl.array(np.ones(3)).sum()",
              RGLNS_FORCE_KERNEL="sse9")
    assert proc.returncode != 0 and "RGLNS_FORCE_KERNEL" in proc.stderr


def test_explicit_unavailable_kernel_is_an_error(native):
    import numpy as np
    if native.info()["avx2_active"]:
        pytest.skip("AVX2 is available here")
    with pytest.raises(ValueError):
        native.reduce_sum(np.ones(4), "avx2")


@pytest.mark.slow
def test_pure_wheel_is_py3_none_any(tmp_path):
    import shutil
    src = tmp_path / "src"
    shutil.copytree(PYTHON_DIR, src, ignore=shutil.ignore_patterns(
        "build", "*.egg-info", "__pycache__", ".pytest_cache", ".hypothesis"))
    out = tmp_path / "wheels"
    proc = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out),
                           str(src)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    [wheel] = os.listdir(out)
    assert wheel.endswith("-py3-none-any.whl"), wheel
    import zipfile
    names = zipfile.ZipFile(out / wheel).namelist()
    assert "regresslens/_boot/sitecustomize.py" in names
    assert "regresslens/telemetry_schema.json" in names
    assert not any(n.endswith((".so", ".pyd")) for n in names)
