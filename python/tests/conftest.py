"""Shared test fixtures.

Every test gets its own trace database (RGLNS_DB_PATH) and config file,
small drift-canary buffers (fast), and a fresh process-wide aggregator,
so no test reads or writes ~/.regresslens.

The package directory is also put on PYTHONPATH, because the CLI runs
measured programs as separate processes, which inherit environment
variables but not this process's sys.path.
"""
import os
import subprocess
import sys

import pytest

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)
_pp = os.environ.get("PYTHONPATH", "")
if PACKAGE_DIR not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = PACKAGE_DIR + (os.pathsep + _pp if _pp else "")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RGLNS_DB_PATH", str(tmp_path / "traces.db"))
    monkeypatch.setenv("RGLNS_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv("RGLNS_CANARY_MEM_BYTES", str(4 * 1024 * 1024))
    monkeypatch.setenv("RGLNS_TELEMETRY", "0")
    for var in ("RGLNS_RUN_ID", "RGLNS_SESSION_ID", "RGLNS_ACTIVE", "RGLNS_NO_TRACE",
                "REGRESSLENS_NO_TRACE", "REGRESSLENS_DB_PATH"):
        monkeypatch.delenv(var, raising=False)
    from regresslens.observe import aggregator
    monkeypatch.setattr(aggregator, "_global", None)
    from regresslens import _array
    monkeypatch.setattr(_array, "_selectivity_db_cache", {})
    yield


@pytest.fixture
def native():
    from regresslens import _accel
    mod = _accel.native()
    if mod is None:
        pytest.skip("regresslens-accel not installed")
    return mod


@pytest.fixture
def avx2(native):
    """The native module, when the selector is free to pick AVX2 kernels."""
    info = native.info()
    if not info["avx2_active"]:
        pytest.skip("AVX2 kernels not active on this CPU")
    if info["forced_kernel"]:
        pytest.skip(f"RGLNS_FORCE_KERNEL={info['forced_kernel']} overrides kernel selection")
    return native


def run_rglns(*args, cwd=None, env=None, timeout=600):
    """Runs the rglns CLI in a subprocess; returns CompletedProcess."""
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run([sys.executable, "-m", "regresslens.cli", *args], cwd=cwd,
                          env=full_env, capture_output=True, text=True, timeout=timeout)


def make_git_repo(path, files, message="initial"):
    """Creates a git repository with `files` ({relpath: text}) committed."""
    os.makedirs(path, exist_ok=True)
    for args in (["init", "-q"], ["config", "user.email", "t@example.com"],
                 ["config", "user.name", "test"], ["config", "core.autocrlf", "false"]):
        subprocess.run(["git", *args], cwd=path, check=True)
    commit(path, files, message)
    return path


def commit(path, files, message):
    for rel, text in files.items():
        full = os.path.join(path, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", newline="\n") as f:
            f.write(text)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=path, check=True)
