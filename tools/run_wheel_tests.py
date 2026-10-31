"""Runs the Python test suite against an installed regresslens-accel wheel.

Used by cibuildwheel (cpp/pyproject.toml). Installs the pure-Python
regresslens package from the source tree, then runs the suite twice:
once with automatic kernel dispatch and once with the AVX2 kernels
disabled (RGLNS_FORCE_KERNEL=scalar), so the scalar kernel set is
tested on every platform, including AVX2 machines.
"""
import os
import subprocess
import sys


def main(project_root):
    python_pkg = os.path.join(project_root, "python")
    subprocess.check_call([sys.executable, "-m", "pip", "install", python_pkg])
    tests = os.path.join(python_pkg, "tests")
    for forced in ("", "scalar"):
        env = dict(os.environ)
        if forced:
            env["RGLNS_FORCE_KERNEL"] = forced
        else:
            env.pop("RGLNS_FORCE_KERNEL", None)
        label = forced or "auto"
        print(f"=== test suite, kernel dispatch: {label} ===", flush=True)
        rc = subprocess.call(
            [sys.executable, "-m", "pytest", tests, "-q", "-m", "not slow"], env=env
        )
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()))
