"""
Ensures the regresslens package directory is on PYTHONPATH for the
whole test session. This matters specifically for test_cli.py: the
CLI spawns the user's pipeline script as a SEPARATE subprocess (see
cli.py's _run_script_n_times), which does NOT automatically inherit
sys.path modifications made within the pytest process itself — only
actual environment variables propagate to a subprocess. Without this,
CLI tests fail with "No module named 'regresslens'" in the child
process, not in the test itself, which is a confusing failure mode
worth preventing explicitly rather than discovering per-environment.
"""
import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _PACKAGE_DIR not in _existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _PACKAGE_DIR + (os.pathsep + _existing_pythonpath if _existing_pythonpath else "")
    )
