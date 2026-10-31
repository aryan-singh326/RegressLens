"""RegressLens observation bootstrap (injected by `rglns run`).

The rglns launcher prepends this directory to PYTHONPATH, so Python
imports this file as `sitecustomize` at startup, in the measured
process and in every child Python process it starts. When RGLNS_ACTIVE=1
it installs observation. Then it chains to the sitecustomize module that
would otherwise have run (conda and many venvs ship one), so that one
is not clobbered.

This file must never raise: a broken observer must not break the
user's program.
"""
import os
import sys


def _ensure_regresslens_importable():
    try:
        import regresslens  # noqa: F401
        return True
    except ImportError:
        pass
    # The measured interpreter may not have regresslens installed (e.g.
    # `rglns run -- /other/python script.py`). Borrow the CLI's copy,
    # but only temporarily: leaving the CLI's site-packages on sys.path
    # would let unrelated packages shadow the user's environment.
    parent = os.environ.get("RGLNS_PKG_PARENT")
    if not parent:
        return False
    sys.path.append(parent)
    try:
        import regresslens  # noqa: F401
        import regresslens.observe.bootstrap  # noqa: F401
        return True
    except ImportError:
        return False
    finally:
        try:
            sys.path.remove(parent)
        except ValueError:
            pass


def _install():
    if os.environ.get("RGLNS_ACTIVE") != "1":
        return
    try:
        if not _ensure_regresslens_importable():
            print("rglns: regresslens is not importable in this interpreter; "
                  "running unobserved", file=sys.stderr)
            return
        from regresslens.observe import bootstrap
        bootstrap.install()
    except Exception as exc:  # never break the user's program
        print(f"rglns: observation disabled ({type(exc).__name__}: {exc})", file=sys.stderr)


def _chain():
    """Executes the next sitecustomize on sys.path (the one ours shadows)."""
    import importlib.machinery
    import importlib.util

    here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    rest = [p for p in sys.path
            if os.path.normcase(os.path.abspath(p or os.curdir)) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"rglns: chained sitecustomize ({spec.origin}) raised "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return
    # Code that inspects sitecustomize afterwards sees the original.
    sys.modules["sitecustomize"] = module


_install()
_chain()
