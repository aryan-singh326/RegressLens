"""Call-site identity (WP0): stable across line shifts, sensitive to
renames, ordinal by NumPy call within the enclosing scope."""
import importlib.util
import os
import sys
import textwrap

import pytest

from regresslens import callsite


def _load(path, name):
    callsite._clear_caches()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE = textwrap.dedent('''
    import sys
    import numpy as np
    from regresslens import callsite

    SITES = []

    def probe(*args, **kwargs):
        frame = sys._getframe(1)
        SITES.append(callsite.resolve_frame(frame, callee="sum"))
        return 0.0

    np_probe = type("NP", (), {"sum": staticmethod(probe), "mean": staticmethod(probe)})()
''')


def _module(tmp_path, body, name="mod"):
    path = tmp_path / f"{name}.py"
    path.write_text(PROBE + textwrap.dedent(body))
    return path


def _sites(tmp_path, body, name="mod"):
    path = _module(tmp_path, body, name)
    mod = _load(str(path), name + str(abs(hash(body))))
    mod.run()
    return mod.SITES


BODY = '''
def run():
    x = [1.0, 2.0]
    np_probe.sum(x)
    np_probe.mean(x)
'''

BODY_SHIFTED = '''


def run():

    x = [1.0, 2.0]


    np_probe.sum(x)
    np_probe.mean(x)
'''

def test_id_unchanged_when_blank_lines_are_inserted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _module(tmp_path, BODY, "same")
    mod = _load(str(path), "same_v1")
    mod.run()
    before = [s.id for s in mod.SITES]
    path.write_text(PROBE + textwrap.dedent(BODY_SHIFTED))
    mod = _load(str(path), "same_v2")
    mod.run()
    after = [s.id for s in mod.SITES]
    assert before == after and len(before) == 2 and before[0] != before[1]


def test_id_changes_when_enclosing_function_is_renamed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _module(tmp_path, BODY, "ren")
    mod = _load(str(path), "ren_v1")
    mod.run()
    before = mod.SITES[0]
    path.write_text(PROBE + textwrap.dedent(BODY.replace("def run():",
                                                         "def run():\n    return go()\n\ndef go():")))
    mod = _load(str(path), "ren_v2")
    mod.run()
    after = mod.SITES[0]
    assert before.qualname == "run" and after.qualname == "go"
    assert before.id != after.id


def test_ordinal_counts_numpy_calls_in_source_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sites = _sites(tmp_path, '''
def run():
    print("a non-NumPy call does not shift ordinals")
    y = [3.0]
    np_probe.mean(y)
    np_probe.sum(y)
''', "ordn")
    assert [s.ordinal for s in sites] == ["0", "1"]


def test_qualnames_for_methods_nested_functions_and_lambdas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sites = _sites(tmp_path, '''
class Pipeline:
    def step(self):
        np_probe.sum([1.0])

def outer():
    def inner():
        np_probe.sum([1.0])
    inner()
    f = lambda: np_probe.sum([2.0])
    f()
    return [np_probe.sum([v]) for v in (1.0, 2.0)]

def run():
    Pipeline().step()
    outer()
''', "quals")
    quals = [s.qualname for s in sites]
    assert quals[:3] == ["Pipeline.step", "outer.<locals>.inner", "outer.<locals>.<lambda>"]
    # Comprehensions belong to the enclosing function on every Python version.
    assert quals[3:] == ["outer", "outer"]
    assert sites[3].id == sites[4].id


def test_relative_path_uses_repository_root(tmp_path):
    from conftest import make_git_repo
    repo = make_git_repo(str(tmp_path / "repo"), {"pkg/mod.py": "x = 1\n"})
    assert callsite.relative_path(os.path.join(repo, "pkg", "mod.py")) == "pkg/mod.py"


def test_user_file_classification():
    import numpy
    assert not callsite.is_user_file(numpy.__file__)
    assert not callsite.is_user_file(os.__file__)
    assert not callsite.is_user_file(callsite.__file__)
    assert callsite.is_user_file(__file__)
    assert callsite.is_user_file("<stdin>")
    assert not callsite.is_user_file("<frozen importlib._bootstrap>")


@pytest.mark.skipif(sys.version_info < (3, 11), reason="needs co_positions")
def test_nested_calls_on_one_line_get_distinct_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sites = _sites(tmp_path, '''
def run():
    np_probe.sum([np_probe.mean([1.0])])
''', "nested")
    inner, outer = sites  # inner call completes first
    assert (inner.ordinal, outer.ordinal) == ("1", "0") and inner.id != outer.id
