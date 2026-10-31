"""Tier D (optional, `rglns run --observe-operators`): observe arithmetic
operators and indexing in the user's own modules.

Operators never pass through a call hook (a + b is a BINARY_OP
instruction), so an import hook rewrites the AST of user modules
instead:

    a + b      ->  __rglns_binop__(SITE, "op.add", a, b)
    x[mask]    ->  __rglns_getitem__(SITE, x, mask)

The helpers evaluate exactly operator.add(a, b) / operator.getitem(x, k)
(same dunder dispatch, same evaluation order) and time the call only
when an ndarray is involved. SITE is a constant tuple computed at
rewrite time. Its id follows the call-site rule (path | qualname |
"op<k>"), so it is stable across runs.

Only load-context subscripts are rewritten (assignment targets and del
are untouched), and only files that are user code. Rewritten modules
bypass the .pyc cache in both directions, so the transformed code is
never persisted. The main script is not imported, so the launcher runs
it through `python -m regresslens.observe.tierd SCRIPT ARGS...`.

Cost: every operator in user code becomes a Python-level call, so
expect noticeably higher overhead than the call-hook backends.
"""
import ast
import builtins
import importlib.abc
import importlib.machinery
import importlib.util
import operator
import os
import sys
import time
import types

from .. import callsite
from . import aggregator as _agg
from . import classify as _cls

_BINOPS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow", ast.MatMult: "matmul",
    ast.LShift: "lshift", ast.RShift: "rshift", ast.BitOr: "or_", ast.BitXor: "xor",
    ast.BitAnd: "and_",
}
_FUNCS = {name: getattr(operator, name) for name in _BINOPS.values()}
_perf = time.perf_counter_ns


# --- runtime helpers ----------------------------------------------------------

def _record(site, opname, arr, ns):
    agg = _agg.get()
    sid, path, qualname, ordinal, line = site
    if sid not in agg.sites:
        agg.note_site(callsite.CallSite(sid, path, qualname, ordinal, line))
    dtype, n, contig = _cls.describe(arr)
    agg.record(sid, opname, dtype, n, contig, "numpy", float(ns))


def _binop(site, opname, a, b):
    fn = _FUNCS[opname]
    np = _cls.numpy_ready()
    if np is None:
        return fn(a, b)
    nd = np.ndarray
    if isinstance(a, nd):
        arr = a
    elif isinstance(b, nd):
        arr = b
    else:
        return fn(a, b)
    t0 = _perf()
    result = fn(a, b)
    _record(site, "op." + opname.rstrip("_"), arr, _perf() - t0)
    return result


def _getitem(site, value, key):
    np = _cls.numpy_ready()
    if np is None or not isinstance(value, np.ndarray):
        return value[key]
    t0 = _perf()
    result = value[key]
    ns = _perf() - t0
    if isinstance(key, np.ndarray) and key.dtype == bool:
        kind = "index.mask"
    elif isinstance(key, (np.ndarray, list)):
        kind = "index.fancy"
    else:
        kind = "index.basic"
    _record(site, kind, value, ns)
    return result


HELPERS = {
    "__rglns_binop__": _binop,
    "__rglns_getitem__": _getitem,
    "__rglns_slice__": builtins.slice,
}


# --- AST rewrite ------------------------------------------------------------------

class _Rewriter(ast.NodeTransformer):
    def __init__(self, relpath):
        self.relpath = relpath
        self.scopes = [("<module>", "module")]
        self.counters = [0]

    def _qualname(self):
        return self.scopes[-1][0]

    def _enter(self, node, name):
        parent, kind = self.scopes[-1]
        if kind == "module":
            q = name
        elif kind == "class":
            q = f"{parent}.{name}"
        else:
            q = f"{parent}.<locals>.{name}"
        self.scopes.append((q, "class" if isinstance(node, ast.ClassDef) else "func"))
        self.counters.append(0)
        self.generic_visit(node)
        self.scopes.pop()
        self.counters.pop()
        return node

    def visit_FunctionDef(self, node):
        return self._enter(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        return self._enter(node, node.name)

    def visit_Lambda(self, node):
        return self._enter(node, "<lambda>")

    def _site(self, node):
        k = self.counters[-1]
        self.counters[-1] += 1
        ordinal = f"op{k}"
        q = self._qualname()
        return ast.Constant((callsite.make_id(self.relpath, q, ordinal), self.relpath, q,
                             ordinal, node.lineno))

    def visit_BinOp(self, node):
        opname = _BINOPS.get(type(node.op))
        site = self._site(node)  # assigned before children: stable pre-order
        self.generic_visit(node)
        if opname is None:
            return node
        call = ast.Call(ast.Name("__rglns_binop__", ast.Load()),
                        [site, ast.Constant(opname), node.left, node.right], [])
        return ast.copy_location(call, node)

    def visit_Subscript(self, node):
        if not isinstance(node.ctx, ast.Load) or _has_starred(node.slice):
            self.generic_visit(node)
            return node
        site = self._site(node)
        self.generic_visit(node)
        key = _slice_expr(node.slice)
        call = ast.Call(ast.Name("__rglns_getitem__", ast.Load()), [site, node.value, key], [])
        return ast.copy_location(call, node)


def _has_starred(expr):
    return any(isinstance(n, ast.Starred) for n in ast.walk(expr))


def _slice_expr(expr):
    """Turns subscript syntax into an ordinary expression: a:b:c becomes
    slice(a, b, c), applied inside tuples (x[1:2, ::3])."""
    if isinstance(expr, ast.Slice):
        none = ast.Constant(None)
        return ast.copy_location(ast.Call(
            ast.Name("__rglns_slice__", ast.Load()),
            [expr.lower or none, expr.upper or none, expr.step or none], []), expr)
    if isinstance(expr, ast.Tuple):
        return ast.copy_location(ast.Tuple([_slice_expr(e) for e in expr.elts], ast.Load()),
                                 expr)
    return expr


def rewrite(source, filename):
    tree = ast.parse(source, filename)
    tree = _Rewriter(callsite.relative_path(filename)).visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec", dont_inherit=True)


# --- import hook ------------------------------------------------------------------

class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = importlib.util.decode_source(self.get_data(self.path))
        return rewrite(source, self.path)

    def exec_module(self, module):
        module.__dict__.update(HELPERS)
        super().exec_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (spec is None or not spec.origin or not spec.origin.endswith(".py")
                or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
                or not callsite.is_user_file(spec.origin)):
            return None
        spec.loader = _Loader(fullname, spec.origin)
        return spec


_finder = None


def install_import_hook():
    global _finder
    if _finder is None:
        _finder = _Finder()
        sys.meta_path.insert(0, _finder)


def run_main(argv):
    """`python -m regresslens.observe.tierd SCRIPT ARGS...`: runs SCRIPT as
    __main__ with operator observation, matching `python SCRIPT ARGS`
    (sys.argv, sys.path[0] = script directory)."""
    if not argv:
        print("usage: python -m regresslens.observe.tierd SCRIPT [ARGS...]", file=sys.stderr)
        return 2
    script = argv[0]
    sys.argv = list(argv)
    sys.path[0] = os.path.dirname(os.path.abspath(script))
    install_import_hook()
    with open(script, "rb") as f:
        code = rewrite(importlib.util.decode_source(f.read()), script)
    main = types.ModuleType("__main__")
    main.__file__ = script
    main.__dict__.update(HELPERS)
    main.__dict__["__builtins__"] = builtins
    sys.modules["__main__"] = main
    exec(code, main.__dict__)
    return 0


if __name__ == "__main__":
    # Delegate to the package module so the import hook and helpers are
    # the same objects the rest of regresslens uses.
    from regresslens.observe import tierd as _tierd
    sys.exit(_tierd.run_main(sys.argv[1:]))
