"""Call-site identity.

    call_site_id = sha1(path relative to repo root | enclosing qualname | ordinal)

where `ordinal` is k for the k-th NumPy call (0-based, source order)
inside the enclosing function, class body or module. Line numbers are
deliberately not part of the identity: they shift whenever code above
the call changes, which would break compare and bisect. The current
line is kept for display only.

Classification is purely syntactic, so an ID does not depend on which
branches ran or on the installed NumPy version. A call counts as a
NumPy call when its callee is:
  * an attribute chain rooted at a name bound to numpy/regresslens by
    an import in the same file (np.convolve, numpy.linalg.norm, rl.array);
  * a bare name imported from numpy/regresslens (`from numpy import diff`);
  * a method whose name is an ndarray or rl.array method (x.sum(),
    arr.rolling_mean(20)). Methods on other objects with the same names
    are counted too, deterministically.
Calls that are observed at runtime but not classified as NumPy calls
statically (for example `f = np.sum; f(x)`) get ordinal "a<i>", i being
their index among all calls in the scope. When no source is available
(exec'd strings, frozen modules) the bytecode offset is used ("@<off>").

Comprehension bodies count as part of their enclosing function on
every Python version (3.12 inlines list comprehensions; 3.9-3.11 do
not), so IDs agree across interpreter versions.
"""
import ast
import hashlib
import linecache
import os
import sys
import sysconfig
import threading
from dataclasses import dataclass

from . import gitutil

_NDARRAY_METHODS = frozenset("""
all any argmax argmin argpartition argsort astype byteswap choose clip compress
conj conjugate copy cumprod cumsum diagonal dot dump dumps fill flatten getfield
item max mean min nonzero partition prod ptp put ravel repeat reshape resize round
searchsorted setfield setflags sort squeeze std sum swapaxes take tobytes tofile
tolist trace transpose var view
""".split())

_RL_ARRAY_METHODS = frozenset(
    "project_affine filter_gt rolling_sum rolling_mean rolling_var to_numpy".split())

_NUMPY_ROOTS = ("numpy", "regresslens")

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

_HAS_POSITIONS = sys.version_info >= (3, 11)


@dataclass(frozen=True)
class CallSite:
    id: str
    path: str
    qualname: str
    ordinal: str
    line: int


def make_id(path, qualname, ordinal):
    return hashlib.sha1(f"{path}|{qualname}|{ordinal}".encode()).hexdigest()


# --- path handling --------------------------------------------------------------

_rel_cache = {}


def relative_path(filename):
    """Path relative to its repository root (forward slashes), so the same
    file has the same path in every git worktree. Outside a repository:
    relative to the working directory when below it, else absolute."""
    cached = _rel_cache.get(filename)
    if cached is not None:
        return cached
    if filename.startswith("<"):
        rel = filename
    else:
        absolute = os.path.abspath(filename)
        root = gitutil.repo_root(absolute)
        base = root or os.getcwd()
        try:
            rel = os.path.relpath(absolute, base)
            if rel.startswith(".."):
                rel = absolute
        except ValueError:  # different drive on Windows
            rel = absolute
        rel = rel.replace(os.sep, "/")
    _rel_cache[filename] = rel
    return rel


def _library_prefixes():
    paths = sysconfig.get_paths()
    prefixes = {paths.get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    prefixes.add(os.path.dirname(os.path.abspath(__file__)))  # regresslens itself
    # NumPy only once the program has imported it: importing it here
    # would run before the program's own `import numpy`.
    numpy = sys.modules.get("numpy")
    if getattr(numpy, "__file__", None):
        prefixes.add(os.path.dirname(os.path.abspath(numpy.__file__)))
    return tuple(os.path.normcase(os.path.abspath(p)) for p in prefixes if p)


_user_cache = {}
_LIB_PREFIXES = None
_PREFIXES_HAVE_NUMPY = False

# NumPy < 1.25 exec()s every __array_function__ dispatcher (np.sum,
# np.concatenate, ...) into this pseudo-file; its frames are NumPy's.
NUMPY_PSEUDO_FILE = "<__array_function__ internals>"


def is_user_file(filename):
    """True for the user's own code: not the standard library, not an
    installed package, not RegressLens or NumPy. RGLNS_USER_PATHS
    (os.pathsep-separated) forces directories to count as user code."""
    cached = _user_cache.get(filename)
    if cached is not None:
        return cached
    global _LIB_PREFIXES, _PREFIXES_HAVE_NUMPY
    if _LIB_PREFIXES is None or (not _PREFIXES_HAVE_NUMPY and "numpy" in sys.modules):
        _PREFIXES_HAVE_NUMPY = "numpy" in sys.modules
        _LIB_PREFIXES = _library_prefixes()
    if filename.startswith("<"):
        result = not filename.startswith(("<frozen", NUMPY_PSEUDO_FILE))
    else:
        norm = os.path.normcase(os.path.abspath(filename))
        forced = [os.path.normcase(os.path.abspath(p))
                  for p in os.environ.get("RGLNS_USER_PATHS", "").split(os.pathsep) if p]
        if any(norm.startswith(p) for p in forced):
            result = True
        elif ("site-packages" in norm or "dist-packages" in norm
              or any(norm.startswith(p) for p in _LIB_PREFIXES)):
            result = False
        else:
            result = True
    _user_cache[filename] = result
    return result


# --- per-file AST index -------------------------------------------------------

class _CallInfo:
    __slots__ = ("node", "scope_qualname", "ordinal", "name")

    def __init__(self, node, scope_qualname, ordinal, name):
        self.node = node
        self.scope_qualname = scope_qualname
        self.ordinal = ordinal
        self.name = name


def _callee_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _root_name(func):
    while isinstance(func, ast.Attribute):
        func = func.value
    return func.id if isinstance(func, ast.Name) else None


class _FileIndex:
    def __init__(self, tree):
        self.module_aliases, self.imported_names = self._imports(tree)
        self.calls = []
        self._by_line = {}
        self._walk_scope(tree, "<module>")

    @staticmethod
    def _imports(tree):
        modules, names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in _NUMPY_ROOTS:
                        modules.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in _NUMPY_ROOTS:
                    for a in node.names:
                        names.add(a.asname or a.name)
        return modules, names

    def _is_numpy_call(self, call):
        func = call.func
        if isinstance(func, ast.Name):
            return func.id in self.imported_names
        if isinstance(func, ast.Attribute):
            root = _root_name(func)
            if root is not None and (root in self.module_aliases
                                     or root in self.imported_names):
                return True
            return func.attr in _NDARRAY_METHODS or func.attr in _RL_ARRAY_METHODS
        return False

    def _walk_scope(self, scope_node, qualname):
        """Collects the Call nodes that belong to this scope (not nested
        scopes) in source order, then recurses into nested scopes. Only
        the body is walked: decorators, defaults and bases of a nested
        definition run in the enclosing scope and are collected there."""
        calls, nested = [], []
        if isinstance(scope_node, ast.Lambda):
            stack = [scope_node.body]
        else:
            stack = list(scope_node.body)
        while stack:
            node = stack.pop()
            if isinstance(node, _SCOPES):
                nested.append(node)
                # Decorators and defaults are evaluated in the enclosing scope.
                stack.extend(getattr(node, "decorator_list", []))
                if not isinstance(node, ast.ClassDef):
                    args = node.args
                    stack.extend(d for d in args.defaults + args.kw_defaults if d)
                else:
                    stack.extend(node.bases)
                continue
            if isinstance(node, ast.Call):
                calls.append(node)
            stack.extend(ast.iter_child_nodes(node))
        calls.sort(key=lambda c: (c.lineno, c.col_offset))
        k = 0
        for i, call in enumerate(calls):
            if self._is_numpy_call(call):
                ordinal = str(k)
                k += 1
            else:
                ordinal = f"a{i}"
            info = _CallInfo(call, qualname, ordinal, _callee_name(call.func))
            self.calls.append(info)
            for line in range(call.lineno, (call.end_lineno or call.lineno) + 1):
                self._by_line.setdefault(line, []).append(info)
        # __qualname__ rules: "C.m" inside a class, "f.<locals>.g" inside a
        # function, bare name at module level.
        if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            child_prefix = qualname + ".<locals>."
        elif isinstance(scope_node, ast.ClassDef):
            child_prefix = qualname + "."
        else:
            child_prefix = ""
        for node in nested:
            name = "<lambda>" if isinstance(node, ast.Lambda) else node.name
            self._walk_scope(node, child_prefix + name)

    def match_position(self, lineno, end_lineno, col, end_col):
        """Innermost Call whose span contains the instruction's span."""
        best = None
        for info in self._by_line.get(lineno, ()):
            n = info.node
            if (n.lineno, n.col_offset) <= (lineno, col) and \
                    (n.end_lineno, n.end_col_offset) >= (end_lineno, end_col):
                if best is None or _span(n) < _span(best.node):
                    best = info
        return best

    def match_line(self, lineno, callee):
        """Fallback without column information (Python < 3.11)."""
        candidates = self._by_line.get(lineno, ())
        named = [c for c in candidates if callee and c.name == callee]
        pool = named or list(candidates)
        if not pool:
            return None
        return min(pool, key=lambda c: (_span(c.node), c.node.col_offset))


def _span(node):
    return ((node.end_lineno or node.lineno) - node.lineno, (node.end_col_offset or 0) -
            node.col_offset)


_index_cache = {}
_index_lock = threading.Lock()


def _file_index(filename):
    if filename in _index_cache:
        return _index_cache[filename]
    with _index_lock:
        if filename in _index_cache:
            return _index_cache[filename]
        index = None
        lines = linecache.getlines(filename)
        if lines:
            try:
                index = _FileIndex(ast.parse("".join(lines), filename))
            except (SyntaxError, ValueError, RecursionError):
                index = None
        _index_cache[filename] = index
        return index


# --- resolution ----------------------------------------------------------------
#
# Caches are keyed by id(code), never by the code object itself: CPython
# does not cache code-object hashes, and hashing one walks its constants
# (measured ~1.4 us per lookup, more than a tiny NumPy call). Each entry
# keeps a reference to its code object and checks identity, so a recycled
# id can never return a stale entry.

_site_cache = {}


def _code_qualname(code):
    return getattr(code, "co_qualname", code.co_name)


def resolve(code, offset=None, lineno=None, callee=None):
    """CallSite for a call made by `code` at bytecode `offset` (3.11+) or
    on `lineno` (older Pythons), to a callable named `callee`."""
    key = (id(code), offset, lineno if offset is None or not _HAS_POSITIONS else None, callee)
    hit = _site_cache.get(key)
    if hit is not None and hit[0] is code:
        return hit[1]
    filename = code.co_filename
    path = relative_path(filename)
    index = _file_index(filename)
    info = None
    line = lineno
    if index is not None:
        if _HAS_POSITIONS and offset is not None and offset >= 0:
            positions = _positions(code)
            idx = offset // 2
            if 0 <= idx < len(positions):
                l, el, c, ec = positions[idx]
                if l is not None and c is not None:
                    info = index.match_position(l, el, c, ec)
                    line = l
        if info is None and line is not None:
            info = index.match_line(line, callee)
    if info is not None:
        site = CallSite(make_id(path, info.scope_qualname, info.ordinal), path,
                        info.scope_qualname, info.ordinal, info.node.lineno)
    else:
        ordinal = f"@{offset}" if offset is not None else f"L{lineno}"
        qual = _code_qualname(code)
        site = CallSite(make_id(path, qual, ordinal), path, qual, ordinal, line or 0)
    _site_cache[key] = (code, site)
    return site


_positions_cache = {}


def _positions(code):
    hit = _positions_cache.get(id(code))
    if hit is not None and hit[0] is code:
        return hit[1]
    pos = list(code.co_positions())
    _positions_cache[id(code)] = (code, pos)
    return pos


def resolve_frame(frame, callee=None):
    """CallSite for the call currently executing in `frame`."""
    if _HAS_POSITIONS:
        return resolve(frame.f_code, offset=frame.f_lasti, callee=callee)
    return resolve(frame.f_code, lineno=frame.f_lineno, callee=callee)


def user_stack(frame, limit=8):
    """Up to `limit` user frames, innermost first, normalized to
    {"path", "qualname", "line"}. Library frames are skipped."""
    out = []
    f = frame
    while f is not None and len(out) < limit:
        code = f.f_code
        if is_user_file(code.co_filename):
            out.append({"path": relative_path(code.co_filename),
                        "qualname": _code_qualname(code),
                        "line": f.f_lineno})
        f = f.f_back
    return out


def first_user_frame(start):
    """Walks outward from `start` to the first user-code frame."""
    f = start
    while f is not None and not is_user_file(f.f_code.co_filename):
        f = f.f_back
    return f


def _clear_caches():
    """For tests that rewrite a source file between resolutions."""
    _site_cache.clear()
    _index_cache.clear()
    _positions_cache.clear()
    _rel_cache.clear()
    _user_cache.clear()
    linecache.clearcache()
    gitutil.repo_root.cache_clear()
