"""Runs measured commands: one session per CLI invocation, one run per
subprocess.

For each run the launcher
  * generates RGLNS_RUN_ID (the session's RGLNS_SESSION_ID is fixed);
  * prepends regresslens/_boot (the injected sitecustomize.py) to
    PYTHONPATH and sets RGLNS_ACTIVE=1, so every Python process the
    command starts is observed without code changes (RGLNS_OBSERVE=0
    keeps the bootstrap but skips observation: same start-up cost, so
    observed and unobserved runs are never mixed in one comparison);
  * drift canaries: for Python commands the measured process samples
    them itself (see observe/bootstrap.py for why the CLI process must
    not); for other commands the CLI samples right before and after;
  * measures wall time around the process and subtracts the time the
    child spent on its own canary samples;
  * stores the run row (wall time, canaries, exit code).
"""
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass

from . import canary as _canary
from . import gitutil, hw, store

BOOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_boot")
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PYTHON_TOKEN = re.compile(r"^python(3(\.\d+)?)?(\.exe)?$", re.IGNORECASE)


def split_command(cmd):
    """argv list for a command given as a list or a shell-like string. A
    leading `python`/`python3` is replaced by this interpreter, so the
    measured program runs in the same environment as rglns."""
    if isinstance(cmd, str):
        argv = shlex.split(cmd, posix=(os.name != "nt"))
        if os.name == "nt":
            argv = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] == '"' else a for a in argv]
    else:
        argv = list(cmd)
    if argv and _PYTHON_TOKEN.match(os.path.basename(argv[0])):
        argv[0] = sys.executable
    return argv


def with_operator_observation(argv):
    """Rewrites `python SCRIPT ARGS` to run SCRIPT through the Tier D
    runner (the main script is not imported, so the import hook alone
    cannot rewrite it). Other command shapes are returned unchanged."""
    if len(argv) >= 2 and argv[0] == sys.executable and not argv[1].startswith("-"):
        return [argv[0], "-m", "regresslens.observe.tierd", *argv[1:]]
    return argv


@dataclass
class RunResult:
    run_id: str
    exit_code: int
    wall_ns: float
    stdout: str = None
    stderr: str = None


class Launcher:
    def __init__(self, cmd, *, label, cwd=None, observe=True, accelerate=False,
                 observe_operators=False, canaries=None, capture=False, quiet=False,
                 meta=None, extra_env=None, session_id=None, timeout=None):
        self.argv = split_command(cmd)
        if observe_operators:
            self.argv = with_operator_observation(self.argv)
        self.label = label
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.observe = observe
        self.accelerate = accelerate
        self.observe_operators = observe_operators
        self.canaries = _canary.enabled() if canaries is None else canaries
        self.capture = capture
        self.quiet = quiet
        self.meta = dict(meta or {})
        self.extra_env = dict(extra_env or {})
        self.timeout = timeout
        self.session_id = session_id
        self._canary = None
        self.python_child = bool(self.argv) and (
            self.argv[0] == sys.executable or bool(_PYTHON_TOKEN.match(os.path.basename(self.argv[0]))))
        self.child_canaries = self.canaries and self.python_child

    def start_session(self, conn=None):
        if self.session_id is not None:
            return self.session_id
        sha, dirty = gitutil.head_info(self.cwd)
        self.meta.setdefault("cmd", self.argv)
        own = conn is None
        conn = conn or store.connect()
        try:
            self.session_id = store.create_session(
                conn, self.label, hw_fingerprint=hw.fingerprint(), git_sha=sha,
                git_dirty=dirty, meta=self.meta)
        finally:
            if own:
                conn.close()
        return self.session_id

    def _env(self, run_id):
        env = os.environ.copy()
        env.update(self.extra_env)
        env["RGLNS_SESSION_ID"] = self.session_id
        env["RGLNS_RUN_ID"] = run_id
        env["RGLNS_DB_PATH"] = os.path.abspath(store.db_path())
        env["RGLNS_PKG_PARENT"] = PKG_PARENT
        if self.observe or self.child_canaries:
            env["RGLNS_ACTIVE"] = "1"
            env["RGLNS_OBSERVE"] = "1" if self.observe else "0"
            env["RGLNS_CHILD_CANARY"] = "1" if self.child_canaries else "0"
            existing = env.get("PYTHONPATH", "")
            parts = [BOOT_DIR] + [p for p in existing.split(os.pathsep) if p and p != BOOT_DIR]
            env["PYTHONPATH"] = os.pathsep.join(parts)
        else:
            env.pop("RGLNS_ACTIVE", None)
        if self.accelerate:
            env["RGLNS_ACCELERATE"] = "1"
        if self.observe_operators:
            env["RGLNS_TIERD"] = "1"
        if self.quiet:
            env["RGLNS_QUIET"] = "1"
        return env

    def _canary_sample(self, where):
        """CLI-side sample, only for non-Python commands (a Python child
        samples its own canaries)."""
        if not self.canaries or self.child_canaries:
            return []
        if self._canary is None:
            self._canary = _canary.Canary()
        return [self._canary.measure(where)]

    def run(self, conn=None):
        """Runs the command once and records the run."""
        self.start_session(conn)
        run_id = store.new_id()
        env = self._env(run_id)
        before = self._canary_sample("start")
        started_at = time.time()
        stdout = stderr = None
        t0 = time.perf_counter_ns()
        try:
            proc = subprocess.run(self.argv, cwd=self.cwd, env=env, timeout=self.timeout,
                                  capture_output=self.capture,
                                  text=True if self.capture else None)
            code = proc.returncode
            if self.capture:
                stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            code = -9
        except OSError as exc:
            code = 127
            stderr = str(exc)
            print(f"rglns: cannot run {self.argv[0]!r}: {exc}", file=sys.stderr)
        wall = float(time.perf_counter_ns() - t0)
        after = self._canary_sample("end")

        own = conn is None
        conn = conn or store.connect()
        try:
            row = conn.execute("SELECT wall_ns, canary_json FROM runs WHERE run_id = ?",
                               (run_id,)).fetchone()
            if row and row[0]:
                wall = row[0]  # the program's own run time, reported at its exit
            else:
                child_samples = json.loads(row[1]) if row and row[1] else []
                wall -= sum(s.get("cost_ns", 0.0) for s in child_samples)
            store.upsert_run(conn, run_id, self.session_id, wall_ns=wall, exit_code=code,
                             started_at=started_at, canary_samples=before + after)
        finally:
            if own:
                conn.close()
        return RunResult(run_id, code, wall, stdout, stderr)
