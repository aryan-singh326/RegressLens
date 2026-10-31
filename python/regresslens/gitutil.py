"""Small git helpers: repository info, revision lists, worktrees.

Everything shells out to the git CLI. Functions that only describe the
environment (repo_root, head_info) return None outside a repository
instead of raising, because RegressLens works on plain directories too.
"""
import functools
import os
import shutil
import subprocess
import tempfile


class GitError(RuntimeError):
    pass


def _git(args, cwd, check=True):
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except OSError as exc:  # git not installed
        raise GitError(f"git unavailable: {exc}") from exc
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


@functools.lru_cache(maxsize=256)
def repo_root(path):
    """Top-level directory of the repository containing `path`, found by
    walking up to the nearest `.git` entry (a directory, or a file in a
    worktree). Pure filesystem walk, no subprocess, because call-site
    resolution calls it on hot paths."""
    d = os.path.abspath(path)
    if os.path.isfile(d):
        d = os.path.dirname(d)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def head_info(path):
    """(sha, dirty) for the repository containing `path`, or (None, None)."""
    root = repo_root(path)
    if root is None:
        return None, None
    try:
        sha = _git(["rev-parse", "HEAD"], root).strip()
        dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], root).strip())
    except GitError:
        return None, None
    return sha, dirty


def rev_parse(rev, cwd):
    return _git(["rev-parse", "--verify", f"{rev}^{{commit}}"], cwd).strip()


def first_parent_range(good, bad, cwd):
    """Commits in good..bad along the first-parent chain, oldest first."""
    out = _git(["rev-list", "--first-parent", "--reverse", f"{good}..{bad}"], cwd)
    return [line.strip() for line in out.splitlines() if line.strip()]


def changed_files(commit, cwd):
    """Paths (repo-relative, forward slashes) changed by `commit` relative
    to its first parent. A root commit lists all of its files."""
    out = _git(["diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                "-m", "--first-parent", commit], cwd)
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def subject(commit, cwd):
    return _git(["log", "-1", "--format=%s", commit], cwd).strip()


def short(sha):
    return sha[:10] if sha else "?"


class Worktree:
    """A detached git worktree in a temporary directory, removed on close.

    Used by compare and bisect so the user's working tree is never
    touched. `checkout()` reuses the same directory for another commit.
    """

    def __init__(self, repo, rev, prefix="rglns-wt-"):
        self.repo = repo_root(repo) or repo
        self.path = tempfile.mkdtemp(prefix=prefix)
        # `git worktree add` needs a non-existent (or empty) target.
        os.rmdir(self.path)
        _git(["worktree", "add", "--detach", "--force", self.path, rev], self.repo)
        self.rev = rev

    def checkout(self, rev):
        _git(["checkout", "--detach", "--force", rev], self.path)
        _git(["clean", "-fdq"], self.path)
        self.rev = rev

    def close(self):
        if self.path and os.path.exists(self.path):
            _git(["worktree", "remove", "--force", self.path], self.repo, check=False)
            shutil.rmtree(self.path, ignore_errors=True)
        _git(["worktree", "prune"], self.repo, check=False)
        self.path = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
