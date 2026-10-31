"""rglns bisect: find the commit that introduced a slowdown.

    rglns bisect GOOD BAD --cmd "python bench.py" [--setup CMD] [--lambda 4]

Probabilistic bisection over the first-parent chain good..bad
(stats/bisection.py). The parts specific to RegressLens:

  * attribution prior: the endpoints are measured with observation on;
    the call site that regressed most is located, and commits touching
    any file in its recorded stack frames get prior weight 1 + lambda;
  * batch size per visit b = clamp(ceil(build_time / run_time), 3, 20),
    because checkout and build cost dominate small batches;
  * stops when one commit holds >= 95% posterior mass, and always
    prints the 95% credible set;
  * unbuildable commits (setup fails) are excluded, and the report
    names the range if the culprit falls next to them;
  * the endpoints are re-measured every 8 visits as drift anchors;
  * after convergence a step model is compared with a linear-drift
    model by Bayes factor, and a "gradual regression, no single commit"
    is reported if drift wins by more than 10x;
  * results are cached by commit SHA (and command), so re-running a
    bisect reuses earlier measurements. Measurement is serialized; only
    one timing runs at a time.
"""
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np

from .. import analysis, gitutil, store
from ..launcher import Launcher
from ..stats.bisection import Bisector
from .common import eprint, write_output

ANCHOR_EVERY = 8


def _cmd_hash(cmd, setup):
    return hashlib.sha1(json.dumps([cmd, setup]).encode()).hexdigest()[:12]


def _tail(text, lines=30):
    """The last lines of a failed command's output, without pip's
    upgrade notices (pip prints the real build error well above them)."""
    kept = [l for l in (text or "").strip().splitlines() if not l.startswith("[notice]")]
    return "\n".join(kept[-lines:])


def _y(run):
    """log(time) - log(canary blend): the blend is the geometric mean of
    the compute and memory canaries, so a uniformly slower machine
    shifts both terms and cancels."""
    t = math.log(run.wall_ns)
    if run.canary_compute_ns and run.canary_mem_ns:
        t -= 0.5 * (math.log(run.canary_compute_ns) + math.log(run.canary_mem_ns))
    return t


class _Measurer:
    def __init__(self, args, repo, commits):
        self.args = args
        self.repo = repo
        self.commits = commits
        self.label = f"bisect:{_cmd_hash(args.cmd, args.setup)}"
        self.worktree = gitutil.Worktree(repo, commits[0], prefix="rglns-bisect-")
        self.workdir = self.worktree.path
        rel = os.path.relpath(os.getcwd(), repo)
        if not rel.startswith(".."):
            self.workdir = os.path.join(self.worktree.path, rel)
        self.built = None
        self.build_s = 0.0
        self.run_s = None
        self.runs_used = 0
        self.unbuildable = set()
        self.failures = {}  # sha -> why it could not be measured
        self.sessions = {}

    def close(self):
        self.worktree.close()

    def _checkout(self, sha):
        if self.built == sha:
            return True
        t0 = time.time()
        self.worktree.checkout(sha)
        if self.args.setup:
            proc = subprocess.run(self.args.setup, shell=True, cwd=self.workdir,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                self.failures[sha] = (f"setup exited with {proc.returncode}\n"
                                      f"{_tail(proc.stderr or proc.stdout)}")
                return False
        self.build_s = time.time() - t0
        self.built = sha
        return True

    def cached(self, conn, sha):
        """y values from earlier bisects of the same command at this SHA."""
        ids = [s.session_id for s in store.find_sessions(conn, label=self.label, git_sha=sha)]
        return [_y(r) for s in store.load_sessions(conn, ids, with_calls=False)
                for r in s.runs if r.wall_ns]

    def measure(self, conn, i, n_runs, attribution=False):
        """n_runs new runs at commit i; returns their y values, or None if
        the commit cannot be built or the command fails there.

        Timing runs are never observed: observation adds start-up cost,
        and every y must be measured under identical conditions.
        `attribution=True` instead takes observed runs whose only purpose
        is locating the regressed call site; they are stored under a
        separate label and never used as timing data."""
        sha = self.commits[i]
        if sha in self.unbuildable:
            return None
        if not self._checkout(sha):
            self.unbuildable.add(sha)
            return None
        label = self.label + (":attribution" if attribution else "")
        launcher = Launcher(self.args.cmd, label=label, cwd=self.workdir,
                            observe=attribution, capture=True, quiet=True,
                            meta={"commit_index": i}, timeout=self.args.timeout)
        launcher.start_session(conn)  # git_sha = the worktree's HEAD = sha
        for _ in range(n_runs):
            r = launcher.run(conn)
            self.runs_used += 1
            if r.exit_code != 0:
                self.failures[sha] = f"command exited with {r.exit_code}\n{_tail(r.stderr)}"
                self.unbuildable.add(sha)
                return None
            if not attribution and self.run_s is None:
                self.run_s = r.wall_ns / 1e9
        if attribution:
            self.sessions.setdefault(i, []).append(launcher.session_id)
            return []
        loaded = store.load_sessions(conn, [launcher.session_id], with_calls=False)
        ys = [_y(r) for s in loaded for r in s.runs if r.wall_ns]
        return ys

    def batch(self):
        if not self.run_s:
            return 3
        return int(min(max(math.ceil(self.build_s / self.run_s), 3), 20))


def _attribution_prior(conn, args, measurer, repo, n):
    """(prior weights for k=1..N or None, description, details) from the
    most-regressed call site between the endpoints. details is None or
    {"files": the site's stack files, "flagged": SHAs of the commits
    touching them}."""
    lam = args.lam
    if lam <= 0:
        return None, "uniform prior (--lambda 0)", None
    base = store.load_sessions(conn, measurer.sessions.get(0, []))
    cur = store.load_sessions(conn, measurer.sessions.get(n, []))
    site_ids = {c.call_site_id for s in base + cur for r in s.runs for c in r.calls}
    if not site_ids:
        return None, "uniform prior (no NumPy calls observed at the endpoints)", None
    info = store.call_site_info(conn, site_ids)
    if args.call_site:
        target = next((k for k in info if k.startswith(args.call_site)), None)
    else:
        report = analysis.compare_sessions(base, cur, info, theta=args.theta,
                                           min_runs=2, with_diagnosis=False)
        candidates = [s for s in report.sites if s.comparison is not None
                      and s.key != analysis.WALL]
        target = max(candidates, key=lambda s: s.comparison.estimate).key if candidates else None
    if target is None or target not in info:
        return None, "uniform prior (no regressed call site found)", None
    files = {info[target]["path"]} | {f["path"] for f in info[target]["stack"]}
    a = np.zeros(n)
    for k in range(1, n + 1):
        touched = set(gitutil.changed_files(measurer.commits[k], repo))
        a[k - 1] = 1.0 if touched & files else 0.0
    site = analysis._site_display(info[target], target)
    details = {"files": sorted(files),
               "flagged": [measurer.commits[k] for k in range(1, n + 1) if a[k - 1]]}
    if not a.any():
        # e.g. the regression is inside a library and the recorded stack
        # holds only the benchmark's own files
        return None, (f"uniform prior (no commit touches the stack files of {site}: "
                      f"{sorted(files)})"), details
    desc = (f"attribution prior from {site}: "
            f"{int(a.sum())}/{n} commits touch its stack files {sorted(files)}")
    return 1.0 + lam * a, desc, details


def run_bisect(args):
    repo = gitutil.repo_root(os.getcwd())
    if repo is None:
        raise SystemExit("error: rglns bisect must run inside a git repository")
    good = gitutil.rev_parse(args.good, repo)
    bad = gitutil.rev_parse(args.bad, repo)
    chain = gitutil.first_parent_range(good, bad, repo)
    if not chain or chain[-1] != bad:
        raise SystemExit(f"error: {args.bad} is not a first-parent descendant of {args.good}")
    commits = [good] + chain
    n = len(commits) - 1
    eprint(f"bisecting {n} commits ({gitutil.short(good)}..{gitutil.short(bad)})")
    measurer = _Measurer(args, repo, commits)
    started = time.time()
    try:
        conn = store.connect()
        try:
            return _bisect(args, conn, measurer, repo, commits, n, started)
        finally:
            conn.close()
    finally:
        measurer.close()


def _bisect(args, conn, measurer, repo, commits, n, started):
    endpoint_runs = max(args.endpoint_runs, 3)
    ys0 = measurer.measure(conn, 0, endpoint_runs)
    ysn = measurer.measure(conn, n, endpoint_runs)
    if ys0 is None or ysn is None:
        failed = [(name, measurer.commits[i]) for name, i, ys in
                  (("good", 0, ys0), ("bad", n, ysn)) if ys is None]
        raise SystemExit("\n".join(
            f"error: cannot bisect: at the {name} endpoint {gitutil.short(sha)}, "
            f"{measurer.failures.get(sha, 'the command failed')}" for name, sha in failed))
    if args.lam > 0 or args.call_site:
        for end in (0, n):
            measurer.measure(conn, end, 3, attribution=True)
    gap = np.mean(ysn) - np.mean(ys0)
    eprint(f"endpoints: bad vs good {math.expm1(gap):+.1%} (canary-adjusted)")
    if gap <= math.log1p(args.theta):
        eprint(f"no regression above {args.theta:.0%} between the endpoints; nothing to bisect")
        return 2

    prior, prior_desc, prior_details = _attribution_prior(conn, args, measurer, repo, n)
    eprint(prior_desc)
    bis = Bisector(n, prior)
    if not args.no_cache:
        reused = 0
        for i in range(1, n):
            ys = measurer.cached(conn, commits[i])
            if ys:
                bis.add(i, ys)
                reused += len(ys)
        if reused:
            eprint(f"reusing {reused} cached runs from earlier bisects (--no-cache to disable)")
    bis.add(0, ys0)
    bis.add(n, ysn)
    b = measurer.batch()
    eprint(f"batch size per visit: {b} (build {measurer.build_s:.1f} s, run "
           f"{(measurer.run_s or 0):.2f} s)")
    visits = 0
    while True:
        k, p = bis.best()
        if p >= args.confidence or measurer.runs_used >= args.max_runs:
            break
        if visits and visits % ANCHOR_EVERY == 0:
            for end in (0, n):
                ys = measurer.measure(conn, end, b)
                if ys:
                    bis.add(end, ys)
        i = bis.next_commit()
        if i is None:
            break
        ys = measurer.measure(conn, i, b)
        if ys is None:
            eprint(f"  commit {i} {gitutil.short(commits[i])}: unbuildable or failing, excluded")
            bis.exclude(i)
            continue
        bis.add(i, ys)
        visits += 1
        k, p = bis.best()
        eprint(f"  visit {visits}: commit {i} {gitutil.short(commits[i])} "
               f"{math.expm1(np.mean(ys) - bis.m_good):+.1%} vs good; best {k} p={p:.2f}")

    result = _result(args, bis, commits, repo, measurer, started, prior_desc, prior_details)
    print(render_text(result))
    if args.json:
        write_output(args.json, json.dumps(result, indent=2, default=float))
    args.verdict_counts = {"converged" if result["converged"] else "inconclusive": 1}
    return 0 if result["converged"] else 3


def _result(args, bis, commits, repo, measurer, started, prior_desc, prior_details):
    k, p = bis.best()
    cred = bis.credible_set(0.95)
    log_bf = bis.drift_log_bayes_factor()
    excluded_idx = sorted(i for i, c in enumerate(commits) if c in measurer.unbuildable)
    # Unbuildable commits just before the culprit make them indistinguishable from it.
    lo = k
    while lo - 1 in excluded_idx:
        lo -= 1
    def describe(i):
        return {"index": i, "sha": commits[i], "subject": gitutil.subject(commits[i], repo)}
    return {
        "converged": p >= args.confidence,
        "culprit": describe(k), "probability": p,
        "credible_set": [describe(i) for i in cred],
        "unbuildable_range": [describe(i) for i in range(lo, k + 1)] if lo < k else None,
        "gradual": log_bf > math.log(10),
        "drift_bayes_factor": math.exp(min(log_bf, 700)) if math.isfinite(log_bf) else 0.0,
        "effect": math.expm1(bis.m_bad - bis.m_good),
        "runs": measurer.runs_used, "elapsed_s": time.time() - started,
        "prior": prior_desc, "commits": len(commits) - 1,
        "prior_files": prior_details["files"] if prior_details else None,
        "prior_flagged": prior_details["flagged"] if prior_details else None,
    }


def render_text(r):
    c = r["culprit"]
    lines = ["rglns bisect: " + ("converged" if r["converged"] else "did not converge "
                                 "(budget exhausted)")]
    if r["gradual"]:
        lines.append(f"  gradual regression, no single commit: a linear drift model fits "
                     f"{r['drift_bayes_factor']:.0f}x better than any single step")
    lines.append(f"  most likely first bad commit: {gitutil.short(c['sha'])} {c['subject']} "
                 f"(p = {r['probability']:.3f}, effect {r['effect']:+.1%})")
    cs = r["credible_set"]
    lines.append(f"  95% credible set ({len(cs)} commit{'s' if len(cs) != 1 else ''}): "
                 + ", ".join(gitutil.short(x["sha"]) for x in cs))
    if r["unbuildable_range"]:
        rng = r["unbuildable_range"]
        lines.append(f"  culprit range includes unbuildable commits: "
                     f"{gitutil.short(rng[0]['sha'])}..{gitutil.short(rng[-1]['sha'])}")
    lines.append(f"  {r['runs']} runs over {r['commits']} commits in {r['elapsed_s']:.0f} s; "
                 f"{r['prior']}")
    return "\n".join(lines)
