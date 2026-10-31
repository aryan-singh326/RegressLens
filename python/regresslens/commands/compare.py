"""rglns compare: interleaved A/B measurement with anytime-valid stopping.

    rglns compare REV_A REV_B --cmd "python bench.py" [--setup CMD]
                  [--min-blocks 4] [--max-blocks 100] [--alpha 0.05] [--theta 0.05]

REV is a git revision, `@` for the working tree, or a directory via
--dir-a/--dir-b. Each git side is checked out into its own temporary
git worktree, so the user's checkout is never touched.

Both sides run in alternating blocks, ABBA or BAAB at random. A
position-balanced block cancels any linear drift within it (thermal
ramp, a backup job starting). Each block gives
d = mean(log t_B) - mean(log t_A), and after every block an
always-valid confidence sequence decides whether to stop
(stats/sequential.py). Obvious cases stop after a handful of blocks.

After stopping:
  * secondary check: the drift-bounded correction (WP3b) using the
    canary samples of A and B runs; a regression that does not survive
    it is downgraded to inconclusive;
  * an exploratory per-call-site drill-down, BH-controlled.
"""
import json
import math
import os
import random
import subprocess
import time

import numpy as np

from .. import analysis, gitutil, store
from ..launcher import Launcher
from ..stats import verdict as V
from ..stats.sequential import ConfidenceSequence
from .common import eprint, write_output

MARKER = "<!-- rglns-compare -->"


class _Side:
    def __init__(self, name, rev, path, worktree=None):
        self.name, self.rev, self.path, self.worktree = name, rev, path, worktree


def _prepare_sides(args, repo):
    sides = []
    for name, rev, directory in (("A", args.rev_a, args.dir_a), ("B", args.rev_b, args.dir_b)):
        if directory:
            sides.append(_Side(name, rev or directory, os.path.abspath(directory)))
        elif rev in (None, "@"):
            if rev is None:
                raise SystemExit(f"error: side {name}: give a revision or --dir-{name.lower()}")
            sides.append(_Side(name, "@", os.getcwd()))  # the working tree as-is
        else:
            if repo is None:
                raise SystemExit("error: git revisions need a git repository "
                                 "(or use --dir-a/--dir-b)")
            sha = gitutil.rev_parse(rev, repo)
            wt = gitutil.Worktree(repo, sha, prefix=f"rglns-{name.lower()}-")
            sides.append(_Side(name, rev, _subdir(wt.path, repo), wt))
    return sides


def _subdir(worktree_path, repo):
    """Runs from the same relative directory inside the worktree as the
    user's current directory inside the repository."""
    rel = os.path.relpath(os.getcwd(), repo)
    return os.path.join(worktree_path, rel) if not rel.startswith("..") else worktree_path


def _setup(side, command):
    if not command:
        return
    eprint(f"setup {side.name} ({side.rev}): {command}")
    proc = subprocess.run(command, shell=True, cwd=side.path)
    if proc.returncode != 0:
        raise SystemExit(f"error: setup failed for side {side.name} (exit {proc.returncode})")


def _canary_check(conn, sessions, theta_log, state, decision):
    """Drift-bounded secondary check on the interval of the mean block
    difference, using each side's canary samples."""
    logs = {}
    for side, sid in sessions.items():
        rows = conn.execute("SELECT canary_compute_ns, canary_mem_ns FROM runs WHERE "
                            "session_id = ? AND exit_code = 0", (sid,)).fetchall()
        c = [math.log(r[0]) for r in rows if r[0]]
        m = [math.log(r[1]) for r in rows if r[1]]
        logs[side] = (np.mean(c) if c else None, np.mean(m) if m else None)
    (ca, ma), (cb, mb) = logs["A"], logs["B"]
    if None in (ca, ma, cb, mb):
        return decision, None
    dc, dm = cb - ca, mb - ma
    low = state.low - max(dc, dm)
    high = state.high - min(dc, dm)
    note = (f"canary shift B vs A: compute {math.expm1(dc):+.1%}, memory {math.expm1(dm):+.1%}; "
            f"drift-bounded interval [{math.expm1(low):+.1%}, {math.expm1(high):+.1%}]")
    if decision == V.REGRESSION and low <= theta_log:
        return V.INCONCLUSIVE, note + " (regression does not survive drift bounds)"
    if decision == V.IMPROVEMENT and high >= -theta_log:
        return V.INCONCLUSIVE, note + " (improvement does not survive drift bounds)"
    return decision, note


def run_compare(args):
    repo = gitutil.repo_root(os.getcwd())
    sides = _prepare_sides(args, repo)
    try:
        return _run(args, sides)
    finally:
        for s in sides:
            if s.worktree is not None:
                s.worktree.close()


def _run(args, sides):
    side = {s.name: s for s in sides}
    for s in sides:
        _setup(s, getattr(args, f"setup_{s.name.lower()}") or args.setup)
    compare_id = store.new_id()[:12]
    launchers = {}
    for s in sides:
        launchers[s.name] = Launcher(
            args.cmd, label=f"compare:{s.name}", cwd=s.path, observe=not args.no_observe,
            capture=True, quiet=True, timeout=args.timeout,
            meta={"compare_id": compare_id, "side": s.name, "rev": s.rev})
    conn = store.connect()
    try:
        for name, l in launchers.items():
            l.start_session(conn)
            # One discarded warm-up run per side (bytecode compilation,
            # first-touch page faults, CPU frequency ramp).
            warm = Launcher(args.cmd, label="compare:warmup", cwd=side[name].path,
                            observe=False, capture=True, quiet=True, canaries=False,
                            timeout=args.timeout)
            r = warm.run(conn)
            if r.exit_code != 0:
                eprint(f"error: command fails on side {name} (exit {r.exit_code}):\n"
                       f"{(r.stderr or '').strip()[-1500:]}")
                return 2

        rng = random.Random(args.seed)
        cs = ConfidenceSequence(args.theta, args.alpha, args.epsilon, args.min_blocks,
                                args.max_blocks)
        decision, started = None, time.time()
        while decision is None:
            order = "ABBA" if rng.random() < 0.5 else "BAAB"
            logs = {"A": [], "B": []}
            for name in order:
                r = launchers[name].run(conn)
                if r.exit_code != 0:
                    eprint(f"error: run on side {name} failed (exit {r.exit_code}):\n"
                           f"{(r.stderr or '').strip()[-1500:]}")
                    return 2
                logs[name].append(math.log(r.wall_ns))
            cs.add(np.mean(logs["B"]) - np.mean(logs["A"]))
            st = cs.state()
            decision = cs.decision()
            if st is not None:
                eprint(f"  block {st.n:3d} {order}: B vs A {math.expm1(st.mean):+.2%}, "
                       f"CS [{math.expm1(st.low):+.2%}, {math.expm1(st.high):+.2%}]")
            else:
                eprint(f"  block {len(cs.blocks):3d} {order}: d = "
                       f"{math.expm1(cs.blocks[-1]):+.2%}")
        st = cs.state()
        sessions = {name: l.session_id for name, l in launchers.items()}
        final, canary_note = _canary_check(conn, sessions, cs.theta_log, st, decision)

        drill = None
        if not args.no_observe:
            loaded = store.load_sessions(conn, [sessions["A"], sessions["B"]])
            site_ids = {c.call_site_id for s in loaded for r in s.runs for c in r.calls}
            info = store.call_site_info(conn, site_ids)
            if site_ids:
                drill = analysis.compare_sessions([loaded[0]], [loaded[1]], info,
                                                  theta=args.theta, alpha=args.alpha,
                                                  exploratory=True, seed=args.seed)
    finally:
        conn.close()

    result = {
        "verdict": final, "sequential_verdict": decision,
        "rev_a": side["A"].rev, "rev_b": side["B"].rev,
        "effect": math.expm1(st.mean), "cs_low": math.expm1(st.low),
        "cs_high": math.expm1(st.high), "blocks": st.n, "runs": 4 * st.n,
        "theta": args.theta, "alpha": args.alpha, "canary_check": canary_note,
        "elapsed_s": time.time() - started, "sessions": sessions,
    }
    print(render_text(result, drill))
    if args.json:
        payload = dict(result)
        if drill is not None:
            payload["drilldown"] = analysis.to_dict(drill)
        write_output(args.json, json.dumps(payload, indent=2, default=float))
    if args.markdown:
        write_output(args.markdown, render_markdown(result, drill))
    args.verdict_counts = {final: 1}
    return 1 if final == V.REGRESSION else 0


def render_text(r, drill):
    lines = [f"rglns compare: {r['rev_a']} (A) vs {r['rev_b']} (B)",
             f"  verdict: {r['verdict'].upper()}",
             f"  effect of B vs A: {r['effect']:+.2%}, anytime-valid {1 - r['alpha']:.0%} "
             f"confidence sequence [{r['cs_low']:+.2%}, {r['cs_high']:+.2%}]",
             f"  {r['blocks']} interleaved blocks ({r['runs']} measured runs), threshold "
             f"{r['theta']:.0%}, {r['elapsed_s']:.0f} s"]
    if r["canary_check"]:
        lines.append(f"  {r['canary_check']}")
    if r["verdict"] == V.INCONCLUSIVE and r["sequential_verdict"] == V.INCONCLUSIVE:
        lines.append("  budget exhausted before the interval excluded the decision "
                     "boundaries; raise --max-blocks or reduce noise")
    if drill is not None:
        lines.append("")
        lines.append(analysis.render_text(drill, "Per-call-site drill-down (exploratory)"))
    return "\n".join(lines)


def render_markdown(r, drill):
    icon = {V.REGRESSION: ":red_circle:", V.IMPROVEMENT: ":green_circle:",
            V.EQUIVALENT: ":white_check_mark:"}.get(r["verdict"], ":grey_question:")
    lines = [MARKER,
             f"### {icon} RegressLens: **{r['verdict']}** "
             f"({r['effect']:+.1%} [{r['cs_low']:+.1%}, {r['cs_high']:+.1%}])",
             "",
             f"`{r['rev_b']}` vs `{r['rev_a']}`: {r['blocks']} interleaved ABBA/BAAB blocks "
             f"({r['runs']} runs), anytime-valid {1 - r['alpha']:.0%} confidence sequence, "
             f"threshold {r['theta']:.0%}."]
    if r["canary_check"]:
        lines += ["", f"Drift check: {r['canary_check']}"]
    if drill is not None:
        rows = [s for s in analysis._sorted_sites(drill) if s.comparison is not None][:15]
        if rows:
            lines += ["", "<details><summary>Per-call-site drill-down (exploratory, "
                      "Benjamini-Hochberg)</summary>", "",
                      "| verdict | effect [95% CI] | call site |", "|---|---|---|"]
            for s in rows:
                lines.append(f"| {s.verdict.verdict} | {analysis._interval(s.comparison)} | "
                             f"`{s.display}` |")
            lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
