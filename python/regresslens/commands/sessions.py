"""rglns run / baseline / check / profile / sessions."""
import datetime
import math
import statistics
from collections import defaultdict

from .. import analysis, hw, store
from ..launcher import Launcher
from .common import command_from_args, eprint, run_n, write_output


def _launcher(args, label, interactive=False):
    """`interactive` (rglns run): the command's output and the coverage
    report go to the terminal. Otherwise (repeated measurement runs) the
    child's output is captured and shown only when a run fails."""
    return Launcher(command_from_args(args), label=label,
                    accelerate=getattr(args, "accelerate", False),
                    observe_operators=getattr(args, "observe_operators", False),
                    canaries=False if getattr(args, "no_canary", False) else None,
                    quiet=getattr(args, "quiet", False) or not interactive,
                    capture=not interactive)


def cmd_run(args):
    """Runs a command with zero-code-change observation."""
    launcher = _launcher(args, args.label or "run", interactive=True)
    results = run_n(launcher, args.runs, verbose=args.runs > 1)
    eprint(f"rglns: session {launcher.session_id[:12]} ({args.runs} run(s), label "
           f"'{launcher.label}'); see `rglns profile --session {launcher.session_id[:12]}`")
    return results[-1].exit_code if results else 1


def cmd_baseline(args):
    launcher = _launcher(args, f"baseline:{args.name}")
    eprint(f"Recording baseline '{args.name}': {args.runs} runs")
    results = run_n(launcher, args.runs)
    ok = sum(1 for r in results if r.exit_code == 0)
    if ok == 0:
        eprint("error: every run failed; no baseline recorded")
        return 1
    conn = store.connect()
    try:
        sessions = store.find_sessions(conn, label=f"baseline:{args.name}")
        session = store.load_sessions(conn, [launcher.session_id])[0]
    finally:
        conn.close()
    sites = {c.call_site_id for r in session.runs for c in r.calls}
    print(f"Baseline '{args.name}' session {launcher.session_id[:12]} stored: {ok}/{args.runs} "
          f"runs, {len(sites)} call sites observed.")
    print(f"  Sessions under this baseline: {len(sessions)}"
          + ("" if len(sessions) >= 3 else
             " (record >= 3 on different occasions for non-advisory verdicts)"))
    print(f"  Hardware: {hw.describe(hw.fingerprint())}")
    return 0


def cmd_check(args):
    conn = store.connect()
    try:
        base = store.find_sessions(conn, label=f"baseline:{args.baseline}",
                                   limit=args.baseline_sessions)
        if not base:
            eprint(f"error: no baseline named '{args.baseline}'. Record one with "
                   f"`rglns baseline --name {args.baseline} -- <command>`.")
            return 2
        if args.current_session:
            cur_ids = [_resolve_session(conn, args.current_session)]
        else:
            launcher = _launcher(args, f"check:{args.baseline}")
            eprint(f"Running {args.runs} runs to compare against baseline '{args.baseline}'")
            results = run_n(launcher, args.runs)
            if not any(r.exit_code == 0 for r in results):
                eprint("error: every run failed")
                return 2
            cur_ids = [launcher.session_id]
        base_full = store.load_sessions(conn, [s.session_id for s in base])
        cur_full = store.load_sessions(conn, cur_ids)
        site_ids = {c.call_site_id for s in base_full + cur_full for r in s.runs
                    for c in r.calls}
        info = store.call_site_info(conn, site_ids)
    finally:
        conn.close()
    report = analysis.compare_sessions(base_full, cur_full, info, theta=args.theta,
                                       alpha=args.alpha, B=args.bootstrap, seed=args.seed)
    print(analysis.render_text(report, f"rglns check vs baseline '{args.baseline}'"))
    if args.json:
        write_output(args.json, analysis.render_json(report))
    args.verdict_counts = _verdict_counts(report)
    return 1 if report.regressions else 0


def _verdict_counts(report):
    counts = defaultdict(int)
    for s in report.sites:
        if s.verdict:
            counts[s.verdict.verdict] += 1
    return dict(counts)


def _resolve_session(conn, prefix):
    rows = conn.execute("SELECT session_id FROM sessions WHERE session_id LIKE ?",
                        (prefix + "%",)).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"error: session '{prefix}' matches {len(rows)} sessions")
    return rows[0][0]


def cmd_profile(args):
    conn = store.connect()
    try:
        if args.session:
            ids = [_resolve_session(conn, args.session)]
        elif args.label:
            ids = [s.session_id for s in store.find_sessions(conn, label=args.label)]
            if not ids:
                eprint(f"error: no sessions labeled '{args.label}'")
                return 2
        else:
            launcher = _launcher(args, "profile")
            eprint(f"Profiling: {args.runs} runs")
            run_n(launcher, args.runs)
            ids = [launcher.session_id]
        sessions = store.load_sessions(conn, ids)
        site_ids = {c.call_site_id for s in sessions for r in s.runs for c in r.calls}
        info = store.call_site_info(conn, site_ids)
    finally:
        conn.close()
    print(render_profile(sessions, info))
    return 0


def render_profile(sessions, info):
    totals = defaultdict(lambda: {"ns": 0.0, "count": 0, "ops": set(), "kernels": defaultdict(int),
                                  "samples": []})
    walls = []
    for s in sessions:
        for run in s.runs:
            if run.wall_ns:
                walls.append(run.wall_ns)
            for c in run.calls:
                t = totals[c.call_site_id]
                t["ns"] += c.sum_ns
                t["count"] += c.count
                t["ops"].add(c.operator)
                t["kernels"][c.kernel] += c.count
                t["samples"].extend(c.samples)
    runs = sum(len(s.runs) for s in sessions)
    observed = sum(t["ns"] for t in totals.values())
    lines = [f"Runtime breakdown by call site ({len(sessions)} session(s), {runs} runs)"]
    if walls:
        lines.append(f"  median wall time per run: {statistics.median(walls) / 1e6:.2f} ms; "
                     f"observed calls: {observed / max(runs, 1) / 1e6:.2f} ms per run "
                     f"({100 * observed / max(sum(walls), 1):.1f}%)")
    lines.append("")
    lines.append(f"  {'share':>6}  {'calls/run':>9}  {'median':>10}  call site")
    for site_id, t in sorted(totals.items(), key=lambda kv: -kv[1]["ns"]):
        share = 100 * t["ns"] / observed if observed else 0
        med = statistics.median(t["samples"]) if t["samples"] else float("nan")
        kern = ",".join(f"{k}" for k in sorted(t["kernels"]))
        disp = analysis._site_display(info.get(site_id), site_id)
        lines.append(f"  {share:5.1f}%  {t['count'] / max(runs, 1):9.1f}  "
                     f"{_fmt_ns(med):>10}  {disp}  [{', '.join(sorted(t['ops']))}; {kern}]")
    return "\n".join(lines)


def _fmt_ns(ns):
    if not math.isfinite(ns):
        return "n/a"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.2f} {unit}"
    return f"{ns:.0f} ns"


def cmd_sessions(args):
    conn = store.connect()
    try:
        sessions = store.find_sessions(conn, label=args.label, limit=args.limit)
        counts = dict(conn.execute("SELECT session_id, COUNT(*) FROM runs GROUP BY session_id"))
    finally:
        conn.close()
    for s in sessions:
        when = datetime.datetime.fromtimestamp(s.started_at).strftime("%Y-%m-%d %H:%M")
        sha = (s.git_sha or "")[:10] + ("*" if s.git_dirty else "")
        print(f"{s.session_id[:12]}  {when}  {counts.get(s.session_id, 0):>3} runs  "
              f"{sha:<11}  {s.label}")
    return 0
