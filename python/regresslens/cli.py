"""The `rglns` command-line tool.

  rglns run [--label X] [--runs N] [--accelerate] [--observe-operators] -- CMD...
  rglns baseline --name NAME [--runs N] -- CMD...
  rglns check --baseline NAME [--runs N] -- CMD...
  rglns profile (-- CMD... | --label X | --session ID)
  rglns compare REV_A REV_B --cmd "CMD" [--setup CMD] [--min-blocks 4] [--max-blocks 100]
  rglns bisect GOOD BAD --cmd "CMD" [--setup CMD] [--lambda 4]
  rglns watch --label X [--k 8] [--threshold 0.9]
  rglns verify [--cases N] [--seed S]
  rglns sessions [--label X]
  rglns telemetry on|off|status
  rglns ci comment --markdown FILE

Every measured command runs as a separate process (once per run), with
observation injected through sitecustomize, so the user's program needs
no changes. Exit codes: 0 ok, 1 regression found (check, compare) or
alert active (watch) or verification failure, 2 usage/infrastructure
error, 3 bisect did not converge.
"""
import argparse
import sys

from . import __version__, telemetry


def _add_run_options(p, runs_default):
    p.add_argument("--runs", type=int, default=runs_default, help="runs per session")
    p.add_argument("--script", help="(legacy) Python script to run; same as `-- python SCRIPT`")
    p.add_argument("--accelerate", action="store_true",
                   help="route eligible np.sum/np.mean calls to the native kernels")
    p.add_argument("--observe-operators", action="store_true",
                   help="also observe a+b and x[mask] in your own modules (Tier D, slower)")
    p.add_argument("--no-canary", action="store_true", help="skip drift canaries")
    p.add_argument("--quiet", action="store_true", help="no per-run coverage report")


def _add_stats_options(p):
    p.add_argument("--theta", "--threshold", dest="theta", type=float, default=0.05,
                   help="practical threshold as a fraction (default 0.05 = 5%%)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--json", help="write a JSON report to this path ('-' for stdout)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rglns", description="RegressLens: performance regression intelligence for "
        "scientific Python/NumPy pipelines.")
    parser.add_argument("--version", action="version", version=f"rglns {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a command with zero-code-change observation")
    p.add_argument("--label", help="session label (default 'run')")
    _add_run_options(p, 1)

    p = sub.add_parser("baseline", help="record a baseline session")
    p.add_argument("--name", required=True)
    _add_run_options(p, 20)

    p = sub.add_parser("check", help="compare a new session against a baseline")
    p.add_argument("--baseline", required=True)
    p.add_argument("--baseline-sessions", type=int, default=None,
                   help="use only the most recent N baseline sessions")
    p.add_argument("--current-session", help="analyze an existing session instead of running")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=None)
    _add_run_options(p, 20)
    _add_stats_options(p)

    p = sub.add_parser("profile", help="per-call-site runtime breakdown")
    p.add_argument("--label")
    p.add_argument("--session")
    _add_run_options(p, 5)

    p = sub.add_parser("sessions", help="list recorded sessions")
    p.add_argument("--label")
    p.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("compare", help="interleaved, anytime-valid comparison of two revisions")
    p.add_argument("rev_a", nargs="?", help="git revision, or @ for the working tree")
    p.add_argument("rev_b", nargs="?")
    p.add_argument("--cmd", required=True, help="benchmark command, e.g. \"python bench.py\"")
    p.add_argument("--setup", help="command run once in each tree before measuring")
    p.add_argument("--setup-a")
    p.add_argument("--setup-b")
    p.add_argument("--dir-a")
    p.add_argument("--dir-b")
    p.add_argument("--min-blocks", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=None,
                   help="equivalence margin (default theta/2)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None, help="per-run timeout (s)")
    p.add_argument("--no-observe", action="store_true",
                   help="do not observe NumPy calls (no per-call-site drill-down)")
    p.add_argument("--markdown", help="write a Markdown report (PR comment) to this path")
    _add_stats_options(p)

    p = sub.add_parser("bisect", help="find the commit that introduced a slowdown")
    p.add_argument("good")
    p.add_argument("bad")
    p.add_argument("--cmd", required=True)
    p.add_argument("--setup", help="build command run after each checkout")
    p.add_argument("--lambda", dest="lam", type=float, default=4.0,
                   help="attribution prior weight (0 = uniform prior)")
    p.add_argument("--call-site", help="call_site_id prefix to attribute (default: auto)")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--max-runs", type=int, default=400)
    p.add_argument("--endpoint-runs", type=int, default=10)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--no-cache", action="store_true", help="ignore cached runs by SHA")
    _add_stats_options(p)

    p = sub.add_parser("watch", help="change-point detection over a label's sessions")
    p.add_argument("--label", required=True)
    p.add_argument("--k", type=int, default=8, help="alert window (sessions)")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--hazard", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--per-call-site", action="store_true")
    p.add_argument("--json")

    p = sub.add_parser("verify", help="check kernel accuracy against exact results")
    p.add_argument("--cases", type=int, default=700)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--ops", help="comma-separated subset of operations")
    p.add_argument("--dtypes", help="comma-separated subset: float32,float64")
    p.add_argument("--max-n", type=int, default=20000)
    p.add_argument("--failures-dir")

    p = sub.add_parser("telemetry", help="opt-in anonymous usage statistics")
    p.add_argument("action", choices=("on", "off", "status"))
    p.add_argument("--endpoint", help="collector URL (with `on`)")

    p = sub.add_parser("ci", help="CI helpers (used by the GitHub Action)")
    ci_sub = p.add_subparsers(dest="ci_command", required=True)
    pc = ci_sub.add_parser("comment", help="create or update the PR comment")
    pc.add_argument("--markdown", required=True)
    pc.add_argument("--repo")
    pc.add_argument("--pr")
    pc.add_argument("--api")
    pc.add_argument("--token-env", default="GITHUB_TOKEN")
    return parser


def _dispatch(args):
    cmd = args.command
    if cmd in ("run", "baseline", "check", "profile", "sessions"):
        from .commands import sessions
        return getattr(sessions, f"cmd_{cmd}")(args)
    if cmd == "compare":
        from .commands.compare import run_compare
        return run_compare(args)
    if cmd == "bisect":
        from .commands.bisect import run_bisect
        return run_bisect(args)
    if cmd == "watch":
        from .commands.watch import run_watch
        return run_watch(args)
    if cmd == "verify":
        from .commands.verify import run_verify
        return run_verify(args)
    if cmd == "ci":
        from .commands.ci import run_ci_comment
        return run_ci_comment(args)
    if cmd == "telemetry":
        if args.action == "status":
            print(telemetry.status_text())
        else:
            telemetry.set_state(args.action == "on", args.endpoint)
            print(telemetry.status_text())
        return 0
    raise AssertionError(cmd)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd_argv = None
    if "--" in argv:
        i = argv.index("--")
        argv, cmd_argv = argv[:i], argv[i + 1:]
    args = build_parser().parse_args(argv)
    args.cmd_argv = cmd_argv
    if args.command not in ("telemetry", "ci"):
        telemetry.maybe_prompt()
    code = _dispatch(args)
    if args.command != "telemetry":
        telemetry.record(args.command, getattr(args, "verdict_counts", None))
    return code


if __name__ == "__main__":
    sys.exit(main())
