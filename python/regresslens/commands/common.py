"""Helpers shared by the rglns commands."""
import os
import sys


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def command_from_args(args):
    """The measured command: `-- cmd...` or the legacy `--script PATH`."""
    if getattr(args, "cmd_argv", None):
        return list(args.cmd_argv)
    script = getattr(args, "script", None)
    if script:
        if not os.path.exists(script):
            raise SystemExit(f"error: script not found: {script}")
        return ["python", script]
    raise SystemExit("error: give the command to measure after `--` (or --script PATH)")


def run_n(launcher, n, label="run", verbose=True):
    """Runs the launcher n times; returns the RunResults. Failed runs are
    reported and kept out of statistics (their exit code is stored)."""
    results = []
    for i in range(n):
        r = launcher.run()
        results.append(r)
        if r.exit_code != 0:
            eprint(f"  warning: {label} {i + 1}/{n} exited with code {r.exit_code}; "
                   "its timings are excluded")
            if r.stderr:
                eprint("  " + r.stderr.strip()[-500:].replace("\n", "\n  "))
        elif verbose:
            eprint(f"  {label} {i + 1}/{n}: {r.wall_ns / 1e6:.1f} ms")
    return results


def write_output(path, text):
    if path == "-":
        print(text)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
