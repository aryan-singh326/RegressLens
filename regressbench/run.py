"""RegressBench runner: blind `rglns bisect` against real regressions.

    python regressbench/run.py [--cases ID,...] [--workdir DIR] [--verify-only]

For each case in regressbench/cases/:
  1. clone (or reuse) the project repository in --workdir;
  2. verify the case: good, bad and the culprit exist, the expected
     first-parent culprit lies on the first-parent chain good..bad, and
     no listed fix commit is already an ancestor of `bad`;
  3. run `rglns bisect GOOD BAD --cmd "python BENCH" [--setup SETUP]`
     WITHOUT telling it the culprit;
  4. score: localization correct? (the most likely commit equals the
     expected first-parent culprit), and is it inside the 95% credible
     set? Plus runs used and wall-clock time.

Each case lists the environment it needs (Python version, pinned
dependencies, build toolchain). This runner uses the current interpreter
and checks nothing beyond importability. Run each case in a matching
environment, e.g. the per-case Docker commands in README.md. Results go
to regressbench/results/<case>.json; --summary prints the table.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def git(args, cwd, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=check).stdout.strip()


def load_cases(ids=None):
    cases = []
    for path in sorted(glob.glob(os.path.join(HERE, "cases", "*.json"))):
        with open(path) as f:
            case = json.load(f)
        if not ids or case["id"] in ids:
            cases.append(case)
    return cases


def checkout(case, workdir):
    path = os.path.join(workdir, case["project"])
    if not os.path.isdir(path):
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", case["repo"], path],
                       check=True)
    return path


def verify(case, repo):
    """List of problems with the case definition (empty when valid)."""
    problems = []
    for key in ("good", "bad", "expected_first_parent"):
        if subprocess.run(["git", "cat-file", "-e", case[key] + "^{commit}"], cwd=repo).returncode:
            problems.append(f"{key} {case[key]} not found")
    if problems:
        return problems
    chain = git(["rev-list", "--first-parent", f"{case['good']}..{case['bad']}"], repo).split()
    if case["expected_first_parent"] not in chain:
        problems.append("expected culprit is not on the first-parent chain good..bad")
    if subprocess.run(["git", "merge-base", "--is-ancestor", case["culprit"]["sha"],
                       case["expected_first_parent"]], cwd=repo).returncode:
        problems.append("culprit commit is not contained in the expected first-parent commit")
    for fix in case.get("fixed_by", []):
        sha = fix.rsplit("/", 1)[-1]
        if "/commit/" in fix and not subprocess.run(
                ["git", "merge-base", "--is-ancestor", sha, case["bad"]], cwd=repo).returncode:
            problems.append(f"fix {sha} is already in `bad`")
    return problems


def rglns_command():
    """The installed `rglns` script, run as a user would run it. With
    `python -m regresslens.cli` the current directory (the project
    checkout) comes first on sys.path, so for the NumPy case the CLI
    would import NumPy's unbuilt source tree."""
    name = "rglns.exe" if os.name == "nt" else "rglns"
    script = os.path.join(sysconfig.get_path("scripts"), name)
    if not os.path.exists(script):
        script = shutil.which("rglns")
    if not script:
        raise SystemExit("error: the rglns script is not installed in this environment")
    return [script]


def run_case(case, repo, args):
    bench = os.path.join(HERE, "bench", case["bench"])
    out = os.path.join(args.results, case["id"] + ".json")
    cmd = [*rglns_command(), "bisect", case["good"], case["bad"],
           "--cmd", f'python "{bench}"', "--json", out + ".bisect",
           "--max-runs", str(args.max_runs)]
    if case.get("setup"):
        cmd += ["--setup", case["setup"]]
    env = dict(os.environ, RGLNS_DB_PATH=os.path.join(args.workdir, "regressbench.db"),
               RGLNS_TELEMETRY="0")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    result = {"id": case["id"], "elapsed_s": elapsed, "exit_code": proc.returncode,
              "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    if os.path.exists(out + ".bisect"):
        with open(out + ".bisect") as f:
            b = json.load(f)
        found = b["culprit"]["sha"]
        result.update(found=found, correct=found == case["expected_first_parent"],
                      in_credible_set=case["expected_first_parent"] in
                      [c["sha"] for c in b["credible_set"]],
                      probability=b["probability"], runs=b["runs"], converged=b["converged"],
                      credible_set_size=len(b["credible_set"]), commits=b["commits"],
                      prior=b.get("prior"),
                      prior_flagged=len(b.get("prior_flagged") or []),
                      prior_flags_culprit=case["expected_first_parent"] in
                      (b.get("prior_flagged") or []))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    return result


def prior_cell(r):
    """How the attribution prior did: uniform, or how many commits it
    flagged and whether the culprit was one of them."""
    if "prior_flags_culprit" not in r:
        return "-"
    if not r["prior_flagged"]:
        return "uniform"
    which = "culprit flagged" if r["prior_flags_culprit"] else "culprit not flagged"
    return f"{r['prior_flagged']} of {r['commits']} flagged, {which}"


def summary(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    print("| case | correct | in 95% set | runs | wall-clock | attribution prior |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if "correct" in r:
            print(f"| {r['id']} | {'yes' if r['correct'] else 'no'} | "
                  f"{'yes' if r['in_credible_set'] else 'no'} | {r['runs']} | "
                  f"{r['elapsed_s']:.0f} s | {prior_cell(r)} |")
        elif "note" in r:
            print(f"| {r['id']} | {r['note']} | | | | |")
        else:
            print(f"| {r['id']} | error (exit {r['exit_code']}) | | | {r['elapsed_s']:.0f} s | |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="comma-separated case ids (default: all)")
    ap.add_argument("--workdir", default=os.path.join(HERE, ".work"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--max-runs", type=int, default=300)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.results)
        return 0
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    cases = load_cases(args.cases.split(",") if args.cases else None)
    bad = 0
    for case in cases:
        repo = checkout(case, args.workdir)
        problems = verify(case, repo)
        status = "ok" if not problems else "; ".join(problems)
        print(f"{case['id']}: definition {status}", flush=True)
        if problems:
            bad += 1
            continue
        if not args.verify_only:
            r = run_case(case, repo, args)
            print(f"  -> correct={r.get('correct')} runs={r.get('runs')} "
                  f"elapsed={r['elapsed_s']:.0f}s", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
