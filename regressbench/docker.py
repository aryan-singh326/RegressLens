"""Runs RegressBench cases in Docker, one fresh container per case, in the
environment the case declares: the python:<version> image, its pinned pip
packages and its apt packages.

    python regressbench/docker.py [--cases ID,...] [--cpus 6] [--max-runs 200]

The repository is mounted read-only; results go to regressbench/results/
(<case>.json, plus the raw bisect output <case>.json.bisect and the
container log <case>.log, both git-ignored). RegressLens is installed with
--no-deps so the case's pinned NumPy stays in place: the 2020 pandas cases
pin NumPy 1.19.5, the oldest with Python 3.9 wheels. Cases whose
environment predates Python 3.9 (which RegressLens needs) are skipped.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run import load_cases  # noqa: E402


def container_script(case, max_runs):
    env = case["environment"]
    lines = ["set -e"]
    if env.get("apt"):
        lines.append("apt-get update -qq >/dev/null && apt-get install -y -qq "
                     + " ".join(env["apt"]) + " >/dev/null")
    lines += [
        "pip install -q --disable-pip-version-check --root-user-action=ignore "
        + " ".join(shlex.quote(p) for p in env["pip"]),
        "cp -r /src/python /tmp/rlpy && pip install -q --disable-pip-version-check "
        "--root-user-action=ignore --no-deps /tmp/rlpy",
        "cp -r /src/regressbench /rb",
        f"cd /rb && python run.py --cases {case['id']} --workdir /work --results /out "
        f"--max-runs {max_runs}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="comma-separated case ids (default: all)")
    ap.add_argument("--cpus", default="6", help="docker --cpus per container")
    ap.add_argument("--max-runs", type=int, default=200)
    args = ap.parse_args()
    results = os.path.join(HERE, "results")
    os.makedirs(results, exist_ok=True)
    failed = 0
    for case in load_cases(args.cases.split(",") if args.cases else None):
        version = case["environment"]["python"]
        if tuple(int(x) for x in version.split(".")) < (3, 9):
            print(f"{case['id']}: skipped (needs Python {version}; RegressLens needs >= 3.9)")
            continue
        cmd = ["docker", "run", "--rm", "--cpus", args.cpus,
               "-v", f"{ROOT}:/src:ro", "-v", f"{results}:/out",
               f"python:{version}", "bash", "-c", container_script(case, args.max_runs)]
        t0 = time.time()
        with open(os.path.join(results, case["id"] + ".log"), "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{case['id']}: container exit {code} after {time.time() - t0:.0f} s", flush=True)
        failed += code != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
