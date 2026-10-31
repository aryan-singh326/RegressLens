"""WP4: rglns compare.

Simulation:
  * always-valid property: 5000 null simulations per sigma in {2%, 5%},
    monitoring after every block up to 100 blocks; P(ever declares
    regression or improvement) must be <= 0.055 at alpha = 0.05. Reported
    at the default theta (5%) and at theta = 0 (any interval excluding 0
    counts), next to a naive "bootstrap CI after every block" strategy;
  * efficiency: +10% at sigma 3%: median blocks <= 8; +3% with theta 5%:
    never a regression in >= 95% of trials.
Real runs (--real):
  * A/A: `rglns compare X X` 200 times on a real script, <= 5% false verdicts;
  * a generated repo with an injected np.convolve slowdown is flagged, also
    with a memory-bandwidth hog started mid-run.

    python validation/compare_sequential.py [--trials 5000] [--real] [--aa 200]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import common
import simlib

BENCH = ("import numpy as np\nfrom signals import smooth\n"
         "x = np.random.default_rng(0).uniform(-1, 1, 300_000)\n"
         "for _ in range(12):\n    smooth(x)\n")
SIGNALS = "import numpy as np\ndef smooth(x):\n    return np.convolve(x, np.ones({w}) / {w}, mode='valid')\n"
HOG = "import numpy as np\na = np.ones(64 * 1024 * 1024 // 8)\nwhile True:\n    b = a.copy()\n"


def make_repo(slow):
    d = tempfile.mkdtemp(prefix="rglns-val-")
    def git(*a):
        subprocess.run(["git", *a], cwd=d, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "v@example.com")
    git("config", "user.name", "v")
    open(os.path.join(d, "bench.py"), "w").write(BENCH)
    open(os.path.join(d, "signals.py"), "w").write(SIGNALS.format(w=5))
    git("add", "-A")
    git("commit", "-qm", "base")
    if slow:
        open(os.path.join(d, "signals.py"), "w").write(SIGNALS.format(w=60))
        git("commit", "-qam", "slower")
    return d


def rglns(args, cwd):
    env = dict(os.environ, RGLNS_DB_PATH=os.path.join(cwd, ".t.db"), RGLNS_TELEMETRY="0")
    return subprocess.run([sys.executable, "-m", "regresslens.cli", *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def real_tests(aa_count):
    out = {}
    repo = make_repo(slow=False)
    false = 0
    for i in range(aa_count):
        p = rglns(["compare", "HEAD", "HEAD", "--cmd", "python bench.py", "--seed", str(i),
                   "--no-observe", "--max-blocks", "40"], repo)
        false += ("REGRESSION" in p.stdout) or ("IMPROVEMENT" in p.stdout)
        if (i + 1) % 20 == 0:
            common.progress(f"A/A {i + 1}/{aa_count}: {false} false verdicts")
    out["aa_runs"], out["aa_false"] = aa_count, false
    slow = make_repo(slow=True)
    p = rglns(["compare", "HEAD~1", "HEAD", "--cmd", "python bench.py", "--seed", "1"], slow)
    out["fixture_flagged"] = p.returncode == 1
    hogs = []
    def start_hogs():
        for _ in range(max(1, (os.cpu_count() or 2) // 4)):
            hogs.append(subprocess.Popen([sys.executable, "-c", HOG]))
    import threading
    timer = threading.Timer(3.0, start_hogs)
    timer.start()
    try:
        p = rglns(["compare", "HEAD~1", "HEAD", "--cmd", "python bench.py", "--seed", "2",
                   "--min-blocks", "8"], slow)
    finally:
        timer.cancel()
        for h in hogs:
            h.kill()
    out["fixture_flagged_with_hog"] = p.returncode == 1
    out["hog_output"] = p.stdout[-1500:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--aa", type=int, default=200)
    args = ap.parse_args()
    res = {"null": []}
    for theta in (0.05, 0.0):
        for sigma in (0.02, 0.05):
            cs = simlib.sequential_null(args.trials, sigma, theta, seed=int(sigma * 1000))
            naive = simlib.sequential_null(max(args.trials // 5, 200), sigma, theta,
                                           seed=int(sigma * 1000) + 1, naive=True)
            res["null"].append({"theta": theta, "sigma": sigma, "cs": cs, "naive": naive})
            common.progress(f"null theta={theta} sigma={sigma}: CS {cs:.4f}, naive {naive:.4f}")
    rng = np.random.default_rng(3)
    blocks = [simlib.sequential_trial(rng, 0.10, 0.03)[1] for _ in range(2000)]
    small = [simlib.sequential_trial(rng, 0.03, 0.03)[0] for _ in range(2000)]
    res["efficiency"] = {"median_blocks_10pct": float(np.median(blocks)),
                         "p90_blocks_10pct": float(np.percentile(blocks, 90)),
                         "regression_rate_3pct": float(np.mean([d == "regression" for d in small]))}
    common.progress(f"efficiency: {res['efficiency']}")
    if args.real:
        res["real"] = real_tests(args.aa)
        common.progress(f"real: { {k: v for k, v in res['real'].items() if k != 'hog_output'} }")
    common.save("wp4_compare", res)


if __name__ == "__main__":
    main()
