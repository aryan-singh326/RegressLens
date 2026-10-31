"""WP3 canary test on a real machine.

Each trial records a baseline session of a memory-bound workload, then a
current session of the SAME code while background processes loop np.copy
over large arrays (memory-bandwidth hogs). The new engine must report
environment drift, not a regression, in >= 95% of trials. The v0.1 engine
(pooled calls, no canaries) is evaluated on the same recorded calls.

    python validation/canary_hog.py [--trials 20] [--hogs N]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import common

WORKLOAD = """
import numpy as np
a = np.ones(8 * 1024 * 1024)        # 64 MB: streams from DRAM
for _ in range(30):
    a.sum()
"""
HOG = """
import numpy as np
a = np.ones(48 * 1024 * 1024)
while True:
    b = a.copy()
"""


def session(path, label, runs):
    from regresslens.launcher import Launcher
    lnch = Launcher(["python", path], label=label, capture=True, quiet=True)
    for _ in range(runs):
        lnch.run()
    return lnch.session_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--hogs", type=int, default=max(2, (os.cpu_count() or 4) // 3))
    args = ap.parse_args()
    from regresslens import analysis, store
    from regresslens.stats import legacy
    d = tempfile.mkdtemp()
    work = os.path.join(d, "workload.py")
    open(work, "w").write(WORKLOAD)
    verdicts, old = {}, 0
    raw_slowdowns = []
    for t in range(args.trials):
        os.environ["RGLNS_DB_PATH"] = os.path.join(d, f"t{t}.db")
        base = session(work, "base", args.runs)
        hogs = [subprocess.Popen([sys.executable, "-c", HOG]) for _ in range(args.hogs)]
        time.sleep(1.5)
        try:
            cur = session(work, "cur", args.runs)
        finally:
            for h in hogs:
                h.kill()
        conn = store.connect()
        b, c = store.load_sessions(conn, [base]), store.load_sessions(conn, [cur])
        info = store.call_site_info(conn, {x.call_site_id for s in b + c for r in s.runs
                                           for x in r.calls})
        report = analysis.compare_sessions(b, c, info, B=2000, with_diagnosis=False)
        site = next(s for s in report.sites if s.key != analysis.WALL and s.comparison)
        verdicts[site.verdict.verdict] = verdicts.get(site.verdict.verdict, 0) + 1
        raw_slowdowns.append(float(np.expm1(site.comparison.estimate)))
        pooled = lambda sess: [v for s in sess for r in s.runs for x in r.calls
                               if x.call_site_id == site.key for v in x.samples]
        old += legacy.evaluate(pooled(b), pooled(c))[0] == "regression"
        conn.close()
        common.progress(f"trial {t + 1}: {site.verdict.verdict} (raw slowdown "
                        f"{raw_slowdowns[-1]:+.1%}, canary shifts compute "
                        f"{np.expm1(site.comparison.drift_c):+.1%} memory "
                        f"{np.expm1(site.comparison.drift_m):+.1%})")
    res = {"trials": args.trials, "hogs": args.hogs,
           "verdicts": {k: v / args.trials for k, v in verdicts.items()},
           "v01_regression_rate": old / args.trials,
           "median_raw_slowdown": float(np.median(raw_slowdowns))}
    common.progress(f"result: {res}")
    common.save("wp3_canary_hog", res)


if __name__ == "__main__":
    main()
