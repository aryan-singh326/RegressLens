"""WP7: top-1 diagnosis accuracy on fault-injection fixtures.

One fixture per cause (strided view, astype(float64), 10x larger n,
selectivity shift, memory-bandwidth hog), 30 trials each: the top-1
diagnosis must be correct in >= 90% of trials. A null fixture must give
no diagnosis in >= 95% of trials. Each trial records a fresh baseline
session and a fresh current session (3 runs each) in its own database.

    python validation/diagnosis_faults.py [--trials 30]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import common

FIXTURE = os.path.join(common.HERE, "..", "python", "tests", "fixtures", "faults.py")
HOG = "import numpy as np\na = np.ones(96 * 1024 * 1024 // 8)\nwhile True:\n    b = a.copy()\n"
CASES = [("strided", "contiguity_loss"), ("float64", "dtype_promotion"),
         ("bigger", "size_regime"), ("selectivity", "kernel_selection"),
         ("hog", "environment_drift"), ("none", None)]


def trial(fault):
    from regresslens import analysis, store
    from regresslens.launcher import Launcher
    os.environ["RGLNS_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
    def session(env_fault):
        l = Launcher(["python", FIXTURE], label=env_fault, capture=True, quiet=True,
                     extra_env={"RGLNS_FAULT": env_fault})
        for _ in range(3):
            l.run()
        return l.session_id
    base = session("none")
    hogs = []
    if fault == "hog":
        hogs = [subprocess.Popen([sys.executable, "-c", HOG])
                for _ in range(max(1, (os.cpu_count() or 2) // 3))]
        import time
        time.sleep(1.0)
    try:
        cur = session("none" if fault == "hog" else fault)
    finally:
        for h in hogs:
            h.kill()
    conn = store.connect()
    b, c = store.load_sessions(conn, [base]), store.load_sessions(conn, [cur])
    info = store.call_site_info(conn, {x.call_site_id for s in b + c for r in s.runs
                                       for x in r.calls})
    report = analysis.compare_sessions(b, c, info, B=1000)
    best = None
    for site in report.sites:
        for h in site.hypotheses:
            if best is None or h.likelihood_ratio > best.likelihood_ratio:
                best = h
    conn.close()
    return best.name if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    from regresslens import _accel
    res = {}
    for fault, expected in CASES:
        if fault == "selectivity" and not (_accel.native() and
                                           _accel.native().info()["avx2_active"]):
            continue  # the kernel choice only changes when AVX2 kernels exist
        tops = [trial(fault) for _ in range(args.trials)]
        rate = sum(t == expected for t in tops) / len(tops)
        res[fault] = {"expected": expected, "correct_rate": rate,
                      "tops": {str(k): tops.count(k) for k in set(tops)}}
        common.progress(f"{fault}: {rate:.2f} correct ({res[fault]['tops']})")
    common.save("wp7_diagnosis", res)


if __name__ == "__main__":
    main()
