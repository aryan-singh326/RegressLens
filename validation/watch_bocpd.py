"""WP6: BOCPD for `rglns watch`.

  * false alarms <= 1 per 500 null sessions;
  * detection delay <= 5 sessions for a 10% shift at sigma = 3%;
  * replay: a regression injected into a recorded history (real runs of
    the validation pipeline under `rglns run --label`) is detected.

    python validation/watch_bocpd.py [--series 400] [--trials 1000]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

import common
import simlib


def replay():
    """Records 30 real sessions, then injects a +15% regression into the
    last 4 by rescaling their stored wall times, and runs `rglns watch`."""
    from regresslens import store
    d = tempfile.mkdtemp()
    env = dict(os.environ, RGLNS_DB_PATH=os.path.join(d, "t.db"), RGLNS_TELEMETRY="0")
    script = os.path.join(d, "job.py")
    open(script, "w").write("import numpy as np\nx = np.random.default_rng(0).normal(size=2_000_000)\n"
                            "for _ in range(5):\n    np.sort(x)\n")
    for _ in range(30):
        subprocess.run([sys.executable, "-m", "regresslens.cli", "run", "--label", "nightly",
                        "--quiet", "--", sys.executable, script], env=env, check=True,
                       capture_output=True)
    os.environ["RGLNS_DB_PATH"] = env["RGLNS_DB_PATH"]
    conn = store.connect()
    sessions = store.find_sessions(conn, label="nightly")
    for s in sessions[-4:]:
        conn.execute("UPDATE runs SET wall_ns = wall_ns * 1.15 WHERE session_id = ?",
                     (s.session_id,))
    conn.commit()
    p = subprocess.run([sys.executable, "-m", "regresslens.cli", "watch", "--label", "nightly"],
                       env=env, capture_output=True, text=True)
    return {"exit_code": p.returncode, "detected": "change detected" in p.stdout,
            "output_tail": p.stdout[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=400)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()
    fa = simlib.bocpd_false_alarms(args.series, seed=1)
    common.progress(f"false alarms per 500 null sessions: {fa:.3f}")
    d = simlib.bocpd_delays(args.trials, seed=2)
    finite = d[np.isfinite(d)]
    delay = {"detected": float(np.mean(np.isfinite(d))), "within_5": float(np.mean(d <= 5)),
             "median": float(np.median(finite)), "p90": float(np.percentile(finite, 90))}
    common.progress(f"delay: {delay}")
    res = {"false_alarms_per_500": fa, "delay": delay}
    if not args.no_replay:
        res["replay"] = replay()
        common.progress(f"replay: detected={res['replay']['detected']}")
    common.save("wp6_watch", res)


if __name__ == "__main__":
    main()
