"""WP2: observation overhead of `rglns run`.

  * validation pipeline (1M-row prices -> diff_log -> rolling -> zscore ->
    filter -> mean): target <= 5%;
  * 100k-tiny-call microloops (np.add on 16 elements, x.sum(), np.sum(x)):
    target <= 25%, number documented per call form.

Three configurations: plain Python, `rglns run` with the C fast path
(regresslens-accel installed), and the pure-Python observer
(RGLNS_PURE_OBSERVER=1). Each is measured best-of-N in fresh processes,
interleaved: every round runs all three, in rotating order. Measuring one
configuration after another instead turns any drift in machine state
(clock, temperature, background load, which core type of a hybrid CPU the
process lands on) into apparent overhead, which once showed as -10% on
the pipeline. Run on a quiet machine.

On Python 3.12+ the microloops are also run under a fourth configuration,
the sys.monitoring floor: CALL/C_RETURN/C_RAISE enabled on the benchmark's
own code only, with a C builtin that does nothing as every callback. That
is CPython's own cost of reporting one call to any observer, which no
implementation of the callbacks can go below.

    python validation/overhead.py [--repeats 15]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import common

MICRO = """
import time
import numpy as np
x = np.ones(16)
for _ in range(2000):
    x.sum()
t0 = time.perf_counter_ns()
for _ in range(100_000):
    np.add(x, x)
t1 = time.perf_counter_ns()
for _ in range(100_000):
    x.sum()
t2 = time.perf_counter_ns()
for _ in range(100_000):
    np.sum(x)
t3 = time.perf_counter_ns()
print((t1 - t0) / 1e5, (t2 - t1) / 1e5, (t3 - t2) / 1e5)
"""

PIPE = """
import sys, time
sys.path.insert(0, {vdir!r})
import pipeline
prices = pipeline.generate_prices(n=1_000_000, seed=1)
pipeline.run_numpy_baseline(prices)
best = float("inf")
for _ in range(10):
    t0 = time.perf_counter()
    pipeline.run_numpy_baseline(prices)
    best = min(best, time.perf_counter() - t0)
print(best * 1e9)
"""


FLOOR = """
import sys
m = sys.monitoring
E = m.events
path = sys.argv[1]
code = compile(open(path).read(), path, "exec")
m.use_tool_id(m.PROFILER_ID, "floor")
for ev in (E.CALL, E.C_RETURN, E.C_RAISE):
    m.register_callback(m.PROFILER_ID, ev, "".format)  # C, any arguments, returns ""
m.set_local_events(m.PROFILER_ID, code, E.CALL | E.C_RETURN | E.C_RAISE)
exec(code, {"__name__": "__main__"})
"""

MODES = ("plain", "fast", "pure")


def run_once(script, mode, db):
    env = dict(os.environ, RGLNS_TELEMETRY="0", RGLNS_QUIET="1", RGLNS_CANARY="0",
               RGLNS_DB_PATH=db)
    if mode == "pure":
        env["RGLNS_PURE_OBSERVER"] = "1"
    if mode == "plain":
        cmd = [sys.executable, script]
    elif mode == "floor":
        cmd = [sys.executable, "-c", FLOOR, script]
    else:
        cmd = [sys.executable, "-m", "regresslens.cli", "run", "--quiet", "--", sys.executable,
               script]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True).stdout
    return [float(v) for v in out.split()]


def measure(script, repeats, modes=MODES):
    """Per mode, the column-wise minimum over `repeats` interleaved rounds."""
    db = os.path.join(tempfile.mkdtemp(), "t.db")
    results = {mode: [] for mode in modes}
    for r in range(repeats):
        for i in range(len(modes)):
            mode = modes[(r + i) % len(modes)]
            results[mode].append(run_once(script, mode, db))
    return {mode: [min(col) for col in zip(*rows)] for mode, rows in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()
    d = tempfile.mkdtemp()
    micro, pipe = os.path.join(d, "micro.py"), os.path.join(d, "pipe.py")
    open(micro, "w").write(MICRO)
    open(pipe, "w").write(PIPE.format(vdir=common.HERE))
    micro_modes = MODES + (("floor",) if sys.version_info >= (3, 12) else ())
    micro_ns = measure(micro, args.repeats, micro_modes)
    pipe_ns = measure(pipe, args.repeats)
    res = {mode: {"micro_ns": micro_ns[mode], "pipeline_ns": pipe_ns[mode][0]} for mode in MODES}
    for mode in MODES:
        common.progress(f"{mode}: {res[mode]}")
    if "floor" in micro_ns:
        res["floor"] = {"micro_ns": micro_ns["floor"],
                        "micro_overhead": [f / p - 1 for f, p in zip(micro_ns["floor"],
                                                                     res["plain"]["micro_ns"])]}
        common.progress(f"floor: {res['floor']}")
    for mode in ("fast", "pure"):
        res[mode]["micro_overhead"] = [o / p - 1 for o, p in zip(res[mode]["micro_ns"],
                                                                 res["plain"]["micro_ns"])]
        res[mode]["pipeline_overhead"] = res[mode]["pipeline_ns"] / res["plain"]["pipeline_ns"] - 1
    res["forms"] = ["np.add(x, x) (16 elements)", "x.sum()", "np.sum(x)"]
    res["repeats"] = args.repeats
    common.save("wp2_overhead", res)


if __name__ == "__main__":
    main()
