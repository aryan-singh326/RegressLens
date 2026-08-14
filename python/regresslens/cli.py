"""
regresslens.cli: the `rglns` command-line tool.

Three commands, per the project brief:
  rglns baseline --name NAME --runs N --script SCRIPT
  rglns check --baseline NAME --runs N --script SCRIPT
  rglns profile --script SCRIPT --runs N

Each command runs the user's script as a SEPARATE PROCESS N times
(not imported and called in-process), so that:
  - the user's script can be an arbitrary standalone program, not
    forced into an importable function
  - traces from each run land in the shared SQLite database exactly
    as they would from normal use, tagged with a run_label the CLI
    controls via an environment variable — no changes to the user's
    script are required
"""
import argparse
import json
import os
import subprocess
import sys
import time

from . import remediation as _remediation
from . import stats as _stats
from . import trace as _trace


def _run_script_n_times(script_path, n, run_label):
    """Runs the user's script n times as subprocesses, each tagged
    with run_label via environment variable so their rl.array calls
    land in the trace database under that label. Returns the number
    of runs that completed successfully; prints a warning for any
    that failed rather than silently ignoring them."""
    succeeded = 0
    env = os.environ.copy()
    env["REGRESSLENS_RUN_LABEL"] = run_label

    for i in range(n):
        result = subprocess.run(
            [sys.executable, script_path], env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(
                f"  warning: run {i+1}/{n} of {script_path} exited with code "
                f"{result.returncode}, skipping this run's data",
                file=sys.stderr,
            )
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()[:500]}", file=sys.stderr)
        else:
            succeeded += 1
    return succeeded


def cmd_baseline(args):
    if not os.path.exists(args.script):
        print(f"error: script not found: {args.script}", file=sys.stderr)
        return 1

    run_label = f"baseline:{args.name}"
    print(f"Running {args.script} {args.runs} times to capture baseline '{args.name}'...")
    succeeded = _run_script_n_times(args.script, args.runs, run_label)

    if succeeded == 0:
        print("error: every run failed, no baseline data captured", file=sys.stderr)
        return 1

    samples = _trace.get_samples(run_label)
    if not samples:
        print(
            "error: script ran successfully but recorded no RegressLens "
            "operations. Does it use regresslens.array?",
            file=sys.stderr,
        )
        return 1

    print()
    print(f"Baseline '{args.name}' stored.")
    print(f"  Operations profiled: {len(samples)}")
    for (operator, dtype, row_count), runtimes in sorted(samples.items()):
        print(f"    {operator} ({dtype}, n={row_count}): {len(runtimes)} samples")
    print(f"  Runs: {succeeded}/{args.runs} succeeded")
    print(f"  Hardware fingerprint: {_trace.get_hardware_fingerprint()}")
    return 0


def _detect_hardware_confound(baseline_hw, current_hw):
    """Compares hardware fingerprint sets from a baseline and current
    session. Returns a warning string, or None if no confound is
    detected. Factored out from cmd_check so this decision logic can
    be tested directly against synthetic fingerprint sets, without
    needing to force two real subprocesses to actually run on
    different hardware (which isn't controllable in a test)."""
    warning = None
    if baseline_hw and current_hw and baseline_hw != current_hw:
        warning = (
            "WARNING: baseline and current runs used DIFFERENT hardware "
            "fingerprints. Any regression reported below may be a hardware "
            "or environment change, not a real code regression. This is a "
            "confound, not noise — investigate before trusting this report."
        )
    if len(current_hw) > 1:
        warning = ((warning + "\n  ") if warning else "") + (
            "WARNING: the current run's OWN sessions used more than one "
            "hardware fingerprint — the machine's effective configuration "
            "changed mid-measurement (e.g. thermal throttling on shared/"
            "noisy infrastructure). Timing comparisons here are less "
            "trustworthy than a stable-hardware run would be."
        )
    return warning


def cmd_check(args):
    if not os.path.exists(args.script):
        print(f"error: script not found: {args.script}", file=sys.stderr)
        return 1

    baseline_label = f"baseline:{args.baseline}"
    baseline_samples = _trace.get_samples(baseline_label)
    if not baseline_samples:
        print(
            f"error: no baseline named '{args.baseline}' found. Run "
            f"'rglns baseline --name {args.baseline} --script {args.script}' first.",
            file=sys.stderr,
        )
        return 1

    current_label = f"check:{args.baseline}:{int(time.time())}"
    print(f"Running {args.script} {args.runs} times for comparison against "
          f"baseline '{args.baseline}'...")
    succeeded = _run_script_n_times(args.script, args.runs, current_label)
    if succeeded == 0:
        print("error: every run failed, no current data captured", file=sys.stderr)
        return 1

    current_samples = _trace.get_samples(current_label)

    # Confound check: did baseline and current actually run on
    # comparable hardware? This was found missing by testing on this
    # project's own noisy sandbox environment — a CPU frequency
    # change between sessions produced a false-positive regression
    # report before this check existed. Per the project brief,
    # environment differences must be flagged, not silently absorbed.
    baseline_hw = _trace.get_hardware_fingerprints_for_label(baseline_label)
    current_hw = _trace.get_hardware_fingerprints_for_label(current_label)
    hw_confound_warning = _detect_hardware_confound(baseline_hw, current_hw)

    # Only compare operations present in BOTH baseline and current —
    # per the trace-grouping limitation documented in trace.py
    # (operator+dtype+row_count is an approximation of "same call
    # site", not real stack-based attribution).
    common_keys = sorted(set(baseline_samples) & set(current_samples))
    if not common_keys:
        print(
            "error: no operations in common between baseline and current run. "
            "Did the script change its array sizes or dtypes?",
            file=sys.stderr,
        )
        return 1

    num_comparisons = len(common_keys)
    print()
    if hw_confound_warning:
        print("  " + "!" * 60)
        for line in hw_confound_warning.split("\n"):
            print(f"  {line}")
        print("  " + "!" * 60)
        print()
    print("Performance regression report:")
    print("  " + "\u2500" * 50)

    any_regression = False
    for operator, dtype, row_count in common_keys:
        result = _stats.evaluate_regression(
            baseline_samples[(operator, dtype, row_count)],
            current_samples[(operator, dtype, row_count)],
            min_pairs=args.min_pairs,
            practical_threshold=args.threshold,
            num_comparisons=num_comparisons,
        )
        print(f"  {operator} ({dtype}, n={row_count})")
        if result["status"] == "insufficient_data":
            print(
                f"    Insufficient data (baseline={result['n_baseline']}, "
                f"current={result['n_current']}, need {args.min_pairs})"
            )
        else:
            pct = lambda x: f"{x*100:+.1f}%"
            print(
                f"    Bootstrap CI for slowdown: [{pct(result['ci_low'])}, "
                f"{pct(result['ci_high'])}]"
            )
            print(f"    Practical threshold: {args.threshold*100:.0f}%")
            if result["status"] == "regression":
                print("    Status: REGRESSION")
                any_regression = True

                # Diagnosis: was this regression accompanied by a
                # contiguity loss? Check the CURRENT session for a
                # non-contiguous trace at this same call shape — per
                # the project brief, this identifies WHERE the first
                # such call happened, not WHY the array lost
                # contiguity upstream (that's out of scope; see
                # trace.capture_call_site's docstring).
                loss = _trace.get_first_contiguity_loss(
                    operator, dtype, row_count, current_label
                )
                if loss is not None:
                    call_site, _ts = loss
                    print()
                    print("    Diagnosis:")
                    print("      Array contiguity lost before this operation.")
                    print(f"      First rl-observed call receiving non-contiguous "
                          f"array: {call_site}")
                    print("      Note: exact upstream cause requires manual "
                          "inspection.")
                    print("            RegressLens v0.1 identifies the first "
                          "affected call site,")
                    print("            not the operation that caused "
                          "contiguity loss.")

                    dtype_bytes = 8 if "float64" in dtype else 4
                    fallback_samples = [
                        rt for rt in current_samples[(operator, dtype, row_count)]
                    ]
                    accel_samples = _trace.get_contiguous_runtime_samples(
                        operator, dtype, row_count
                    )
                    rem = _remediation.estimate_remediation(
                        row_count, dtype_bytes, fallback_samples, accel_samples,
                        safety_margin=args.remediation_margin,
                    )
                    print()
                    print("    Remediation — contiguity fix:")
                    if rem["reason"]:
                        print(f"      {rem['reason']}")
                    else:
                        ms = lambda ns: f"{ns/1e6:.2f}ms"
                        print(f"      Estimated copy cost:   {ms(rem['copy_cost_ns'])} "
                              f"({row_count} {dtype} elements)")
                        print(f"      Estimated savings:     "
                              f"{ms(rem['estimated_savings_ns'])} per call")
                        print(f"      Net estimated benefit: "
                              f"{ms(rem['net_benefit_ns'])} per call")
                        if rem["recommend_apply"]:
                            print("      Recommendation: applying "
                                  "np.ascontiguousarray() before this call is "
                                  "likely worth it.")
                        else:
                            print("      Recommendation: copy cost likely "
                                  "exceeds benefit for a single call — only "
                                  "worth it if this array is reused many "
                                  "times.")
                        print("      RegressLens does not apply this "
                              "automatically. Fix your pipeline's array "
                              "handling directly (e.g. add "
                              "np.ascontiguousarray() at the identified call "
                              "site).")
            else:
                print("    Status: within noise")
        print("  " + "\u2500" * 50)

    print(f"  Multiple-comparison correction: Bonferroni ({num_comparisons} operations)")
    print("  Comparison method: two independent measurement sessions "
          "(NOT interleaved — see regresslens.stats module docstring "
          "for why true interleaving isn't done against a stored "
          "baseline in v0.1)")
    print(f"  Baseline hardware fingerprint(s): {sorted(baseline_hw)}")
    print(f"  Current hardware fingerprint(s):  {sorted(current_hw)}")

    return 1 if any_regression else 0


def cmd_profile(args):
    if not os.path.exists(args.script):
        print(f"error: script not found: {args.script}", file=sys.stderr)
        return 1

    run_label = f"profile:{int(time.time())}"
    print(f"Running {args.script} {args.runs} times to profile...")
    succeeded = _run_script_n_times(args.script, args.runs, run_label)
    if succeeded == 0:
        print("error: every run failed", file=sys.stderr)
        return 1

    samples = _trace.get_samples(run_label)
    if not samples:
        print(
            "error: script ran but recorded no RegressLens operations",
            file=sys.stderr,
        )
        return 1

    total_ns = sum(sum(runtimes) for runtimes in samples.values())

    print()
    print("Pipeline runtime breakdown:")
    ranked = sorted(samples.items(), key=lambda kv: -sum(kv[1]))
    for (operator, dtype, row_count), runtimes in ranked:
        op_total = sum(runtimes)
        pct = 100 * op_total / total_ns if total_ns > 0 else 0
        median_ns = sorted(runtimes)[len(runtimes) // 2]
        print(f"  {operator} ({dtype}, n={row_count}): {pct:.1f}% of total "
              f"(median {median_ns:.0f}ns/call, {len(runtimes)} calls)")

    print()
    print(f"Hardware: {_trace.get_hardware_fingerprint()}")
    print(f"Runs: {succeeded}/{args.runs}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rglns",
                                      description="RegressLens performance "
                                      "regression intelligence CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_baseline = sub.add_parser("baseline", help="store current performance profile")
    p_baseline.add_argument("--name", required=True)
    p_baseline.add_argument("--runs", type=int, default=30)
    p_baseline.add_argument("--script", required=True)
    p_baseline.set_defaults(func=cmd_baseline)

    p_check = sub.add_parser("check", help="compare against baseline")
    p_check.add_argument("--baseline", required=True)
    p_check.add_argument("--runs", type=int, default=30)
    p_check.add_argument("--script", required=True)
    p_check.add_argument("--min-pairs", type=int, default=20)
    p_check.add_argument("--threshold", type=float, default=0.05)
    p_check.add_argument("--remediation-margin", type=float, default=2.0,
                          help="only recommend applying a contiguity fix "
                          "when estimated savings exceed copy cost by this "
                          "multiple (default 2.0x, per the project brief)")
    p_check.set_defaults(func=cmd_check)

    p_profile = sub.add_parser("profile", help="operation-level breakdown")
    p_profile.add_argument("--script", required=True)
    p_profile.add_argument("--runs", type=int, default=50)
    p_profile.set_defaults(func=cmd_profile)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
