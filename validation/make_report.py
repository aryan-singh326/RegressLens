"""Builds docs/validation.md from validation/results/*.json and
regressbench/results/*.json. Every number in that document comes from
these files.

    python validation/make_report.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")


def load(name):
    path = os.path.join(RESULTS, name + ".json")
    return json.load(open(path)) if os.path.exists(path) else None


def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def meta(r, script=None):
    m = r.get("_meta", {})
    src = f"`{script}`; " if script else ""
    return f"*{src}{m.get('machine', '?')}, Python {m.get('python', '?')}, {m.get('when', '?')}*"


def wp2(out):
    r = load("wp2_overhead")
    out.append("## WP2: observation overhead")
    if not r:
        out.append("Not yet run (`python validation/overhead.py`).\n")
        return
    out.append(meta(r, "validation/overhead.py") + "\n")
    out.append(f"Each number is the best of {r.get('repeats', 7)} fresh processes; the three "
               f"configurations run interleaved, in rotating order, round by round. Target: "
               f"<= 5% on the validation pipeline and <= 25% on 100k-call microloops.\n")
    out.append("| workload | plain | C fast path | overhead | pure Python | overhead | "
               "sys.monitoring floor |")
    out.append("|---|---|---|---|---|---|---|")
    p, f, q, fl = r["plain"], r["fast"], r["pure"], r.get("floor")
    for i, form in enumerate(r["forms"]):
        floor = (f"+{fl['micro_ns'][i] - p['micro_ns'][i]:.0f} ns ({pct(fl['micro_overhead'][i])})"
                 if fl else "n/a")
        out.append(f"| 100k x {form} | {p['micro_ns'][i]:.0f} ns/call | {f['micro_ns'][i]:.0f} | "
                   f"{pct(f['micro_overhead'][i])} | {q['micro_ns'][i]:.0f} | "
                   f"{pct(q['micro_overhead'][i])} | {floor} |")
    out.append(f"| validation pipeline (1M rows) | {p['pipeline_ns'] / 1e6:.2f} ms | "
               f"{f['pipeline_ns'] / 1e6:.2f} ms | {pct(f['pipeline_overhead'])} | "
               f"{q['pipeline_ns'] / 1e6:.2f} ms | {pct(q['pipeline_overhead'])} | |\n")
    if fl:
        out.append("*sys.monitoring floor*: CPython's own cost of reporting one call to an "
                   "observer, measured with no-op C callbacks enabled only on the benchmark's "
                   "code. The microloop target leaves an observer only the difference between "
                   "25% and this floor; see docs/observation.md#overhead.\n")


def wp3(out):
    r = load("wp3_check_engine")
    out.append("## WP3: measurement core (`rglns check`)")
    if not r:
        out.append("Not yet run.\n")
        return
    out.append(meta(r, "validation/check_engine.py") + "\n")
    out.append(f"Simulated runs with run-level noise sigma_run and 20 calls per run (10% "
               f"per-call noise); one session per side; {r['trials']} trials per cell; theta "
               f"= 5%. *False regression* is the share of trials flagged as a regression "
               f"when the true effect is at most theta (requirement: <= 5% + 1.5 pp). "
               f"*Coverage* is the share of 95% intervals containing the true effect "
               f"(requirement: 93-97%). The v0.1 engine (calls pooled, percentile "
               f"bootstrap of the ratio of means) ran on the same data.\n")
    out.append("| sigma_run | effect | runs | false regression (new) | coverage (new) | "
               "false regression (v0.1) | coverage (v0.1) |")
    out.append("|---|---|---|---|---|---|---|")
    for c in r["grid"]:
        fr_new = pct(c["new_regression"]) if c["is_null"] else f"power {pct(c['new_regression'])}"
        fr_old = pct(c["old_regression"]) if c["is_null"] else f"power {pct(c['old_regression'])}"
        out.append(f"| {pct(c['sigma'], 0)} | {pct(c['effect'], 0)} | {c['runs']} | {fr_new} | "
                   f"{pct(c['new_coverage'])} | {fr_old} | {pct(c['old_coverage'])} |")
    out.append("")
    nulls = [c for c in r["grid"] if c["is_null"]]
    out.append(f"Rows with a 10% effect show *power* instead. v0.1's higher power there "
               f"comes from pooling calls: the same intervals cover the true effect as "
               f"rarely as {pct(min(c['old_coverage'] for c in r['grid']))} of the time, and "
               f"v0.1 raises false regressions in up to "
               f"{pct(max(c['old_regression'] for c in nulls))} of null trials (new engine: "
               f"{pct(max(c['new_regression'] for c in nulls))}). The new engine answers "
               f"*inconclusive* in those cases, with an estimate of the runs needed.\n")
    out.append(f"**Drift canaries (simulated bandwidth hog, code unchanged):** new engine "
               f"verdicts {', '.join(f'{k} {pct(v)}' for k, v in r['canary_sim'].items())}; "
               f"v0.1 called it a regression in {pct(r['canary_sim_v01_regression'])} of trials.\n")
    bh = r["bh"]
    out.append(f"**Benjamini-Hochberg** (200 operations, 10 with a true +10% effect, sigma 3%, "
               f"{bh.get('B', '?')} bootstrap replicates per operation): FDR "
               f"{pct(bh['fdr'])} (requirement <= 5%), power {pct(bh['power_bh'])} vs "
               f"Bonferroni {pct(bh['power_bonferroni'])}.\n")
    out.append(f"**Size normalization** (baseline n in [1e5, 1e6], current at 2x n, identical "
               f"cost model): regression flagged in {pct(r['size_false_regression'])} of trials.\n")
    hog = load("wp3_canary_hog")
    if hog:
        out.append(f"**Real canary test** ({meta(hog, 'validation/canary_hog.py')}): a memory-bound workload measured "
                   f"quietly, then again with {hog['hogs']} np.copy bandwidth hogs running "
                   f"(median raw slowdown {pct(hog['median_raw_slowdown'])}). Verdicts over "
                   f"{hog['trials']} trials: "
                   f"{', '.join(f'{k} {pct(v)}' for k, v in hog['verdicts'].items())} "
                   f"(requirement: environment drift >= 95%). The v0.1 engine called it a "
                   f"regression in {pct(hog['v01_regression_rate'])}.\n")


def wp4(out):
    r = load("wp4_compare")
    out.append("## WP4: `rglns compare`")
    if not r:
        out.append("Not yet run.\n")
        return
    out.append(meta(r, "validation/compare_sequential.py") + "\n")
    out.append("Null simulations (effect 0), monitored after every block up to 100 blocks: "
               "share of runs that ever declared a regression or improvement. Requirement "
               "<= 5.5% at alpha = 0.05. *theta = 0* counts any interval excluding zero "
               "(the strict test of time-uniform coverage). The naive strategy recomputes a "
               "fixed-n bootstrap interval after every block.\n")
    out.append("| theta | sigma (block) | confidence sequence | naive repeated bootstrap |")
    out.append("|---|---|---|---|")
    for row in r["null"]:
        out.append(f"| {pct(row['theta'], 0)} | {pct(row['sigma'], 0)} | {pct(row['cs'], 2)} | "
                   f"{pct(row['naive'], 1)} |")
    e = r["efficiency"]
    out.append("")
    out.append(f"**Efficiency:** +10% at sigma 3%: median {e['median_blocks_10pct']:.0f} blocks "
               f"to a regression verdict (requirement: median <= 8; 90th percentile "
               f"{e['p90_blocks_10pct']:.0f}). +3% with theta 5%: declared a regression in "
               f"{pct(e['regression_rate_3pct'], 2)} of trials (requirement <= 5%).\n")
    real = r.get("real")
    if real:
        out.append(f"**Real runs:** A/A (`rglns compare HEAD HEAD`, {real['aa_runs']} times): "
                   f"{real['aa_false']} false verdicts ({pct(real['aa_false'] / real['aa_runs'])}; "
                   f"requirement <= 5%). Injected np.convolve slowdown flagged: "
                   f"{real['fixture_flagged']}; flagged with a bandwidth hog started mid-run: "
                   f"{real['fixture_flagged_with_hog']}.\n")


def wp5(out):
    r = load("wp5_bisect")
    out.append("## WP5: `rglns bisect` (simulation)")
    if not r:
        out.append("Not yet run.\n")
        return
    out.append(meta(r, "validation/bisect_sim.py") + "\n")
    out.append(f"Endpoints: 20 runs each for every method. RegressLens: stop at 95% posterior, "
               f"3 runs per visit, {r['trials']} trials per cell, at most 4000 runs. "
               f"Baselines: the smallest fixed runs-per-step reaching >= 95% correct "
               f"localization ({r['search_trials']} trials per setting; '-' = never reached "
               f"with up to 144 runs per step). *Calibration* = true commit inside the 95% "
               f"credible set (requirement >= 94%).\n")
    out.append("| N | effect | sigma | RegressLens runs | accuracy | calibration | "
               "naive bisection runs | Mann-Whitney runs |")
    out.append("|---|---|---|---|---|---|---|---|")
    for c in r["grid"]:
        o = c["ours"]
        nv = f"{c['naive']['runs']:.0f}" if c["naive"] else "-"
        mw = f"{c['mwu']['runs']:.0f}" if c["mwu"] else "-"
        out.append(f"| {c['N']} | {pct(c['effect'], 0)} | {pct(c['sigma'], 0)} | "
                   f"{o['runs']:.0f} | {pct(o['accuracy'])} | {pct(o['calibration'])} | {nv} | {mw} |")
    a = r["prior_ablation"]
    out.append("")
    out.append(f"**Attribution prior** (N = 128, +10%, sigma 5%): uniform prior "
               f"{a['uniform']['runs']:.0f} runs ({pct(a['uniform']['accuracy'])} correct); "
               f"informative for the culprit with probability 0.8: "
               f"{a['informative_0.8']['runs']:.0f} runs ({pct(a['informative_0.8']['accuracy'])}); "
               f"misleading (culprit never flagged): {a['misleading']['runs']:.0f} runs "
               f"({pct(a['misleading']['accuracy'])}), so it still converges.\n")


def _prior_cell(r):
    # Same wording as regressbench/run.py --summary.
    if "prior_flags_culprit" not in r:
        return "-"
    if not r["prior_flagged"]:
        return "uniform"
    which = "culprit flagged" if r["prior_flags_culprit"] else "culprit not flagged"
    return f"{r['prior_flagged']} of {r['commits']} flagged, {which}"


def regressbench(out):
    out.append("## RegressBench")
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "regressbench", "results", "*.json"))):
        rows.append(json.load(open(path)))
    if not rows:
        out.append("No results yet.\n")
        return
    out.append("*`regressbench/run.py`, one Docker container per case with the environment "
               "listed in the case file; see regressbench/README.md*\n")
    out.append("Blind `rglns bisect`: it is given the good and bad commits and the benchmark "
               "command, never the culprit.\n")
    out.append("| case | correct | culprit in 95% set | runs | wall-clock | attribution prior |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        if "correct" in r:
            out.append(f"| {r['id']} | {'yes' if r['correct'] else 'no'} | "
                       f"{'yes' if r['in_credible_set'] else 'no'} | {r['runs']} | "
                       f"{r['elapsed_s']:.0f} s | {_prior_cell(r)} |")
        else:
            why = r.get("note") or f"error (exit {r.get('exit_code')}); see its log"
            out.append(f"| {r['id']} | {why} | | | | |")
    out.append("")
    out.append("*Attribution prior*: the commits touching a file in the regressed call "
               "site's recorded stack get prior weight 1 + lambda (lambda = 4). *Uniform* "
               "means no commit in the range touched those files.\n")


def wp6(out):
    r = load("wp6_watch")
    out.append("## WP6: `rglns watch`")
    if not r:
        out.append("Not yet run.\n")
        return
    d = r["delay"]
    out.append(meta(r, "validation/watch_bocpd.py") + "\n")
    text = (f"False alarms: {r['false_alarms_per_500']:.3f} per 500 null sessions "
            f"(requirement <= 1). 10% shift at sigma 3%: {pct(d['detected'])} detected, "
            f"{pct(d['within_5'])} within 5 sessions, median delay {d['median']:.0f}, "
            f"90th percentile {d['p90']:.0f} sessions (requirement: delay <= 5).")
    if r.get("replay"):
        text += (f" Replay (30 recorded real sessions, +15% injected into the last 4): "
                 f"{'detected' if r['replay']['detected'] else 'not detected'}.")
    out.append(text + "\n")


def wp7(out):
    r = load("wp7_diagnosis")
    out.append("## WP7: diagnosis")
    if not r:
        out.append("Not yet run.\n")
        return
    out.append(meta(r, "validation/diagnosis_faults.py") + "\n")
    out.append("Fault-injection fixtures (tests/fixtures/faults.py); each trial records a fresh "
               "baseline and current session (3 runs each). Requirement: top-1 correct in >= 90% "
               "of trials; null fixture: no diagnosis in >= 95%.\n")
    out.append("| fault | expected top diagnosis | correct | observed top diagnoses |")
    out.append("|---|---|---|---|")
    for fault, v in r.items():
        if fault.startswith("_"):
            continue
        out.append(f"| {fault} | {v['expected'] or 'none'} | {pct(v['correct_rate'])} | "
                   f"{', '.join(f'{k}: {n}' for k, n in v['tops'].items())} |")
    out.append("")


def text_section(out, title, path, script):
    out.append(f"## {title}")
    out.append(f"*`{script}`*\n")
    if os.path.exists(path):
        out.append("```")
        out.append(open(path).read().rstrip())
        out.append("```\n")
    else:
        out.append("Not yet run.\n")


def main():
    out = ["# Validation results", "",
           "Generated by `validation/make_report.py` from `validation/results/` and "
           "`regressbench/results/`. Each section names the script that produced it.", ""]
    wp2(out)
    wp3(out)
    wp4(out)
    wp5(out)
    regressbench(out)
    wp6(out)
    wp7(out)
    text_section(out, "WP8: `rglns verify` (accuracy policy)", os.path.join(RESULTS, "wp8_verify.txt"),
                 "rglns verify --cases 14000 --seed 777")
    text_section(out, "WP8: fifth-kernel gate", os.path.join(RESULTS, "wp8_kernel_gate.txt"),
                 "tools/kernel_gate.py")
    path = os.path.join(ROOT, "docs", "validation.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
