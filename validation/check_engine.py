"""WP3 Monte Carlo: the check engine against the v0.1 engine.

  * grid sigma_run in {1%, 3%, 8%} x effect in {0, 3%, 10%} x runs in
    {10, 20, 40}, 2000 trials each: false-regression rate (effect <=
    theta) must be <= 5% (+1.5 pp Monte Carlo slack), CI coverage 93-97%;
  * canary (simulated bandwidth hog): environment drift, not regression;
  * BH: 200 operations, 10 with +10%, sigma 3%: FDR <= 5%, power vs
    Bonferroni;
  * size normalization: baseline n in [1e5, 1e6], current at 2x n.

    python validation/check_engine.py [--trials 2000]
"""
import argparse
import math
import os
import sys

import numpy as np

import common
import simlib


def size_test(trials):
    sys.path.insert(0, os.path.join(common.HERE, "..", "python", "tests"))
    from regresslens import analysis, hw
    from test_stats import _fake_sessions
    hw.cache_sizes = lambda: {"l1d": 1e15, "l2": 2e15, "l3": 3e15}
    flagged = 0
    rng = np.random.default_rng(0)
    for t in range(trials):
        sizes = [int(x) for x in np.geomspace(1e5, 1e6, 6)]
        rep = analysis.compare_sessions(_fake_sessions(rng, sizes, "b"),
                                        _fake_sessions(rng, [2 * n for n in sizes], "c"),
                                        {}, seed=t, with_diagnosis=False, B=1000)
        site = next(s for s in rep.sites if s.key == "site")
        flagged += site.verdict.verdict == "regression"
    return flagged / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--bh-trials", type=int, default=200)
    ap.add_argument("--bh-B", type=int, default=10_000,
                    help="bootstrap replicates per operation in the BH test; with fewer "
                         "than m/alpha = 4000 the smallest attainable p-value is above "
                         "Bonferroni's cutoff, which would make Bonferroni powerless by "
                         "construction rather than by its own conservatism")
    ap.add_argument("--bh-only", action="store_true",
                    help="rerun only the BH test, updating results/wp3_check_engine.json")
    args = ap.parse_args()
    if args.bh_only:
        import json
        path = os.path.join(common.RESULTS, "wp3_check_engine.json")
        res = json.load(open(path))
        res["bh"] = simlib.bh_experiment(args.bh_trials, seed=7, B=args.bh_B)
        res["bh"]["B"] = args.bh_B
        res.pop("_meta", None)
        common.progress(f"BH: {res['bh']}")
        common.save("wp3_check_engine", res)
        return
    grid = []
    seed = 0
    for sigma in (0.01, 0.03, 0.08):
        for effect in (0.0, 0.03, 0.10):
            for runs in (10, 20, 40):
                seed += 1
                cell = simlib.check_cell(sigma, effect, runs, args.trials, seed=seed)
                cell.update(sigma=sigma, effect=effect, runs=runs)
                grid.append(cell)
                common.progress(f"sigma={sigma} effect={effect} runs={runs}: "
                                f"new reg {cell['new_regression']:.3f} cov {cell['new_coverage']:.3f}"
                                f" | old reg {cell['old_regression']:.3f} cov {cell['old_coverage']:.3f}")
    canary, canary_old = simlib.canary_cell(args.trials // 4, seed=99)
    common.progress(f"canary: {canary} (v0.1 regression rate {canary_old:.3f})")
    bh = simlib.bh_experiment(args.bh_trials, seed=7, B=args.bh_B)
    bh["B"] = args.bh_B
    common.progress(f"BH: {bh}")
    size = size_test(200)
    common.progress(f"size test: regression flagged in {size:.3f} of trials")
    common.save("wp3_check_engine", {"grid": grid, "canary_sim": canary,
                                     "canary_sim_v01_regression": canary_old, "bh": bh,
                                     "size_false_regression": size, "trials": args.trials})


if __name__ == "__main__":
    main()
