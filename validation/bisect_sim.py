"""WP5: probabilistic bisection vs fixed-sample baselines (simulation).

Grid N in {16, 128, 1024} x effect in {3%, 10%, 30%} x sigma in
{2%, 5%, 10%}. For RegressLens (stop at 95% posterior, batch 3) report
accuracy, runs and calibration (true commit inside the 95% credible set,
target >= 94%). For baseline A (classic bisection, threshold at the
endpoint midpoint) and baseline B (fixed-sample Mann-Whitney, Pinpoint
style), report the runs needed to reach >= 95% correct localization,
searching the runs-per-step setting. Prior ablation: attribution flags
informative for the culprit with probability 0.8 vs a uniform prior, and
a misleading prior (culprit never flagged).

    python validation/bisect_sim.py [--trials 2000] [--search-trials 400]
"""
import argparse

import numpy as np

import common
import simlib


def ours(rng, N, effect, sigma, trials, prior_p=None):
    res = []
    for _ in range(trials):
        k = int(rng.integers(1, N + 1))
        prior = simlib.prior_vector(rng, N, k, prior_p) if prior_p is not None else None
        res.append(simlib.probabilistic_bisect(rng, N, k, effect, sigma, prior=prior))
    return {"accuracy": float(np.mean([r["correct"] for r in res])),
            "calibration": float(np.mean([r["in_credible"] for r in res])),
            "runs": float(np.mean([r["runs"] for r in res]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--search-trials", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(5)
    grid = []
    for N in (16, 128, 1024):
        for effect in (0.03, 0.10, 0.30):
            for sigma in (0.02, 0.05, 0.10):
                cell = {"N": N, "effect": effect, "sigma": sigma,
                        "ours": ours(rng, N, effect, sigma, args.trials),
                        "naive": simlib.runs_to_95(simlib.naive_bisect, rng, N, effect, sigma,
                                                   args.search_trials),
                        "mwu": simlib.runs_to_95(simlib.mwu_bisect, rng, N, effect, sigma,
                                                 args.search_trials)}
                grid.append(cell)
                common.progress(f"N={N} eff={effect} sigma={sigma}: ours {cell['ours']} | "
                                f"naive {cell['naive']} | mwu {cell['mwu']}")
    ablation = {}
    for label, p in (("uniform", None), ("informative_0.8", 0.8), ("misleading", 0.0)):
        ablation[label] = ours(rng, 128, 0.10, 0.05, args.trials // 2, prior_p=p)
        common.progress(f"prior {label}: {ablation[label]}")
    common.save("wp5_bisect", {"grid": grid, "prior_ablation": ablation,
                               "trials": args.trials, "search_trials": args.search_trials})


if __name__ == "__main__":
    main()
