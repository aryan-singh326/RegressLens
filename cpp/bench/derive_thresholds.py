"""Derives kernel-selector thresholds from a characterize CSV.

    ./characterize > results/characterize-<host>.csv
    python derive_thresholds.py results/characterize-<host>.csv

A kernel is preferred only where it is at least MIN_GAIN faster than the
alternative (median vs median). Inside that margin the simpler kernel
wins, because a 2% difference measured on one machine does not survive
moving to another. Prints the per-operation evidence and the constants
for src/kernel_selector.cpp.
"""
import csv
import sys
from collections import defaultdict

MIN_GAIN = 0.05


def load(path):
    rows = defaultdict(dict)  # (op, dtype, n, param_base) -> {kernel: median}
    with open(path) as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for r in reader:
            # Thread counts are not a grouping dimension: mt rows join the
            # single-threaded rows of the same (op, dtype, n, param).
            param = "+".join(t for t in r["param"].split("+") if t and
                             not t.startswith("threads="))
            rows[(r["operation"], r["dtype"], int(r["n"]), param)][r["kernel"]] = float(
                r["median_ns"])
    return rows


def faster(a, b):
    """True when a is at least MIN_GAIN faster than b."""
    return a < (1.0 - MIN_GAIN) * b


def _pairs(rows, op, a, b):
    """(n, a at least MIN_GAIN faster than b) for every measurement of op."""
    return [(n, faster(k[a], k[b])) for (o, _d, n, _p), k in sorted(rows.items())
            if o == op and a in k and b in k]


def _report(rows, op, a, b):
    for (o, dtype, n, param), k in sorted(rows.items()):
        if o == op and a in k and b in k:
            print(f"  {dtype} n={n:>10,} {param:<10} {a} {k[b] / k[a]:.2f}x vs {b}")


def main(path):
    rows = load(path)
    out = {}
    for op in ("reduction", "projection", "rolling"):
        print(f"== {op}: avx2 vs scalar (speedup = scalar time / avx2 time)")
        _report(rows, op, "avx2", "scalar")
        out[f"{op}_avx2_min_n"] = min_n_where_always(_pairs(rows, op, "avx2", "scalar"))
    print("== reduction: mt_avx2 vs avx2")
    _report(rows, "reduction", "mt_avx2", "avx2")
    out["reduction_mt_min_n"] = min_n_where_always(_pairs(rows, "reduction", "mt_avx2", "avx2"))

    print("== filter: avx2 vs scalar by selectivity (all sizes, both dtypes)")
    by_sel = defaultdict(list)
    for (op, _dtype, _n, param), k in sorted(rows.items()):
        if op == "filter":
            by_sel[float(param.split("=")[1])].append(faster(k["avx2"], k["scalar"]))
    band = []
    for sel in sorted(by_sel):
        share = sum(by_sel[sel]) / len(by_sel[sel])
        print(f"  sel={sel:.2f}: avx2 faster in {share:.0%} of size/dtype combinations")
        if share >= 0.75:
            band.append(sel)
    out["filter_avx2_band"] = (min(band), max(band)) if band else None
    print("== filter: mt_avx2 vs avx2 (inside the band)")
    mt = [(n, faster(k["mt_avx2"], k["avx2"])) for (o, _d, n, p), k in sorted(rows.items())
          if o == "filter" and "mt_avx2" in k and band
          and band[0] <= float(p.split("=")[1]) <= band[-1]]
    out["filter_mt_min_n"] = min_n_where_always(mt)

    print()
    print("== derived selector decisions (None: never reliably faster)")
    for key, value in out.items():
        print(f"  {key}: {value}")
    return 0


def min_n_where_always(pairs):
    """Smallest n such that the faster-flag holds for every measurement at
    that n and at every larger n; None if it fails at the largest n."""
    by_n = defaultdict(list)
    for n, ok in pairs:
        by_n[n].append(ok)
    best = None
    for n in sorted(by_n, reverse=True):
        if all(by_n[n]):
            best = n
        else:
            break
    return best


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
