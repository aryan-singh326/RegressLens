#pragma once
#include <cmath>

#include "regresslens/kernels.hpp"

// Error-free compensated summation for the SCALAR translation unit.
//
// Knuth's TwoSum recovers the exact rounding error of every addition,
// so a running (s, c) pair represents the true sum to roughly
// eps^2 * n * sum|x|. That is what lets RegressLens state a testable
// accuracy guarantee ("never less accurate than NumPy against an
// exact reference", see docs/accuracy.md): NumPy's pairwise summation
// has error growing with log2(n), plain sequential summation with n.
//
// Requires strict IEEE semantics. Never compile kernels with
// -ffast-math or /fp:fast — reassociation turns TwoSum's error term
// into an algebraic zero.
//
// The AVX2 translation unit must NOT call these member functions: an
// inline function emitted by both TUs is folded into one COMDAT copy
// by the linker, and if the AVX2-compiled copy wins, the scalar path
// executes VEX-encoded instructions on CPUs without AVX. The AVX2 TU
// has its own static helpers (see avx2_common.hpp) and exchanges plain
// SumParts values across the boundary instead.

namespace regresslens {

struct CompSum {
    double s = 0.0;
    double c = 0.0;

    void add(double x) {
        double t = s + x;
        double bp = t - s;
        c += (s - (t - bp)) + (x - bp);
        s = t;
    }

    void merge(const SumParts& o) {
        add(o.s);
        c += o.c;
    }
    void merge(const CompSum& o) { merge(SumParts{o.s, o.c}); }

    // Once s is non-finite the error term is meaningless (Inf - Inf
    // produces NaN in c), and IEEE semantics of the plain sum are what
    // NumPy reports, so s alone is returned.
    double value() const { return std::isfinite(s) ? s + c : s; }

    SumParts parts() const { return SumParts{s, c}; }
};

inline double finish_sum(const SumParts& p) {
    return std::isfinite(p.s) ? p.s + p.c : p.s;
}

// --- Double-double helpers (scalar TU / bindings only) -----------------
//
// A value hi + lo with |lo| <= ulp(hi)/2 carries ~106 bits. Used where
// a result must be rounded only ONCE: means (sum / n) and variances.

struct DD {
    double hi;
    double lo;
};

inline DD two_sum(double a, double b) {
    double s = a + b;
    double bp = s - a;
    return DD{s, (a - (s - bp)) + (b - bp)};
}

// Dekker's exact product a * b = hi + lo without FMA (the scalar TU has
// no FMA). Exact unless a * b overflows; callers keep |values| < 1e150.
inline DD two_prod(double a, double b) {
    const double split = 134217729.0;  // 2^27 + 1
    double p = a * b;
    double ca = split * a, cb = split * b;
    double ahi = ca - (ca - a), alo = a - ahi;
    double bhi = cb - (cb - b), blo = b - bhi;
    return DD{p, ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo};
}

inline DD dd_normalize(double hi, double lo) { return two_sum(hi, lo); }

// (hi + lo) / d with a single final rounding (up to O(eps^2) error).
inline double dd_div(DD a, double d) {
    a = dd_normalize(a.hi, a.lo);
    double q1 = a.hi / d;
    DD p = two_prod(q1, d);
    double r = ((a.hi - p.hi) - p.lo) + a.lo;  // a.hi - p.hi is exact (Sterbenz)
    return q1 + r / d;
}

inline DD dd_div_dd(DD a, double d) {
    a = dd_normalize(a.hi, a.lo);
    double q1 = a.hi / d;
    DD p = two_prod(q1, d);
    double r = ((a.hi - p.hi) - p.lo) + a.lo;
    return two_sum(q1, r / d);
}

// Mean of a compensated sum, rounded once. Non-finite sums follow IEEE.
inline double mean_from_parts(const SumParts& p, double n) {
    if (!std::isfinite(p.s)) return p.s / n;
    return dd_div(DD{p.s, p.c}, n);
}

}  // namespace regresslens
