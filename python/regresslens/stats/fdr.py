"""Multiple-comparison control across call sites.

Benjamini-Hochberg controls the false discovery rate (the expected
fraction of flagged call sites that are not real regressions). It
replaced Bonferroni, which controls the chance of even one false flag
and at 200+ call sites shrinks each site's alpha so far that real
regressions go undetected.
"""
import numpy as np


def benjamini_hochberg(pvalues, alpha=0.05):
    """Boolean mask of rejected hypotheses (BH step-up procedure)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresholds = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresholds
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.max(np.nonzero(passed)[0])
        reject[order[:k + 1]] = True
    return reject


def bonferroni(pvalues, alpha=0.05):
    p = np.asarray(pvalues, dtype=float)
    return p <= alpha / max(len(p), 1)
