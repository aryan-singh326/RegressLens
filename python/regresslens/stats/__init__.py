"""Statistics for RegressLens.

  bootstrap   run-level hierarchical bootstrap, drift bounds, BCa intervals
  verdict     regression / equivalent / inconclusive / environment drift
  fdr         Benjamini-Hochberg across call sites
  costmodel   size-normalized cost model t(n) = alpha + beta*n (Theil-Sen)
  sequential  anytime-valid confidence sequences for `rglns compare`
  bisection   probabilistic bisection posterior for `rglns bisect`
  bocpd       Bayesian online change-point detection for `rglns watch`
  legacy      the v0.1 engine (validation comparisons only)
"""
