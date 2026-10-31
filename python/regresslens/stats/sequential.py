"""Anytime-valid confidence sequence for interleaved comparisons (WP4).

`rglns compare` observes one difference per ABBA/BAAB block,
    d = mean(log t_B) - mean(log t_A),
and checks after every block whether to stop. Checking a fixed-n
interval after every block inflates the false-positive rate far beyond
alpha. Every look is another chance to cross by luck. A confidence
sequence is valid simultaneously over all n, so it can be checked
after every block (normal mixture / mSPRT boundary):

    h_n = sqrt( 2 s2 (s2 + n tau2) / (n^2 tau2) * ln( sqrt(1 + n tau2 / s2) / alpha ) )

CS_n = mean(d_1..d_n) +/- h_n, with s2 the variance of one block
difference and tau the mixture scale (default: theta on the log scale).
s2 is estimated from the blocks (>= 3), inflated and floored at
(0.1%)^2, so the guarantee is approximate. The null simulations in
docs/validation.md are the evidence that it holds in practice.

Inflation: max(1.15, (n-1) / chi2_{0.2, n-1}), the 80% upper confidence
bound on the variance. A sample variance from 3-5 blocks is often far
too small, and a flat 1.15 then let the sequence exclude zero under the
null in ~9% of runs (theta = 0, sigma = 5%). The chi-square factor is
~3 at 4 blocks, ~1.3 at 30 and falls to 1.15 by ~100 blocks.


Stop rules (theta, epsilon on the log scale, epsilon < theta):
    CS inside (theta, inf)       regression
    CS inside (-eps, eps)        equivalent
    CS inside (-inf, -theta)     improvement
    max blocks reached           inconclusive (with the interval)
"""
import math
from dataclasses import dataclass

import numpy as np

from .verdict import EQUIVALENT, IMPROVEMENT, INCONCLUSIVE, REGRESSION

VARIANCE_INFLATION = 1.15
SIGMA_FLOOR = 0.001
VARIANCE_UCB_LEVEL = 0.80
DEFAULT_TAU = math.log1p(0.05)


def chi2_ppf(p, k):
    """Chi-square quantile (Wilson-Hilferty; within ~1% for k >= 2)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p)
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def variance_inflation(n):
    """Small-sample correction for an estimated block variance."""
    k = n - 1
    return max(VARIANCE_INFLATION, k / chi2_ppf(1 - VARIANCE_UCB_LEVEL, k))


def halfwidth(n, sigma2, tau2, alpha):
    if n <= 0:
        return float("inf")
    ratio = 1.0 + n * tau2 / sigma2
    return math.sqrt(2.0 * sigma2 * (sigma2 + n * tau2) / (n * n * tau2)
                     * math.log(math.sqrt(ratio) / alpha))


@dataclass
class State:
    n: int
    mean: float
    halfwidth: float
    sigma: float

    @property
    def low(self):
        return self.mean - self.halfwidth

    @property
    def high(self):
        return self.mean + self.halfwidth


class ConfidenceSequence:
    def __init__(self, theta=0.05, alpha=0.05, epsilon=None, min_blocks=4, max_blocks=100,
                 tau=None):
        self.theta_log = math.log1p(theta)
        self.eps_log = math.log1p(epsilon) if epsilon is not None else self.theta_log / 2
        if self.eps_log >= self.theta_log and self.theta_log > 0:
            raise ValueError("epsilon must be below theta")
        self.alpha = alpha
        self.min_blocks = max(min_blocks, 3)
        self.max_blocks = max_blocks
        self.tau = tau if tau is not None else (self.theta_log if self.theta_log > 0
                                                else DEFAULT_TAU)
        self.blocks = []

    def add(self, d):
        self.blocks.append(float(d))

    def state(self):
        n = len(self.blocks)
        if n < 3:
            return None
        arr = np.asarray(self.blocks)
        s2 = max(arr.var(ddof=1) * variance_inflation(n), SIGMA_FLOOR ** 2)
        h = halfwidth(n, s2, self.tau ** 2, self.alpha)
        return State(n, float(arr.mean()), h, math.sqrt(s2))

    def decision(self):
        """A verdict once a stop rule fires, else None."""
        st = self.state()
        if st is None or st.n < self.min_blocks:
            return None
        if st.low > self.theta_log:
            return REGRESSION
        if st.high < -self.theta_log:
            return IMPROVEMENT
        if st.low > -self.eps_log and st.high < self.eps_log:
            return EQUIVALENT
        if st.n >= self.max_blocks:
            return INCONCLUSIVE
        return None
