"""Probabilistic performance bisection (WP5).

Candidates c_0..c_N along the first-parent chain: c_0 is known good and
c_N known bad. Hypothesis H_k (k = 1..N): c_k is the first bad commit.
Each run at commit i gives y = log(time) - log(canary blend), modeled as

    y ~ N(m_G, s^2)  if i <  k   (good)
    y ~ N(m_B, s^2)  if i >= k   (bad)

    p_k  propto  pi_k * prod_ij N(y_ij | mu_i(k), s^2)

m_G, m_B and s are unknown, and plugging in point estimates makes the
posterior overconfident when the effect is small relative to the noise
(measured: 73% correct at a claimed 95% for a 3% effect, 5% noise). They
are therefore integrated out: flat priors on the means and a Jeffreys
prior on s give the closed form

    p_k  propto  pi_k * (n_G n_B)^(-1/2) * RSS_k^(-(n - 2) / 2)

where n_G / n_B count the runs on each side of the split and RSS_k is
the within-group residual sum of squares. With per-commit sufficient
statistics (n_i, sum y, sum y^2) and prefix sums, the whole posterior
costs O(N) per update.

Prior: pi_k propto 1 + lambda * a_k, with a_k = 1 when commit k touches a
file in the regressed call site's recorded stack (runtime attribution).
Data overrides a wrong prior; it only changes how fast the posterior
converges.

The next commit to measure is the posterior median: one measurement
at i reveals only whether k <= i, so the median is near-optimal. Effort
goes into the batch size per visit instead (see cli bisect).
"""
import math

import numpy as np


class Bisector:
    def __init__(self, n_commits, prior=None, excluded=()):
        """n_commits = N (hypotheses 1..N). `prior`: weights for k=1..N."""
        self.N = n_commits
        self.n = np.zeros(n_commits + 1)
        self.s = np.zeros(n_commits + 1)
        self.q = np.zeros(n_commits + 1)
        prior = np.ones(n_commits) if prior is None else np.asarray(prior, dtype=float)
        self.log_prior = np.log(prior / prior.sum())
        self.excluded = set(excluded)
        self.m_good = self.m_bad = None
        self.sigma = None
        self._post = None

    # -- data ----------------------------------------------------------------

    def add(self, i, ys):
        ys = np.asarray(ys, dtype=float)
        ys = ys[np.isfinite(ys)]
        self.n[i] += len(ys)
        self.s[i] += ys.sum()
        self.q[i] += (ys ** 2).sum()
        self._post = None
        self._estimate()

    def exclude(self, i):
        self.excluded.add(i)

    def _split_stats(self):
        """Per hypothesis k = 1..N: (n_G, S_G, Q_G, n_B, S_B, Q_B)."""
        cn, cs, cq = np.cumsum(self.n), np.cumsum(self.s), np.cumsum(self.q)
        tn, ts, tq = cn[-1], cs[-1], cq[-1]
        ng, sg, qg = cn[:-1], cs[:-1], cq[:-1]          # commits 0..k-1
        return ng, sg, qg, tn - ng, ts - sg, tq - qg

    def _ready(self):
        return self.n[0] >= 2 and self.n[self.N] >= 2

    def _posterior(self):
        if self._post is not None:
            return self._post
        ng, sg, qg, nb, sb, qb = self._split_stats()
        rss = (qg - sg ** 2 / ng) + (qb - sb ** 2 / nb)
        total = ng[0] + nb[0]
        rss = np.maximum(rss, 1e-300)
        logp = (self.log_prior - 0.5 * np.log(ng) - 0.5 * np.log(nb)
                - 0.5 * (total - 2) * np.log(rss))
        logp -= logp.max()
        p = np.exp(logp)
        self._post = p / p.sum()
        return self._post

    def _estimate(self):
        """Posterior-weighted means and pooled sigma, for reporting."""
        if not self._ready():
            return
        post = self._posterior()
        ng, sg, qg, nb, sb, qb = self._split_stats()
        self.m_good = float(np.sum(post * sg / ng))
        self.m_bad = float(np.sum(post * sb / nb))
        rss = (qg - sg ** 2 / ng) + (qb - sb ** 2 / nb)
        self.sigma = float(math.sqrt(max(np.sum(post * rss) / max(ng[0] + nb[0] - 2, 1),
                                         1e-24)))

    def posterior(self):
        """P(first bad commit = k) for k = 1..N (index 0 -> k=1)."""
        if not self._ready():
            p = np.exp(self.log_prior)
            return p / p.sum()
        return self._posterior()

    def best(self):
        p = self.posterior()
        k = int(np.argmax(p))
        return k + 1, float(p[k])

    def credible_set(self, level=0.95):
        """Smallest set of commits (k values) holding >= level mass."""
        p = self.posterior()
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        cut = int(np.searchsorted(cum, level)) + 1
        return sorted(int(k) + 1 for k in order[:cut])

    def next_commit(self):
        """Commit index to measure next: the posterior median, moved to the
        nearest buildable commit if needed. Measuring commit i tells
        whether k <= i, so candidates are 1..N-1."""
        if self.N <= 1:
            return None
        cdf = np.cumsum(self.posterior())                # cdf[j] = P(k <= j+1)
        candidates = [i for i in range(1, self.N) if i not in self.excluded]
        if not candidates:
            return None
        return min(candidates, key=lambda i: abs(cdf[i - 1] - 0.5))

    # -- model check -----------------------------------------------------------

    def drift_log_bayes_factor(self):
        """log(marginal L_drift / marginal L_step), both with unknown sigma
        (Jeffreys) and two flat-prior mean parameters: the step model's
        (m_G, m_B) per split, marginalized over k with its prior, and the
        drift model's line mu_i = a + b*i fitted to all runs."""
        if not self._ready():
            return float("-inf")
        idx = np.arange(self.N + 1, dtype=float)
        w = self.n
        if (w > 0).sum() < 3:
            return float("-inf")
        sw, sx, sxx = w.sum(), (w * idx).sum(), (w * idx * idx).sum()
        sy, sxy, syy = self.s.sum(), (idx * self.s).sum(), self.q.sum()
        det = sw * sxx - sx * sx
        if det <= 0:
            return float("-inf")
        b = (sw * sxy - sx * sy) / det
        a = (sy - b * sx) / sw
        rss_line = max(syy - a * sy - b * sxy, 1e-300)
        total = sw
        ll_drift = -0.5 * math.log(det) - 0.5 * (total - 2) * math.log(rss_line)

        ng, sg, qg, nb, sb, qb = self._split_stats()
        rss = np.maximum((qg - sg ** 2 / ng) + (qb - sb ** 2 / nb), 1e-300)
        terms = (self.log_prior - 0.5 * np.log(ng * nb) - 0.5 * (total - 2) * np.log(rss))
        ll_step = float(np.logaddexp.reduce(terms))
        return float(ll_drift - ll_step)
