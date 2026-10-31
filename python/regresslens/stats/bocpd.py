"""Bayesian online change-point detection (Adams & MacKay 2007) for
`rglns watch` (WP6).

Each session of a recurring pipeline is one observation (canary-adjusted
log time). The posterior over the current run length r_t (sessions
since the last change) is updated online with:
  * a Normal-Inverse-Gamma conjugate prior, giving a Student-t
    predictive per run length;
  * a constant hazard H = 1/100 (a change every ~100 sessions a priori);
  * truncation at 500 run lengths.
An alert fires when the posterior mass on "a change within the last k
sessions", P(r_t <= k), exceeds 0.9 (r_t counts the observations since
the change).

The prior is set from a warm-up window (median and MAD of the first
sessions), so it is scale-free: the same defaults work for 1 ms and
1 h pipelines. Prior art: MongoDB's E-divisive change-point work; this
is here because the recurring-pipeline user needs it, not as a novelty.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Alert:
    index: int          # session index at which the alert fired
    probability: float  # P(change within the last k sessions)
    onset: int          # most probable change location (index of first changed session)


class BOCPD:
    def __init__(self, mu0, sigma0, hazard=0.01, max_run=500, kappa0=0.1, alpha0=2.0):
        self.hazard = hazard
        self.max_run = max_run
        self.prior = (float(mu0), float(kappa0), float(alpha0),
                      float(alpha0) * float(sigma0) ** 2)
        self.log_p = np.array([0.0])            # log P(r_t = r)
        mu, k, a, b = self.prior
        self.mu, self.kappa = np.array([mu]), np.array([k])
        self.alpha, self.beta = np.array([a]), np.array([b])
        # alpha_r = alpha0 + r/2, so the Student-t normalizing constants
        # depend only on r: precompute them once.
        df = 2 * alpha0 + np.arange(max_run + 2)
        self._lg = np.array([math.lgamma(v / 2 + 0.5) - math.lgamma(v / 2) for v in df])

    def _log_pred(self, x):
        mu, k, a, b = self.mu, self.kappa, self.alpha, self.beta
        df = 2 * a
        scale2 = b * (k + 1) / (a * k)
        z = (x - mu) ** 2 / (df * scale2)
        lg = self._lg[:len(df)]
        return lg - 0.5 * np.log(df * math.pi * scale2) - (df + 1) / 2 * np.log1p(z)

    def update(self, x):
        log_pred = self._log_pred(x)
        log_h, log_1mh = math.log(self.hazard), math.log1p(-self.hazard)
        growth = self.log_p + log_pred + log_1mh
        change = np.logaddexp.reduce(self.log_p + log_pred + log_h)
        log_p = np.concatenate([[change], growth])
        # Posterior parameters: r = 0 restarts from the prior.
        mu0, k0, a0, b0 = self.prior
        mu, k, a, b = self.mu, self.kappa, self.alpha, self.beta
        new_mu = (k * mu + x) / (k + 1)
        new_b = b + k * (x - mu) ** 2 / (2 * (k + 1))
        self.mu = np.concatenate([[mu0], new_mu])
        self.kappa = np.concatenate([[k0], k + 1])
        self.alpha = np.concatenate([[a0], a + 0.5])
        self.beta = np.concatenate([[b0], new_b])
        if len(log_p) > self.max_run:
            log_p = log_p[:self.max_run]
            for name in ("mu", "kappa", "alpha", "beta"):
                setattr(self, name, getattr(self, name)[:self.max_run])
        self.log_p = log_p - np.logaddexp.reduce(log_p)
        return np.exp(self.log_p)


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    return mad if mad > 0 else (np.std(x) if np.std(x) > 0 else 1e-6)


def detect(series, k=8, threshold=0.9, hazard=0.01, max_run=500, warmup=8):
    """Runs BOCPD over a series. Returns (alerts, probabilities) where
    probabilities[t] = P(change within the last k sessions) after
    observation t (NaN during warm-up) and alerts are the rising edges of
    probabilities > threshold."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    probs = np.full(n, np.nan)
    if n < warmup + 2:
        return [], probs
    head = x[:warmup]
    model = BOCPD(np.median(head), robust_scale(head), hazard, max_run)
    alerts, above = [], False
    for t in range(n):
        p = model.update(x[t])
        if t < warmup + k:
            continue  # run length is necessarily short at the start
        # r_t counts the observations in the current segment (Adams &
        # MacKay's convention), so "a change within the last k sessions"
        # is r_t <= k.
        probs[t] = float(p[:k + 1].sum())
        if probs[t] > threshold and not above:
            onset = t - int(np.argmax(p[:k + 1])) + 1
            alerts.append(Alert(t, probs[t], onset))
        above = probs[t] > threshold
    return alerts, probs
