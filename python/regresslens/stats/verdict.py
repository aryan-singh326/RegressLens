"""Verdicts for a set of per-call-site comparisons.

Three verdicts, plus one label for drift:

  regression          the drift-bounded worst case still exceeds +theta:
                      BH rejects "effect <= theta" for this site
  equivalent          the whole effect interval lies within +/- theta
  inconclusive        neither; comes with an estimate of the runs needed
  environment_drift   not a regression after drift correction, although
                      the uncorrected comparison would have flagged one
                      and the canaries moved. The machine changed, not
                      the code.

v0.1 reported "within noise", which conflated "proven fine"
(equivalent) with "can't tell" (inconclusive).
"""
import math
from dataclasses import dataclass

from .fdr import benjamini_hochberg

REGRESSION = "regression"
EQUIVALENT = "equivalent"
INCONCLUSIVE = "inconclusive"
ENVIRONMENT_DRIFT = "environment_drift"
IMPROVEMENT = "improvement"


@dataclass
class Verdict:
    verdict: str
    advisory: bool          # < 3 sessions on a side: cross-session estimate only
    runs_needed: int = None
    reason: str = ""


def assign(comparisons, theta=0.05, alpha=0.05, low_confidence=None):
    """Verdicts for a list of bootstrap.Comparison, with BH across them.
    `low_confidence[i]` (e.g. an extrapolated size model) forces
    inconclusive for that site."""
    theta_log = math.log1p(theta)
    low_confidence = low_confidence or [None] * len(comparisons)
    reject = benjamini_hochberg([c.p_value for c in comparisons], alpha)
    reject_raw = benjamini_hochberg([c.p_raw for c in comparisons], alpha)
    out = []
    for c, rej, rej_raw, low in zip(comparisons, reject, reject_raw, low_confidence):
        advisory = not c.hierarchical
        if low:
            out.append(Verdict(INCONCLUSIVE, advisory, c.runs_needed(theta_log, alpha), low))
        elif rej:
            out.append(Verdict(REGRESSION, advisory))
        elif rej_raw and (abs(c.drift_c) > 0 or abs(c.drift_m) > 0):
            out.append(Verdict(ENVIRONMENT_DRIFT, advisory, reason=(
                f"canaries moved (compute {math.expm1(c.drift_c):+.1%}, memory "
                f"{math.expm1(c.drift_m):+.1%}); the drift-bounded effect is "
                f"[{math.expm1(c.ci_low):+.1%}, {math.expm1(c.ci_high):+.1%}]")))
        elif math.isfinite(c.ci_low) and c.ci_low > -theta_log and c.ci_high < theta_log:
            out.append(Verdict(EQUIVALENT, advisory))
        elif math.isfinite(c.ci_high) and c.ci_high < -theta_log:
            out.append(Verdict(IMPROVEMENT, advisory))
        else:
            out.append(Verdict(INCONCLUSIVE, advisory, c.runs_needed(theta_log, alpha)))
    return out
