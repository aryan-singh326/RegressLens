# Statistics

Everything below is validated by simulation and on real hardware; see
[validation.md](validation.md).

## `rglns check`: two groups of sessions

**Unit of analysis: the run.** Each run contributes one value per call site,
the median log time of that run's calls (reservoir samples weighted by call
count). v0.1 resampled individual calls. Calls in one run share that run's
machine state, so treating them as independent pseudo-replicates made its
intervals far too narrow: in simulation v0.1 covered the true effect only
47-94% of the time, and called a +3% change a >5% regression up to 13% of the
time.

**Hierarchy.** With at least 3 sessions on both sides, bootstrap replicates
resample sessions first and runs within them, so session-to-session variation
(a different machine state each invocation) enters the interval. With fewer,
runs are pooled and the verdict is marked *advisory (cross-session)*. That
variation cannot be estimated from one session, which is exactly the flaw that
produced the original false positive when the CPU frequency changed between
sessions.

**Intervals.** Log-ratios of run-level means, BCa intervals (jackknife
acceleration), with Hesterberg's small-sample expansion. Plain BCa under-covers
at 10 runs (~90-94%).

**Drift-bounded correction.** Two canaries are timed inside the measured
process: right after the program imports NumPy, at exit, and every ~30 s
between observed calls on long runs. *compute* runs `np.exp` on an L1-resident
array; *memory* streams a sum over at least 4x the LLC. If an operation's time
scales as c^w * m^(1-w) * effect for an unknown w in [0, 1], the effect lies
between r/c and r/m. Each replicate computes
`lower = d - max(dc, dm)` and `upper = d - min(dc, dm)` on the log scale.
The effect interval is [BCa lower of `lower`, BCa upper of `upper`]; no
fitted model, honest about what it doesn't know.

Canaries are taken in the measured process, not the CLI. A waiting CLI process
gets scheduled differently: on a hybrid CPU its canary read 3-9x slower while
the workload itself was unchanged.

**Verdicts.**

| verdict | meaning |
|---|---|
| regression | Benjamini-Hochberg rejects "effect <= theta" at alpha; p = P*(lower <= theta) |
| equivalent | the whole effect interval lies within +/- theta |
| inconclusive | neither; reports runs needed, n ~ (z sigma / margin)^2 |
| environment drift | the uncorrected comparison would flag a regression, but the drift-bounded one does not: the machine changed |
| improvement | the whole interval lies below -theta |

**Multiple comparisons.** Benjamini-Hochberg across call sites, controlling
the false discovery rate. With 200 call sites, 10 of them truly +10% slower,
BH finds 98.0% of the regressions and Bonferroni 89.7%, both with no false
discoveries ([validation](validation.md#wp3-measurement-core-rglns-check)).
The bootstrap needs enough replicates for this: a p-value from B replicates
cannot fall below 1/B, and Bonferroni's per-site level here is 0.05/200, so
with B = 600 it could never reject anything.

**Size normalization.** Per call site, the baseline's time-vs-size points are
fitted as t(n) = alpha + beta n, piecewise by cache regime (breakpoints are the
detected L1/L2/L3 sizes), with Theil-Sen per piece. Samples are divided by the
baseline prediction at their own n, so 2x more data is not reported as 2x
slower. The report says whether alpha (fixed overhead) or beta (per-element
cost) moved. Sizes more than 2x outside the baseline range are labeled
*extrapolated, low confidence*.

**Diagnosis.** Independent detectors (contiguity loss, dtype promotion, kernel
selection change, NumPy fallback rate, size regime, environment drift) each
return hypotheses with likelihood ratios. These are combined naive-Bayes style,
by summing log-LRs as if the evidence were independent (an approximation,
stated as such), and ranked.

## `rglns compare`: interleaved, anytime-valid

Both revisions run in ABBA blocks, flipped to BAAB at random, so linear drift
cancels within a block. Each block gives d = mean(log t_B) - mean(log t_A).
After every block, a normal-mixture confidence sequence

    h_n = sqrt( 2 s2 (s2 + n tau2) / (n^2 tau2) * ln( sqrt(1 + n tau2 / s2) / alpha ) )

with tau = theta (log scale) is valid at every n simultaneously, so it can
stop as soon as it excludes the decision boundaries. Checking a fixed-n
bootstrap interval after every block instead declares a difference under the
null in ~54% of runs (validation.md). s2 is the sample variance of block
differences, inflated by max(1.15, (n-1)/chi2_{0.2,n-1}) and floored at
(0.1%)^2. The chi-square factor fixes under-coverage at 3-5 blocks, where a
flat 1.15 let 9% of null runs exclude zero at theta = 0.

Stop rules (epsilon = theta/2 by default): CS in (theta, inf) is a
regression, CS in (-epsilon, epsilon) is equivalent, CS in (-inf, -theta) is
an improvement, and an exhausted budget is inconclusive. Afterwards, the
drift-bounded correction is applied to the A/B canary samples as a secondary
check, and a per-call-site drill-down (BH, labeled exploratory) is reported.

## `rglns bisect`: probabilistic bisection with an attribution prior

Hypothesis H_k is "c_k is the first bad commit" on the first-parent chain.
Each run gives y = log(time) - log(canary blend), and
y ~ N(m_G, s^2) before k and N(m_B, s^2) from k on. The means and s are
unknown. Plugging in point estimates made the posterior overconfident (73%
correct at a claimed 95%), so they are integrated out: flat priors on the
means and a Jeffreys prior on s give

    p_k  propto  pi_k * (n_G n_B)^(-1/2) * RSS_k^(-(n-2)/2)

computed for all k with prefix sums, O(N) per update.

* **Prior:** pi_k propto 1 + lambda a_k (lambda = 4), with a_k = 1 if commit k
  touches a file in the regressed call site's recorded stack. The call site is
  found by observed runs at the endpoints. These runs are never used as timing
  data, because observation start-up must not mix into the measurements. When
  the project being bisected is itself a library (pandas, xarray), its files in
  the worktree count as user code, so the stack reaches into it. On
  RegressBench the prior flagged the culprit in 5 of 8 cases, flagged only other
  commits in 2 (those bisects still converged on the right commit), and was
  uniform in 1, where the regression is in NumPy's C code and the stack holds
  only the benchmark file ([validation](validation.md#regressbench)).
* **Next commit:** the posterior median; b = clamp(ceil(build/run), 3, 20) runs
  per visit.
* **Stop:** max p_k >= 0.95; the 95% credible set is always printed.
* **Reality:** unbuildable commits are excluded and reported as a range;
  endpoint anchors are re-measured every 8 visits; a linear-drift model is
  compared by Bayes factor (> 10x reports "gradual regression, no single
  commit"); results are cached by SHA.

## `rglns watch`: change-point detection

Bayesian online change-point detection (Adams & MacKay) over the
canary-adjusted per-session series: a Normal-Inverse-Gamma prior set from a
warm-up window, a Student-t predictive, hazard 1/100, and run lengths
truncated at 500. It alerts when P(change within the last k sessions) =
P(r_t <= k) > 0.9, with k = 8. At k = 5 the posterior's uncertainty about the
exact onset session left 13% of 10% shifts unalerted. This is not novel
(MongoDB's E-divisive work is prior art); it is here because recurring
pipelines need it.
