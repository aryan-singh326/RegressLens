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

**Intervals.** Log-ratios of run-level means, BCa intervals with jackknife
acceleration ([Efron 1987](https://doi.org/10.1080/01621459.1987.10478410)).
For small samples the nominal level is first widened by Hesterberg's
expanded-percentile adjustment
([Hesterberg 2015](https://doi.org/10.1080/00031305.2015.1089789)). Plain BCa
under-covers at 10 runs (~90-94%).

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
the false discovery rate
([Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)).
With 200 call sites, 10 of them truly +10% slower,
BH finds 98.0% of the regressions and Bonferroni 89.7%, both with no false
discoveries ([validation](validation.md#wp3-measurement-core-rglns-check)).
The bootstrap needs enough replicates for this: a p-value from B replicates
cannot fall below 1/B, and Bonferroni's per-site level here is 0.05/200, so
with B = 600 it could never reject anything.

**Size normalization.** Per call site, the baseline's time-vs-size points are
fitted as t(n) = alpha + beta n, piecewise by cache regime (breakpoints are the
detected L1/L2/L3 sizes), with Theil-Sen per piece
([Sen 1968](https://doi.org/10.1080/01621459.1968.10480934)). Samples are divided by the
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
After every block, the interval mean(d) +/- h_n is a normal-mixture
confidence sequence ([Robbins 1970](https://doi.org/10.1214/aoms/1177696786);
[Howard et al. 2021](https://doi.org/10.1214/20-AOS1991)):

    h_n = sqrt( 2 s2 (s2 + n tau2) / (n^2 tau2) * ln( sqrt(1 + n tau2 / s2) / alpha ) )

with tau = theta (log scale). If the block differences are independent and
Gaussian (or sub-Gaussian) with a *known* variance s2, it holds at every n
simultaneously with probability at least 1 - alpha, so it can be checked after
every block and stop as soon as it excludes the decision boundaries.

The variance is not known. s2 is estimated from the blocks themselves, so the
guarantee is approximate, in the sense of asymptotic confidence sequences
([Waudby-Smith et al. 2024](https://doi.org/10.1214/24-AOS2408)). The first few
blocks are the weak point, so s2 is the sample variance of block differences
inflated by max(1.15, (n-1)/chi2_{0.2,n-1}) (its 80% upper confidence bound),
floored at (0.1%)^2, and no verdict is given before 4 blocks. The chi-square
factor fixes under-coverage at 3-5 blocks, where a flat 1.15 let 9% of null
runs exclude zero at theta = 0.

The evidence that the approximation holds is empirical
([validation](validation.md#wp4-rglns-compare)). With i.i.d. Gaussian block
differences monitored after every block up to 100 blocks, the sequence ever
excluded zero in 2.7% (sigma 2%) and 3.5% (sigma 5%) of null runs at
alpha = 0.05, and 200 real A/A comparisons gave no false verdict. Checking a
fixed-n bootstrap interval after every block instead declares a difference
under the null in ~54% of runs.

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

computed for all k with prefix sums, O(N) per update. Measuring next at the
posterior median is Horstein's probabilistic bisection
([Horstein 1963](https://doi.org/10.1109/TIT.1963.1057832);
[Waeber, Frazier & Henderson 2013](https://doi.org/10.1137/120861898)), which
assumes a known error rate. Here the noise level is unknown, so it is
integrated out instead.

* **Prior:** pi_k propto 1 + lambda a_k (lambda = 4), with a_k = 1 if commit k
  touches a file in the regressed call site's recorded stack. The call site is
  found by observed runs at the endpoints. These runs are never used as timing
  data, because observation start-up must not mix into the measurements. When
  the project being bisected is itself a library (pandas, xarray), its files in
  the worktree count as user code, so the stack reaches into it. Of the 9
  RegressBench cases, 8 ran (a 2015 SciPy cannot be built). In those 8 the
  prior flagged the culprit in 5 and flagged only other commits in 2, and
  those bisects still converged on the right commit. It was uniform in 1,
  where the regression is in NumPy's C code and the stack holds only the
  benchmark file ([validation](validation.md#regressbench)).
* **Next commit:** the posterior median; b = clamp(ceil(build/run), 3, 20) runs
  per visit.
* **Stop:** max p_k >= 0.95; the 95% credible set is always printed.
* **Reality:** unbuildable commits are excluded and reported as a range;
  endpoint anchors are re-measured every 8 visits; a linear-drift model is
  compared by Bayes factor (> 10x reports "gradual regression, no single
  commit"); results are cached by SHA.

## `rglns watch`: change-point detection

Bayesian online change-point detection
([Adams & MacKay 2007](https://arxiv.org/abs/0710.3742)) over the
canary-adjusted per-session series: a Normal-Inverse-Gamma prior set from a
warm-up window, a Student-t predictive, hazard 1/100, and run lengths
truncated at 500. It alerts when P(change within the last k sessions) =
P(r_t <= k) > 0.9, with k = 8. At k = 5 the posterior's uncertainty about the
exact onset session left 13% of 10% shifts unalerted. Change-point detection
on performance data is not novel: MongoDB runs E-divisive
([Matteson & James 2014](https://doi.org/10.1080/01621459.2013.849605)) on its
CI performance results
([Daly et al. 2020](https://doi.org/10.1145/3358960.3375791)). It is here
because recurring pipelines need it.

## Revision notes (2026-11-08)

* `rglns compare`: the confidence sequence was described as "valid at every n
  simultaneously". That holds only with a known variance. The section now
  says the guarantee is approximate because s2 is estimated, cites asymptotic
  confidence sequences, and gives the empirical null rates from validation.md,
  as the docstring of `stats/sequential.py` already did.
* `rglns bisect`: "5 of 8 cases" now says that 8 of the 9 RegressBench cases
  ran.
* `rglns watch`: the prior-art note now says what MongoDB does (E-divisive
  on its CI performance results) and cites it.
* Added citations for methods named without one: BCa, the expanded-percentile
  adjustment, Benjamini-Hochberg, Theil-Sen, the normal-mixture boundary,
  probabilistic bisection, BOCPD and E-divisive. All were checked against
  Crossref or arXiv (see also [novelty.md](novelty.md)).
