# What is (and is not) new

The claim is the combination, not any single piece.

## Not novel

* **Bootstrap and multiple testing:** BCa intervals
  ([Efron 1987](https://doi.org/10.1080/01621459.1987.10478410)) with
  Hesterberg's expanded-percentile adjustment for small samples
  ([Hesterberg 2015](https://doi.org/10.1080/00031305.2015.1089789)), and
  Benjamini-Hochberg FDR control
  ([Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)).
* **Confidence sequences and anytime-valid A/B testing:** the normal-mixture
  boundary that `rglns compare` uses goes back to
  [Robbins (1970)](https://doi.org/10.1214/aoms/1177696786); time-uniform
  nonparametric confidence sequences
  ([Howard et al. 2021](https://doi.org/10.1214/20-AOS1991)). With an
  estimated variance, as in `rglns compare`, a confidence sequence is valid
  only asymptotically
  ([Waudby-Smith et al. 2024](https://doi.org/10.1214/24-AOS2408)). mSPRT and
  always-valid inference for A/B tests
  ([Johari et al. 2022](https://doi.org/10.1287/opre.2021.2135)), and their
  use in a production platform
  ([Maharaj et al. 2023](https://arxiv.org/abs/2302.10108)).
* **Anytime-valid benchmarking with an equivalence margin:**
  [seqbench](https://castlaboratory.github.io/seqbench/) (Leite, Ospina,
  Ferraz) is an R package (v0.1.0) that monitors a confidence sequence on the
  paired losses of two algorithms and decides superiority or practical
  equivalence against a pre-registered margin. Its companion article is
  submitted to The R Journal, so it is not yet peer reviewed. Its
  documentation does not mention wall-clock timing, interleaving or machine
  drift.
* **Interleaving to cancel machine drift:** airspeed velocity, the NumPy
  ecosystem's own benchmark runner, interleaves rounds across commits
  (`asv run` / `asv continuous --interleave-rounds`,
  [docs](https://asv.readthedocs.io/en/stable/commands.html)). Randomized
  multiple interleaved trials address noisy clouds
  ([Abedi & Brecht 2017](https://doi.org/10.1145/3030207.3030229)). Duet
  benchmarking runs both versions at once on a shared machine and compares
  them together ([Bulej et al. 2020](https://doi.org/10.1145/3358960.3379132)).
* **Change-point detection:** Bayesian online change-point detection
  ([Adams & MacKay 2007](https://arxiv.org/abs/0710.3742)), and change-point
  detection on CI performance data at MongoDB
  ([Daly et al. 2020](https://doi.org/10.1145/3358960.3375791)) with
  E-divisive ([Matteson & James 2014](https://doi.org/10.1080/01621459.2013.849605)).
* **Noisy and probabilistic bisection in the abstract:**
  [Horstein (1963)](https://doi.org/10.1109/TIT.1963.1057832),
  [Waeber, Frazier & Henderson (2013)](https://doi.org/10.1137/120861898),
  [Karp & Kleinberg (2007)](https://dblp.org/rec/conf/soda/KarpK07).
* **Performance bisection in practice:** Chromium's
  [Pinpoint](https://chromium.googlesource.com/chromium/src/+/HEAD/docs/speed/bisects.md),
  and `asv find`, an adaptive binary search that assumes the regression is
  mostly monotonic over the range. PRCA
  ([Heger, Happe & Farahbod 2013](https://doi.org/10.1145/2479871.2479879))
  runs git bisect with an ANOVA test at each step, then analyzes the call tree
  of the change it found.
  [Ocariza Jr. (2022)](https://doi.org/10.1007/s10664-022-10152-3) models the
  probability that a bisection localizes a performance regression. Checked
  against 300+ bug reports, bisection can be highly effective, but it is
  sensitive to the choice of baseline and to the measurement distribution at
  each commit.
* **Bisection with a non-uniform prior:**
  * Flake-aware culprit finding at Google
    ([Henderson et al. 2023](https://doi.org/10.1109/ICST57152.2023.00041)) is
    Bayesian bisection for flaky *test failures*. Its production prior weights
    a commit by how close, in the static build graph, the files it changed are
    to the failing test. This follows the suspiciousness metric of
    [Ziftci & Reardon (2017)](https://doi.org/10.1109/ICSE-SEIP.2017.13).
  * [BISCEPTER](https://arxiv.org/abs/2610.02995) (Gao, Wüst, Jiang, Su;
    arXiv preprint, Oct 2026) weights pivots by *historical*
    bug-inducing-commit latency. It reports 25.75% fewer iterations on average
    on GCC, the Linux kernel and MariaDB.
* **Linking runtime traces to commits:** PerfImpact
  ([Luo, Poshyvanyk & Grechanik 2016](https://doi.org/10.1145/2901739.2901765))
  ranks code changes by their estimated impact on execution traces. It does
  not bisect.

## Candidate novelty (combination; provisional)

1. **Interleaved, anytime-valid wall-clock comparison with drift-bounded
   correction** (`rglns compare`): ABBA/BAAB blocks, plus a normal-mixture
   confidence sequence on block differences that stops at a regression,
   equivalence or improvement decision, plus a secondary bound from paired
   compute and memory canaries. That bound brackets the effect between r/c and
   r/m without fitting a model. Interleaving (asv, RMIT, duet) and sequential
   decisions with an equivalence margin (seqbench) each exist. Their
   combination with canary drift bounds is what we have not found.
2. **Bisection whose prior comes from runtime attribution of a performance
   regression** (`rglns bisect`): the files on the regressed call site's
   recorded stack mark the commits that touch them, and the posterior
   integrates out the unknown before and after means of noisy timings.
   Code-relevance priors are not new: Google's flake-aware culprit finding
   derives one from static build-graph distance to a failing test. What
   differs is the source of the prior (runtime attribution of a slowdown) and
   the target (performance, not pass/fail). PRCA analyzes call trees only
   after its bisection, and PerfImpact maps traces to changes but does not
   bisect. BISCEPTER's latency prior is complementary and could be multiplied
   in. Evidence so far: 8 of the 9 RegressBench cases ran (the ninth, a 2015
   SciPy, cannot be built). In those 8 the prior flagged the culprit in 5,
   flagged only other commits in 2, and was uniform in 1, and all 8 bisects
   converged on the right commit. In simulation, a misleading prior cost about
   one run (69 vs 68) and 1.2 points of accuracy (98.0% vs 99.2% correct)
   ([validation.md](validation.md), WP5).
3. **RegressBench**: real NumPy-ecosystem performance regressions with known
   culprit commits, verified brackets and runnable reproducers
   (regressbench/). It has not yet been compared with existing datasets of
   performance bugs.

## Status of the search

Searched so far: the web and arXiv; the asv documentation; ICPE (PRCA, RMIT,
duet, MongoDB), MSR (PerfImpact), ICST (flake-aware culprit finding),
ICSE-SEIP (Ziftci & Reardon) and EMSE (Ocariza Jr.). Before claiming novelty
publicly, still search the ICSE and FSE main tracks, Mozilla Perfherder,
conbench, Bencher, and existing performance-bug datasets (for claim 3). Cite
flake-aware culprit finding, BISCEPTER, PRCA and asv/RMIT/duet as the closest
prior art.

## Citation review (2026-11-08)

Every reference was checked against its DOI record (Crossref), its arXiv
record or the project's own documentation. For the two papers closest to
claim 2 (flake-aware culprit finding, PRCA), the sections on the prior and
the bisection were read.

Corrections:

* **Ocariza Jr. (EMSE 2022)** was described as quantifying "how often naive
  bisection finds the right commit". It models the probability that bisection
  succeeds and studies what that depends on. The description now says so and
  names the author.
* **Flake-aware culprit finding** was cited only as noisy bisection. Its
  production prior comes from the files each commit changed, which is close to
  claim 2. Claim 2 is narrowed accordingly, and the paper is named as the
  closest prior art.
* **Interleaving** was presented as missing from the prior work. asv
  (`--interleave-rounds`), randomized multiple interleaved trials and duet
  benchmarking already interleave on noisy machines. Claim 1 now rests on the
  combination with the canary drift bounds.
* **Confidence-sequence validity:** the doc now notes that with an estimated
  variance, as `rglns compare` uses, validity is only asymptotic
  (Waudby-Smith et al. 2024).
* **RegressBench evidence:** "5 of 8 real cases" now says that 8 of the 9
  cases ran. "A misleading prior costs runs, not correctness" now gives the
  simulated numbers, since it costs a little of both.
* **seqbench:** it is now identified as an R package whose article is
  submitted, not yet peer reviewed. "Does not address" now refers to its
  documentation.
* **Pinpoint:** dropped "statistical", which its documentation does not state,
  and linked that documentation.

Replaced links: two ResearchGate pages became the DOI of Waeber, Frazier &
Henderson (SIAM J. Control Optim. 51(3), 2013) and the dblp record of Karp &
Kleinberg (SODA 2007, pp. 881-890).

Added primary sources for methods that were named without a citation: Efron
1987, Hesterberg 2015, Benjamini & Hochberg 1995, Robbins 1970, Howard et al.
2021, Johari et al. 2022, Adams & MacKay 2007, Daly et al. 2020, Matteson &
James 2014, Horstein 1963.

Added prior art that was missing: Waudby-Smith et al. 2024, Abedi & Brecht
2017, Bulej et al. 2020, asv's `--interleave-rounds` and `asv find`, Heger et
al. 2013, Ziftci & Reardon 2017, Luo et al. 2016.

Verified unchanged: BISCEPTER (authors, arXiv 2610.02995, 25.75% on GCC, the
Linux kernel and MariaDB) and Maharaj et al. 2023 (WWW '23 Companion).
