# What is (and is not) new

The claim is the combination, not any single piece.

## Not novel

* Bootstrap and BCa intervals, Benjamini-Hochberg.
* Mixture-martingale / mSPRT confidence sequences and anytime-valid A/B
  testing ([Anytime-Valid Confidence Sequences in an Enterprise A/B Testing
  Platform](https://arxiv.org/abs/2302.10108)).
* Anytime-valid *benchmarking* with an equivalence margin:
  [seqbench](https://castlaboratory.github.io/seqbench/) (Leite, Ospina, Ferraz)
  monitors confidence sequences on paired losses of two algorithms with
  superiority/equivalence decisions. It does not address wall-clock timing,
  interleaving or machine drift.
* Bayesian online change-point detection (Adams & MacKay) and change-point
  detection on performance data (MongoDB's E-divisive work).
* Probabilistic / noisy bisection in the abstract (Horstein; [Bisection Search
  with Noisy Responses](https://www.researchgate.net/publication/266715465_Bisection_Search_with_Noisy_Responses);
  [Noisy binary search and its applications](https://www.researchgate.net/publication/220779539_Noisy_binary_search_and_its_applications),
  Google's flake-aware culprit finding), and statistical performance bisection
  inside large organizations (Chromium's Pinpoint).
* Bisection with a non-uniform prior: [BISCEPTER](https://arxiv.org/abs/2610.02995)
  (Gao, Wüst, Jiang, Su, Oct 2026) weights pivots by *historical
  bug-inducing-commit latency* and reports 25.75% fewer iterations on GCC,
  Linux and MariaDB.
* The empirical study [On the Effectiveness of Bisection in Performance
  Regression Localization](https://link.springer.com/article/10.1007/s10664-022-10152-3)
  (EMSE 2022) quantifies how often naive bisection finds the right commit for
  performance regressions.

## Candidate novelty (combination; provisional)

1. **Interleaved, anytime-valid wall-clock comparison with drift-bounded
   correction**: ABBA/BAAB blocks plus a confidence sequence on block
   differences, with a secondary bound from paired compute and memory canaries
   that brackets the effect between r/c and r/m without fitting a model
   (`rglns compare`). seqbench and the A/B-testing literature supply the
   sequential statistics, but not the interleaving or the drift bounds for
   noisy machines.
2. **Bisection whose prior comes from runtime call-site attribution**: the
   regressed call site's recorded stack files mark the commits that touch them
   (`rglns bisect`). BISCEPTER uses a prior from *historical* commit-latency
   statistics; the two are complementary and could be multiplied. Evidence so
   far: on RegressBench it flagged the true culprit in 5 of 8 real cases, and
   the 3 others still converged correctly; a misleading prior costs runs, not
   correctness (simulated in validation.md, WP5).
3. **RegressBench**: real NumPy-ecosystem performance regressions with known
   culprit commits, verified brackets and runnable reproducers
   (regressbench/).

## Status of the search

A first search of the web and arXiv found the related work above. Before
claiming novelty publicly, still search ICPE, MSR and ICSE/FSE proceedings,
Mozilla Perfherder, airspeed velocity (`asv find`), conbench and Bencher, and
cite BISCEPTER and seqbench as the closest prior art.
