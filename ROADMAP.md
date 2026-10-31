# RegressLens Roadmap

## v0.1 (released as 0.1.0, August 2026)

Scope, locked: float32/float64, contiguous arrays only, x86-64 Linux, four operations
(projection, filter, sum/mean reduction, rolling sum/mean). Scalar + AVX2 + MT-AVX2
kernels, a hand-tuned (not learned) kernel selector, SQLite trace persistence, bootstrap
CI regression detection, contiguity-loss attribution, cost-aware remediation
suggestions, and the `rglns baseline/check/profile` CLI.

## v0.2 (planned during v0.1; status in 0.2.0 below each item)

Items below are ordered roughly by how directly they were motivated by real findings
during v0.1's own validation, not by difficulty.

### A fifth kernel: shifted-subtraction / diff-style operations
Real, reproducible finding from the Phase 4 validation pipeline: `diff_log`-style
operations (`np.diff(np.log(x))`) were consistently the single largest unaccelerated
cost in a realistic pipeline — on two different machines. The subtraction half of this
pattern (elementwise, reading two overlapping offset views) is structurally close to the
existing `project_affine` kernel; the log transform is a separate question. Worth
scoping as its own characterization pass before committing to an implementation.

**0.2.0: measured, not built.** `tools/kernel_gate.py` finds `np.diff` at 30-33% of `diff_log` on Windows and ~11% on Linux, but most of that is page-faulting the freshly allocated output, which a native kernel would pay too; the part a kernel could speed up is 8-9% ([docs/kernels.md](docs/kernels.md)).

### NaN-safe operations
v0.1 requires explicit validation at array-wrap time and raises on NaN presence. Real
financial and scientific data frequently has missing values. This needs its own
characterization pass — NaN-aware reductions and filters have different vectorization
characteristics than the current NaN-free kernels, not just an added branch.

**0.2.0: done.** Every operation follows NumPy's NaN/Inf semantics; rolling windows use a NaN-segmented algorithm; property tests inject NaN/Inf into every operation ([docs/accuracy.md](docs/accuracy.md)).

### Rolling variance / standard deviation
The validation pipeline's `zscore` step needed `std()`, which v0.1 has no accelerated
kernel for — it fell back to NumPy every time, a real and measured cost. A rolling
variance kernel is the natural companion to the existing rolling sum/mean kernel and
would close this gap directly.

**0.2.0: done.** `rl.array(x).rolling_var(window, ddof)`, correctly rounded on the `rglns verify` corpus ([docs/accuracy.md](docs/accuracy.md)).

### Freeze/export execution policy
Adaptive systems that learn from runtime history make execution depend on prior runs —
a reproducibility concern for anyone producing paper results from a RegressLens-
accelerated pipeline. Export the current kernel selection policy to a static
configuration file that runs deterministically without trace-history dependency. Users
get adaptation during development, frozen determinism for final experiments. This
becomes more directly relevant once the deferred adaptive-selector research (see below)
actually produces a learned policy — right now the "policy" is already static, so
freeze/export is lower urgency than it will be later.

**0.2.0: not done.** The selector is still a static policy, now derived from characterization data ([docs/kernels.md](docs/kernels.md)), and `RGLNS_FORCE_KERNEL` pins a kernel set; exporting a policy file matters once a learned selector exists.

### Selectivity feedback loop for filter — extend beyond current-shape matching
v0.1's selectivity feedback (added post-launch) averages historical selectivity across
all past runs matching the same `(dtype, row_count)` shape. This is a reasonable
approximation but conflates genuinely different call sites that happen to share a shape.
Real per-call-site historical tracking (using the call-site attribution already
captured in every trace) would be more precise.

**0.2.0: done.** Selectivity history is kept per call site; within a process, observed selectivity takes over once a site has been seen twice.

### Real call-site-based trace grouping, beyond `(operator, dtype, row_count)`
Both `rglns check`'s comparison and the validation pipeline's reporting currently group
traces by `(operator, dtype, row_count)` as an approximation of "same call site." Two
genuinely different call sites that happen to produce the same operator/dtype/size will
be conflated. Call-site data is already captured (see `trace.capture_call_site()`) but
not yet used as the primary grouping key — doing so properly means deciding how to
handle the case where the SAME call site produces different array sizes across runs
(common — e.g. filter's output size is inherently variable), which needs real design
work, not just a query change.

**0.2.0: done.** Calls are grouped by call site (file relative to the repository root, qualified name, ordinal), and a per-site cost model normalizes for varying sizes ([docs/data-model.md](docs/data-model.md), [docs/statistics.md](docs/statistics.md)).

## Explicitly deferred, not abandoned

### The adaptive kernel selector research question
Originally scoped as "Phase 2": does an online adaptive selector beat the hand-tuned
heuristic once all its own overhead (interception, exploration, persistence) is
honestly counted? This was deliberately decoupled from the v0.1 product release — the
kernel selector sits behind a single swappable interface
(`cpp/include/regresslens/kernel_selector.hpp`) specifically so this research can
produce a different implementation later without requiring changes to anything
downstream (the Python integration layer, trace persistence, or `rglns` itself already
depend only on that interface, not on how a decision is made behind it).

**0.2.0: still deferred.**

### True interleaved baseline/current measurement
The project's original methodology called for interleaving baseline and current
execution within a single measurement session, to cancel out thermal throttling and
background-activity drift. This is not implementable against a STORED baseline from a
prior session without also storing and being able to re-execute the baseline's exact
code — which needs git integration. Documented as a known limitation in
`regresslens/stats.py` rather than silently dropped; worth revisiting once/if git
integration is in scope.

**0.2.0: done** as `rglns compare`: both revisions run interleaved (ABBA/BAAB) from git worktrees, with an anytime-valid confidence sequence. `regresslens/stats.py` no longer exists.

### CI integration, team dashboards, fleet comparison
Per the project's original business-model framing: these are enterprise-tier features,
built only after the open-source release demonstrates real demand for them — not
speculatively ahead of that signal.

**0.2.0:** CI integration done (the GitHub Action in `action.yml`); dashboards and fleet comparison still deferred.
