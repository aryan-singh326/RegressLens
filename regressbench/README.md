# RegressBench

Real performance regressions from the scientific Python ecosystem, each with a
known culprit commit and a runnable reproducer, for evaluating
`rglns bisect` (or any regression-localization tool) **blind**: the tool gets
only a good commit, a bad commit and a benchmark command.

| case | project | slowdown | culprit (PR) | build per visit |
|---|---|---|---|---|
| [xarray-7376-groupby-map-multiindex](cases/xarray-7376-groupby-map-multiindex.json) | xarray | groupby(...).map on a MultiIndex, ~680x | `3ead17ea` ([#5692](https://github.com/pydata/xarray/pull/5692)) | none |
| [xarray-10683-interp-vectorized](cases/xarray-10683-interp-vectorized.json) | xarray | DataArray.interp, ~76x | `29fe679a` ([#9881](https://github.com/pydata/xarray/pull/9881)) | none |
| [xarray-11305-to-dataframe-multiindex](cases/xarray-11305-to-dataframe-multiindex.json) | xarray | to_dataframe / Coordinates.to_index, ~40x | `a13a2556` ([#10694](https://github.com/pydata/xarray/pull/10694)) | none |
| [pandas-57366-to-datetime-float-unit](cases/pandas-57366-to-datetime-float-unit.json) | pandas | to_datetime(float, unit="s"), 42-48x | `d9f70b39` ([#56037](https://github.com/pandas-dev/pandas/pull/56037)) | meson |
| [pandas-38038-groupby-rolling](cases/pandas-38038-groupby-rolling.json) | pandas | groupby().rolling().mean(), ~10x | `bad52a9d` ([#34052](https://github.com/pandas-dev/pandas/pull/34052)) | setup.py |
| [pandas-36757-groupby-fillna](cases/pandas-36757-groupby-fillna.json) | pandas | groupby().fillna(ffill), ~400x | `be6a3bc4` ([#30679](https://github.com/pandas-dev/pandas/pull/30679)) | setup.py |
| [numpy-26807-log-complex64](cases/numpy-26807-log-complex64.json) | NumPy | np.log on complex64, ~1.6x (Linux/GCC only) | `1cd92a14` ([#24085](https://github.com/numpy/numpy/pull/24085)) | meson |
| [scipy-5953-ckdtree-linf](cases/scipy-5953-ckdtree-linf.json) | SciPy | cKDTree.query(p=inf), effectively hangs | `58a10c26` in merge `ee8eba76` ([#4890](https://github.com/scipy/scipy/pull/4890)) | setup.py + Fortran (2015) |
| [sklearn-29229-columntransformer-njobs](cases/sklearn-29229-columntransformer-njobs.json) | scikit-learn | ColumnTransformer(n_jobs=2), quadratic | `51fca394` ([#28822](https://github.com/scikit-learn/scikit-learn/pull/28822)) | meson |

Every case was taken from a closed issue in which the culprit was identified
(by bisection or by the maintainers), and links that issue. Each culprit was
resolved to its full SHA in the project's history. The `good`/`bad` brackets
are commits on the first-parent chain, roughly 20 commits before and 15 after
the culprit, and before any fix. The exception is scikit-learn, whose `good` is
moved later (to commit 17 of 32 before the culprit) because the earlier commits
cannot be built with any current Meson (`good_note` in the case file). When a culprit landed inside a merged branch
(SciPy), the expected answer is the merge commit, because a first-parent
bisection can only name that. `python run.py --verify-only` re-checks all of
this against the live repositories.

Reproducers come from the issues. Where a case says so in its benchmark file,
it was scaled down to run in seconds, or changed to avoid an unrelated error
introduced by the same commit.

## Running

```
python regressbench/run.py --verify-only                # check every case definition
python regressbench/docker.py                           # every case, each in its own container
python regressbench/docker.py --cases xarray-10683-interp-vectorized
python regressbench/run.py --summary                    # table of results/
```

`docker.py` starts one fresh `python:<version>` container per case and
installs exactly the environment the case declares. `run.py` can also run a
case directly in an environment you prepared yourself.

Each case lists its environment (`environment` in the JSON): Python version,
pinned dependencies, system packages (`apt`, NumPy only) and, for compiled
projects, a `setup` build command that `rglns bisect` runs after every
checkout. Where a build needed a workaround for today's toolchains, the case
says why in `setup_note` or `environment.note`. Run each case in a matching
environment; `docker.py` does that for every case.

## Results

See [docs/validation.md](../docs/validation.md#regressbench) for the blind
results.
