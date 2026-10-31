# regresslens-accel

Optional native kernels for [RegressLens](../README.md). Install them with

```
pip install "regresslens[accel]"
```

This package contains one extension module, `regresslens_accel._native`, built from
the C++ sources in this directory. It holds two kernel sets: a portable scalar set,
compiled with no architecture flags, and an AVX2+FMA set, which is compiled only on
x86-64. The extension chooses between them at import time from runtime CPU
detection, so the same wheel is safe on any CPU of its architecture. Set
`RGLNS_FORCE_KERNEL=scalar|avx2` to override the choice for testing.

Development build with the C++ tests:

```
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```
