cmake_minimum_required(VERSION 3.20)
project(regresslens CXX)

# One CMake project serves two builds:
#   * the regresslens-accel wheel (scikit-build-core sets SKBUILD; see
#     pyproject.toml), which builds only the Python extension;
#   * a development build with the C++ tests and characterization
#     tools.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(SKBUILD)
  set(_rl_dev_default OFF)
else()
  set(_rl_dev_default ON)
endif()
option(REGRESSLENS_BUILD_PYTHON "Build the regresslens_accel._native extension" ON)
option(REGRESSLENS_BUILD_TESTS "Build the C++ test executables" ${_rl_dev_default})
option(REGRESSLENS_BUILD_CHARACTERIZE "Build the characterize driver" ${_rl_dev_default})
option(REGRESSLENS_BUILD_BENCH "Build Google Benchmark targets (fetches gbench)" OFF)
option(REGRESSLENS_DISABLE_AVX2 "Build without the AVX2 kernel set" OFF)

# The AVX2 translation unit exists only on x86-64. Everywhere else
# (aarch64 Linux, Apple silicon) the extension is scalar-only.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _rl_proc)
if(CMAKE_OSX_ARCHITECTURES)
  string(TOLOWER "${CMAKE_OSX_ARCHITECTURES}" _rl_proc)
endif()
if(NOT REGRESSLENS_DISABLE_AVX2 AND _rl_proc MATCHES "^(x86_64|amd64|x64)$")
  set(REGRESSLENS_HAVE_AVX2_TU ON)
else()
  set(REGRESSLENS_HAVE_AVX2_TU OFF)
endif()
message(STATUS "regresslens: processor='${_rl_proc}' AVX2 kernels=${REGRESSLENS_HAVE_AVX2_TU}")

# Flags for code that may use AVX2. Applied per source file / per
# test target, never project-wide (see src/kernels_avx2.cpp).
if(MSVC)
  set(REGRESSLENS_AVX2_FLAGS /arch:AVX2)
else()
  set(REGRESSLENS_AVX2_FLAGS -mavx2 -mfma)
endif()

# Warnings plus strict IEEE semantics; optimization comes from the build
# type (Release: -O3 / /O2). -ffp-contract=off keeps the compiler from
# fusing a*b+c on its own (it would on aarch64, where FMA is baseline),
# so scalar kernels round exactly like NumPy. Never add -ffast-math:
# compensated summation depends on exact rounding.
if(MSVC)
  add_compile_options(/W3 /fp:precise)
else()
  add_compile_options(-Wall -Wextra -ffp-contract=off)
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)

if(REGRESSLENS_BUILD_PYTHON)
  add_subdirectory(python_bindings)
endif()

if(REGRESSLENS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(REGRESSLENS_BUILD_CHARACTERIZE OR REGRESSLENS_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(NOT SKBUILD)
  # Toolchain sanity check: reports what this machine supports.
  add_executable(cpu_check src/cpu_check.cpp)
  target_link_libraries(cpu_check PRIVATE regresslens_kernels)
endif()
