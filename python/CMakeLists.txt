# Minimal CMake config used ONLY by `pip install .` (via
# scikit-build-core, see pyproject.toml). This is deliberately
# separate from cpp/CMakeLists.txt, which additionally builds the
# full test suite, benchmarks, and Google Benchmark via FetchContent
# — none of which a normal user installing RegressLens should have
# to wait for or need network access to fetch. This file builds
# exactly one thing: the _regresslens_native extension module, from
# the same source files cpp/CMakeLists.txt uses, so there is only one
# place kernel/selector logic actually lives.
cmake_minimum_required(VERSION 3.20)
project(regresslens_native_wheel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wextra)

find_package(Threads REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(REGRESSLENS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)

pybind11_add_module(_regresslens_native
    ${REGRESSLENS_CPP_ROOT}/python_bindings/bindings.cpp
    ${REGRESSLENS_CPP_ROOT}/src/kernel_selector.cpp
)
target_include_directories(_regresslens_native PRIVATE
    ${REGRESSLENS_CPP_ROOT}/include
)
target_link_libraries(_regresslens_native PRIVATE Threads::Threads)
target_compile_options(_regresslens_native PRIVATE -mavx2 -mfma)

# scikit-build-core maps this install destination into the wheel
# alongside the pure-Python package — DESTINATION regresslens means
# the .so lands in the same directory as array.py, cli.py, etc.,
# exactly where regresslens/__init__.py expects to find it via
# `from . import _regresslens_native`.
install(TARGETS _regresslens_native LIBRARY DESTINATION regresslens)
