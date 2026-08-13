cmake_minimum_required(VERSION 3.20)
project(regresslens CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Scalar code must NOT get -march=native — it's the portable baseline.
# AVX2 code opts in per-target below, not globally.
add_compile_options(-O3 -Wall -Wextra)

find_package(Threads REQUIRED)

# Let CMake find pybind11's config even though it was installed via pip.
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.0
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(python_bindings)

# Step 2 sanity-check binary: confirms toolchain + AVX2 availability.
add_executable(cpu_check src/cpu_check.cpp)
