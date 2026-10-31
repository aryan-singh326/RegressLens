# Neither target gets architecture flags: both reach AVX2 kernels only
# through the runtime kernel tables (see characterize.cpp).
if(REGRESSLENS_BUILD_CHARACTERIZE)
  add_executable(characterize characterize.cpp)
  target_link_libraries(characterize PRIVATE regresslens_kernels)
endif()

if(REGRESSLENS_BUILD_BENCH)
  include(FetchContent)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.0
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(bench_kernels bench_kernels.cpp)
  target_link_libraries(bench_kernels PRIVATE regresslens_kernels benchmark::benchmark)
endif()
