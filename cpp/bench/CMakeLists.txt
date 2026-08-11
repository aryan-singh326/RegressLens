add_executable(bench_reduction bench_reduction.cpp)
target_link_libraries(bench_reduction PRIVATE regresslens_core benchmark::benchmark)

add_executable(bench_projection bench_projection.cpp)
target_link_libraries(bench_projection PRIVATE regresslens_core benchmark::benchmark)
target_compile_options(bench_projection PRIVATE -mavx2 -mfma)
