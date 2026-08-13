add_executable(bench_reduction bench_reduction.cpp)
target_link_libraries(bench_reduction PRIVATE regresslens_core benchmark::benchmark)

add_executable(bench_projection bench_projection.cpp)
target_link_libraries(bench_projection PRIVATE regresslens_core benchmark::benchmark)
target_compile_options(bench_projection PRIVATE -mavx2 -mfma)

add_executable(bench_filter bench_filter.cpp)
target_link_libraries(bench_filter PRIVATE regresslens_core benchmark::benchmark)
target_compile_options(bench_filter PRIVATE -mavx2)

add_executable(bench_rolling bench_rolling.cpp)
target_link_libraries(bench_rolling PRIVATE regresslens_core benchmark::benchmark)
target_compile_options(bench_rolling PRIVATE -mavx2)

add_executable(characterize characterize.cpp)
target_link_libraries(characterize PRIVATE regresslens_core Threads::Threads)
target_compile_options(characterize PRIVATE -mavx2 -mfma)
