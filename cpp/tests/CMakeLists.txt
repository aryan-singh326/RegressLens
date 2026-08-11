add_executable(test_column test_column.cpp)
target_link_libraries(test_column PRIVATE regresslens_core)

add_executable(test_correctness_harness test_correctness_harness.cpp)
target_link_libraries(test_correctness_harness PRIVATE regresslens_core)

add_executable(test_reduction_scalar test_reduction_scalar.cpp)
target_link_libraries(test_reduction_scalar PRIVATE regresslens_core)

add_executable(test_reduction_avx2 test_reduction_avx2.cpp)
target_link_libraries(test_reduction_avx2 PRIVATE regresslens_core)
# AVX2 opt-in applied ONLY to this target, not globally, so scalar
# code stays portable per the project's kernel isolation policy.
target_compile_options(test_reduction_avx2 PRIVATE -mavx2)
