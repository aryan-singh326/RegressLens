# Portable tests: no architecture flags.
function(regresslens_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE regresslens_kernels)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Tests that include the AVX2 kernel headers directly. They are
# compiled with AVX2 flags and exit with code 77 (reported as skipped)
# when the CPU lacks AVX2. Kernel-set parity on any CPU is covered by
# test_dispatch, which goes through the runtime dispatcher instead.
function(regresslens_avx2_test name)
  if(NOT REGRESSLENS_HAVE_AVX2_TU)
    return()
  endif()
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE regresslens_kernels)
  target_compile_options(${name} PRIVATE ${REGRESSLENS_AVX2_FLAGS})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

regresslens_test(test_column)
regresslens_test(test_correctness_harness)
regresslens_test(test_reduction_scalar)
regresslens_test(test_projection_scalar)
regresslens_test(test_filter_scalar)
regresslens_test(test_rolling_scalar)
regresslens_test(test_rolling_var)
regresslens_test(test_nonfinite)
regresslens_test(test_kernel_selector)
regresslens_test(test_dispatch)

regresslens_avx2_test(test_reduction_avx2)
regresslens_avx2_test(test_reduction_mt_avx2)
regresslens_avx2_test(test_projection_avx2)
regresslens_avx2_test(test_filter_avx2)
regresslens_avx2_test(test_filter_mt_avx2)
regresslens_avx2_test(test_rolling_avx2)
