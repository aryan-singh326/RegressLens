# regresslens_kernels: every kernel set plus runtime dispatch and the
# kernel selector. Only kernels_avx2.cpp gets AVX2 flags.
set(_rl_sources
    kernels_scalar.cpp
    cpu_features.cpp
    dispatch.cpp
    kernel_selector.cpp
)
if(REGRESSLENS_HAVE_AVX2_TU)
  list(APPEND _rl_sources kernels_avx2.cpp)
  set_source_files_properties(kernels_avx2.cpp PROPERTIES
      COMPILE_OPTIONS "${REGRESSLENS_AVX2_FLAGS}")
endif()

add_library(regresslens_kernels STATIC ${_rl_sources})
target_include_directories(regresslens_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(regresslens_kernels PUBLIC
    REGRESSLENS_HAVE_AVX2_TU=$<BOOL:${REGRESSLENS_HAVE_AVX2_TU}>)
target_link_libraries(regresslens_kernels PUBLIC Threads::Threads)
set_target_properties(regresslens_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
