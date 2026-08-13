# regresslens_headers: pure header-only kernel/struct interfaces.
add_library(regresslens_headers INTERFACE)
target_include_directories(regresslens_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

# regresslens_core: adds compiled sources (currently just the
# selector). Real library, not interface-only, now that there's a
# .cpp file involved.
add_library(regresslens_core STATIC
    kernel_selector.cpp
)
target_link_libraries(regresslens_core PUBLIC regresslens_headers)
