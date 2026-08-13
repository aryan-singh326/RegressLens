find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_regresslens_native bindings.cpp)
target_link_libraries(_regresslens_native PRIVATE regresslens_core Threads::Threads)
target_compile_options(_regresslens_native PRIVATE -mavx2 -mfma)

# Drop the built extension module directly into the Python package
# directory so `import regresslens` finds it without a separate
# install step during development.
set_target_properties(_regresslens_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../python/regresslens
)
