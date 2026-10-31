find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Development builds outside pip: ask the active interpreter where
  # its pybind11 CMake config lives.
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE _rl_pybind11_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED HINTS "${_rl_pybind11_dir}")
endif()

# No architecture flags: this module must import on any CPU.
pybind11_add_module(_native bindings.cpp monitor.cpp)
target_link_libraries(_native PRIVATE regresslens_kernels)

# The wheel installs the module next to regresslens_accel/__init__.py.
install(TARGETS _native LIBRARY DESTINATION regresslens_accel)
