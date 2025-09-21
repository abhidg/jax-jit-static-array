cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(JAX_FFI_EXAMPLE_ENABLE_CUDA "Enable CUDA support" OFF)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

set(
  JAX_JIT_STATIC_CPU_PROJECTS
  "rms_norm"
  "matmul"
)

foreach(PROJECT ${JAX_JIT_STATIC_CPU_PROJECTS})
  nanobind_add_module("_${PROJECT}" NB_STATIC "src/jax_jit_static_array/${PROJECT}.cc")
  target_include_directories("_${PROJECT}" PUBLIC ${XLA_DIR})
  install(TARGETS "_${PROJECT}" LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endforeach()
