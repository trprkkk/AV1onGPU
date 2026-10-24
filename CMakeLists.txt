cmake_minimum_required(VERSION 3.16)
project(av1gpu VERSION 0.1.0 LANGUAGES CXX)

option(AV1GPU_BUILD_TESTS "Build the unit tests" ON)
option(AV1GPU_BUILD_TOOLS "Build av1gpu_info / av1gpu_bench / av1gpu_play" ON)
option(AV1GPU_ENABLE_OPENCL "Build the OpenCL backend" ON)
option(AV1GPU_ENABLE_CUDA "Build the CUDA backend" ON)
option(AV1GPU_ENABLE_OPENMP "Use OpenMP in the CPU reference backend" ON)
option(AV1GPU_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

if(AV1GPU_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

if(AV1GPU_ENABLE_OPENMP)
  find_package(OpenMP QUIET COMPONENTS CXX)
endif()

# ---------------------------------------------------------------------------
# kernel embedding: the shared header is inlined into the device sources, which
# are then compiled into the library (no runtime file lookups).
# ---------------------------------------------------------------------------
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})
set(KERNEL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/kernels/av1gpu_kernels.h)

if(AV1GPU_ENABLE_OPENCL)
  set(OPENCL_SOURCE ${GENERATED_DIR}/opencl_source.cpp)
  add_custom_command(
    OUTPUT ${OPENCL_SOURCE}
    COMMAND ${CMAKE_COMMAND}
            -DHEADER=${KERNEL_HEADER}
            -DBODY=${CMAKE_CURRENT_SOURCE_DIR}/kernels/av1_opencl.cl
            -DLANGUAGE=opencl
            -DOUTPUT=${OPENCL_SOURCE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${KERNEL_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/av1_opencl.cl
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    COMMENT "Embedding OpenCL kernels")
endif()

if(AV1GPU_ENABLE_CUDA)
  set(CUDA_SOURCE ${GENERATED_DIR}/cuda_source.cpp)
  add_custom_command(
    OUTPUT ${CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND}
            -DHEADER=${KERNEL_HEADER}
            -DBODY=${CMAKE_CURRENT_SOURCE_DIR}/kernels/av1_cuda.cu
            -DLANGUAGE=cuda
            -DOUTPUT=${CUDA_SOURCE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${KERNEL_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/av1_cuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    COMMENT "Embedding CUDA kernels")
endif()

# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------
set(AV1GPU_SOURCES
    src/core/tables.cpp
    src/core/engine.cpp
    src/core/factory.cpp
    src/core/synthetic.cpp
    src/core/post_filters.cpp
    src/core/kernel_source.cpp
    src/backends/cpu/cpu_backend.cpp
)

if(AV1GPU_ENABLE_OPENCL)
  list(APPEND AV1GPU_SOURCES src/backends/opencl/opencl_backend.cpp ${OPENCL_SOURCE})
else()
  list(APPEND AV1GPU_SOURCES src/core/no_opencl.cpp)
endif()

if(AV1GPU_ENABLE_CUDA)
  list(APPEND AV1GPU_SOURCES src/backends/cuda/cuda_backend.cpp ${CUDA_SOURCE})
else()
  list(APPEND AV1GPU_SOURCES src/core/no_cuda.cpp)
endif()

add_library(av1gpu STATIC ${AV1GPU_SOURCES})

target_include_directories(av1gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

target_compile_definitions(av1gpu PUBLIC
    $<$<BOOL:${AV1GPU_ENABLE_OPENCL}>:AV1GPU_HAVE_OPENCL=1>
    $<$<BOOL:${AV1GPU_ENABLE_CUDA}>:AV1GPU_HAVE_CUDA=1>)

if(TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(av1gpu PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(av1gpu PUBLIC AV1GPU_HAVE_OPENMP=1)
endif()

if(UNIX)
  target_link_libraries(av1gpu PUBLIC ${CMAKE_DL_LIBS})
endif()

# ---------------------------------------------------------------------------
# tools and tests
# ---------------------------------------------------------------------------
if(AV1GPU_BUILD_TOOLS)
  foreach(tool av1gpu_info av1gpu_bench av1gpu_play)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE av1gpu)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  endforeach()
endif()

if(AV1GPU_BUILD_TESTS)
  enable_testing()
  foreach(test test_kernels test_pipeline)
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE av1gpu)
    set_target_properties(${test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ${test} COMMAND ${test})
  endforeach()

  # Compiles the real OpenCL source with a real OpenCL compiler and runs a few
  # jobs. Needs pyopencl + a runtime; exits with 77 when unavailable, which ctest
  # reports as a skip.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME opencl_selftest
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_selftest.py)
    set_tests_properties(opencl_selftest PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
