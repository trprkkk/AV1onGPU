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
    src/frontend/cdf.cpp
    src/frontend/ivf.cpp
    src/frontend/obu.cpp
    src/frontend/headers.cpp
    src/frontend/range_decoder.cpp
    src/frontend/tiles.cpp
    src/frontend/tile_decoder.cpp
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
  foreach(tool av1gpu_info av1gpu_bench av1gpu_play av1gpu_dump av1gpu_probe av1gpu_decode)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE av1gpu)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  endforeach()
endif()

if(AV1GPU_BUILD_TESTS)
  enable_testing()
  foreach(test test_kernels test_pipeline test_frontend test_entropy test_gpu_sources)
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE av1gpu)
    set_target_properties(${test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_definitions(${test} PRIVATE
      AV1GPU_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
      AV1GPU_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ${test} COMMAND ${test})
  endforeach()

  # Bit exact streams: the whole front end plus the CPU backend against hashes
  # of the reference decoder's output (see tests/data/README.md). lw_t2 and
  # lw_t3 are the only fixtures that code local warped motion and OBMC,
  # palette_sc the only screen content one (palette coded blocks), intrabc the
  # only one that codes intra block copy, inter_intra the only one that codes
  # inter/intra compound, with the taper and with wedge masks, and seg_altq the
  # only one that codes segmentation (a different quantiser per segment); every
  # frame of all six is bit exact.
  if(TARGET av1gpu_decode)
    foreach(stream lw_t2 lw_t3 palette_sc intrabc inter_intra seg_altq)
      set(frames 30)
      if(stream STREQUAL "palette_sc")
        set(frames 10)
      elseif(stream STREQUAL "intrabc")
        set(frames 16)
      elseif(stream STREQUAL "inter_intra")
        set(frames 110)
      endif()
      add_test(NAME stream_${stream}
               COMMAND av1gpu_decode --backend cpu --frames ${frames}
                       ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/${stream}.ivf
                       --hash-check ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/${stream}.hash)
    endforeach()
  endif()

  # Compiles the real OpenCL source with a real OpenCL compiler and runs a few
  # jobs. Needs pyopencl + a runtime; exits with 77 when unavailable, which ctest
  # reports as a skip.
  #
  # Those two modules are often only present in a virtualenv, so the interpreter
  # can be pointed at one: -DAV1GPU_TEST_PYTHON=/path/to/venv/bin/python. Without
  # it the system interpreter is used, which usually has neither module, and the
  # tests report themselves as skipped instead of silently passing.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  set(AV1GPU_TEST_PYTHON "" CACHE FILEPATH
      "Python for the OpenCL tests; must have numpy and pyopencl")
  set(AV1GPU_TEST_PY "${Python3_EXECUTABLE}")
  if(AV1GPU_TEST_PYTHON)
    set(AV1GPU_TEST_PY "${AV1GPU_TEST_PYTHON}")
  endif()
  if(AV1GPU_TEST_PY)
    add_test(NAME opencl_selftest
             COMMAND ${AV1GPU_TEST_PY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_selftest.py)
    set_tests_properties(opencl_selftest PROPERTIES SKIP_RETURN_CODE 77)

    # Replays real decode plans through the OpenCL kernels and compares whole
    # frames against the CPU reference decode. It needs the dump tool, which is
    # only built with the tools, and pyopencl + a runtime.
    if(TARGET av1gpu_dump)
      add_test(NAME opencl_pipeline
               COMMAND ${AV1GPU_TEST_PY}
                       ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_pipeline.py
                       --dumper $<TARGET_FILE:av1gpu_dump>)
      set_tests_properties(opencl_pipeline PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()

    # The same replay for real streams, where the plans come from the decoder
    # itself, so the reference pool slot mapping is the real one:
    #   palette_sc      the only fixture that codes palette blocks, so every
    #                   palette prediction runs through the device kernels
    #   intrabc         the only one that codes intra block copy, whose job has
    #                   a negative reference slot and reads the frame being
    #                   decoded, so the device kernels' `cur` path is the one
    #                   under test
    #   lw_t2, lw_t3    the only ones that code local warped motion and OBMC,
    #                   whose overlap candidates carry their own reference slot
    #   inter_intra     the only one that codes inter/intra compound, so every
    #                   blend runs through the device kernels too
    #   seg_altq        the only one that codes segmentation, so the per segment
    #                   quantiser reaches the device kernels
    #   svt_av1_320x240 a different encoder, so a different mix of tools
    # All of it on a machine with no GPU at all.
    if(TARGET av1gpu_decode)
      foreach(stream palette_sc intrabc lw_t2 lw_t3 inter_intra seg_altq
                     svt_av1_320x240)
        set(frames 30)
        if(stream STREQUAL "palette_sc")
          set(frames 10)
        elseif(stream STREQUAL "intrabc")
          set(frames 16)
        elseif(stream STREQUAL "inter_intra")
          set(frames 110)
        elseif(stream STREQUAL "svt_av1_320x240")
          set(frames 6)
        endif()
        add_test(NAME opencl_${stream}
                 COMMAND ${AV1GPU_TEST_PY}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_pipeline.py
                         --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/${stream}.ivf
                         --frames ${frames} --decoder $<TARGET_FILE:av1gpu_decode>)
        set_tests_properties(opencl_${stream} PROPERTIES SKIP_RETURN_CODE 77
                             TIMEOUT 600)
      endforeach()
    endif()

    # Verification on a machine that really has a GPU: the OpenCL/CUDA backends
    # against the CPU reference and against a reference decoder's frame hashes.
    # Off by default because it needs a device; without one the script exits with
    # 77 and ctest reports a skip. See docs/GPU.md.
    set(AV1GPU_GPU_TESTS OFF CACHE BOOL "run tests/gpu_verify.py in ctest (needs a GPU)")
    if(AV1GPU_GPU_TESTS)
      add_test(NAME gpu_verify
               COMMAND ${AV1GPU_TEST_PY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu_verify.py
                       --build ${CMAKE_BINARY_DIR})
      set_tests_properties(gpu_verify PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)
    endif()
  endif()
endif()
