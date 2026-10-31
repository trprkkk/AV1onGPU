cmake_minimum_required(VERSION 3.16)
project(av1gpu VERSION 0.1.0 LANGUAGES CXX)

option(AV1GPU_BUILD_TESTS "Build the unit tests" ON)
option(AV1GPU_BUILD_TOOLS "Build av1gpu_info / av1gpu_bench / av1gpu_play" ON)
option(AV1GPU_ENABLE_OPENCL "Build the OpenCL backend" ON)
option(AV1GPU_ENABLE_CUDA "Build the CUDA backend" ON)
option(AV1GPU_ENABLE_OPENMP "Use OpenMP in the CPU reference backend" ON)
# The DirectShow AV1 decoder filter (dsfilter/): Windows and MSVC only, since
# it is a COM in process server built against dshow.h.
if(WIN32 AND MSVC)
  set(AV1GPU_DSFILTER_DEFAULT ON)
else()
  set(AV1GPU_DSFILTER_DEFAULT OFF)
endif()
option(AV1GPU_BUILD_DSFILTER "Build the DirectShow AV1 decoder filter"
       ${AV1GPU_DSFILTER_DEFAULT})
# ASan + UBSan, with UBSan configured to halt (see the block below). The
# sweep it is meant for - every fixture decoded, 38/38 tests, no findings -
# is written down in docs/ACCURACY.md.
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
  # UBSan recovers by default: it prints a diagnostic and carries on, so a ctest
  # run stays green while stderr fills with undefined behaviour. Halting turns
  # every finding into a failed test, which is the only useful setting for CI.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                      -fno-omit-frame-pointer -g)
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
    src/core/grain.cpp
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
# the DirectShow filter itself (see dsfilter/README.md for registering and for
# wiring it into MPC-BE / MPC-HC / PotPlayer / graph editors)
# ---------------------------------------------------------------------------
if(AV1GPU_BUILD_DSFILTER)
  if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "AV1GPU_BUILD_DSFILTER needs Windows and MSVC")
  endif()
  add_library(av1gpu_dshow SHARED
    dsfilter/decode_core.cpp
    dsfilter/av1gpu_dshow.cpp
    dsfilter/av1gpu_dshow.def)
  target_link_libraries(av1gpu_dshow PRIVATE av1gpu ole32 strmiids uuid advapi32)
  target_compile_definitions(av1gpu_dshow PRIVATE _CRT_SECURE_NO_WARNINGS)
  set_target_properties(av1gpu_dshow PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  target_link_options(av1gpu_dshow PRIVATE "/MAP:${CMAKE_BINARY_DIR}/av1gpu_dshow.map")
endif()

# ---------------------------------------------------------------------------
# tools and tests
# ---------------------------------------------------------------------------
if(AV1GPU_BUILD_TOOLS)
  foreach(tool av1gpu_info av1gpu_bench av1gpu_play av1gpu_dump av1gpu_probe av1gpu_decode av1gpu_quartz)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE av1gpu)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  endforeach()
  # the quartz graph-build reproduction harness touches real DirectShow
  # the daily car chase clips montage: real coupling here, but the batch spirit
  if(TARGET av1gpu_quartz)
    target_link_libraries(av1gpu_quartz PRIVATE ole32 strmiids uuid)
  endif()

  # the filter's pipeline through the real decode core (no DirectShow)
  add_executable(av1gpu_pipebench tools/av1gpu_pipebench.cpp dsfilter/decode_core.cpp)
  target_link_libraries(av1gpu_pipebench PRIVATE av1gpu)
  target_include_directories(av1gpu_pipebench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(av1gpu_pipebench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
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

  # The DirectShow decode core without DirectShow: the same NV12 bytes as the
  # canonical decode walk, incremental OBU parsing, and (when a GPU is present)
  # bit exact GPU output. Runs on the CPU everywhere; the GPU parity check
  # only runs when a device is there.
  if(AV1GPU_BUILD_DSFILTER)
    add_executable(test_ds_core tests/test_ds_core.cpp dsfilter/decode_core.cpp)
    target_link_libraries(test_ds_core PRIVATE av1gpu)
    set_target_properties(test_ds_core PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(test_ds_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(test_ds_core PRIVATE
      AV1GPU_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
    add_test(NAME test_ds_core COMMAND test_ds_core)
    set_tests_properties(test_ds_core PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)

    # The real COM surface of the filter, exercised through DllGetClassObject:
    # factory -> filter -> media type agreement -> allocator handshake ->
    # samples pushed through a fake source into Receive, frames counted at a
    # minimal sink. The DLL is loaded with LoadLibrary, the way a player does.
    add_executable(av1gpu_dsgraph tools/av1gpu_dsgraph.cpp)
    target_link_libraries(av1gpu_dsgraph PRIVATE av1gpu ole32 uuid strmiids)
    set_target_properties(av1gpu_dsgraph PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_definitions(av1gpu_dsgraph PRIVATE
      AV1GPU_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
    if(TARGET av1gpu_dshow)
      # Fixtures whose every frame is shown, with the frame counts the
      # stream_* tests above rely on.
      foreach(stream lw_t2:30 lr:8 tiles:4 grain:30)
        string(REGEX REPLACE ":.*" "" name ${stream})
        string(REGEX REPLACE ".*:" "" frames ${stream})
        add_test(NAME dsgraph_${name}
                 COMMAND av1gpu_dsgraph --dll $<TARGET_FILE:av1gpu_dshow> --stream
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/${name}.ivf
                         --expect ${frames})
        set_tests_properties(dsgraph_${name} PROPERTIES TIMEOUT 300)
      endforeach()
    endif()
  endif()

  # Bit exact streams: the whole front end plus the CPU backend against hashes
  # of the reference decoder's output (see tests/data/README.md). lw_t2 and
  # lw_t3 are the only fixtures that code local warped motion and OBMC,
  # palette_sc the only screen content one (palette coded blocks), intrabc the
  # only one that codes intra block copy, inter_intra the only one that codes
  # inter/intra compound, with the taper and with wedge masks, seg_altq the only
  # one that codes segmentation (a different quantiser per segment), superres
  # the only one that codes a picture narrower than it displays (so the
  # upscaling process and reference scaling both run), grain the only one
  # that carries film grain parameters (so the generator, the AR filter and the
  # stripe blending run), grain_inherit the only one where the grain
  # parameters are re-used from a reference (update_grain = 0), lr the only one
  # that codes loop restoration (the Wiener and the self guided filter, on the
  # luma and on the chroma planes, in frames that code the type per unit and in
  # frames that code it switchable), lr_10bit the same clip at 10 bit (where the
  # restoration filters' bit depth scaling is not a no-op), and the three aom_
  # ones the only ones that are not 8 bit 4:2:0: aom_10bit_320x240 is 10 bit,
  # aom_444_10bit_320x240 is 4:4:4 and aom_12bit_320x240 is 12 bit, and lossless
  # the only one whose frames code no quantiser at all (so the frame header
  # carries no loop filter levels, no CDEF strengths and no restoration unit, and
  # the decoder must not filter the picture), odd_size the only one whose
  # picture is not a multiple of eight in either direction (so the 4x4 grid the
  # tiles code on is the frame size rounded up to eight, and the filters and the
  # prediction reach into the border), and edge_grid the only one whose planes
  # end *inside* the four sample grid the loop filter walks (66 luma rows and 33
  # chroma rows: the filter covers 68 and 36, which is neither the picture size
  # nor the eight sample grid, and CDEF and restoration read the difference),
  # and tiles / tiles_grid the only ones cut into more than one tile per frame
  # (2x2 on 256x160, and 4x2 on a 320x188 frame whose tile widths are 64 + 64 +
  # 64 + 128 and whose tile rows are 64 + 124 samples, so the last row ends at
  # the coded extent, two samples past the picture), palette_10bit the only one
  # that codes a palette in a picture deeper than eight bits (so a colour does
  # not fit in a byte: the fixture was recorded because a uint8_t colour array
  # turned 880 into 112 while the range decoder stayed in step), and
  # error_resilient the only one that codes error_resilient_mode on inter frames
  # (primary_ref_frame = PRIMARY_REF_NONE, frame ids with a delta per reference,
  # and the seven reference names interleaved with them).
  # Every frame of all twenty four is bit exact.
  if(TARGET av1gpu_decode)
    foreach(stream lw_t2 lw_t3 palette_sc intrabc inter_intra seg_altq superres grain
                 grain_inherit lr lr_10bit lr_12bit lr_444 lr_422 lossless
                 odd_size edge_grid tiles tiles_grid aom_10bit_320x240
                 aom_444_10bit_320x240 aom_12bit_320x240 palette_10bit
                 error_resilient)
      set(frames 30)
      # deeper than 8 bit: the hashes are over the samples themselves, so the
      # decoder has to write them at full precision
      set(stream_flags "")
      if(stream STREQUAL "palette_sc")
        set(frames 10)
      elseif(stream STREQUAL "intrabc")
        set(frames 16)
      elseif(stream STREQUAL "inter_intra")
        set(frames 110)
      elseif(stream STREQUAL "grain_inherit")
        set(frames 10)
      elseif(stream STREQUAL "lr")
        set(frames 8)
      elseif(stream STREQUAL "lr_444" OR stream STREQUAL "lr_422")
        # 4:4:4 and 4:2:2: restoration on a plane whose subsampling is not
        # 4:2:0, which is what makes the chroma unit grid (lr_uv_shift only
        # exists for 4:2:0) and the difference weighted mask's luma fallback
        # observable
        set(frames 8)
      elseif(stream STREQUAL "lr_12bit")
        # 12 bit is the one depth the Wiener filter rounds differently
        # (InterRound0 = 5, InterRound1 = 9)
        set(frames 8)
        set(stream_flags --yuv16)
      elseif(stream STREQUAL "lossless")
        # four frames, three of them inter: no quantiser, so the whole in-loop
        # filter chain is absent from the header
        set(frames 4)
      elseif(stream STREQUAL "odd_size")
        # 130x100: not a multiple of eight either way, so MiCols/MiRows round up
        set(frames 6)
      elseif(stream STREQUAL "edge_grid")
        # 130x66: the loop filter's four sample grid extends past the picture in
        # both planes (68 luma rows, 36 chroma rows)
        set(frames 4)
      elseif(stream STREQUAL "tiles")
        # 256x160 in 2x2 tiles (128+128 across, 128+32 down): the only stream
        # whose frames are cut into more than one tile, so the only one that
        # exercises the per tile range decoder, CDF and contexts, the
        # context_update_tile_id copy, and is_inside being the tile rather than
        # the frame in chroma availability and in the warp sample scan
        set(frames 4)
      elseif(stream STREQUAL "tiles_grid")
        # 320x188 in 4x2 tiles: unequal tile widths (64, 64, 64, 128 from
        # tile_log2) and a last tile row that is clamped to the coded extent
        # (192 luma samples for a 188 sample picture), which is what the mi
        # grid of the tile decoder and of the planner have to agree on
        set(frames 4)
      elseif(stream STREQUAL "lr_10bit")
        # the same clip at 10 bit, where the restoration filters' bit depth
        # scaling (7.17.3's Round2(a, 2 * (BitDepth - 8))) actually does
        # something: at 8 bit it is a no-op
        set(frames 8)
        set(stream_flags --yuv16)
      elseif(stream MATCHES "^aom_")
        set(frames 12)
        set(stream_flags --yuv16)
      elseif(stream STREQUAL "palette_10bit")
        # 10 bit: the palette colours are samples, not bytes
        set(frames 2)
        set(stream_flags --yuv16)
      elseif(stream STREQUAL "error_resilient")
        # four frames, three of them error resilient inter frames whose
        # references do not all name the same slot
        set(frames 4)
      endif()
      add_test(NAME stream_${stream}
               COMMAND av1gpu_decode --backend cpu --frames ${frames} ${stream_flags}
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
  # Which OpenCL device the Python tests compile and run on. Empty means the
  # first one pyopencl reports, which on a machine that has both a card and a
  # software implementation (pocl, for the driverless tests) is not necessarily
  # the card. `tests/opencl_selftest.py --list-devices` names them.
  set(AV1GPU_TEST_OPENCL_DEVICE "" CACHE STRING
      "OpenCL device index for the OpenCL tests (empty: the first device)")
  set(AV1GPU_TEST_OPENCL_ARGS "")
  if(NOT AV1GPU_TEST_OPENCL_DEVICE STREQUAL "")
    set(AV1GPU_TEST_OPENCL_ARGS --device ${AV1GPU_TEST_OPENCL_DEVICE})
  endif()
  set(AV1GPU_TEST_PY "${Python3_EXECUTABLE}")
  if(AV1GPU_TEST_PYTHON)
    set(AV1GPU_TEST_PY "${AV1GPU_TEST_PYTHON}")
  endif()
  if(AV1GPU_TEST_PY)
    # Single byte mutations of the fixtures, decoded through the decoder under
    # test: a corrupt stream has to come back as an error, not as a crash or a
    # sanitizer report. Standard library only, so it runs wherever Python does -
    # and it is the test that carries the palette cache overflow and the
    # transform's 64 bit multiplies (see docs/ACCURACY.md).
    if(TARGET av1gpu_decode)
      # one mutation per fixture on top of the known one, two frames each: the
      # sweep is a smoke test, not a fuzzing campaign, and it has to stay cheap
      # enough to run with every build (about half a minute normally and three
      # with the sanitizers)
      add_test(NAME fuzz_mutations
               COMMAND ${AV1GPU_TEST_PY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz_decode.py
                       --decoder $<TARGET_FILE:av1gpu_decode> --per-file 1 --frames 2)
      set_tests_properties(fuzz_mutations PROPERTIES TIMEOUT 900)
    endif()

    add_test(NAME opencl_selftest
             COMMAND ${AV1GPU_TEST_PY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_selftest.py
                     ${AV1GPU_TEST_OPENCL_ARGS})
    set_tests_properties(opencl_selftest PROPERTIES SKIP_RETURN_CODE 77)

    # Replays real decode plans through the OpenCL kernels and compares whole
    # frames against the CPU reference decode. It needs the dump tool, which is
    # only built with the tools, and pyopencl + a runtime.
    if(TARGET av1gpu_dump)
      add_test(NAME opencl_pipeline
               COMMAND ${AV1GPU_TEST_PY}
                       ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_pipeline.py
                       ${AV1GPU_TEST_OPENCL_ARGS}
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
    #   superres        the only one that is coded narrower than it displays, so
    #                   the upscaling process and the scaled motion
    #                   compensation reach the device kernels
    #   lr              the only one that codes loop restoration, so the two
    #                   source pictures a unit reads - the CDEF output and the
    #                   deblocked one before it - and the stripes that take
    #                   samples from both reach the device kernels
    #   lr_10bit        the same at 10 bit, where the self guided filter's bit
    #                   depth scaling runs at all (at 8 bit those Round2 calls
    #                   shift by zero, so a missing scale costs nothing there)
    #   lr_12bit        and at 12 bit, the depth the Wiener filter rounds
    #                   differently (InterRound0 5, InterRound1 9)
    #   grain           the only one that carries film grain parameters, so the
    #                   noise image and the per plane curves reach the device
    #                   kernels, and the chroma planes have to be blended before
    #                   the luma ones
    #   grain_inherit   the only one where a frame re-uses a reference's grain
    #                   parameters instead of coding them
    #   the three aom_  the only streams that are not 8 bit 4:2:0, so the only
    #   streams         ones where the deblocking thresholds' 1 << (BitDepth - 8)
    #                   scaling, the deeper reconstruction and the wider chroma
    #                   planes run at all; aom_10bit_320x240's first frame codes
    #                   a zero luma level with the reference deltas enabled, so
    #                   it is also the one that catches a decoder filtering a
    #                   frame the loop filter never runs on
    #   lossless        the only one whose frames code no quantiser at all, so the
    #                   frame header has no filter parameters and every sample a
    #                   block reconstructs has to reach the kernels untouched
    #   odd_size        the only one whose picture is not a multiple of eight, so
    #                   the block grid, the CDEF halo and the prediction clamps
    #                   reach into the border of the plane
    #   edge_grid       the only one whose planes end inside the four sample grid
    #                   the loop filter walks, so the filter's extent and the
    #                   rows CDEF and restoration copy from it show up in the
    #                   last rows of the picture
    #   tiles           the only one cut into more than one tile, so the only one
    #                   whose plans carry blocks from four independently coded
    #                   tiles and whose CDEF and restoration tests follow them
    #   tiles_grid      the same with an unequal grid (4x2, 64/64/64/128 wide)
    #                   on a frame whose height is not a superblock multiple
    #   palette_10bit   the only one that codes palette blocks in a picture
    #                   deeper than eight bits, so the palette's colours are
    #                   samples and the job's colours have to carry them whole
    #   error_resilient the only one whose inter frames are error resilient, so
    #                   the plans are built with every reference slot resolved
    #                   from a header that names no primary reference
    #   svt_av1_320x240 a different encoder, so a different mix of tools
    # All of it on a machine with no GPU at all.
    if(TARGET av1gpu_decode)
      foreach(stream palette_sc intrabc lw_t2 lw_t3 inter_intra seg_altq
                     superres lr lr_10bit lr_12bit lr_444 lr_422 lossless
                     odd_size edge_grid tiles tiles_grid grain grain_inherit
                     aom_10bit_320x240 aom_444_10bit_320x240 aom_12bit_320x240
                     palette_10bit error_resilient svt_av1_320x240)
        set(frames 30)
        if(stream STREQUAL "palette_sc")
          set(frames 10)
        elseif(stream STREQUAL "intrabc")
          set(frames 16)
        elseif(stream STREQUAL "inter_intra")
          set(frames 110)
        elseif(stream STREQUAL "svt_av1_320x240")
          set(frames 6)
        elseif(stream STREQUAL "lossless")
          set(frames 4)
        elseif(stream STREQUAL "odd_size")
          set(frames 6)
        elseif(stream STREQUAL "edge_grid")
          set(frames 4)
        elseif(stream STREQUAL "tiles")
          set(frames 4)
        elseif(stream STREQUAL "tiles_grid")
          set(frames 4)
        elseif(stream STREQUAL "palette_10bit")
          set(frames 2)
        elseif(stream STREQUAL "error_resilient")
          set(frames 4)
        elseif(stream STREQUAL "grain_inherit")
          set(frames 10)
        elseif(stream STREQUAL "lr")
          set(frames 8)
        elseif(stream STREQUAL "lr_10bit")
          set(frames 8)
        elseif(stream STREQUAL "lr_12bit")
          set(frames 8)
        elseif(stream STREQUAL "lr_444")
          set(frames 8)
        elseif(stream STREQUAL "lr_422")
          set(frames 8)
        elseif(stream MATCHES "^aom_")
          set(frames 12)
        endif()
        add_test(NAME opencl_${stream}
                 COMMAND ${AV1GPU_TEST_PY}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_pipeline.py
                         ${AV1GPU_TEST_OPENCL_ARGS}
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
