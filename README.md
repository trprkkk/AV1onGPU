# AV1onGPU

**GPU-accelerated AV1 reconstruction for GPUs that have no AV1 decoder block.**

Every GPU made before roughly 2018 — and most cheap ones since — has no fixed
function AV1 decoder, so the browser or player falls back to software decoding
and the CPU fans spin up. AV1onGPU takes the other road: it does the whole
reconstruction pipeline (prediction, inverse transform, loop filters, film
grain, colour conversion) with **compute kernels** on the same shaders the GPU
already has, so an old card can decode AV1 while the CPU stays nearly idle.

* **Backends:** OpenCL 2.0, CUDA (via NVRTC, no CUDA SDK needed to build) and a
  bit exact CPU reference implementation of the same maths.
* **No AV1 bitstream parser required on the GPU side.** The caller parses with
  whatever front-end it likes (libaom, dav1d, an in-house parser) and hands over
  the *decode syntax* of a frame: partition, prediction modes, motion vectors,
  dequantised coefficients, filter parameters.
* **No SDK to build.** OpenCL and CUDA entry points are resolved with
  `dlopen`/`dlsym` at run time; a missing or broken driver degrades to "backend
  unavailable" instead of failing to start.
* **Bit exactness.** All three backends run the *same* scalar code
  (`kernels/av1gpu_kernels.h`), which is compiled twice: once by the host
  compiler for the CPU backend, once by the device compiler for the kernels.
  The tests require the GPU output to be identical to the CPU output, pixel for
  pixel.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requirements: a C++17 compiler and CMake 3.16. OpenMP is used when found. No
OpenCL or CUDA SDK is needed; both backends are compiled in and pick up the
driver at run time.

| option | default | meaning |
| --- | --- | --- |
| `AV1GPU_ENABLE_OPENCL` | `ON` | build the OpenCL backend and embed its kernels |
| `AV1GPU_ENABLE_CUDA` | `ON` | build the CUDA backend and embed its kernels |
| `AV1GPU_ENABLE_OPENMP` | `ON` | thread the CPU reference backend |
| `AV1GPU_BUILD_TOOLS` | `ON` | the three tools below |
| `AV1GPU_BUILD_TESTS` | `ON` | unit + pipeline tests |
| `AV1GPU_SANITIZE` | `OFF` | ASan/UBSan build |

## Tools

```sh
build/bin/av1gpu_info                    # devices, kernel sources, feature flags
build/bin/av1gpu_bench --all             # fps, CPU share and PSNR per backend
build/bin/av1gpu_play --rgb --output out.rgb    # raw frames, pipe into a player
```

`av1gpu_bench` is the interesting one for this project: it reports the **CPU
share** next to fps. Software decoding at 1080p pegs a core or two; the point of
this library is that the reconstruction work moves to the GPU and the host only
pays for submission and synchronisation.

```sh
build/bin/av1gpu_bench --backend cpu --width 1280 --height 720 --frames 30
backend    device                        ms/frame       fps    cpu %   psnr Y   launch
cpu        CPU reference                    56.49      16.3    194.9    44.57        0
```

(The `cpu %` column is above 100 because the CPU *reference* backend is
threaded — it is the baseline, not the target configuration.)

## Using the library

```cpp
#include "av1gpu/av1gpu.hpp"

av1gpu::BackendOptions options;      // BackendKind::Auto: CUDA, OpenCL, CPU
std::string error;
std::unique_ptr<av1gpu::Decoder> dec =
    av1gpu::Decoder::create(options, &error);

av1gpu::FrameSyntax syntax;          // filled in by your bitstream parser
av1gpu::Frame out;
int slot = dec->decode_reference(syntax, &out, &error);   // slot: device resident
// ... later frames reference it through syntax.reference_slots ...
dec->release_reference(slot);
```

`decode_reference` keeps the decoded picture on the device, so an inter frame
can predict from it without a round trip through host memory — that is what
keeps the CPU out of the loop. `Decoder::decode` is the simple variant that
drops the device copy.

Colour conversion has a device path too (`Decoder::to_rgb`), so the player is
one upload and one readback per frame.

## Layout

```
include/av1gpu/av1gpu.hpp     public API: Frame, FrameSyntax, Decoder
include/av1gpu/synthetic.hpp  synthetic clip generator + closed loop encoder
kernels/av1gpu_kernels.h      the maths: all stages, shared by every backend
kernels/av1_opencl.cl         OpenCL entry points (one work item per output)
kernels/av1_cuda.cu           CUDA entry points, mirroring the OpenCL file
src/core/engine.cpp           syntax -> job plan, wavefront scheduler, Decoder
src/core/tables.cpp           transform bases, subpel filters, grain, colour
src/core/post_filters.cpp     host deblock/CDEF/grain, shared with encoders
src/backends/cpu/             reference backend (the correctness baseline)
src/backends/opencl/          loader (dlopen) + backend
src/backends/cuda/            loader (dlopen) + backend (NVRTC at run time)
tools/                        info, bench, player, plan dumper (av1gpu_dump)
tests/                        kernel unit tests, pipeline tests, OpenCL parity
```

## How a frame is scheduled

Intra prediction reads reconstructed neighbours, so the blocks of a frame cannot
all run at once. The engine records which block owns which part of the frame on
a coarse grid and gives every block the earliest wave in which its neighbours
are already reconstructed:

```
wave(block) = 1 + max(wave(above), wave(left), wave(above-left),
                      [wave(co-located luma) for chroma-from-luma])
```

Inter blocks only read reference frames, so they all land in wave 0 — an inter
frame is typically **one** wave, i.e. a handful of kernel launches for the whole
picture. Waves run in order; inside a wave the jobs are completely independent,
which is what a GPU wants. `BackendOptions::dependency_cell` trades launch count
against parallelism.

## Tests

```sh
cd build && ctest --output-on-failure
```

* `test_kernels` — transform bases, intra/inter prediction, subpel filters,
  deblock, CDEF, grain, colour conversion, on the host.
* `test_pipeline` — closed loop decode of a synthetic clip on every available
  backend; the decoded frames must match the encoder's own reconstruction
  exactly, and the wavefront cell size must not change the output.
* `opencl_selftest.py` — compiles `kernels/av1_opencl.cl` with a real OpenCL
  compiler (pocl works) and checks a job of every kernel against values computed
  in Python. This is the test that catches kernel *source* errors, which the C++
  tests cannot see. It needs `pyopencl`; without it the test reports "skipped".
* `opencl_pipeline.py` — the same, one level up: `av1gpu_dump` writes the real
  decode plans of a clip, the test replays every job of every wave through the
  actual OpenCL kernels in the order the backend submits them (waves, then
  deblocking vertical/horizontal, CDEF, grain), and the whole frame has to match
  the CPU reference decode bit for bit. It exercises job arrays, prefix tables,
  launch counts, reference slots and loop filter scheduling together — the
  things the per-kernel check cannot see. Also needs `pyopencl`.

```sh
pip install pyopencl          # or pyopencl + pocl-binary-distribution
python3 tests/opencl_selftest.py
python3 tests/opencl_pipeline.py --frames 4 --qstep 6
```

`tests/` and this section assume Linux or macOS. On a machine without an ICD
loader both OpenCL tests exit with 77 ("skipped") instead of failing, because
that is a missing driver and not a wrong kernel; with pocl the two tests run on
the CPU and still compile and execute the real kernel sources.

```sh
# PoCL in a scratch prefix, no ICD loader in the container
export LD_LIBRARY_PATH=$POCL_LIBS:$PYOPENCL_LIBS
export OPENCL_VENDOR_PATH=$PYOPENCL_LIBS
ctest --test-dir build -R opencl --output-on-failure
```

## What is not here

There is no entropy decoder and no bitstream parser in this tree: the library
starts where parsing ends. `av1gpu::synthetic_encode` is a small closed loop test
encoder that produces valid syntax from a generated clip, and it is what the
tools and tests decode. See `docs/ROADMAP.md` for the intended front-end work
and `docs/ACCURACY.md` for exactly where this implementation approximates the
AV1 specification and where it is bit exact.

## License

BSD-3-Clause. See `LICENSE`.
