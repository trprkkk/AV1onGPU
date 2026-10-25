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
* **The GPU never parses bits.** The caller hands over the *decode syntax* of a
  frame: partition, prediction modes, motion vectors, dequantised coefficients,
  filter parameters. A small front-end for AV1 container and header parsing is
  included (`src/frontend/`), and any other parser (libaom, dav1d, an in-house
  one) can produce the same `FrameSyntax`.
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
| `AV1GPU_BUILD_TOOLS` | `ON` | the tools below |
| `AV1GPU_BUILD_TESTS` | `ON` | unit + pipeline tests |
| `AV1GPU_SANITIZE` | `OFF` | ASan/UBSan build — `docs/ACCURACY.md` has the sweep |

## Tools

```sh
build/bin/av1gpu_info                    # devices, kernel sources, feature flags
build/bin/av1gpu_bench --all             # fps, CPU share and PSNR per backend
build/bin/av1gpu_play --rgb --output out.rgb    # raw frames, pipe into a player
build/bin/av1gpu_probe clip.ivf          # what is in this bitstream?
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

`Decoder::to_nv12` is the same idea for the layout a video surface actually has:
planar in, one buffer of interleaved Cb/Cr out, rows `pitch` bytes wide - a
VA-API surface's pitch is not the picture width, which is why it is a parameter -
so the pack that a compositor imports is done by the kernels and the planes never
come back to the host. Bytes between the picture and the pitch are written as
zero, NV12 is an 8 bit format so a deeper stream is shifted down like
`frame_to_yuv8` does, and a frame that is not 4:2:0 is refused instead of being
truncated. `av1gpu::frame_to_nv12` is the host function with the same output (the
CPU backend and the tests use it as the reference). `av1gpu_decode --nv12` writes
that layout at a pitch of the picture width, and its output is byte identical to
`ffmpeg -i stream.ivf -pix_fmt nv12 -f rawvideo out.nv12` on the fixtures, which
is a second opinion on the interleaving and the plane order.

## Layout

```
include/av1gpu/av1gpu.hpp     public API: Frame, FrameSyntax, Decoder
include/av1gpu/synthetic.hpp  synthetic clip generator + closed loop encoder
kernels/av1gpu_kernels.h      the maths: all stages, shared by every backend
kernels/av1_opencl.cl         OpenCL entry points (one work item per output)
kernels/av1_cuda.cu           CUDA entry points, mirroring the OpenCL file
src/core/engine.cpp           syntax -> job plan, wavefront scheduler, Decoder
src/core/tables.cpp           transform bases, subpel filters, colour
src/core/grain.cpp            film grain: noise, AR filter, stripe blending
src/core/post_filters.cpp     host deblock/CDEF/grain, shared with encoders
src/backends/cpu/             reference backend (the correctness baseline)
src/backends/opencl/          loader (dlopen) + backend
src/backends/cuda/            loader (dlopen) + backend (NVRTC at run time)
src/frontend/                 IVF/OBU demuxing, bit reader, header parsing
tools/                        info, bench, player, probe, plan dumper
tests/                        kernel unit tests, pipeline tests, OpenCL parity
```

## The bitstream front-end

`src/frontend/` turns bytes into the parameters the engine needs:

| file | what it does |
| --- | --- |
| `bitreader.hpp` | literal bits: `f(n)`, `su(n)`, `ns(n)`, `uvlc()`, leb128 |
| `obu.{hpp,cpp}` | OBU header, extension header, leb128 sizes, temporal unit splitting |
| `ivf.{hpp,cpp}` | the IVF container used by the AV1 test tools |
| `headers.{hpp,cpp}` | sequence header and frame header syntax |
| `range_decoder.{hpp,cpp}` | the boolean range decoder: symbols, `L(n)`, `ns(n)`, overrun handling |
| `cdf.{hpp,cpp}`, `cdf_tables*.hpp` | the adaptive CDFs and the specification's default tables |
| `tile_decoder.cpp` | the tile payload: partitions, modes, motion vectors, coefficients, filter and restoration units |
| `dequant.hpp`, `reconstruct.hpp`, `warp.hpp`, `restoration.hpp` | dequantisation, prediction and the reconstruction helpers the tile decoder calls |

What parses today is the whole stream: the complete sequence header (profiles
0/1/2, 8/10/12 bit, 4:2:0 / 4:2:2 / 4:4:4, operating points, timing info, all
the feature flags), KEY, INTRA_ONLY, INTER and SWITCH frame headers up to the
byte where the arithmetic coded tile data starts - plus `show_existing_frame`,
segmentation, delta Q / delta LF, the tile grid, quantiser, deblocking levels,
CDEF strengths, the loop restoration units, reference and skip mode decisions
and the transform set - and then the tile payload itself, which comes out as the
`FrameSyntax` a backend consumes.

Inter frames need to know what came before them - a frame that inherits its size
from a reference (`found_ref`), that is allowed to use skip mode, or whose
references have to be verified, depends on the frames the decoder has already
stored - so the parser takes the `RefPool` the caller maintains
(`refresh_ref_pool()` applies a parsed frame's refresh flags). That
pool is also what reference frame ids are checked against, and what
`frame_refs_short_signaling` derives the five uncoded references from
(`set_frame_refs()`, 7.8): see `docs/ACCURACY.md`.

All of it is parsed: loop restoration (7.17) included, both the frame header's
`restoration_params` and each unit's filter, though the filter that is *applied*
lives in the backends rather than the parser (see `docs/ROADMAP.md`).

The tile payload is coded with AV1's boolean range decoder, which
`src/frontend/range_decoder.hpp` implements: adaptive Q15 symbol CDFs, boolean
and equiprobable bits, `L(n)` and `ns(n)`, and the padded end-of-tile tail. It is
bit identical to dav1d's `msac`, and the default CDF tables on top of it are
checked against dav1d's own (`tests/data/tile_trace.hpp` replays the first 480
range coder events dav1d logged while decoding a key frame). `tile_decoder.cpp`
then walks the partition tree of every superblock, so what comes out is the
syntax of a whole frame - which is why `av1gpu_decode` decodes the fixtures
without a second parser.

A frame may be cut into a grid of tiles and each tile is coded independently, so
`decode_tiles` decodes every tile of a parsed tile group - in any order - into
the frame wide maps: the range decoder, the CDF tables, the above/left contexts,
`CurrentQIndex` and the restoration predictors start over at each tile (5.11.2),
while the block sizes, references, motion field, segment ids, CDEF index grid and
restoration units are the frame's, and the tables it hands on are those of the
tile `context_update_tile_id` names (5.9.2). Neighbour availability is the tile
everywhere 7.10 asks, which is where two boundary bugs lived until
`tests/data/tiles.ivf` (2x2) pinned them - `docs/ACCURACY.md` has both.

The parser is checked against a real stream (`tests/data/svt_av1_320x240.ivf`,
encoded with SVT-AV1) whose fields were cross-checked with FFmpeg's
`trace_headers` bitstream filter, and it was fuzz-compared against dav1d's
sequence header parser; see `docs/ACCURACY.md`.

```sh
$ build/bin/av1gpu_probe tests/data/svt_av1_320x240.ivf
tests/data/svt_av1_320x240.ivf: IVF, 320x240, 6 frame(s), timebase 30/1
temporal unit 0 (15495 bytes)
  obu temporal_delimiter   0 bytes
  obu sequence_header      11 bytes
sequence header
  profile 0, 8 bit, 4:2:0, 320x240 max
  superblock 64, cdef yes, restoration no, superres no
  order hint yes, warped motion yes, dual filter no, masked compound no
  film grain parameters absent
  obu frame                15477 bytes
  key frame, 320x240, base_q_idx 51, 1 tile(s) (1x1)
  deblock 2/2 luma, 1/1 chroma, sharpness 0; cdef damping 3, y strengths 0
  delta Q per superblock (res 0)
  tile data: 15466 bytes at offset 11 (320x240 at 0,0)
  tile decoded: 306 block(s), 1075 transform block(s), 64301 coefficient(s), 15466 of 15466 bytes used
  first block: 32x32 at mi (0,0), partition 0, skip 0, modes 10/10, transform 32x32, q 53
temporal unit 1 (491 bytes)
  ...
  inter frame (not shown), 320x240, base_q_idx 138, 1 tile(s) (1x1)
  references: last=slot 2 (hint 0) ... intra=slot 2 (hint 0)
  refresh flags 0x01, primary ref 0, order hint 4, not error resilient
  compound reference select, skip mode off, ref frame mvs, free integer motion, low precision
  global motion: none
  deblock 7/7 luma, 3/3 chroma, sharpness 0; cdef damping 5, y strengths 62
  tile data: 147 bytes at offset 16 (320x240 at 0,0)
  tile decoded: 87 block(s), 261 transform block(s), 22 coefficient(s), 147 of 147 bytes used
  first block: 64x64 at mi (0,0), partition 0, skip 1, modes 0/0, transform 64x64, q 138
```

The bytes the tile decoder consumed are part of the output because a tile that
ends before its last byte is a decoder that stopped early: the counts above are
how `docs/ACCURACY.md` records what a fixture actually codes.

`tests/test_frontend.cpp` writes the same syntax with a small bit writer and
checks it field by field, including the error paths. The sequence header parser
was additionally cross-checked against dav1d's parser over 4000 random inputs:
every header dav1d accepts is parsed with the same field values and the same bit
position (it found two bugs while it ran, see `docs/ACCURACY.md`).

## How a frame is scheduled

Intra prediction reads reconstructed neighbours, so the blocks of a frame cannot
all run at once. The engine records which block owns every sample of the coded
picture — the frame rounded up to the block grid — and gives every block the
earliest wave in which the samples its prediction reads are already
reconstructed:

```
wave(block) = 1 + max(wave(owner(sample)) for every sample the block reads)
```

The samples are the plain left column and above row, plus the second block of
either when the mode actually reads it (a directional block reads above samples
two blocks wide below 90 degrees, left samples that far above 180), plus the
co-located luma block for chroma from luma. A palette block predicts nothing and
reads nothing. All three planes share one dependency grid.

Inter blocks only read reference frames, so they all land in wave 0 — an inter
frame is typically **one** wave, i.e. a handful of kernel launches for the whole
picture. Waves run in order; inside a wave the jobs are completely independent,
which is what a GPU wants. An intra frame costs about the length of its block
diagonal — 31 to 125 waves across the streams in `tests/data`, roughly half of
what a coarser schedule gave (the numbers are in `docs/ACCURACY.md`).

## Tests

```sh
cd build && ctest --output-on-failure
```

* `test_kernels` — transform bases, intra/inter prediction, subpel filters,
  deblock, CDEF, grain, colour conversion, on the host.
* `test_entropy` — the boolean range decoder: symbol decoding over adaptive
  Q15 CDFs, `L(n)`/`ns(n)`, frozen (`disable_cdf_update`) tables, and end of
  tile handling. The expectations are golden vectors decoded by dav1d's `msac`,
  not round trips against our own encoder.
* `test_frontend` — bit reader, OBU framing, IVF demuxing, sequence and frame
  header parsing, and the error paths of all of them.
* `test_pipeline` — closed loop decode of a synthetic clip on every available
  backend; the decoded frames must match the encoder's own reconstruction
  exactly, and the wavefront cell size must not change the output.
* `fuzz_decode.py` — single byte mutations of every fixture, decoded through
  `av1gpu_decode`: a stream that is not AV1 has to come back as an error, not as
  a crash and not as a sanitizer report. Run against the sanitizer build it is
  the test that carries the palette cache overflow and the transform's 64 bit
  multiplies (see `docs/ACCURACY.md`).
* `opencl_selftest.py` — compiles `kernels/av1_opencl.cl` with a real OpenCL
  compiler (pocl works) and checks a job of every kernel against values computed
  in Python. This is the test that catches kernel *source* errors, which the C++
  tests cannot see. It needs `pyopencl`; without it the test reports "skipped".
* `opencl_pipeline.py` — the same, one level up: `av1gpu_dump` (synthetic clip)
  or `av1gpu_decode --dump-plans` (a real stream) writes the real decode plans,
  the test replays every job of every wave through the actual OpenCL kernels in
  the order the backend submits them (waves, then deblocking
  vertical/horizontal, CDEF, upscale, loop restoration, grain), and the whole
  frame has to match the CPU reference decode bit for bit. It exercises job arrays, prefix tables, launch
  counts, reference slots, palette maps, OBMC candidates and loop filter
  scheduling together — the things the per-kernel check cannot see. Also needs
  `pyopencl`. Before it runs it checks every job against the pool it indexes, so
  a plan that does not fit its buffers fails the test instead of corrupting
  memory.

* `gpu_verify.py` — the same question asked of a machine that really has a GPU:
  `av1gpu_bench` has to report a GPU result bit identical to the CPU reference,
  and the fixture streams have to decode bit exactly through the GPU backend with
  `--require-device`, so a backend that fell back to the CPU cannot pass. Run it
  on the machine with the card, see `docs/GPU.md`.

```sh
pip install pyopencl          # or pyopencl + pocl-binary-distribution
python3 tests/opencl_selftest.py
python3 tests/opencl_pipeline.py --frames 4 --qstep 6
python3 tests/opencl_pipeline.py --input tests/data/lw_t2.ivf --frames 30 \
        --decoder build/bin/av1gpu_decode
python3 tests/gpu_verify.py   # on the machine with the GPU
```

`tests/` and this section assume Linux or macOS. On a machine without an ICD
loader both OpenCL tests exit with 77 ("skipped") instead of failing, because
that is a missing driver and not a wrong kernel; with pocl the two tests run on
the CPU and still compile and execute the real kernel sources. The tests use the
loader pyopencl bundles, so the Python layer needs nothing else; the C++ backend,
which `dlopen`s `libOpenCL.so.1`, needs a system (or built) ICD loader as well —
with one in place `--backend opencl` decodes the fixtures bit exactly on a PoCL
CPU device. `-DAV1GPU_TEST_PYTHON=/path/to/venv/bin/python` points ctest at the
interpreter that has numpy and pyopencl, instead of the system one.

```sh
# PoCL in a scratch prefix, no ICD loader in the container
export LD_LIBRARY_PATH=$POCL_LIBS:$PYOPENCL_LIBS
export OPENCL_VENDOR_PATH=$PYOPENCL_LIBS
ctest --test-dir build -R opencl --output-on-failure
```

## What is not here

Every inter and intra tool AV1 defines decodes bit exactly - including palette
coded blocks (`tests/data/palette_sc.ivf`), inter/intra compound
(`tests/data/inter_intra.ivf`), intra block copy (`tests/data/intrabc.ivf`) and
skip mode, which the `lw_t2`/`lw_t3`/`inter_intra` fixtures code thousands of
times, and segmentation (`tests/data/seg_altq.ivf`, a different quantiser per
segment), superres, where a frame is coded narrower than it is displayed and
upscaled after CDEF (`tests/data/superres.ivf`), loop restoration, both filters
and the per unit switchable form, which reads the two pictures 7.17 keeps
(`tests/data/lr.ivf`, and the same clip at 4:4:4 and 4:2:2 as `lr_444` /
`lr_422`), and film grain, whose generator, auto-regressive filter
and stripe blending follow 7.18.3 sample for sample (`tests/data/grain.ivf`), and frames cut into tiles, which `decode_tiles`
decodes tile by tile into one picture (`tests/data/tiles.ivf`, 2x2, checked
against dav1d and through the OpenCL replay).
`av1gpu::synthetic_encode` is a small closed loop test encoder that produces
valid syntax from a generated clip and does not need any of those.

The CUDA backend compiles the same kernel source as the OpenCL and CPU backends
with NVRTC but has had no NVIDIA device in the development environment, so it is
checked by construction and by the shared source, not by a run: only the OpenCL
backend and the CPU backend have been run end to end on real streams here.
`docs/ACCURACY.md` records exactly where this implementation approximates the
specification and where it is bit exact.

## License

BSD-3-Clause. See `LICENSE`.
