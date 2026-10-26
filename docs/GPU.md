# Verifying on a real GPU

Everything in this tree is written so that it can be checked on a machine with no
GPU at all — the kernels are one C source that all three backends compile, and a
software OpenCL implementation (pocl) runs the real kernel source. What that
cannot show is whether a **driver** accepts this backend: device queries, buffer
sizes, the job layouts the kernels declare, launch shapes, and how much CPU time
is left over while the card works. That is what this page is for, and
`tests/gpu_verify.py` is the one command that answers it.

If you have a GPU machine and five minutes, run this and keep the log:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAV1GPU_BUILD_TOOLS=ON
cmake --build build -j
./build/bin/av1gpu_info                       # has to list an [opencl] or [cuda] device
python3 tests/gpu_verify.py -v 2>&1 | tee gpu_verify.log
```

The last line is either `gpu_verify: all checks passed on opencl0` (or `cuda0`)
or a list of failures. [What to send back](#what-to-send-back) says what to put
in a report; [when it fails](#when-it-fails) maps the messages to a cause.

## The layers, and which of them need the card

The evidence comes in five layers. The first four run anywhere — they are what
CI runs, with pocl standing in for a driver — and only the last one needs a card.

| layer | command | proves | needs a card |
| --- | --- | --- | --- |
| kernel maths | `build/bin/test_kernels` | every stage against hand computed values, on the host compiler | no |
| launch ABI | `build/bin/test_gpu_sources` | the hand written argument lists match the kernels they launch, in both languages | no |
| kernel source | `python3 tests/opencl_selftest.py` | `kernels/av1_opencl.cl` compiles with a real OpenCL C compiler and one job of every kernel returns what Python computes | no (pocl) |
| whole frames | `python3 tests/opencl_pipeline.py` | real decode plans replayed job by job through the device kernels match the CPU decode bit for bit | no (pocl) |
| the driver | `python3 tests/gpu_verify.py` | a real card initialises, accepts every buffer and launch shape, and produces the same pixels — at a useful speed and CPU load | **yes** |

The middle three are worth re-running on the GPU machine even though they already
pass at home: with a card installed they compile the kernel source with *that*
driver instead of pocl, which is a stricter test of the source than pocl is (see
[the Python layers on the card](#the-python-layers-on-the-card)).

## Checklist

Each step has a result you can recognise. Stop at the first one that does not
match: the later ones cannot pass if an earlier one failed.

1. **Build with the tools.**
   ```sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
         -DAV1GPU_BUILD_TOOLS=ON -DAV1GPU_BUILD_TESTS=ON
   cmake --build build -j
   ```
   Result: `build/bin/` holds `av1gpu_info`, `av1gpu_decode`, `av1gpu_bench`,
   `av1gpu_dump`, `av1gpu_probe`, `av1gpu_play`.

2. **The loader finds a driver.**
   ```sh
   ls /etc/OpenCL/vendors          # one .icd per vendor; missing here = no OpenCL
   ./build/bin/av1gpu_info
   ```
   Result: a line like
   `[opencl 0] <card name> OpenCL 3.0 CUDA ...` followed by its driver version,
   compute units and workgroup limit. `[cpu]` alone means the loader or the ICD
   file is missing — see [prerequisites](#what-you-need).

3. **The device accepts the kernel source and a launch.**
   ```sh
   ./build/bin/av1gpu_bench --backend opencl --frames 2 --width 320 --height 180
   ```
   Result: a table row for `opencl` naming the device, with the same `psnr Y`
   as `cpu` (the decode is bit identical, so the numbers have to be equal). A row
   that reads `no OpenCL loader found`, `no CUDA driver library` or a compile
   error is the whole answer for steps 5–8, and it arrives in seconds instead of
   after a 400 frame run: this is where an NVRTC/OpenCL source problem surfaces.

4. **The card decodes one fixture correctly.**
   ```sh
   ./build/bin/av1gpu_decode --backend opencl --require-device \
       tests/data/lr.ivf --hash-check tests/data/lr.hash
   ```
   Result: `8 of 8 decoded frame(s) match the recorded hashes` and
   `backend: opencl (...)` naming the card, not `cpu`. `lr` is the fixture to
   start with: loop restoration is the stage with the most buffers and the most
   two-picture reads, so it fails first when a layout is wrong.

5. **The one command.**
   ```sh
   python3 tests/gpu_verify.py -v 2>&1 | tee gpu_verify.log
   ```
   Result: `gpu_verify: all checks passed on opencl0`. A failure names the stream
   and the backend; see [when it fails](#when-it-fails).

6. **The same thing under ctest** (optional, so the result is part of the suite):
   ```sh
   cmake -S . -B build -DAV1GPU_GPU_TESTS=ON -DAV1GPU_TEST_PYTHON=$(which python3)
   cmake --build build -j
   ctest --test-dir build -R gpu_verify --output-on-failure
   ```
   Result: `100% tests passed`. Without a device the test reports a skip (exit
   77) rather than a failure, which is the correct answer for a machine that
   cannot run it.

7. **The Python layers, on the card** (optional but cheap):
   ```sh
   python3 tests/opencl_selftest.py --list-devices     # which index is the card?
   python3 tests/opencl_selftest.py --device 0
   python3 tests/opencl_pipeline.py --device 0 --frames 4
   python3 tests/opencl_pipeline.py --device 0 --input tests/data/tiles_grid.ivf \
           --frames 4 --decoder build/bin/av1gpu_decode
   ```
   Result: `all OpenCL kernel checks passed`, then `all 4 frames match the CPU
   reference decode bit for bit`.

8. **Numbers.** `av1gpu_bench` at a resolution you care about, for both backends,
   plus the CPU share — [what to measure](#performance-numbers-to-collect).

## What you need

**A build with the tools** (step 1 above). Nothing else is compiled in: both GPU
backends are always built and resolve their entry points with `dlopen`/`dlsym` at
run time, so a missing driver is "backend unavailable", not a build failure, and
no SDK is needed to build.

**OpenCL.** An ICD loader (`ocl-icd-libopencl1` on Debian/Ubuntu and their
derivatives) plus the vendor's ICD file in `/etc/OpenCL/vendors`. The backend
`dlopen`s `libOpenCL.so.1`, so nothing has to be linked at build time.

| vendor | driver package | ICD file | notes |
| --- | --- | --- | --- |
| NVIDIA | the driver (`nvidia-driver-*`, or a data centre driver) | `nvidia.icd` | The same driver provides `libcuda.so.1` for the CUDA backend. |
| AMD | `mesa-opencl-icd` (RADV/Radeon) or `rocm-opencl-runtime` | `mesa.icd` / `amdocl64.icd` | ROCm's ICD needs the user in the `render`/`video` group. |
| Intel | `intel-opencl-icd` (NEO) or `mesa-opencl-icd` | `intel.icd` / `mesa.icd` | Gen9+ via NEO; Mesa's `rusticl` needs `RUSTICL_ENABLE=iris`. |
| pocl (software) | `pocl-opencl-icd` | `pocl.icd` | Runs on the CPU: enough for steps 3 and 7, useless for step 8. |

`clinfo -l` is the quickest way to see what the loader itself can enumerate; it
is not a build or test dependency. If `av1gpu_info` says *no OpenCL loader found*
the loader is missing; if it prints no `[opencl]` line the ICD file, the driver
or the permissions are.

**CUDA.** The driver's `libcuda.so.1` and the toolkit's NVRTC — NVRTC compiles the
kernel source at run time, so `nvcc` is not needed and nothing has to be present
at build time. The loader tries `libnvrtc.so.12`, `.so.11.2`, `.so.11.1`,
`.so.11.0`, `.so.10.2` and finally the unversioned `libnvrtc.so`, so a toolkit
that is newer than this list needs the unversioned symlink (or a `cuda-nvrtc`
package that ships one). It takes the compute capability from the device and
reports the architecture it tried when a compile fails.

**In a container.** The loader and the ICD file have to be inside the image, and
the device has to be passed through:

```sh
# NVIDIA: the container toolkit mounts the driver (libcuda.so.1, the driver's
# nvidia.icd and libnvidia-opencl.so); the image supplies libnvrtc and the loader.
docker run --gpus all -v "$PWD:/w" -w /w nvidia/cuda:12.4.0-devel-ubuntu22.04 \
    bash -lc 'apt-get update && apt-get install -y ocl-icd-libopencl1 cmake g++ python3 &&
              cmake -S . -B build && cmake --build build -j &&
              ./build/bin/av1gpu_info && python3 tests/gpu_verify.py'

# AMD or Intel: no toolkit needed, just the device nodes and the groups
docker run --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
    -v "$PWD:/w" -w /w ubuntu:24.04 \
    bash -lc 'apt-get update && apt-get install -y ocl-icd-libopencl1 mesa-opencl-icd \
              cmake g++ && cmake -S . -B build && cmake --build build -j &&
              ./build/bin/av1gpu_info'
```

pocl is *not* what you want in that image if there is a card: it would enumerate a
CPU device, and on a machine that has both the first device is not always the
card (see [the Python layers on the card](#the-python-layers-on-the-card)).

## One command: `gpu_verify.py`

```sh
python3 tests/gpu_verify.py                 # every GPU backend this machine has
python3 tests/gpu_verify.py --backends cuda # that one is a requirement, not a wish
python3 tests/gpu_verify.py --device 1      # the second card
python3 tests/gpu_verify.py --stream-frames 2   # a quick pass over the fixtures
python3 tests/gpu_verify.py --fixtures ~/av1    # plus your own streams
python3 tests/gpu_verify.py -v 2>&1 | tee gpu_verify.log
```

| option | default | meaning |
| --- | --- | --- |
| `--build DIR` | `build/` next to the repo | where `bin/` is |
| `--backends A,B` | every GPU backend present | which to verify; naming one makes it mandatory (exit 1 when absent) |
| `--device N` | `0` | device index for both tools |
| `--bench-frames N` | `20` | frames in the synthetic clip |
| `--bench-size WxH` | `1280x720` | its size |
| `--qstep N` | `8` | its quantiser |
| `--stream-frames N` | `0` (the whole stream) | temporal units per fixture |
| `--fixtures DIR` | — | extra directory of `.ivf` + `.hash` or `_ref.yuv` |
| `-v` | off | print every command it runs |

It decodes the whole fixture set twice (CPU reference and card), so expect
minutes rather than seconds; `--stream-frames 2` is the quick pass while
iterating on a driver problem, and `--fixtures` adds your own streams.

It does two things per backend:

1. **Synthetic parity.** `av1gpu_bench --all` decodes a generated clip on the CPU
   reference and on the card. The result has to be **bit identical**, and the
   table reports ms/frame, fps, the CPU share and the launch count.
2. **Real streams.** `av1gpu_decode --backend <gpu> --require-device
   --hash-check` on all 24 fixtures that carry a reference decoder's frame
   hashes (404 hashes in all). Every shown frame has to match, so a whole front
   end plus reconstruction ran on the device. `svt_av1_320x240.ivf` is
   parser-only and is skipped; `--fixtures` adds your own streams.

`--require-device` is the important part. Without it a backend that fails to
initialise silently falls back to the CPU reference and a verification run would
"pass" while proving nothing; the script refuses a fallback and also refuses a
device whose backend is not the one it asked for. A stream that stops early is a
failure too, not just a mismatch.

Exit codes: **0** everything verified, **1** a mismatch, a backend failure, or a
backend named on the command line that this machine does not have, **2** the
tools are not built, **77** no GPU backend could be initialised and none was
asked for by name (ctest reports that as a skip). The 77/1 split is deliberate:
"this machine has no GPU" is a skip, but a job that said `--backends cuda` must
not pass on a machine without one.

### Reading the output

A run on a machine with a real card looks like this (the numbers below are from
pocl on a two core CPU, where only the `status` column means anything):

```
AV1onGPU GPU verification
  build: /home/user/AV1onGPU/build
  devices:
    [cpu] CPU reference                portable ()
    [opencl0] pthread-Intel(R) Xeon(R) Processor @ 2.60GHz OpenCL 3.0 PoCL ...

1. synthetic clip: 1280x720, 20 frames, qstep 8 (av1gpu_bench --all)
   backend  device                          ms/frame      fps   cpu %   psnr Y  launch  status
   cpu      CPU reference                    111.70      7.2   180.6    59.61       0  reference
   opencl   pthread-Intel(R) Xeon(R) Pro    170.58      5.3   158.9    59.61     172  bit identical
   opencl: 0.65x the CPU backend's ms/frame, CPU load 158.9% vs 180.6%

2. real streams: 24 stream(s), every temporal unit
   stream         backend  device                             frames ms/frame  status
   lw_t2          cpu      cpu (CPU reference)                 32/32    13.95  reference
   lw_t2          opencl   opencl (pthread-Intel(R) Xeon(      32/32    18.32  bit exact
   ...

gpu_verify: all checks passed on opencl0
```

* **`frames`**: matched/reference hashes for the real streams. A short count
  means the decode stopped early (a syntax or allocation error), which is a
  failure even if the frames it did produce were right.
* **`status`**: `bit exact`/`bit identical` is the only passing value;
  `MISMATCH` never appears with a passing exit code.
* **`ms/frame`, `cpu %`, `launch`**: performance, not correctness. On a CPU
  OpenCL device the ratio is meaningless (it is the CPU, one thread per work
  item); on a card the interesting pair is ms/frame *and* CPU share, because the
  point of the project is that the host stays idle while the card works.
* **`launch`**: kernel launches per frame on the synthetic clip. An inter frame
  should be a handful of waves; an intra frame pays one wave per block diagonal,
  which is why the count is high. `AV1GPU_PLAN_STATS=1` prints the per frame
  breakdown (`waves=…, intra=…, itx=…, deblock=…, cdef=…`).

## The Python layers on the card

Both scripts pick the **first** OpenCL device the loader reports, which on a
machine that has a card *and* pocl is not always the card. `--list-devices` names
them, `--device N` picks one:

```sh
python3 tests/opencl_selftest.py --list-devices
# [0] pthread-Intel(R) Xeon(R) Processor @ 2.60GHz (OpenCL 3.0 PoCL ...) on Portable Computing Language
# [1] NVIDIA GeForce GTX 1060 (OpenCL 3.0 CUDA) on NVIDIA CUDA
python3 tests/opencl_selftest.py --device 1
python3 tests/opencl_pipeline.py --device 1 --input tests/data/lr_10bit.ivf \
        --frames 8 --decoder build/bin/av1gpu_decode
```

For ctest the same choice is a cache variable:

```sh
cmake -S . -B build -DAV1GPU_TEST_OPENCL_DEVICE=1 \
      -DAV1GPU_TEST_PYTHON=/path/to/venv/bin/python
ctest --test-dir build -R opencl -j4 --output-on-failure
```

Running the 27 `opencl_*` tests on the card (the self test, the synthetic replay
and one replay per fixture) adds two things `gpu_verify` does not do: the self
test compares **each kernel's output against values computed in
Python** (so a difference is localised to a kernel rather than to a frame), and
the replay feeds the device the **real job arrays of every wave** of a real
stream, including the plans for 4:4:4, 12 bit, superres, tiles and film grain,
where `gpu_verify` sees only the finished picture. If `gpu_verify` reports a
mismatch, run these first: a passing self test narrows the problem to scheduling,
buffers or the driver rather than to the arithmetic.

## What each fixture proves on the device

The `opencl_*` ctest cases and the stream step of `gpu_verify` cover the same
streams. What each one is there *for*, on the device:

| stream | what only the device can show |
| --- | --- |
| `palette_sc`, `palette_10bit` | palette prediction and the colour map buffer, at 8 and at 10 bit |
| `intrabc` | the "read the frame being decoded" path (a negative reference slot) |
| `lw_t2`, `lw_t3` | local warped motion and OBMC candidates that carry their own reference slot |
| `inter_intra` | inter/intra compound: every blend of the taper and wedge masks |
| `seg_altq` | the per segment quantiser reaching the dequantiser on the device |
| `superres` | the upscaling pass and motion compensation at a scaled reference |
| `lr`, `lr_444`, `lr_422` | loop restoration: the two source pictures a unit reads and the stripe that takes rows from both |
| `lr_10bit`, `lr_12bit` | the restoration filters' bit depth scaling, which is a no-op at 8 bit |
| `grain`, `grain_inherit` | the noise image, the per plane curves, chroma before luma, and inherited parameters |
| `lossless` | blocks whose samples reach the device untouched (no filter chain at all) |
| `odd_size` | block grids, CDEF halo and clamps that reach past the picture |
| `edge_grid` | planes that end inside the loop filter's four sample grid |
| `tiles`, `tiles_grid` | plans carrying blocks from independent tiles, and CDEF/restoration following them |
| `aom_10bit_320x240`, `aom_444_10bit_320x240`, `aom_12bit_320x240` | deeper and wider planes: 10/12 bit reconstruction, 4:4:4 chroma, `1 << (BitDepth - 8)` thresholds |
| `error_resilient` | references resolved without a primary reference, frame-id named slots |
| `svt_av1_320x240` | different encoder, different tool mix (parser check only; no frame hashes) |

## Performance numbers to collect

Correctness first; then, in the order that matters for this project:

```sh
./build/bin/av1gpu_bench --backend cpu    --width 1280 --height 720 --frames 60
./build/bin/av1gpu_bench --backend opencl --width 1280 --height 720 --frames 60
./build/bin/av1gpu_bench --backend opencl --width 1920 --height 1080 --frames 60
./build/bin/av1gpu_bench --backend opencl --width 3840 --height 2160 --frames 20
```

1. **CPU share while the card works.** The synthetic clip is bound on the parse
   side by design, so compare against `--backend cpu` on the same machine rather
   than against a fixed number: what matters is how much of the host is left
   over for everything else.
2. **ms/frame** at 720p, 1080p and — if the card has the memory — 2160p. The
   reference pool is sized from the frame geometry, so a 4K run is also a
   memory test.
3. **launches per frame** from the `launch` column, and `AV1GPU_PLAN_STATS=1` for
   the wave breakdown. This is the number to attack if an intra frame is slow
   (`docs/ROADMAP.md` section 5).
4. **Bus traffic**: a frame should cost no host round trip except the output read
   back, because reference frames stay in the pool. Watch it with
   `nvidia-smi dmon` or `radeontop` while the bench runs — a steady stream of
   reads back to the host means something is being copied that should not be.

Report the machine, the driver and the numbers together — a ratio without the
driver version is not comparable.

## When it fails

| message | meaning | next step |
| --- | --- | --- |
| `no OpenCL loader found` | `libOpenCL.so.1` is not installed | install the ICD loader (`ocl-icd-libopencl1`) |
| no `[opencl]` line in `av1gpu_info` | the loader found no ICD file | check `/etc/OpenCL/vendors/*.icd`, the vendor package, and group membership |
| a crash inside the loader, or `undefined symbol: clGetPlatformIDs` | a hand assembled ICD setup (a wheel's `libpocl.so` pointed at directly, or a renamed `libOpenCL.so.1`) | use the distribution's `ocl-icd-libopencl1` with `pocl-opencl-icd` instead of the pip wheel, and let the ICD file hold an absolute path |
| `device workgroup limit too small` | the device query returned nothing usable | check the driver; the limit is logged by `av1gpu_info` |
| `asked for opencl, got cpu` | the request was refused and fell back | run `av1gpu_bench --backend opencl --frames 1` for the driver's reason |
| `kernel build failed` | NVRTC or the OpenCL compiler rejected the source | rerun with `-v` and keep the compiler log; the message names the `--gpu-architecture` that was tried |
| `out of device memory` | an allocation in `prepare()` or `decode()` failed | the frame plus its reference pool does not fit; try a smaller `--width/--height` |
| `CUDA kernel execution failed (cuCtxSynchronize = N)` | a kernel faulted or a copy targeted a stale buffer | the driver error number says which; this used to be silent and decoded into whatever the buffer held |
| `gpu_verify: <backend> was requested but no such device is present` | `--backends` named a backend this machine cannot initialise | `av1gpu_info` lists what it has |
| `device N does not exist: M OpenCL device(s) found` | `--device` is out of range | `python3 tests/opencl_selftest.py --list-devices` |
| a frame hash mismatch | the driver computes different pixels | run `opencl_selftest.py` and `opencl_pipeline.py` on that device first: if they pass, the arithmetic is right and the difference is device or driver specific — report device, driver and backend |

## The launch ABI

`cuLaunchKernel` takes a `void**` and OpenCL's `clSetKernelArg` takes a
`(void*, size_t, size_t, const void*)`: neither is type checked against the
kernel being launched, and every argument in both backends is packed by hand. A
missing argument, a swapped pair or an index set twice therefore does not fail to
build - it reads whatever the stack held, which on a GPU shows up as wrong pixels
somewhere in the frame or as a device fault a few frames later.

`tests/test_gpu_sources.cpp` makes those mistakes fail on any machine, in
milliseconds, from the sources the binary actually embeds:

| check | what it catches |
| --- | --- |
| the CUDA and OpenCL sources declare the same kernels with the same parameter types, in order | an argument added to one language only (which is how the palette argument could have landed) |
| every launch site passes as many arguments as the kernel it names declares | a forgotten `add_int`/`add_mem`, or a kernel that grew a parameter |
| the OpenCL indices are exactly `0..n-1`, none twice | a hand written `set_mem(k, 7, ...)` where 6 was meant - the classic silent read of the wrong argument |
| every kernel a backend launches exists in that backend's source | a rename that only reached one of the two files |
| neither source uses the other language's constructs, and the shared header is inlined exactly once | a `get_global_id` in the CUDA source, which only NVRTC would complain about, at run time, on the user's machine |
| both backends rebase the same reference-pool offsets before launching | a new `*_off` field wired up in one backend only, which silently reads another reference frame |

The test is part of the default `ctest` run and does not care whether a backend is
enabled: a build without CUDA checks only the OpenCL half. It is not a substitute
for a run on a card — it checks the shape of the calls, not what the driver does
with them.

## What to send back

A report that can be acted on needs the log and the machine, not a summary:

```sh
{
  uname -a
  ./build/bin/av1gpu_info
  ctest --test-dir build -N            # test count, so a skip is visible
} > environment.txt 2>&1
clinfo -l >> environment.txt 2>&1                 # optional
nvidia-smi >> environment.txt 2>&1                # or: rocm-smi, intel_gpu_top
python3 tests/gpu_verify.py -v > gpu_verify.log 2>&1
python3 tests/opencl_selftest.py --list-devices > devices.log 2>&1
python3 tests/opencl_selftest.py > selftest.log 2>&1
```

and, for a performance claim, the four `av1gpu_bench` command lines from
[performance](#performance-numbers-to-collect) with their output. That set plus
the GPU model, the driver version and the backend name is enough to tell a
driver bug from a kernel bug from a slow configuration.

## Recording the result

When a run succeeds, the useful thing to leave behind is one filled row per
machine. Keep it with whatever report you write (an issue, a PR comment, this
file):

| date | GPU | driver | backend | `gpu_verify` | 720p ms/frame | 720p CPU % | 1080p ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

Two rules that make the numbers mean something: compare the CPU share against
`--backend cpu` on the *same* machine, and say which driver version it was —
the same card on two driver branches is two different devices for this purpose.

## What is still unverified here

This repository was developed on a machine with no GPU at all, so:

* the **CUDA backend has never run**. It compiles the same kernel source with
  NVRTC and uses the same job layouts, so it is right by construction, by review
  and by the ABI checks above, not by measurement — `gpu_verify.py --backends
  cuda` is the check that would change that. What the review of that path fixed,
  besides the two ABI mistakes listed above: a device-side fault was never
  reported (`cuCtxSynchronize`'s result was dropped, so a bad launch decoded into
  whatever the buffer held and the frame was returned as if it were right - the
  OpenCL backend already checked its `clFinish`); the coefficient, residual,
  palette, film grain and snapshot copies plus the frame read back were unchecked
  in the same way; a failed allocation in `prepare()` was ignored until it faulted;
  the palette map and the second prefix table were never freed on shutdown; and
  NVRTC's `GetPTXSize`/`GetPTX` results were trusted, so a toolchain that produced
  no PTX produced a `cuModuleLoadDataEx` failure with no explanation instead of a
  message;
* **every timing number in the documentation** was measured on the CPU reference
  backend;
* `av1gpu_play` (the SDL/terminal player) has not been run with a real device
  either.

The OpenCL backend has been run end to end, but only on pocl, a CPU device: that
proves the plumbing, the layouts and the arithmetic, and says nothing about
speed or about any vendor's driver. Running the checklist above on a machine
with a card closes all of it.
