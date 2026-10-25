# Verifying on a real GPU

This tree is built so that everything except the last mile can be checked without
a device: the kernels are one C source that the CPU, OpenCL and CUDA backends all
compile, `tests/opencl_selftest.py` compiles that source with a real OpenCL
compiler and checks a job of every kernel (palette prediction included), and
`tests/opencl_pipeline.py` replays real decode plans through it. All of that
passes on a machine with nothing but PoCL (an OpenCL implementation that runs on
the CPU):

```sh
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install pyopencl pocl-binary-distribution
cmake -S . -B build -DAV1GPU_TEST_PYTHON=/tmp/venv/bin/python
cmake --build build -j4 && ctest --test-dir build
```

`tests/test_gpu_sources.cpp` covers the part of the launch path that neither of
those reaches: the hand written argument lists. It needs no device at all - see
[the launch ABI](#the-launch-abi) below.

What none of it can show is whether a **driver** accepts this backend: the device
queries, buffer allocations, the job layouts the kernels declare, the launch
shapes, and how much CPU time is actually left over while the GPU works. That is
what this page is for, and `tests/gpu_verify.py` is the one command that does it.

## What you need

* A build with the tools:

  ```sh
  cmake -S . -B build -DAV1GPU_BUILD_TOOLS=ON -DAV1GPU_BUILD_TESTS=ON
  cmake --build build -j4
  ./build/bin/av1gpu_info          # has to list your device
  ```

* OpenCL: an ICD loader (`ocl-icd-libopencl1` on Debian/Ubuntu and their
  derivatives) plus the vendor's ICD file in `/etc/OpenCL/vendors` — the NVIDIA
  driver installs `nvidia.icd`, the AMD and Mesa drivers install theirs. The
  backend `dlopen`s `libOpenCL.so.1` at run time, so nothing has to be linked at
  build time. If `av1gpu_info` says "no OpenCL loader found", the loader is
  missing; if it lists no device, the ICD file or the driver is.
* CUDA: the driver's `libcuda.so.1` and the toolkit's `libnvrtc.so` (NVRTC
  compiles the kernel source at run time; no `nvcc` needed at build time).

## One command

```sh
python3 tests/gpu_verify.py
```

It verifies every GPU backend `av1gpu_info` reports (or the ones you name with
`--backends opencl,cuda`), and exits non-zero if anything is off. Under ctest:

```sh
cmake -S . -B build -DAV1GPU_GPU_TESTS=ON -DAV1GPU_TEST_PYTHON=$(which python3)
cmake --build build -j4
ctest --test-dir build -R gpu_verify --output-on-failure
```

`AV1GPU_GPU_TESTS` is off by default because it needs a device; a build without
one has the script exit 77 and ctest reports a skip instead of a failure.

## What it checks

| step | what runs | what has to hold |
| --- | --- | --- |
| 1. synthetic clip | `av1gpu_bench --all` on every backend | the GPU result is **bit identical** to the CPU reference decode, and the ms/frame, CPU load and launch count are printed |
| 2. real streams | `av1gpu_decode --backend <gpu> --require-device` on every fixture | **every shown frame matches the reference decoder's hashes**, so a whole front end plus reconstruction ran on the device |

`--require-device` is the important part. Without it a backend that fails to
initialise silently falls back to the CPU reference, and a verification run would
"pass" while proving nothing; the script refuses a fallback and also refuses a
device whose backend is not the one it asked for. A stream that stops early is a
failure too, not just a mismatch.

Streams come from `tests/data` (two 30 frame streams that code local warped motion
and OBMC); `--fixtures DIR` adds a directory of your own, each `name.ivf` with
either `name.hash` or `name_ref.yuv` next to it.

## Reading the output

A run on a machine with a real GPU should look like this (`opencl` on an actual
card; the numbers here are from PoCL on a 2 core CPU, where the last column is
what matters and the timings are not):

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

2. real streams: 2 stream(s), every temporal unit
   stream         backend  device                             frames ms/frame  status
   lw_t2          cpu      cpu (CPU reference)                 30/30    13.95  reference
   lw_t2          opencl   opencl (pthread-Intel(R) Xeon(      30/30    18.32  bit exact

gpu_verify: all checks passed on opencl0
```

On a CPU OpenCL device the speed ratio is meaningless (it is the CPU, on one
thread per work item); on a real GPU the interesting numbers are **ms/frame** and
**cpu %** — the project exists to keep the second one low while the first one
stays small enough to play video.

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
enabled: a build without CUDA checks only the OpenCL half.

Two more bugs came out of the same review and are fixed:
`ensure()` asked `cuMemAlloc` for zero bytes whenever a frame coded no palette (a
request the driver rejects), which failed the *whole* decode of any ordinary
stream on the CUDA path; and `load_nvrtc`/`load_driver` passed a hard coded name
count to `open_first()`, reading past the end of the Windows list.

## When it fails

| message | meaning | next step |
| --- | --- | --- |
| `no OpenCL loader found` | `libOpenCL.so.1` is not installed | install the ICD loader |
| `device workgroup limit too small` | the device query returned nothing usable | check the driver; the limit is logged in `av1gpu_info` |
| `asked for opencl, got cpu` | the request was refused and fell back | run `av1gpu_bench --backend opencl --frames 1` for the driver's reason |
| `kernel build failed` | NVRTC/the OpenCL compiler rejected the source | rerun with `-v` and keep the compiler log; that is a driver or language version issue; the message names the `--gpu-architecture` that was tried |
| `out of device memory` | an allocation in `prepare()` or `decode()` failed | the frame does not fit in device memory together with its reference pool; the pool is sized from the frame geometry |
| `CUDA kernel execution failed (cuCtxSynchronize = N)` | a kernel faulted or a copy targeted a stale buffer | the driver error number says which; this used to be silent and decoded into whatever the buffer held |
| a frame hash mismatch | the driver computes different pixels | compare with the CPU-only layers above: if `opencl_selftest.py` and `opencl_pipeline.py` pass, the arithmetic is right and the difference is device specific (report which device, driver and backend) |

## Performance notes

`av1gpu_bench --backend opencl --frames 120 --width 1920 --height 1080` is the
quick way to get a playable-resolution number, and `av1gpu_play` shows the same
thing as a picture. Things worth measuring on a real card, in the order they are
likely to matter:

1. **CPU share while the GPU runs** — the whole point. The synthetic clip is CPU
   bound on the parse side by design, so compare against `--backend cpu` on the
   same machine.
2. **Launches per frame** (`launch` column): an intra frame needs one wave per
   block diagonal, which is where `docs/ROADMAP.md` section 5 comes in.
3. **Wave scheduling for inter frames**: they should be a handful of waves.
4. **Device-to-device transfers**: reference frames stay in the pool, so a frame
   should cost no host round trip except the output read back.

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
* every timing number in the documentation was measured on the CPU reference
  backend;
* `av1gpu_play` (the SDL/terminal player) has not been run with a real device
  either.

Running `python3 tests/gpu_verify.py -v` on a GPU machine and keeping the output
closes all three.
