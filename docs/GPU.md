# Verifying on a real GPU

This tree is built so that everything except the last mile can be checked without
a device: the kernels are one C source that the CPU, OpenCL and CUDA backends all
compile, `tests/opencl_selftest.py` compiles that source with a real OpenCL
compiler and checks a job of every kernel, and `tests/opencl_pipeline.py` replays
real decode plans through it. All of that passes on a machine with nothing but
PoCL (an OpenCL implementation that runs on the CPU).

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

## When it fails

| message | meaning | next step |
| --- | --- | --- |
| `no OpenCL loader found` | `libOpenCL.so.1` is not installed | install the ICD loader |
| `device workgroup limit too small` | the device query returned nothing usable | check the driver; the limit is logged in `av1gpu_info` |
| `asked for opencl, got cpu` | the request was refused and fell back | run `av1gpu_bench --backend opencl --frames 1` for the driver's reason |
| `kernel build failed` | NVRTC/the OpenCL compiler rejected the source | rerun with `-v` and keep the compiler log; that is a driver or language version issue |
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
  NVRTC and uses the same job layouts, so it is right by construction and by
  review, not by measurement — `gpu_verify.py --backends cuda` is the check that
  would change that;
* every timing number in the documentation was measured on the CPU reference
  backend;
* `av1gpu_play` (the SDL/terminal player) has not been run with a real device
  either.

Running `python3 tests/gpu_verify.py -v` on a GPU machine and keeping the output
closes all three.
