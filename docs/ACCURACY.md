# Accuracy notes

This file records, honestly, where AV1onGPU follows the AV1 specification bit
exactly and where it deliberately approximates it. The guiding rule is that
**every backend is bit exact against every other backend**: the GPU result and
the CPU reference result are identical, pixel for pixel, because both compile
the same scalar code in `kernels/av1gpu_kernels.h`. The deviations below are
therefore deviations from the *specification*, applied consistently everywhere,
not disagreements between backends.

## Bit exact, by construction

* **Inverse transform architecture.** Two separable passes of
  `out[i] = (Σ_k in[k]·B[i][k] + 2048) >> 12` with a Q12 synthesis basis, giving
  the same intermediate rounding structure as the spec's staged butterflies.
* **Residual add and clipping.** `clip(pred + residual, 0, maxv)`, in the
  pipeline order: prediction, then transform, then loop filters.
* **Subpel interpolation arithmetic.** Q7 taps, two separable passes, each
  rounded `(acc + 64) >> 7`, the second pass normalised by the filter's DC gain
  so that flat areas stay flat.
* **Deblocking structure.** The job carries `level`, `sharpness`, tap length
  (4/6/8), plane and direction; the filter only ever moves samples towards each
  other and is a no-op on flat areas (checked by the unit tests). Vertical edges
  are filtered before horizontal ones, as the specification orders them, because
  a vertical and a horizontal edge cross: running both at once is a data race,
  and it made `test_pipeline` fail roughly one run in five until the two passes
  were separated.
* **Film grain application.** Per 128×128 tile, per luma bucket LUT, added on
  top of the reconstructed picture with clipping.
* **Colour conversion.** One integer matrix per (matrix, range, bit depth),
  Q16 coefficients with the level/range offsets folded into the constant term,
  so limited-range black is (0,0,0) and limited-range white is 255.
* **Scheduling.** The wavefront order is a scheduling detail only: the tests
  require the output to be independent of `dependency_cell`, and it is.

## Approximations (deliberate, documented, tested)

* **Transform bases are derived, not copied.** The DCT-II and DST-IV matrices
  are generated from their definitions in Q12 (`src/core/tables.cpp`). They are
  orthonormal to within 3.3e-4, which is inside the Q12 rounding of the
  transform itself, so a round trip is exact to ±2 LSB over the full sample
  range (asserted in `test_kernels`). Because the bases are derived instead of
  copied, every size the spec defines exists (4..64 for DCT, 4..16 for DST-IV)
  and no basis table in the source is magic data.
* **Subpel filters use a windowed sinc, not the AV1 tap tables.** The spec
  ships four fixed 8-tap filters (regular, smooth, sharp, bilinear) sampled at
  16 phases; this implementation generates Lanczos-windowed sinc filters at
  **1/8 pel precision** (8 phases) and normalises each phase to a DC gain of
  128. Consequences: chroma/luma interpolation is smoother than the spec's
  sharpest filters, motion vectors are quantised to 1/8 pel, and a reference
  decoder fed a real AV1 stream would produce slightly different pixels. The
  filter generator is a stand-in, not a conformance implementation — replacing
  it with the spec tables is a self-contained change in one function.
* **CDEF weights match the spec, its strength selection does not.** The tap
  weights (primary 4/2, secondary 2/1, 8 directions, secondary directions at
  ±2 and ±6 from the primary, thresholds halved for the far tap, `(sum + 8) >> 4`
  and the `min`/`max` clamp over the neighbourhood) follow the spec. What the
  library does not do is *choose* the strengths/damping: those arrive in the
  syntax. The synthetic encoder picks them from the quantiser step rather than
  by a rate/distortion search, so enabling CDEF there lowers PSNR against the
  clean source (see the measurements below) — that is the synthetic encoder's
  choice, not a filter defect.
* **Film grain template.** The AV1 grain model is a large AR filter with
  per-plane coefficients and overlap blending driven by an LFSR. This
  implementation generates a zero-mean triangular noise template from a
  seeded xorshift (`tables::grain_template`), scaled per luma bucket through the
  32-entry LUT, tiled at 128×128 with optional overlap. Grain therefore *looks*
  right and costs the right amount of work, but it is not bitstream compatible.
* **Intra predictors.** DC, V, H, Paeth and bilinear-style smooth/V/H follow
  their spec definitions. `av1g_smooth_w(i, n) = ((i + 1) * 256) / (n + 1)` is a
  synthesised approximation of the spec's smooth-mode weight table. Chroma-from-
  luma is simplified: the DC part of the luma AC contribution is subtracted
  (`aux->ac_dc`) and the alpha is applied as `(alpha * ac) >> 3`, without the
  spec's per-block scaling and sign handling of the full CfL search.
* **Compound prediction.** Average and distance-weighted blending are
  implemented with Q6 weights; the spec's mask-based (wedge/difference) modes
  are not.

## Closed loop invariant

An encoder that drives this decoder **must** apply the same loop filters to its
own reconstruction, or every following inter frame drifts (measured: 9–13 dB
PSNR loss after five frames when the filters were skipped). `synthetic_encode`
does this through `internal::apply_post_filters_host`, the same job builder the
decoder uses. `test_pipeline` pins the property: the decoder output and the
encoder's reconstruction agree to 99.00 dB — i.e. they are identical — for every
frame of the clip, with deblock, CDEF and grain enabled.

## How the GPU path is checked without a GPU

Everything above is only worth as much as the evidence behind it, so there are
three layers, all of them runnable on a machine with no GPU driver:

1. `test_kernels` / `test_pipeline` — the shared scalar code on the host, and a
   closed loop decode through the CPU backend.
2. `tests/opencl_selftest.py` — compiles `kernels/av1_opencl.cl` with a real
   OpenCL compiler (pocl) and checks one job of every kernel.
3. `tests/opencl_pipeline.py` — `av1gpu_dump` writes the real decode plans of a
   clip, the test replays *every* job of *every* wave through the real kernels in
   backend order (waves, deblock vertical, deblock horizontal, CDEF, grain) and
   compares whole frames against the CPU reference decode. Four frames, intra
   and inter, all filters on: **bit identical**.

The third layer is what found the deblocking group index bug: the kernel was
handed a group number where the shared helper expects a sample offset, so edges
longer than four samples were filtered in the wrong place. A flat test pattern
cannot see that, and the per-kernel check had used one; the frame level test with
a real edge and several groups does, and `opencl_selftest.py` now includes that
case as well.

## Measured quality (synthetic clip, 640×360, CPU reference backend)

PSNR of the decoded picture against the *clean source*, frames 1–11:

| qstep | deblock | + CDEF | + grain | + CDEF + grain |
| --- | --- | --- | --- | --- |
| 4 | 45.62 dB | 42.53 dB | 40.68 dB | 39.56 dB |
| 16 | 42.00 dB | 34.81 dB | — | — |

* Decoder output vs encoder reconstruction: **99.00 dB (identical)** in every
  configuration above.
* The CDEF and grain columns are lower because the synthetic encoder enables
  them at a fixed strength instead of choosing where they help, and because
  grain is noise added to a clean source — PSNR is expected to drop when grain
  is on. They exist to exercise the filter path, not to model a codec's
  rate/distortion decisions.
* At qstep 4 the deblock-only result is 45.6 dB with a **1.9 dB** gap between
  the worst and best frame, i.e. no visible drift over the clip.

## Precision and portability notes

* Samples are stored as 16 bit unsigned in every backend regardless of bit
  depth, so 8/10/12 bit share one kernel build and one buffer layout.
* The transform and prediction accumulation uses `long long` for the dot
  products; a 64×64 DCT with 12 bit input cannot overflow it.
* The kernels never rely on cross work item communication: jobs are independent
  inside a wave by construction, which is why the same code is correct on a CPU
  thread pool, on a GPU with no shuffles and on a GPU with a 64×64 workgroup
  limit alike.
* OpenCL C does not allow a private pointer to global memory, so every helper
  that reads a buffer through a pointer takes an explicitly qualified
  `AV1G_GLOBAL` parameter (see `kernels/av1gpu_kernels.h`). CDEF deliberately
  does not stage through `__local` memory.
