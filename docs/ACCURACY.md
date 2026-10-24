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
* **The boolean range decoder.** `src/frontend/range_decoder.cpp` is a port of
  the reference algorithm (dav1d's `msac`, which is the same arithmetic as
  libaom's `od_ec`): a 64 bit window over complemented input bytes, a range
  renormalised into [2^15, 2^16), `EC_PROB_SHIFT` 6 and `EC_MIN_PROB` 4, and a
  tail that pads with one bits once the tile data runs out. There is no
  tolerance knob here on purpose: the tile payload is either decoded exactly
  like a reference decoder or it desynchronises within a few symbols.
* **Scheduling.** The wavefront order is a scheduling detail only: the tests
  require the output to be independent of `dependency_cell`, and it is.
* **The chroma of a sub-8x8 block is predicted per luma cell.** In 4:2:0 the
  chroma block of a luma block that is one 4x4 wide (or high) covers two luma
  blocks in that direction, and each of them predicts its half with its own
  vector, reference frames and interpolation filters
  (`compute_prediction()`: regions of `Block_Width >> subX` by
  `Block_Height >> subY` samples, taken from the cells at `candRow`/`candCol`).
  `reconstruct` therefore emits one prediction block per region and hangs the
  plane's transforms off the first one. A single prediction block per plane -
  the obvious shape, since a block's motion vector is a block property - is
  wrong by exactly one 4x4 chroma cell per sub-8x8 block, which is what the
  `wg` and `w_cu4` fixtures used to show.

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

The front-end has its own cross-checks. Two of them are one-off development
experiments, and one is part of the test suite:

* `src/frontend/headers.cpp` was compared against dav1d's sequence header parser
  over 4000 random byte strings. Every header dav1d accepts parsed to the same
  field values at the same bit position.
* A real stream (`tests/data/svt_av1_320x240.ivf`, six frames from SVT-AV1) has
  all eight of its frame headers cross-checked against FFmpeg's `trace_headers`
  bitstream filter, which prints every syntax element with its bit position.
  Quantiser, deblock levels, CDEF strengths, reference slots, order hints,
  refresh flags, skip mode and tile data offsets all agree, including the four
  inter frames and both `show_existing_frame` OBUs.
* `tests/test_frontend.cpp` parses that same stream end to end on every run, so
  the fixture keeps the parser honest as it grows.

The dav1d comparison found two real bugs:

* profile 2 with 8/10 bit is 4:2:2 signalled implicitly; the parser defaulted
  `subsampling_y` to 1, read two chroma bits that were not there, and lost sync
  from that point on,
* the sequence header ends with `trailing_bits()` (a one bit, then zeros), not
  with plain zero padding; being strict about it is what catches a desync.

The only disagreements left are cases where dav1d's *strict* mode additionally
requires the header to end exactly on a byte boundary; the default dav1d accepts
them the same way this parser does.

### The entropy decoder

The range decoder is checked the same way, and it had to be, because a range
coder cannot be tested by round trip: an encoder that is wrong in exactly the
same way as the decoder passes every round trip and still cannot read a real
stream. Instead:

* A differential harness decoded identical payloads with dav1d's `msac` and with
  this decoder: random but conforming Q15 tables, alphabets from 2 to 16
  symbols, adaptation on, 192,802 symbols and 120,000 boolean/equiprobable
  reads over 200 runs. **Zero mismatches**, and not just in the symbol values:
  the adapted CDF was compared entry by entry after every trial, because that
  table, not the symbol, is what keeps the next block in sync. The same harness
  also covers the "no symbol fits" case where the padding dominates.
* `tests/entropy_vectors.hpp` freezes that result. It holds six symbol streams
  (2, 3, 4, 8, 11 and 16 symbols, payloads picked so every symbol is exercised),
  the tables they adapt to, plus boolean, equiprobable, `L(n)` and `ns(n)`
  payloads. The expected values were produced by dav1d's `msac`, and
  `tests/test_entropy.cpp` re-decodes them on every run.
* `disable_cdf_update`, the overflow of the adaptation counter at 32, and the
  end-of-tile contract (`overrun()` becomes true, decoding stays in the
  alphabet, `bytes_used()` never passes the tile size) are unit tested directly.
* **A real tile decodes bit exactly.** That was the missing piece, and it is now
  measured rather than argued: dav1d's `msac` was patched to log every decode it
  performs (symbol or bit, the value, the exact CDF it used, and its window,
  range and counter afterwards) while decoding the key frame of
  `tests/data/svt_av1_320x240.ivf`. Replaying that log through AV1onGPU's
  `RangeDecoder` reproduces dav1d for **all 112,648 events** of the tile:
  78,090 symbols, 2,173 boolean reads and 32,385 equiprobable bits, with the
  range coder state (`dif`, `rng`, `cnt`) identical at every event. The first
  480 events are frozen into `tests/data/tile_trace.hpp` and replayed by
  `tests/test_entropy.cpp`, including the final state, so a regression in the
  window handling, the probability shift or the adaptation is caught without
  dav1d being present.

  Two things this found, both invisible to round-trip testing:

  * The last symbol of a table has no stored probability. Its interval is
    whatever range is left, and the reference decoders select it by walking one
    entry past the probabilities into the adaptation counter, which is at most
    32 and therefore contributes no range at the 6 bit probability resolution.
    **12,245 of the 78,090 symbols in that frame - one in seven - are the last
    symbol**, so treating "walked past the table" as corruption, as this decoder
    originally did, cannot decode a real tile at all.
  * The specification's prose gives the adaptation rate as
    `3 + (cdf[N] > 15) + (cdf[N] > 31) + Min(FloorLog2(N), 2)` while dav1d and
    libaom both use `4 + (cdf[N] >> 4) + (N > 2)`. The two agree for alphabets of
    2, 4, 8 and 16 symbols and differ for 3, 5, 6, 7, 9-15. AV1onGPU implements
    the reference decoder formula, because the reference decoders are what real
    encoders target and what a stream has to be decoded with.

* **The default tables are checked against a real stream too.** The first symbol
  of that tile is a 64x64 partition, so dav1d's recorded table for it has to
  equal `init_default_cdf()`'s row for block level 64x64, context 0, entry for
  entry - and decoding the symbol out of *our* table has to return the same
  value and leave the same window/range/counter. That one symbol pins the table
  convention (`32768 - x`, counter after the probabilities, symbol counts per
  block level) before any syntax exists to get confused by.

* **The syntax layer is checked against the same trace.** The first block of
  that frame is decoded end to end by `decode_intra_tile`: a 64x64 split, then a
  32x32 block with skip 0, a quantiser delta of 2 (51 -> 53), y and chroma mode
  SMOOTH_VERTICAL and a 32x32 transform - exactly the sequence dav1d's trace
  shows for those bytes, in the same order. `tests/test_frontend.cpp` asserts
  it, and asserts that parsing stops at the coefficient syntax rather than
  inventing values for it.

Until the coefficient and inter layers land (see `docs/ROADMAP.md`), the
front-end can read the tile bytes but not yet interpret them; the trace replay
above is what makes "the symbol reader is right" a demonstrated fact rather than
a claim.

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
