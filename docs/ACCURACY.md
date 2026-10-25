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
* **Subpel interpolation arithmetic.** `Subpel_Filters` and `Warped_Filters`
  are the specification's tables (Q7, 16 phases for the subpel filters and 193
  offset rows for the warp), not approximations of them. Two separable passes,
  each rounded `(acc + 64) >> 7`, the second pass normalised by the filter's DC
  gain so that flat areas stay flat, with the specification's InterRound0 /
  InterRound1 / InterPostRound shifts.
* **Deblocking structure.** The job carries `level`, `sharpness`, tap length
  (4/6/8), plane and direction; the filter only ever moves samples towards each
  other and is a no-op on flat areas (checked by the unit tests). Vertical edges
  are filtered before horizontal ones, as the specification orders them, because
  a vertical and a horizontal edge cross: running both at once is a data race,
  and it made `test_pipeline` fail roughly one run in five until the two passes
  were separated.
* **When the loop filter runs at all.** `decode_frame_wrapup` only invokes it
  when the frame coded `loop_filter_level[0]` or `[1]` non-zero. A frame that
  codes 0 and 0 is left unfiltered even though the per-reference and per-mode
  deltas are enabled and would scale a level of 0 into something a filter would
  act on, so a decoder that applies the deltas unconditionally filters edges the
  reference decoders leave alone. It is a small error - a few thousand samples
  off by one or two - and it only appears on streams where the encoder codes no
  level at all, which is why it survived until a 10 bit libaom clip was compared:
  `tests/data/aom_10bit_320x240.ivf`'s first frame is exactly that case and the
  fixture is checked sample for sample.
* **Bit depth and chroma.** The thresholds scale with `1 << (BitDepth - 8)`
  (7.14.6), the reconstruction carries more headroom as the bit depth rises, and
  a 4:4:4 frame's chroma planes are as large as its luma one. Each of those is
  pinned by its own fixture - 10 bit 4:2:0, 10 bit 4:4:4 and 12 bit 4:2:0, all
  three from libaom and all three sample for sample equal to dav1d's, on the CPU
  backend and through the OpenCL kernels. The 4:4:4 one is also the only test
  that reaches `frame_to_yuv8` with planes that are not 4:2:0 sized, where the
  8 bit output path used to size its buffer as `width * height * 3 / 2` and run
  off the end of it.
* **Film grain synthesis.** The generator, the auto-regressive filter and the
  overlap blending are the specification's, sample for sample, including the
  chroma coupling that reads the luma grain (7.18.3); the blend is one multiply,
  one `Round2` and one `Clip3` per sample. The sequential half runs on the host
  because neither the AR filter nor the stripe stitching parallelises across
  samples, and the per sample map runs in the kernels, so every backend adds the
  same noise. Chroma is filtered before luma because 7.18.3.6 reads `OutY` as it
  was *before* the luma grain went in, and the noise image is carried as 16 bit
  samples: the grain range is `1 << (BitDepth - 1)` either side of zero, which
  stops fitting in a byte at 10 bit (plan format 6). `tests/data/grain.ivf` pins
  the coded form against dav1d: 30 frames, every sample equal.
* **Inherited film grain parameters.** An inter frame may code `update_grain = 0`
  and name a reference slot instead of a parameter set (5.9.30); 7.18.3.1 then
  loads that slot's copy - the parameters the frame that filled the slot decoded
  with - and keeps only the new `grain_seed`, so the noise differs while the
  curve does not. The syntax has a conformance requirement that the coded index
  is one of the frame's own `ref_frame_idx[]`, and dav1d rejects a frame that
  breaks it (it is a header parse error), so `make_grain.py --inherit-from`
  takes the slot from the probe's trace rather than guessing. The slot's stored
  parameters are per *output order* slot: a frame whose `apply_grain` is 0
  clears them, i.e. a slot filled by a grained frame hands grain on and a slot
  filled by a plain frame hands nothing on. `tests/data/grain_inherit.ivf` pins
  that path: 10 frames, every sample equal to dav1d's.
* **NV12 output.** The pack that turns the planar picture into one interleaved
  buffer is the same scalar function everywhere (`av1g_nv12_px`, one output byte
  per work item), so `Decoder::to_nv12` on a device and `frame_to_nv12` on the
  host write the same bytes - padding included, and the padding of a row is
  written as zero rather than left as whatever the surface held. The pitch is the
  caller's (a VA surface's is not the picture width), a frame that is not 4:2:0
  is refused instead of truncated, and a stream deeper than 8 bits is shifted
  down because NV12 is 8 bit. `test_kernels` checks the interleaving, the
  padding, the odd height (a 5 row frame has 3 chroma rows), the 10 bit shift and
  the three refusals against hand computed bytes; `test_pipeline` decodes a frame
  that carries film grain - so the pack has to read the picture *after* 7.20 -
  and compares the backend's bytes with the host function's, on every backend
  this machine has.
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
* **Scheduling.** The wavefront order is a scheduling detail only, and the
  output must not depend on it: two decodes of the same syntax have to produce
  the same picture (and the same number of waves). The schedule itself is exact
  sample ownership rather than a grid: every sample is owned by the block that
  writes it, over the coded extent, and a block waits for the owners of the
  samples its prediction reads - the plain left column and above row always, the
  second block of either only when the mode reads it (a directional block reads
  above samples two blocks wide below 90 degrees and left samples that far above
  180; DC, the smooths, Paeth, filter intra and chroma from luma read the plain
  edges, and a palette block reads no edge at all). All three planes share one
  dependency grid, so an intra frame costs the length of its diagonal instead of
  one diagonal per plane. Measured on the fixtures, as the largest number of
  waves in a frame:

  | fixture | before | after |
  | --- | --- | --- |
  | `lossless` (160x120, 4 intra frames) | 201 | 77 |
  | `lr` (320x240, one intra frame) | 250 | 118 |
  | `odd_size` (130x100) | 96 | 47 |
  | `inter_intra` (320x240) | 142 | 105 |
  | `intrabc` | 89 | 38 |
  | `palette_sc` | 102 | 31 |
  | `lw_t2` | 209 | 125 |

  What each part of that is worth, measured on `lr`'s intra frame: gating the
  second block on the mode is 7 waves (125 -> 118), the palette skip and the
  corner guard change nothing on these fixtures (they still do not wait for
  anything they do not read), and the rest of the drop from 250 comes from
  dropping the coarse per-cell dependency the schedule used before, which waited
  for whole cells instead of the samples the predictor reads.

  Every fixture stays bit exact through the CPU backend and through the OpenCL
  replay, which runs the jobs of a wave as separate dispatches with no ordering
  between them - a dependency that was dropped by mistake shows up there as a
  wrong sample, not as a slow frame.
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
  range (asserted in `test_kernels`). Where that bites is content far from the
  fixtures: a libaom `testsrc2` encode of one pixel lines and hard edges puts
  400 to 800 luma samples per frame 1-2 LSB away from dav1d while *every* range
  coder event of all 30 frames still matches (251,289 of them, see the
  segmentation section), so the syntax is exact and the difference is this
  rounding. That is why only streams that decode exactly are promoted to
  fixtures. Because the bases are derived instead of
  copied, every size the spec defines exists (4..64 for DCT, 4..16 for DST-IV)
  and no basis table in the source is magic data.
* **CDEF weights match the spec, its strength selection does not.** The tap
  weights (primary 4/2, secondary 2/1, 8 directions, secondary directions at
  ±2 and ±6 from the primary, thresholds halved for the far tap, `(sum + 8) >> 4`
  and the `min`/`max` clamp over the neighbourhood) follow the spec. What the
  library does not do is *choose* the strengths/damping: those arrive in the
  syntax. The synthetic encoder picks them from the quantiser step rather than
  by a rate/distortion search, so enabling CDEF there lowers PSNR against the
  clean source (see the measurements below) — that is the synthetic encoder's
  choice, not a filter defect.
* **Intra predictors.** DC, V, H, Paeth and bilinear-style smooth/V/H follow
  their spec definitions. `av1g_smooth_w(i, n) = ((i + 1) * 256) / (n + 1)` is a
  synthesised approximation of the spec's smooth-mode weight table. Chroma-from-
  luma is simplified: the DC part of the luma AC contribution is subtracted
  (`aux->ac_dc`) and the alpha is applied as `(alpha * ac) >> 3`, without the
  spec's per-block scaling and sign handling of the full CfL search.
* **Compound prediction.** Average, distance weighted, wedge and difference
  weighted blending are implemented. The mask-based modes follow 7.11.3.11-14:
  the wedge mask selects an entry of the master table per block shape and
  applies the specification's flipSign rule, the difference mask is
  `Clip3(0, 64, 38 + Round2(abs(pred0 - pred1), (BitDepth - 8) + InterPostRound)
  / 16)`, and both are evaluated on the luma grid and averaged down for chroma
  samples. Warped motion rides the same blend: global (`useWarp` per reference
  list, 7.11.3.5-7) when the frame's model is steeper than a translation, and
  local (7.11.3.8, the least squares fit to the neighbours, in
  `src/frontend/warp.hpp`) when the block's `motion_mode` says so - in both
  cases only for a prediction region of at least 8x8, and in both cases through
  the same setup shear and kernel.

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
   OpenCL compiler (pocl) and checks one job of every kernel. That now includes
   the two filters that had only ever been checked through a whole-frame replay:
   a Wiener job, a self guided job at 8 bit and the same job at 10 bit (where
   the box filter's sums are scaled back to 8 bit instead of being used as they
   are), and one superres upscale job - each compared with the host build of the
   same kernel, sample for sample.
3. `tests/opencl_pipeline.py` — `av1gpu_dump` writes the real decode plans of a
   clip, the test replays *every* job of *every* wave through the real kernels in
   backend order (waves, deblock vertical, deblock horizontal, CDEF, upscale,
   loop restoration, grain) and compares whole frames against the CPU reference
   decode. Four frames, intra
   and inter, all filters on: **bit identical**. With `--input` the plans come
   from `av1gpu_decode --dump-plans` for a real stream instead, so the reference
   pool slots are the decoder's own: `lw_t2` and `lw_t3` (local warped motion and
   OBMC, 30 frames each), `palette_sc` (102 palette blocks, 10 frames),
   `inter_intra` (23 inter/intra compound blocks, 110 frames), `seg_altq`
   (two quantiser classes), `superres` (320 columns coded, 480 displayed), `lr`
   (Wiener and SGR on both luma and chroma, three of its frames switchable) and
   `svt_av1_320x240` (a different encoder, 6 frames) are all **bit identical**
   too. `superres` is the one where the replay needs the plan to carry the
   display geometry as well as the coded one: the upscale job reads the coded
   frame and writes the displayed one, so the test snapshots the former before
   the job runs.

   The replay sizes every buffer from the plan, and it checks the plan before it
   starts: each inverse transform job needs `w * h` ints of scratch at its
   `tmp_off`, which is *not* the size of its coefficient area (only the low
   32x32 quarter of a 64 point transform is coded, so the coefficient raster is
   clipped to `Min(32, w)` wide). Sizing the scratch buffer from the coefficient
   pool therefore writes past the end of it, which on a CPU OpenCL
   implementation is the host heap - that bug showed up as a malloc abort a
   frame later, not as a mismatch. `plan_dump.hpp` writes the real `tmp_size`
   (plan format 3) and `check_buffers()` fails the test if any job reaches past
   its pool.

   Film grain is where the plan turned into more than job arrays. The noise image
   and the three 256 entry scaling curves are built on the host (plan format 5
   carries them as `noise`/`grain_lut` with a per plane offset and stride; format
   6 widens the noise samples from one byte to two, so a 10 or 12 bit stream's
   grain is not clipped by the container),
   because the generator and the auto-regressive filter are sequential; the
   replay uploads them and runs the same two passes the backends do, chroma
   first. It also has to copy a frame into its reference pool *before* those
   passes - 7.20 stores the decoded picture, not the displayed one - or every
   inter frame after the first comes back different.

To those a fourth layer can be added where a software OpenCL device exists: the
C++ backend itself, not just the kernel source. PoCL (`pip install pyopencl
pocl-binary-distribution`, plus any ICD loader) exposes the host CPU as an
OpenCL device, and

```sh
av1gpu_decode --backend opencl --frames 30 tests/data/lw_t2.ivf \
    --hash-check tests/data/lw_t2.hash
```

then checks the whole backend — context, buffers, job upload, launch order,
reference copies — against a reference decoder's pictures. `tests/gpu_verify.py`
does that for every backend and every fixture in one command, with the CPU
fallback disabled so a run that quietly used the CPU cannot pass; `docs/GPU.md`
is the page to hand to whoever has the GPU. Every stream in the
fixture set decodes bit exactly that way on a two-core PoCL device (14 streams),
and `av1gpu_bench` reports the same PSNR for `--backend cpu` and `--backend
opencl`. This is an arithmetic and plumbing check, not a performance one: a CPU
OpenCL device runs the kernels on the CPU.

Running that layer for the first time found two bugs the kernel tests could not
see, because nothing in this tree had ever created a device:

* `CL_DEVICE_MAX_WORK_GROUP_SIZE` is a `size_t` and was queried into a `cl_uint`.
  PoCL answers `CL_INVALID_VALUE` for a four byte buffer, which read as a limit
  of zero and rejected a perfectly working device with "device workgroup limit
  too small". The query now uses `size_t`, as the API specifies.
* The decode path copied a finished picture into its reference slot with the
  *slot* offset given as the source offset: `clEnqueueCopyBuffer(frame_,
  ref_pool_, slot * size, 0, size, ...)`. Slot 0 worked by accident; every other
  slot asked the driver to read past the end of `frame_` and failed with
  "reference copy failed", so the second reference of a stream could not be kept.

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

### Whole frames against a reference decoder

The end-to-end check is a set of libaom encodes decoded by both this decoder and
dav1d and compared frame by frame, in coding order, on all three planes. Every
one of them is **bit exact, every frame**, as of the inter prediction work in
`docs/ROADMAP.md` section 3. (`tests/data/svt_av1_320x240.ivf` is used for the
*parser* cross-checks above, not for pixel comparison: it is an older encode
whose frame count and reference handling are not what the fixture set below
was built for.)

| fixture | frames | what it exercises |
| --- | --- | --- |
| `q5`, `xc5`, `xc6`, `xc7`, `s`, `xt` | 6 each | general intra and inter streams |
| `t_4_32`, `t_8_32` | 15 each | transform shapes, masked compound |
| `ab` | 12 | masked compound (wedge and difference) |
| `wg` | 15 | the chroma of sub-8x8 blocks |
| `w_cu4` | 30 | global motion, sub-8x8 chroma |
| `w_cu2` | 30 | ROTZOOM global motion, a difference mask on chroma |
| `lwn2`, `lwn3` | 30 each | local warped motion (`motion_mode = LOCALWARP`) |
| `lw_t2`, `lw_t3` | 30 each | local warped motion **and** OBMC (in `tests/data/`) |
| `palette_sc` | 10 | palette coded blocks (screen content, in `tests/data/`) |
| `inter_intra` | 110 | inter/intra compound: the taper and wedge masks (in `tests/data/`) |
| `intrabc` | 16 | intra block copy, as all-key-frame screen content (in `tests/data/`) |
| `seg_altq` | 30 | segmentation, a different quantiser per segment |
| `superres` | 30 | a picture coded narrower than it displays (upscaling, scaled prediction) |
| `grain`, `grain_inherit` | 30, 10 | film grain, coded per frame and inherited from a reference |
| `lr`, `lr_10bit`, `lr_12bit` | 8 each | loop restoration at 8, 10 and 12 bit |
| `lr_444`, `lr_422` | 8 each | loop restoration on chroma planes that are not 4:2:0 |
| `lossless` | 4 | `all_lossless`: no filter parameters in the header, no filtering |
| `odd_size` | 6 | a picture that is not a multiple of eight in either direction |
| `edge_grid` | 4 | a picture whose planes end inside the loop filter's four sample grid |

The four local-warp streams are the ROTZOOM clip re-encoded with libaom at
`--cpu-used=2` and `--cpu-used=3`, `--crf=30`, `--lag-in-frames=25`,
`--enable-restoration=0 --tiles=1x1`, the `lwn*` pair with `--enable-obmc=0` and
the `lw_t*` pair with OBMC left on: they code about 60, 140, 64 and 139
local-warp blocks respectively, and the `lw_t*` pair codes 4x4-unit-aligned OBMC
neighbours on top of that. They are the only streams in the set that code
`motion_mode` as anything but `SIMPLE` - which is why the feature went
unimplemented for so long without anything failing.

`tests/data/lw_t2.ivf` and `tests/data/lw_t3.ivf` are the first two streams that
are kept in the repository for a *pixel* check rather than a parser check,
because a full YUV reference would be 3.4 MB per stream for 30 frames of 320x240.
`av1gpu_decode --hash-out F` writes one FNV-1a 64 hash of each shown frame's
8 bit planar output in output order, and `--hash-check F` decodes and compares
against such a list, naming the first frame that differs and exiting non zero;
`ctest` runs exactly that (`stream_lw_t2`, `stream_lw_t3`). The recorded values
come from dav1d's output for the same streams, so they are a reference decoder's
pictures and not this decoder's. Regenerate one with

```
./build/bin/av1gpu_decode tests/data/lw_t2.ivf --hash-out lw_t2.hash
```

but only after checking the stream against dav1d, or the file would record our
own mistakes.

## Inter/intra compound

An inter block of 8x8 to 32x32 may blend its inter prediction with an intra one
(7.11.4). Both halves are arithmetic, not approximation, and the CPU and GPU
backends share the code:

* **The intra half is predicted first**, over the block's full size, from the
  *reconstructed* neighbours of the frame being decoded - the same edges an
  intra block of that size reads, and only the plain ones: an inter/intra block
  cannot use a directional or recursive mode, so the above-right, below-left and
  corner extensions never affect it.
* **The mask** is `Ii_Weights_1d[ i * sizeScale ]` with
  `sizeScale = MAX_SB_SIZE / Max(w, h)`, indexed by the row (`II_V`), the column
  (`II_H`) or `Min(i, j)` (`II_SMOOTH`), and the constant 32 for `II_DC` - the
  specification writes that one as a number rather than as a table entry, which
  is a real trap: indexing the table at 64/2 gives 32 only by coincidence of the
  rounding, and not at every size. With `wedge_interintra` the mask is the
  codebook's wedge instead, with a sign of zero because the syntax codes none.
* **The mask is built per plane.** A chroma plane uses `Mask[y][x]` of an array
  built over the *plane's* block size, so its `sizeScale` is
  `128 / Max(w >> subX, h >> subY)`; it is not the luma mask averaged down 2x2,
  which is the wedge's rule (`interintra && !wedge_interintra` picks the plane's
  own array in 7.11.3.14).
* **The blend is `Round2(m * intra + (64 - m) * inter, 6)`** with the inter half
  clipped and rounded at `InterPostRound` first (zero for a single 8 bit
  reference), so the pair is rounded once. Scheduling counts an inter/intra
  block with the intra ones, because its intra half needs its neighbours.

`tests/data/inter_intra.ivf` is the fixture: 23 inter/intra blocks over 110
frames, `interintra_mode` in its V, H and smooth variants, five of them wedge
masks (indices 4, 8, 12), on 16x16 up to 32x32 blocks. Every frame is bit exact
against dav1d and through the device kernels. The DC mode does not appear in any
encoding of that clip, so `test_kernels` pins all four mask rules and the chroma
sampling directly.

`tests/data/palette_sc.ivf` is the screen content fixture: a synthetic desktop
style clip (flat colour panels, thin lines, text-like glyph runs, chroma flat
areas) encoded with libaom `tune-content=screen --enable-intrabc=0`, so every
intra block that wants one codes a *palette* instead of a prediction mode. It is
10 frames of 320x240, 6.7 KB, and it codes palettes on both planes: luma-only,
luma+chroma, and 2, 3 and 4 colour palettes with cache reuse from the block
above and to the left. Intra block copy is switched off there on purpose - it
would take blocks away from the palette - and the intrabc fixture below is the
one that codes it; the palette stream is the shortest thing that walks the whole
palette syntax - `palette_mode_info`, `palette_tokens`, the colour map's three
neighbour contexts, and the neighbour palette cache.

`tests/data/intrabc.ivf` is the intra block copy fixture: 16 frames of 320x240,
6 KB, all intra (`-g 1`) and encoded with libaom `tune-content=screen
--enable-intrabc=1 --enable-restoration=0 --enable-cdef=1`, so every frame is a
key frame and intrabc is available on all of them. It codes **239 intra block
copy blocks** over the 16 frames, from 8x32 to 32x32: 204 take a neighbour's
vector off the candidate stack and 35 fall back to the "one superblock up"
offset, and the coded vectors end up on four distinct displacements (-40, -32,
-65/-64 and -49/-64 samples), so both the residual path and the clip path run.
Every frame is bit exact against dav1d and through the device kernels.

What makes intrabc unlike every other tool is that it is an *inter* block in an
*intra* frame:

* **Its motion vector context is its own.** 7.11.3.7 codes the vector against
  `mv_ctx = 1` for an intra block copy block and 0 for everything else, and the
  two contexts adapt separately. Both start from the same default table, so a
  stream that codes no block copy decodes identically either way - and a stream
  that codes one desynchronises at its very first vector if the second context is
  missing. `tests/test_entropy.cpp` pins both halves: the tables start equal, and
  reading a symbol through one context leaves the other untouched.
* **`force_integer_mv` is forced on.** `FrameIsIntra` sets it, which also means
  `allow_high_precision_mv` is never read, so the vector has no fractional or
  half pel symbols: `read_mv_component` is called with `mv_prec = -1` and reads
  the sign, the class, `class0_bit` and either `class0_fr` or the class N bits -
  never `class0_hp` or `hp`.
* **The vector is clipped, not trusted.** 7.11.5 moves the source rectangle until
  it lies inside the decoded part of the tile and outside the superblock being
  decoded, which is what lets a decoder read whole samples of the picture it is
  still building. The clipping is the interesting part: it is what the wavefront
  scheduler has to reproduce as a dependency, and the reference decoder errors
  out rather than clipping when the rectangle cannot be moved.
* **The block reads the frame it belongs to.** In the plan a block copy job has a
  negative reference slot, and both the CPU and the device kernels already read
  the current picture for that case (the same path an inter/intra block's intra
  half uses). The chroma planes shift the vector one bit further, because the
  same luma vector is a half sample offset on a subsampled axis - a bug here
  shows up as a chroma block copied from the wrong place while luma is exact.

The `MvCtx` dimension is the trap here: a single set of vector tables keeps a
stream without block copies in sync indefinitely, and desynchronises on the first
vector of the first block copy block - one symbol, then everything after it is
noise. The tables are compared with dav1d's `cdf.mv` row by row in
`tests/test_entropy.cpp`, together with the check that the two contexts adapt
independently and that a reset restores both.

Two bugs came out of this fixture and neither is visible without pixels:

* **The delta width of a palette colour was CeilLog2, not FloorLog2 + 1.** A
  colour is coded as a delta from the previous one, and the width of the next
  delta is `Min(bits, FloorLog2(MAX_VALUE - prev - (plane ? 0 : 1)) + 1)`. The
  two functions agree everywhere except on exact powers of two (a reachable
  range of 2 needs *two* bits, not one), so the stream stayed in sync until a
  block happened to land on one, then desynchronised - which is why the first
  two frames were bit exact and the third was not.
* **Palette CDFs kept their adaptation counters across a frame boundary.** The
  counters (`cdf[n_symbols]`, saturating at 32) are *not* part of a saved frame
  context: `load_cdfs` loads the probabilities and sets every counter back to 0,
  so the next frame adapts at the base rate. `reset_cdf_counts()` was zeroing
  the counters of the tables that existed when it was written and had never been
  extended to `pal_y`, `pal_uv`, `pal_sz`, the key frame mode table, the intrabc
  flag and the duplicate partition/skip tables. A stale counter of 32 makes the
  adaptation rate `4 + (32 >> 4) + (n > 2)` instead of `4 + (n > 2)`, so the
  probabilities drift by a few Q15 units per update and, several hundred
  symbols later, a symbol decodes differently. Every table `init_default_cdf`
  fills is now reset, together with a comment saying why.

A single sample of drift is worth chasing rather than tolerating: the last three
differences on `w_cu2` were one level on four chroma samples spread over three
frames, and the cause was the difference weighted mask being rebuilt from
*translational* luma predictions in a region the luma plane warped (see
`mask_warp` in `Av1gInterJob`). Display-order comparison cannot even see that:
the stream reorders with `show_existing_frame`, so the dumps are taken in coding
order (`AV1GPU_DUMPCODED`, dav1d's `--outputinvisible`) and matched on their
frame signatures.

The third layer is what found the deblocking group index bug: the kernel was
handed a group number where the shared helper expects a sample offset, so edges
longer than four samples were filtered in the wrong place. A flat test pattern
cannot see that, and the per-kernel check had used one; the frame level test with
a real edge and several groups does, and `opencl_selftest.py` now includes that
case as well.

## Skip mode

Skip mode is the only inter tool with no mode syntax at all. `skip_mode_params`
picks a *pair* of references out of the seven the frame already references - the
nearest one in the past and the nearest one in the future, or, for a frame whose
references all lie in the past, the two nearest past ones - and a block that sets
the flag predicts from that pair at the top of the candidate stack, at the
frame's vector precision, blended with the plain average. It codes no `is_inter`,
no reference, no mode, no filter and no residual: the flag comes first, and
everything else follows from it.

Three things are easy to get wrong and each of them is silent:

* **The pair is in *reference type* space, not slot space.**
  `SkipModeFrame[0]` is `LAST_FRAME + Min(forwardIdx, backwardIdx)`, so it is a
  frame type and its slot comes from `ref_frame_idx[]`. `tests/test_frontend.cpp`
  checks both members of the pair against the header's own reference list, on
  every inter frame of `inter_intra.ivf`, including the all-past fallback
  (a frame whose second member is itself a forward reference).
* **It is a compound block with the average blend, and the two halves have their
  own references.** A blend that dropped one half, or read one slot twice, is a
  pixel error no entropy check can see. `test_kernels` pins the arithmetic
  directly: a flat 100 against a flat 200 in two different pool slots comes out
  at exactly 150, once, at `InterPostRound + 1` - which also means the backend's
  slot rebase is exercised, because without it both halves read slot 0 and the
  result is 100.
* **`has_subpel_filter` is false**, so no filter symbol is coded and the block
  keeps the frame's filter - and it is still a *skipped* block, so it carries no
  residual and its coefficient context is reset.

The fixtures cover it densely: `inter_intra.ivf` codes 1033 skip-mode blocks,
`lw_t2.ivf` 2220, `lw_t3.ivf` 2042 and `svt_av1_320x240.ivf` 43, and every frame
of all four is bit exact against dav1d and through the OpenCL kernels. Replacing
the stack entry the pair is taken from with the second one - a one character
change - decodes to only 4 of 110 matching frames on `inter_intra.ivf` while the
bitstream stays perfectly in sync, which is exactly the kind of error the hash
tests exist for.

## Frame sizes taken from a reference

A frame that codes `frame_size_override_flag` either says its size outright or
names one of its references with `found_ref` and copies that frame's size, render
size and superres decision (`frame_size_with_refs`, 5.9.4). The two were
conflated here until now: the parser read an explicit size whenever the override
was set - so a stream that scaled a frame against its reference desynchronised at
the first such frame - and treated a frame *without* the override as inheriting
its reference's size, where 5.9.5 says it is the sequence's maximum size.

Both are fixed, and `test_frame_size_with_refs` pins them with written headers: a
frame that takes 320x240 and a 512x384 render size from a half size reference, a
`found_ref` on the fifth reference (the bits stop at the first set one), a
frame that codes its own size when no `found_ref` bit is set, an error resilient
frame - which codes no `found_ref` bits at all, because it cannot rely on the
pool it is re-stating - and a `found_ref` naming an empty slot, which is an
error rather than a size of zero. The real streams here are all coded at their
sequence's maximum size with the override off, which is exactly why the old
misreading decoded them bit exactly and why this needed written headers.

## Reference frame ids and short signalling

Two header features change which stored frame a reference *names*, and both were
missing until now. Neither can be produced by the encoders available here, so
they are pinned by written headers rather than by a stream - and, for the part
that is a choice between equally valid-looking candidates, against the reference
decoder's own algorithm.

**Short signalling** (`frame_refs_short_signaling`, 7.8) codes two slot indices -
the frames the encoder committed to as LAST and GOLDEN - and derives the other
five from the order hints in the reference pool: ALTREF is the frame with the
highest output order at or after this one, BWDREF the closest of what remains,
ALTREF2 the next closest, and LAST2/LAST3 plus any leftovers are filled in
anti-chronological order. `set_frame_refs()` implements it directly from the
specification text; it is exposed so `av1gpu_probe` and the tests can show the
list a frame derived.

The scans have three places where two candidates tie: several slots are allowed
to carry the same order hint. The specification only says the reference
implementation breaks ties by reference frame index, which is not enough to
implement from. `test_set_frame_refs_matches_reference` therefore carries a
transcription of dav1d's `src/obu.c`, which resolves the same list with a
different structure - plain signed differences and a "used" sentinel instead of
shifted hints and a used array - and compares the two over 400 pseudo-random
pools (random hints, random current hint, random LAST/GOLDEN). All 400 agree on
all seven references. The directed cases next to it pin the shapes that matter:
two future frames (ALTREF is the furthest, BWDREF the nearest), no future frame
at all (everything from the forward list, deepest first), and a pool with a
single usable slot (every derived reference lands on it - wrong, but defined,
where -1 would index out of bounds).

**Frame ids** (`frame_id_numbers_present_flag`) give every frame an id and make
each reference state how far back its id must be: `delta_frame_id_minus_1` plus
one, subtracted from the current id modulo `1 << idLen`. A mismatch means the
slot does not hold the frame the encoder meant, and the parser returns Invalid
instead of predicting from the wrong picture. Three rules keep the pool honest
and all three are implemented:

* a shown key frame empties the pool (`RefValid[]`, `RefFrameId[]` and
  `RefOrderHint[]`), because it starts a new coded video sequence,
* `mark_ref_frames()` empties every slot whose id has drifted out of the
  `1 << deltaFrameIdLen` window, which is how a decoder notices a lost frame -
  both branches of its two-sided test are exercised, including the wrap,
* an error resilient frame re-states the order hint of all eight slots and a slot
  that disagrees is emptied.

`show_existing_frame` carries `display_frame_id` instead - the id the shown frame
was stored with - and it has to match the slot; the frame type and order hint are
then taken from the slot, not coded. `test_frame_ids` walks a small stream
through all of it: a key frame that fills the pool, an inter frame whose
references match, a frame whose distance is wrong (rejected), a frame whose id
has pushed the oldest slot out of the window (dropped, and the surviving slot
still usable), a display of a stored frame with the right and the wrong id, and
an error resilient frame that drops a slot whose re-stated hint disagrees.

## Segmentation

Segmentation is the one place where the *block* decides its quantiser, its loop
filter offsets and two of its prediction decisions, and it has to be read before
anything else about the block: the segment id comes first (before the skip flag
when the frame says so), and a segment can then force "skip", "intra" or
"GLOBALMV from this reference" instead of the block coding those symbols.

**Header (5.11.9).** Three details are silent when wrong:

* A frame whose `primary_ref_frame` is `PRIMARY_REF_NONE` does not code
  `segmentation_update_map`, `segmentation_temporal_update` or
  `segmentation_update_data` at all - they are 1, 0 and 1 by definition. Reading
  them costs one to three bits and desynchronises the range coder at the first
  tile byte. The first version of this code did read them and every frame of a
  segmented stream came out as noise.
* Feature widths are per feature and a signed feature is *one bit wider than its
  table entry*: `Segmentation_Feature_Bits` is `{8, 6, 6, 6, 6, 3, 0, 0}`, a
  signed value is `su(1 + bits)`, so `SEG_LVL_ALT_Q` is a 9 bit two's complement
  value and -65 is legal. The value is then `Clip3(±Segmentation_Feature_Max)`.
  Reading `su(bits)` and a table of `{..., 3, 1, 1}` - which is what was here
  before - misreads every negative quantiser offset and shifts the rest of the
  header with it.
* `SEG_LVL_SKIP` and `SEG_LVL_GLOBALMV` carry no value: enabling the feature is
  the whole signal, so nothing is read for them.

Three values are derived per frame from those features: `LastActiveSegId` (the
highest segment with any feature at all), `SegIdPreSkip` (true when a feature at
or above `SEG_LVL_REF_FRAME` is on, because the tile data then codes the id
before the skip flag) and `LosslessArray[]`, computed per segment from
`base_q_idx + ALT_Q` clipped to 0..255. Lossless is therefore a property of the
*segment*: the transform size, the transform type and whether a residual may be
coded at all follow from it block by block, not from the frame.

**Tile data.** The id of a block is coded against a prediction - the above and
left neighbours of this frame's map (three comparisons give context 0, 1 or 2),
or, when `segmentation_temporal_update` is set, the id at the same place in the
primary reference frame's stored map, each with a one bit "predicted?" symbol in
between - and the symbol itself is the distance to that prediction, resolved with
`neg_deinterleave(diff, pred, LastActiveSegId + 1)`. A block that is skipped
codes no symbol and inherits its neighbours' prediction. The map of every decoded
frame is kept (`TileDecodeResult::seg_map`) and stored in the reference slot it
refreshes, which is what makes the temporal prediction and
`segmentation_update_data == 0` (inherit the features of a frame's primary
reference) work.

**What is verified.** `tests/data/seg_altq.ivf` is a 30 frame libaom
`--aq-mode 3` encode: two per superblock classes, `ALT_Q` -65 and -80 on top of
`base_q_idx` 160 (quantiser 95 and 80), temporal update on every frame, 207 of
its 2,586 blocks in segment 2 and 15 in segment 1. Both offsets need the ninth
bit, so the stream regresses the width bug above. Every frame is bit exact
against dav1d *and* through the OpenCL kernels (`stream_seg_altq`,
`opencl_seg_altq`).

A second `--aq-mode 3` encode of a different clip (320x240 `testsrc2`) is not a
fixture, but it is the one used for the symbol trace: with `AV1GPU_TRACE_SYM=1`
against dav1d's `DAV1D_MSACLOG=-` all **251,289** range coder events of its 30
frames agree in value, range and bit count, so the segment id syntax, both
predictors and the derived values are right event for event.

What no encoder here emits is the rest: `ALT_LF` (libaom uses only `ALT_Q`, SVT-AV1
does not segment at all) and the forced `REF_FRAME` / `SKIP` / `GLOBALMV`. Those
are pinned by written headers in `test_segmentation` - the feature widths,
`su(1 + bits)` including the clipping at the limit, `LastActiveSegId`,
`SegIdPreSkip`, the per segment `LosslessArray`, and inheritance with
`update_data == 0` - and by `test_segment_deblock_level`, which adds the
segment's `ALT_LF` to the plane's level, clips it at zero and shows that a
disabled feature is ignored. Returning 0 from `read_segment_id_symbol` while
still reading the symbol - so that every frame parses to the end of its tile -
decodes only 2 of `seg_altq`'s 30 frames to the recorded pixels (the two before
segmentation turns on): a pixel error no entropy check would see.

## Superres

A superres frame is coded narrower than it is displayed. `frame_size()` codes
`use_superres` and a three bit `coded_denom` when the sequence allows it, and the
coded width is then derived rather than coded:
`FrameWidth = max((UpscaledWidth * 8 + SuperresDenom / 2) / SuperresDenom, min(16, UpscaledWidth))`
(5.9.5, `SuperresDenom = coded_denom + 9`). Everything the tiles touch - `MiCols`,
the superblock grid, the loop filters - is in that coded width, and only the
output is 1.5x wider.

**Header.** One bit is not where it looks. `allow_intrabc` is only coded when
`UpscaledWidth == FrameWidth` (5.9.2), so a superres frame has no such field; a
parser that reads one shifts the rest of the header by a bit and desynchronises
the range coder at the first tile byte. Superres used to be refused by name here,
which hid that. Two more bits depend on the decision: a reference frame whose
stored width differs from this frame's is *scaled* (5.9.2 `is_scaled`), and for a
scaled reference the tile syntax codes `use_obmc` in place of the `motion_mode`
symbol and global warped motion is disallowed (7.11.9, 7.11.10) - both of which
the reference pool already carried once `RefUpscaledWidth` was added to it.

**Upscale (7.16).** Horizontal only, after CDEF and before loop restoration, with
an eight tap filter from `Upscale_Filter[64][8]` - the same table as dav1d's
`dav1d_resize_filter` with the encoder's sign convention flipped back. Per
plane:

```
stepX = ((downscaled << 14) + upscaled / 2) / upscaled
initialSubpelX = (-((upscaled - downscaled) << 13) + upscaled / 2) / upscaled
                 + 128 - err / 2        (err = upscaled * stepX - (downscaled << 14))
srcX(x) = -16384 + (initialSubpelX & 16383) + x * stepX
```

and the output sample is `Round2(sum of tap[t] * Clip3(0, src_w - 1, (srcX >> 10) + t - 3), 7)`.
The `& 16383` is not cosmetic: the negative start is what makes the first output
columns read the pixels before the corresponding coded column. For 320 -> 480
(`SuperresDenom` 12) the numbers are `stepX = 10923` and `initialSubpelX = 13702`,
and the kernel reproduces dav1d's luma plane exactly, sample for sample.
`src_w` in that clip is the *coded* width: the filter is clipped to the picture
it reads, not to the picture it writes.

**Scaled reference prediction (7.11.3.2).** A reference stored 1.5x wider is
predicted from with a per sample step, not with a single motion vector offset:

```
xScale  = ((RefUpscaledWidth << 14) + (FrameWidth >> 1)) / FrameWidth
originX = (base_x << 4) + ((2 * mvx) >> subX) + 8
baseX   = originX * xScale - (8 << 14)
startX  = Round2Signed(baseX, 14 + 4 - 10) + 32
stepX   = Round2Signed(xScale, 14 - 10)
px(x)   = startX + stepX * x          (1/1024 of a reference sample)
```

and the same for the vertical axis; the horizontal phase `px` selects the row of
the eight tap filter and the integer part the eight samples it reads, clipped to
the reference, with `Round2(sum, 3)` between the passes and `Round2(sum, 11)` at
the end (the same rounding the unscaled path uses). `AV1G_REF_SCALE_SHIFT` is the
14 bit scale fraction, `AV1G_SCALE_SUBPEL_BITS` the 10 bit subpel position.

Two things here are easy to get wrong, and both were:

* The subpel origin is the *patch* origin plus the vector; a sample adds
  `stepX * x` to it. Building it as `(base_x + x) << 4` - the unscaled
  behaviour, with the sample index baked into the origin - is right only when
  `stepX` is exactly 1024. That one made every superres frame differ from dav1d
  in about 115,000 samples while the same kernel passed every unscaled stream.
* The half sample offset that goes with a scaled accumulation is
  `AV1G_SCALE_SUBPEL_OFF` = 32, not the unscaled path's 16. With both in place
  the kernel agrees with a line by line transcription of the 7.11.3.2 pseudocode
  over 128 taps, including the clipping at both edges (`mismatches 0 of 128`).

**What is verified.** `tests/data/superres.ivf` is a 30 frame stream relabelled
to 320 coded / 480 displayed (see `tests/data/README.md` - no encoder here can
signal superres, so the headers are patched). Every frame is bit exact against
dav1d, both through the CPU backend (`stream_superres`) and through the OpenCL
kernels (`opencl_superres`), and the reference frames it predicts from are the
upscaled ones that were stored. The plan format carries both geometries (`v4`
added `out_width`/`out_height` and the per plane display sizes next to the coded
ones, plus the upscale jobs), which is what lets the replay check the displayed
picture rather than the coded one.

## Loop restoration

Loop restoration (7.17) is the last reconstruction filter, and the only one that
reads a picture the frame no longer holds: a restoration unit is filtered from
the CDEF output *and* from the deblocked picture before CDEF, because the two
lines either side of a stripe's border are taken from the latter (`get_source_sample`,
7.17.6). The backend therefore keeps two snapshots - the frame as it leaves the
deblocker and the frame as it leaves CDEF - and every unit reads them, never the
picture being written, so units cannot observe each other's output.

Three things about the geometry decide the result:

* Stripes are 64 luma rows, but the first stripe of a frame starts at
  `-8 >> subY` so that it begins two luma rows above the tile (7.17.1). Every
  later stripe is `stripe_start += 64 >> subY`.
* `StripesPerTile` is `ceil((MiRows * MI_SIZE + 8) >> 3) / 8`, and a unit whose
  top left corner is outside the visible plane is skipped only when it also has
  no stripe start in this tile - the last unit of a row therefore covers the
  short remainder rather than being dropped.
* The unit a row belongs to is not `row / unitSize`: 7.17.1 offsets by the same
  eight luma samples the stripes are shifted by, so the last eight rows of a unit
  belong to the next one.

Two filters, both pointwise once their taps are known. The Wiener filter
(7.17.4) is separable with `FILTER_BITS = 7` - not the 3 of the interpolation
filters - and its intermediate is the *horizontal* pass, rounded by
`InterRound0` and clipped to
`[-(1 << (BitDepth + FILTER_BITS - InterRound0 - 1)),
(1 << (BitDepth + 1 + FILTER_BITS - InterRound0)) - 1]`, which is what the
specification's `Min(15, BitDepth + 5)` bit note works out to at 8 bit. Feeding
3 there instead flattens the picture to a sixteenth of its range. The self guided
filter (7.17.3) is the unit's two box filters, radii 1 and 2, with `Sgr_Params`
giving each a radius and an `eps`: each 4x4 cell becomes a local variance `p`, a
weight `A = SgrMathematicalFormula(p * s)` and an offset
`B = Round2(((1 << SGRPROJ_SGR_BITS) - A) * sum * oneOverN, SGRPROJ_RECIP_BITS)`,
and the unit's two `SgrXqd` weights project the results back together with
`Round2(..., SGRPROJ_PRJ_BITS)`. A radius of 0 means "use the input", i.e. the
filter is skipped for that pass rather than run with a one sample radius.

**What is verified.** `tests/data/lr.ivf` is the fixture: three of its frames code
the type per unit (`SWITCHABLE`) and the other five one filter per plane, and
across it every combination runs - Wiener and SGR on luma and on both chroma
planes, units coded `RESTORE_NONE` inside a frame that restores, and units whose
stripe is split by the stripe grid. Every frame is bit exact against dav1d (and
libaom) through the CPU backend (`stream_lr`) and through the OpenCL kernels
(`opencl_lr`), and the CPU backend's `AV1GPU_NO_LR` - the same debug bitmask as
its `AV1GPU_NO_DEBLOCK` - switches restoration off per plane so the pre-LR
picture can be compared with dav1d's, which is how the two snapshots were checked
separately from the filters. The plan format carries the restoration jobs
(`v7`; `v8` added the plane they belong to), because a replay needs both
snapshots; the replay keeps them exactly where
the backend does, including the second upscale pass a superres frame needs to
bring the pre-CDEF picture into the displayed geometry.

**Chroma that is not 4:2:0.** `tests/data/lr_444.ivf` and `lr_422.ivf` are the
same clip at 4:4:4 and 4:2:2, because restoration on a chroma plane that is not
subsampled the 4:2:0 way is not the same path: the chroma unit grid only halves
for 4:2:0 (`lr_uv_shift` in the plan), at 4:2:2 only its horizontal step does,
and at 4:4:4 the chroma planes are the size of the luma one. The 4:4:4 stream
reproduces `switchable` chroma units - the type coded per unit on a plane that is
not subsampled - and both streams code Wiener and SGR on chroma. Every frame of
both is bit exact against dav1d and ffmpeg, through the CPU backend
(`stream_lr_444`, `stream_lr_422`) and through the kernels (`opencl_lr_444`,
`opencl_lr_422`).

The 4:4:4 one found a bug that had nothing to do with restoration itself, and
every 4:2:0 fixture was blind to it: the compound difference weighted mask is the
*luma* prediction difference (7.11.3.12 is plane 0 only), and the decoder decided
"rebuild it from the luma predictions" from the subsampling flags. That is right
for 4:2:0 - a subsampled plane has no luma predictions of its own - and wrong at
4:4:4, where one chroma sample covers exactly one luma sample but the mask still
has to come from the luma picture. `Av1gInterJob` carries the plane now (plan
format 8) and the kernel branches on it instead of on `ss_x`/`ss_y`.

**The unit tests.** `test_kernels` covers the two filters as properties rather
than as fixtures, because a wrong filter breaks them immediately: the Wiener
taps always sum to 128 and a self guided unit is an identity on a flat block, so
a flat plane has to come back flat at 8, 10 and 12 bit, exactly, for any
coefficients, any strength and any pair of projection weights. Next to that sit
the two things only restoration does: a unit reads samples from two different
pictures (`get_source_sample`, checked with a vertical filter whose only taps are
-2 and +2, so an output row shows which picture each neighbour came from) and a
pass with radius zero uses the sample itself instead of filtering. The last check
is the one the 8 bit fixtures cannot make: the same box at 8 and at 10 bit has to
give the same `A` and a `B` exactly four times as large, which is exactly what the
historical bug in this filter broke. All of them were confirmed by mutation -
dropping either bit depth scaling, building `B` from the scaled sum, or swapping
the two snapshots each fail a named check.

**Bit depth.** The box filter's local sums are scaled back to 8 bit before they
become a variance (`a = Round2(a, 2 * (BitDepth - 8))`, `d = Round2(b,
BitDepth - 8)`), and at 8 bit those two shifts are zero - which is exactly why a
stream that forgets them looks right at 8 bit. The offset is the other way round:
`B` is built from the *raw* sum `b`, not from the rounded `d` that went into the
variance, and again the two are the same number at 8 bit. `tests/data/lr_10bit.ivf`
is the same clip at 10 bit for both, and with the raw `b` missing the chroma
planes of every frame differed by up to 88 and the luma drifted from the sixth
frame on (the reference frames had already inherited the chroma error), while
every 8 bit fixture stayed bit exact. `stream_lr_10bit` and `stream_lr_12bit`
pin both depths, and 12 bit is also the one depth where the Wiener filter's
rounding constants move (`InterRound0` 5, `InterRound1` 9) - which is why the
fixtures are coded at all three depths rather than at 8 bit alone.

The one combination no fixture here codes is superres *and* restoration at once:
making an LR stream superres means changing the *unit grid* (7.17.1 sizes the
units on the upscaled picture), so unlike `superres.ivf` the header of an LR
stream cannot simply be patched, and libaom's superres mode is not reachable
through the ffmpeg wrapper in this sandbox. It is covered by construction
instead: `tests/test_pipeline.cpp` builds the syntax of a clip coded narrower
than it displays - `make_clip` writes the blocks, the test adds the restoration
units - and compares every restored sample against its own implementation of the
specification. Three things are pinned that no single-picture reference can
express:

* **7.17.6 chooses between two pictures.** A row inside the stripe comes from
  the CDEF output, a row outside it from the picture *before* CDEF, and the third
  row outside is a copy of the second. The reference the test carries
  (`wiener_sample_striped`) takes one picture per row, so both choices and the
  clamp are checkable: the clip is decoded twice, once with CDEF and once with
  the CDEF parameters removed - the second decode *is* the deblocked picture -
  and the expected sample of a row is built from the right one. With CDEF on the
  two pictures really differ (110 samples along the stripe borders of that clip,
  and the test asserts that they do, because a wrong picture that no sample can
  tell apart would not be caught); with CDEF off they are equal and only the
  clamp is exercised. CDEF on also pins the second upscale, because the pre-CDEF
  snapshot has to be brought into the displayed geometry as well.
* **The two pictures are swapped.** Nothing else in the suite can see that
  mistake: with CDEF off the two pictures are the same picture, so the swap is a
  no-op - it is the CDEF-on decode that turns it into 1575 wrong samples.
* **The unit grid, not just the filter.** Every unit carries *different*
  coefficients (`wiener[0][0] = 2 + 2*row`, `wiener[1][0] = 1 + 2*col`, so the
  vertical and horizontal sets differ too), and the reference picks them per
  sample with 7.17.1's rule. A grid that is one row or one column off, or a unit
  count that does not stretch the last unit to the plane edge, then produces
  wrong samples instead of the same picture. Eight geometries are swept -
  128x96 over superres 112, 128x100, 128x68, 120x68, 160x96 with one 128-sample
  unit, 96x72 with a 64-wide unit and a 32-wide last one, 32x24 and 16x16, i.e.
  smaller than one unit - and every case also asserts that the filter actually
  changed samples (196 to 1349 of them), because a case where nothing moves could
  not tell a wrong grid from a right one.
* **A no-op has to be a no-op.** With an identity filter in every luma unit (taps
  `{0,0,0,128,0,0,0}`) and a self guided one on both chroma planes, which on this
  ramp is an identity as well, the picture has to come back bit identical to the
  same clip with restoration switched off - the whole frame, chroma included.

Mutations of the engine and of the kernel were then run against the test, each
one restoring the old numbers when reverted:

| mutation | samples wrong |
| --- | --- |
| the stripe starts at the frame edge instead of eight rows above | 39 / 9 / 30 / 26 / 37 / 35 (176 in all), and 227 across a border |
| `unitRow` drops the `+ 8` of 7.17.1 | 572 / 561, and 558 across a border |
| the last unit is not stretched to the plane edge | 315 (160x96) |
| every row is taken from the CDEF picture | 31 (the border case, CDEF on) |
| rows below a stripe are never borrowed from the earlier picture | 26 / 2 / 32 / 19, and 26 across a border |
| the borrowed row is not clamped to `StripeStartY - 2` | 6 / 11 / 9 / 1 / 2, and 5 across a border |
| neither border of the borrowed rows is clamped | 32 / 2 / 11 / 9 / 33 / 21, and 31 across a border |
| the two pictures are swapped | 1575 (the border case, CDEF on) |
| the pre-CDEF snapshot is not upscaled | 120 / 324, and 319 across a border |
| the identity filter's intermediate clip is removed | 0 - not observable |

The geometry numbers are in the order the eight cases run (`128x96` over superres
112, `128x100`, `128x68`, `120x68`, `160x96`, `96x72`, `32x24`, `16x16`), and the
two that stay at 0 are the ones with no stripe border and no partial unit inside
the block the mutation affects.

The last row is worth writing down: `InterRound0` clips the horizontal pass to
`[-(1 << (BitDepth + 7 - InterRound0)), (1 << (BitDepth + 8 + 7 - InterRound0)) -
1]`, and with real coefficients the final `Clip1` hides the difference, so no
sample of any fixture or of the superres clip can tell the two apart. The clip is
still implemented (the specification bounds the intermediate so it fits a
`Min(15, BitDepth + 5)` bit accumulator), but it is not what a test can pin.

## Frames whose size is not a multiple of eight

Every fixture used to be a multiple of eight in both directions, and that is the
size where three roundings agree - so nothing noticed that the decoder had two
of them wrong.

**The 4x4 grid a tile codes on** is the frame size rounded up to eight:
`MiCols = 2 * ((FrameWidth + 7) >> 3)` (7.4.4). Rounding to four, i.e.
`(FrameWidth + 3) >> 2`, gives the same number for every multiple of eight and
one mi column too few for a width 1 to 4 samples past a multiple of eight - and
the block loop of the last superblock column then reads a different number of
blocks than the encoder wrote. It is not a sample or two: a two frame 130x96
clip decoded to 23,796 wrong bytes out of 37,440, i.e. the frame was garbage.
The tile decoder, the deblocking grid, the CDEF grid and the reconstruction grid
all derive the size from one place now (`mi_dim()` in
`src/frontend/headers.hpp`).

**The samples the filters and the prediction read outside the picture** are
inside that grid, and they belong to blocks as well: a block on the last row band
is predicted from the left column *below* it, CDEF copies and writes whole 8x8
blocks, and the deblocker reads `q3` three samples past the edge. Two things
followed. The engine's wave scheduler tracked sample ownership over the visible
plane only, so those reads were not ordered after the block that writes them -
the result was a handful of samples that depended on scheduling (91 wrong samples
on a 128x99 frame, deterministic in practice, wrong nevertheless). And the CDEF
job was given the visible size as its bound, so the last partial block filtered
against a halo that stopped at the picture edge. Both now use the coded extent
(`((FrameWidth + 7) & ~7) >> subX`), which is also what the prediction jobs
already clamped to.

`tests/data/odd_size.ivf` is 130x100 - two samples past 128 in every row, a
partial stripe at the bottom (100 = 64 + 36), and six frames so the error would
drift - and it is bit exact against dav1d and ffmpeg, through the CPU backend and
through the OpenCL replay. `tests/data/make_odd_size.py` rebuilds it byte for
byte.

**The four sample grid the loop filter walks** is the third rounding, and it is
neither of the two above: 7.14 iterates mi rows and steps them by `1 << subY` on
chroma, so each plane is filtered up to *its own* size rounded up to four
samples, and that is the extent the reference decoders use. For a 130x66 4:2:0 frame that is 132x68 for luma and 68x36 for
chroma, so the last two luma rows and the last three chroma rows lie outside the
picture but inside the grid the filter covers - and 7.15 copies whole 8x8 blocks
that include them, and 7.17.6 borrows two lines across a stripe border from them,
so a decoder that stops at the visible edge leaves samples the deringer and the
restoration filter then read at level zero. That was the last of the edge
residuals: a 160x116 clip at `-crf 18` differed in 25 and 36 samples, all of them
by one and all of them in the chroma rows within three of the bottom of the
plane; 128x68 in one to three, a 130x66 clip at `-crf 45` in one. The deblocker
walks `(plane_size + 3) & ~3` now, in all three planes.

`tests/data/edge_grid.ivf` pins it: 130x66, where luma ends two rows and chroma
one row inside its four sample grid, four frames at `-crf 40` with hard edged
panels against the bottom and right edges so that the loop filter, CDEF and
restoration all run at a non-zero level there - a smoother clip codes those rows
at level zero and then the two roundings agree again. It is bit exact against
dav1d and ffmpeg, through the CPU backend and through the OpenCL replay. The
sweep that found the earlier edge bugs is clean with all three fixes: widths 124
to 139 at height 64, 130x64, 130x65, 130x66, 130x68, 130x69, 130x70, 130x71,
130x72, 130x73 and 130x75 (so every `height % 8`), 128x97, 128x98, 128x99,
128x100, 128x101, 128x102, 128x104, 132x102, 134x102, 160x116, 126x98, 120x66,
96x60, 64x52 and 48x36 at `-crf 32`, plus 321x241 and 640x361 at `-crf 45`,
131x67 and 255x129 at `-crf 40`, and 128x68, 128x71, 130x68, 130x70, 130x71 and
130x66 at `-crf 18`, `24` and `45` - two frames each, decoded against dav1d.

One negative result is worth keeping: the CDEF taps themselves must *not* be
clamped to the visible plane. Doing that - treating samples the mi grid says are
available as unavailable - with everything else fixed turns 0 wrong samples into
160 and 199 on the 160x116 clip, 588 on 128x68 and 75 over the four frames of
`edge_grid`: the specification makes availability an mi unit question (7.15.3's
`is_inside_filter_region` against `MiCols`/`MiRows`), not a picture size one.

## Frames cut into tiles

Until this fixture, every stream in `tests/data` coded one tile per frame, and
the decoder said so: `decode_tile` returned `Unsupported` for any frame whose
`tiles.cols` or `tiles.rows` was not one. The tile *group* parser had split
payloads into tiles from the start, so what was missing was the per tile state
and the merge of the frame's maps.

**What starts over at a tile.** 5.11.2's `decode_tile` is explicit about it,
and each of these was a bug waiting to happen: the range decoder reads the
tile's own bytes; the CDF tables are the ones the *frame* started from (a tile
adapts its own copy and never sees another tile's adaptations); `CurrentQIndex`
is `BaseQIdx` again; `DeltaLF` is zero; `clear_above_context` runs once and
`clear_left_context` at every superblock row; and `RefSgrXqd`/`RefLrWiener` go
back to their mid points. **What does not**: `MiSizes`, `RefFrames`, the motion
field, the segment ids, the CDEF index grid and the restoration units are the
frame's, and each restoration unit is coded by the tile that holds its top left
corner. The temporal motion vector grid is saved once, at the end of the frame.

**Which tile's tables are kept** is not "the last one decoded": 9.x's
`frame_end_update_cdf` copies the tables of the tile whose index is
`context_update_tile_id` (5.9.2), and only when `disable_frame_end_update_cdf`
is 0. A frame with one tile names tile 0, so a single tile frame publishes
exactly what it always did.

**Two boundary bugs, both the same mistake.** `is_inside` means the tile, and
the frame's maps are frame wide, so a sample across a tile boundary is a real
reconstructed sample that the bitstream says is *not* a neighbour. Two places
read the frame where the specification means the tile:

  * `AvailUChroma`/`AvailLChroma`. For a block one 4x4 wide (or high) in a
    subsampled frame the chroma is shared with the neighbouring luma block, so
    5.11.5 asks `is_inside(r - 2, c)` / `is_inside(r, c - 2)` - two mi units
    away. The planner tested `mi_row >= 2` and `mi_col >= 2`, i.e. the frame's
    origin, so the second column of chroma in a tile predicted from the tile to
    its left. It shows as a constant offset per chroma block (an error in the
    average luma a chroma-from-luma block subtracts) that then propagates
    through the blocks that predict from it: 18,447 of `tiles.ivf`'s 245,760
    bytes.
  * `find_warp_samples`' `add_sample` (7.10.4.2) starts with
    `is_inside(mvRow, mvCol)`. It tested the frame's bounds instead, so a
    locally warped block on a tile edge fitted its affine model to a motion
    vector from the next tile - and only blocks that use `LOCALWARP` notice:
    3,676 of those bytes.

Both were invisible in every single tile fixture, and both *decode*: the range
coder stays in sync because nothing about the syntax changes, so the frame comes
out plausible and slightly wrong. Reverting either one makes `tiles.ivf` differ
from dav1d again by the numbers above, which is how the fixture is known to pin
them.

`tests/data/tiles.ivf` is 256x160 cut into 2x2 tiles - 128 + 128 across and
128 + 32 down, so the bottom two tiles are only 32 samples high and every one of
the four tiles has a neighbour from another tile on two sides. Four frames, the
first a key frame, at `-crf 28`; the content pans and zooms, which is what makes
libaom code local warped motion (699 blocks of the four frames), and the tile row
boundary runs through a dark band where the encoder spends 4x4 blocks - the
sizes whose chroma is coded the sub-8x8 way. It is bit exact against dav1d and
ffmpeg, through the CPU backend and through the OpenCL replay, and
`tests/test_frontend.cpp` checks the two properties that do not need pixels: a
tile group that covers only three of the four tiles is refused rather than
decoded with a quarter of the maps left as they were, and decoding the same
frame's tiles back to front produces the same blocks and the same CDEF index
grid.

**The grid has to divide the frame, and it usually does not.** 5.9.1's
`tile_log2` distributes the superblock columns over the tile columns, so a frame
that is not a multiple of the tile width ends with a wider tile - and the mi
extent of the last tile row or column stops at the *coded* grid
(`MiRows = 2 * ((FrameHeight + 7) >> 3)`), which for a picture that is not a
multiple of eight is past the picture. `tests/data/tiles_grid.ivf` is 320x188 in
4x2 tiles: the tile columns are 64, 64, 64 and 128 samples wide, the tile rows 64
and 124, and the last row's extent is 192 luma samples for a 188 sample picture.
It is bit exact against dav1d and ffmpeg through the CPU backend and through the
OpenCL replay, and `tests/test_frontend.cpp` checks the eight rectangles and the
mi grid they are cut on (`mi_dim(188) == 48`). `tiles.ivf` cannot show any of
this: its four tiles are equal rectangles on a frame that is a superblock
multiple in both directions.

The sweep behind both: 2x2, 2x1, 1x2 and 4x2 layouts; 130x66, 192x128, 256x144,
320x240 and 512x288; `-crf` 24, 30, 36 and 40; a lossless stream (`-crf 0`, so
no loop filter, CDEF or restoration at all); one with `-enable-restoration 0`;
one with `-enable-cdef 0`; and an all-intra stream (`-g 1`). Every one of them is
bit exact against dav1d. A 10 bit stream was checked the same way but not added:
`av1gpu_decode -o` writes 16 bit samples for those and dav1d's yuv muxer writes
them too, so the comparison needs `--yuv16` on our side (see the README), and
the 10 bit path is what `aom_10bit_320x240` already pins.

## Film grain

Film grain is the one filter that runs after the picture is finished: it is part
of the output process (7.18.3), not of the reconstruction, and it is the only one
that is not a local neighbourhood operation. The noise comes from an
auto-regressive filter over a pseudo-random template - every sample depends on
the ones already filtered above it and to its left - and the 32-sample-high
stripes are then stitched together with weights that depend on how far into the
frame they are. Neither half parallelises across samples, so both run once per
frame on the host (`src/core/grain.cpp`); the kernels are left with the per
sample map (scale one noise sample by the plane's curve, add, clip), which is why
every backend adds the same noise.

Four things decide the result and all four are easy to misread:

* `GrainCenter = 128 << (BitDepth - 8)`, so the grain lives in
  `[-128, 127]` at 8 bit and is kept as `signed char`; `GrainMin`/`GrainMax` clip
  the AR filter's output back into that range.
* Two different shifts: `ScalingShift = grain_scaling_minus_8 + 8` for the blend
  and `ar_coeff_shift_minus_6 + 6` for the AR filter. Feeding one where the other
  belongs shifts the grain by up to four bits - visible, but not obviously wrong.
* The chroma grain arrays are **44×38** at 4:2:0 (82×73 at 4:4:4), not the
  frame's chroma size: the largest block offset (6 + 15) plus a 34 wide block has
  to fit. Allocating them as `(width + 1) >> 1` by `(height + 1) >> 1` also
  allocates them at the *wrong stride*, so the noise blobs come out a different
  shape rather than out of range.
* `Round2` here is the unsigned rounding - `(v + (1 << (n - 1))) >> n` over a
  signed value, so -3 rounds to -1 at `n = 1`. It is *not* the `Round2Signed`
  that motion compensation uses, and the two differ for every negative sample.

**Blend order.** The chroma planes are blended first. Their `averageLuma` is the
luma of the picture *before* any grain was added, so the luma plane has to be
untouched while the chroma jobs run; the plan keeps two job lists
(`grain_chroma`, then `grain_luma`) for exactly that reason.

**References are grained later.** The reference frame update (7.20) stores the
decoded picture, which is the one *before* the grain, because grain belongs to
the output process. So a backend copies the frame into the reference pool first
and runs the grain kernels afterwards, and a request for "the picture just
decoded" reads the display frame rather than the pool. Getting this the other way
round is invisible on an intra-only stream and corrupts every inter frame of a
grained one.

**What is verified.** `tests/data/grain.ivf` is a 30 frame stream with the grain
parameters patched into its frame headers (see `tests/data/README.md`; no encoder here can signal
film grain). Every sample of every frame is equal to dav1d's, through the CPU
backend (`stream_grain`) and through the OpenCL kernels (`opencl_grain`).
`tests/data/make_grain.py --profile` was what made that possible: it switches on
one part of the synthesis at a time - a constant luma curve with no AR filter and
no overlap, then the AR filter, then the overlap blending, then a real curve,
then chroma points, then chroma-from-luma - and each profile is now bit exact
against dav1d on its own. Bisecting that way turns "everything differs" into two
one-line bugs instead of a parameter hunt: `av1g_clip` takes `(v, lo, hi)` here
while the specification writes `Clip3(lo, hi, v)`, and the grain code had copied
the specification's order into the call, so every sample clipped to `GrainMax`
and the noise was a flat +127 at 8 bit; and the chroma grain arrays were
allocated at the frame's chroma size instead of 44×38.

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

## Undefined behaviour and the sanitizer build

`AV1GPU_SANITIZE=ON` builds every target with AddressSanitizer and UBSan, and the
whole fixture set is decoded through it:

```sh
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DAV1GPU_SANITIZE=ON
cmake --build build-san -j
export ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1
ctest --test-dir build-san -j
```

The normal and the sanitizer build are both expected to stay at 53/53 (`ctest -N`
prints the list), and neither is allowed to print a `runtime error` line.
The option compiles with `-fno-sanitize-recover=all` for exactly that reason:
UBSan recovers by default, so without it a finding is a line on stderr and a
green test, and `ctest` never sees it. With halting, a finding aborts the
process and the test that ran it fails. `av1gpu_decode` is still worth wiring into
a sweep of its own, since it is the only thing that decodes every fixture -
`ctest` splits them across the CTest tests and a decode that is not one of them
would not run at all:

```sh
for f in tests/data/*.ivf; do
  build-san/bin/av1gpu_decode "$f" -o /dev/null 2>&1 | grep -E 'runtime error|AddressSanitizer'
done
```

That pipeline has to print nothing for the whole fixture set.

The first sweep over the fixtures found nothing but one defect, wherever
a signed value is scaled by a power of two: a **left shift of a negative value**,
which is undefined in C++ but is what the specification's arithmetic says
(`x * 2^s` for a signed `x`). Every one is a multiplication now, and the decoded
pictures are unchanged - the point of the change is that a compiler is no longer
allowed to assume the value was positive:

* `kernels/av1gpu_kernels.h`, the directional intra predictor: `idx` is a signed
  tap index that goes negative for the acute angles, so
  `(idx << upsample_above) >> 1` became `(idx * (1 << upsample_above)) >> 1`
  (two call sites, one per half of the angle range).
* `src/core/tables.cpp`, `gm_setup_shear`: the four shear parameters it derives
  are signed, and so is `params[4]` before the 16 bit scaling.
* `src/frontend/tile_decoder.cpp` (a negative `delta_q` is scaled by
  `delta_q_res`) and `src/frontend/headers.cpp` (a global motion parameter
  scaled by `prec_diff`).
* `src/frontend/deblock.hpp`, the loop filter's reference and mode deltas at
  `level >> 5` (this one shows up on `lr.ivf`: a reference delta of -1).
* `src/core/synthetic.cpp`, the synthetic encoder's analysis transform (a pixel
  scaled into 64 bit before the separable pass) and the motion vectors it writes
  into a block.
* `kernels/av1gpu_kernels.h`, the film grain chroma blend: `(offset - 256) <<
  shift` scales a signed value, and 7.18.3.5's `offset` is nine bits - so half of
  its range is below 256 and shifts a negative number. A *valid* stream reaches
  this one; no fixture happened to (they all code an offset above 256). It came
  out of the mutation sweep below, `test_kernels` blends a chroma sample with
  `cr_offset = 200` twice now (once with a flat curve, once with a curve that
  makes the index visible), and the sanitizer build is what says so.

**A mutation sweep** is the other half: the decoder is the only place that reads
bytes it did not produce, so the sanitizer build also decodes single byte
mutations of every fixture. `tests/fuzz_decode.py` is the part of that which
stays, and `ctest` runs it as `fuzz_mutations`. It found the film grain shift
above, and two things that no fixture (or any stream an encoder writes on
purpose) would have reported:

* `read_palette_plane` merged the palettes above and to the left into an *eight*
  entry buffer on the stack. Sixteen is what that merge produces - 6.10.3's
  `get_palette_cache` concatenates two ascending colour lists of up to eight
  entries each - so a stream with two disjoint eight colour neighbour palettes
  writes past the end, and after that the effect is whatever the neighbouring
  stack frame held. The cache is `2 * kPaletteColors` now
  (`kPaletteCache`); the palette itself is still eight colours.
* the inverse transform multiplied a coefficient (up to 21 bits with sign) by a
  Q12 constant (13 bits) in `int`. 7.13.3 asks a *conformant* stream to keep
  every multiply/add pair inside 31+sign bits, so a stream that follows the
  specification cannot overflow here - but a 12 bit one that codes 19+sign bit
  coefficients does, and dav1d's `itx_1d.c` has the same note from the other
  side (it rewrites its own pairs to stay in range). Signed overflow is
  undefined in C++ and in OpenCL C alike, so `av1g_bfly`, `av1g_adst4`, the
  identity transform's 5793/11586 scaling and the 2:1 rectangle pre-scale (2896)
  form their products in 64 bits and round back into an `int`
  (`av1g_round2_ll`): the same number a conformant stream got before, and no
  undefined behaviour on the ones the specification excludes. The device
  backends compile that same expression, so CPU, OpenCL and CUDA still agree -
  the `opencl_*` replay tests are what keeps saying so.

The sanitizer build also removed a compiler warning that had been noise for a
while: `read_palette_plane` sorts its eight byte colour cache up to `size`, and
gcc's inliner could not see the bound and warned about an out of bounds read in
`std::sort`. It is an insertion sort now - linear on two ascending runs anyway,
which is what the comment above it already claimed.

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
  that reads a *buffer* through a pointer takes an explicitly qualified
  `AV1G_GLOBAL` parameter (see `kernels/av1gpu_kernels.h`). The exception is a
  pointer into a job struct - `av1g_warp_tap` takes the warp matrix that way -
  because the device wrappers copy a job into private memory before calling in,
  so those pointers live in the generic address space (OpenCL 2.0). CDEF
  deliberately does not stage through `__local` memory.
