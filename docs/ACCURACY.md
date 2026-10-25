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
   OpenCL compiler (pocl) and checks one job of every kernel.
3. `tests/opencl_pipeline.py` — `av1gpu_dump` writes the real decode plans of a
   clip, the test replays *every* job of *every* wave through the real kernels in
   backend order (waves, deblock vertical, deblock horizontal, CDEF, grain) and
   compares whole frames against the CPU reference decode. Four frames, intra
   and inter, all filters on: **bit identical**. With `--input` the plans come
   from `av1gpu_decode --dump-plans` for a real stream instead, so the reference
   pool slots are the decoder's own: `lw_t2` and `lw_t3` (local warped motion and
   OBMC, 30 frames each), `palette_sc` (102 palette blocks, 10 frames),
   `inter_intra` (23 inter/intra compound blocks, 110 frames) and
   `svt_av1_320x240` (a different encoder, 6 frames) are all **bit identical**
   too.

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
above and to the left. Intra block copy is switched off because intrabc blocks
are out of scope (the parser reads the flag and stops with a named error); the
stream is otherwise the shortest thing that walks the whole palette syntax -
`palette_mode_info`, `palette_tokens`, the colour map's three neighbour
contexts, and the neighbour palette cache.

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
  that reads a *buffer* through a pointer takes an explicitly qualified
  `AV1G_GLOBAL` parameter (see `kernels/av1gpu_kernels.h`). The exception is a
  pointer into a job struct - `av1g_warp_tap` takes the warp matrix that way -
  because the device wrappers copy a job into private memory before calling in,
  so those pointers live in the generic address space (OpenCL 2.0). CDEF
  deliberately does not stage through `__local` memory.
