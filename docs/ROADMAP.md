# Roadmap

The library stops at the entropy decoder boundary on purpose: it consumes
reconstruction syntax and produces pictures. Everything below is future work in
the order it unlocks the most.

## 1. A front-end that produces `FrameSyntax`

This is the one thing standing between the library and playing real `.ivf` /
`.obu` files. The work is an AV1 parser (symbol decoder, partition tree,
reference frame management) that emits `av1gpu::FrameSyntax`; it does *not*
touch the GPU.

**Done: containers and headers.** `src/frontend/` reads IVF files and raw OBU
streams and parses the complete sequence header and every frame header type -
KEY, INTRA_ONLY, INTER, SWITCH and `show_existing_frame` - up to the byte where
the tile data starts: tile grid, quantiser, segmentation, delta Q / delta LF,
deblock levels, CDEF strengths, reference and skip mode decisions, transform set.
Inter frames read a `RefPool` the caller keeps up to date, so a size inherited
through `found_ref` (`frame_size_with_refs`) and skip mode are derived, not
guessed. That pool carries each slot's order hint and
frame id, which is what `frame_refs_short_signaling` (7.8) and the frame id
checks of 5.9.2 - including `mark_ref_frames` and the order hints an error
resilient frame re-states - work on. Features that are not implemented (superres,
restoration, film grain data) are reported with a message instead of being
misparsed.
`av1gpu_probe` shows what it found in a real file, and
`tests/data/svt_av1_320x240.ivf` keeps the parser honest against bytes from a
real encoder.

**Done: the boolean range decoder and the tile group.** `src/frontend/range_decoder.hpp`
reads the tile payload: symbols over adaptive Q15 CDFs, boolean and equiprobable
bits, `L(n)` and `ns(n)`, with `overrun()`/`bytes_used()`/`state()` for tile end
handling and diagnostics. `src/frontend/tiles.hpp` parses the tile group -
`tile_start_and_end_present_flag`, the interleaved `tile_size_minus_1` fields,
one byte range per tile and the luma rectangle each tile covers - so a payload
can be split into independently decodable tiles.

The symbol reader is bit identical to dav1d's `msac`: replaying dav1d's own
per-symbol trace of the fixture's key frame reproduces all **112,648 events**
and the range coder state at every one of them (see `docs/ACCURACY.md`).

**Done: a whole intra frame, bit exact.** `tests/data/svt_av1_320x240.ivf`'s key
frame now decodes to the same picture dav1d produces, sample for sample, with
the deblocking filter and CDEF enabled. `tests/test_frontend.cpp` pins two
hashes of that picture (and of two patched variants that push every loop filter
level to its maximum, so the wide filter, the flat masks and the per pass, per
plane level selection are all covered), which means the parser, the coefficient
decoder, prediction, chroma from luma, the inverse transforms, the deblocking
filter and CDEF are all exercised by one reference picture.

**Done: the tile syntax.** `src/frontend/tile_decoder.hpp` walks a key
frame's tile the way the bitstream describes it: superblocks, the partition
tree (including the edge cases where only one direction can be split), skip,
per-superblock quantiser deltas, intra modes with their angle deltas, chroma
from luma, filter intra and the transform size, keeping the per-4x4 neighbour
maps the contexts are built from. It produces a block tree, which is what the
prediction and residual layers consume: the whole tile, including the
coefficient syntax (end of block, base tokens, the extended levels, signs), the
dequantisation, the `cdef_idx` grid and the loop filter parameters. The default
CDF tables in `src/frontend/cdf.hpp` cover every symbol it reads, in the
inverse-CDF layout the range decoder consumes, and `tests/test_entropy.cpp`
still pins the convention (32768 - x, counter after the probabilities) against
the first symbol of the fixture's key frame.

**Next, in order:**

1. *Inter frames.* Frames 1-5 of the fixture are inter frames, so this is what
   turns a still picture into a video. The syntax side is partly in:
   `decode_tile` reads an inter frame's blocks, and `TileBlock` carries the
   reference types, motion vector, inter mode, motion mode and subpel filter.
   What is in place:

   * the inter frame preamble: skip mode, skip, `cdef`,
     the per superblock deltas, and either an intra block (whose y and uv modes
     use the inter frame's own tables) or `is_inter`, the reference tree,
     the dynamic reference list, the motion vector residual, inter/intra, the
     motion mode and the subpel filter;
   * the candidate scan `find_ref_mvs`, ported from the specification: the
     above and left neighbours with their weights, the top/right and top/left
     corners, the secondary rows and columns, the global motion filler and the
     sorting the mode contexts depend on. The per 4x4 motion reference map it
     reads is filled by `mark_inter`, so the scan walks the same state the
     specification describes;
   * a reference store (`RefPool`) that resolves `ref_frame_idx` and knows
     whether a reference is an intra frame, which is what decides if temporal
     candidates exist at all.

   * *Variable transform size* and the inter transform type sets, both of which
     are now read: an inter block of 8x8 or more codes a tree of transform
     splits (`read_var_tx_size`, whose leaves fill `InterTxSizes`; the luma
     residual then walks that map instead of a uniform grid), and its luma
     transform type comes from the three inter sets, with the chroma type
     inherited from the luma one.

   With those in, the first inter frame of the fixture parses to its last byte
   with the same block tree, references, motion vectors, motion modes and
   coefficients as dav1d - 249 blocks, all of the tile's 2004 bytes used - and
   the inter frames of every other fixture do the same. What is left, in the
   order the gateways appear when decoding the rest of the file:

   * *Projected motion vectors.* Done: `RefPool` slots carry a frame's 8x8
     motion vector grid (`RefMvs`) and the order hints of the frames it referred
     to, `RefSlot::ref_mvs` is filled by `refresh_ref_pool`/the tile's
     `save_temporal_mvs`, and `find_ref_mvs` projects the contributing reference
     frames' grids onto the current one (`setup_temporal_mvs`,
     `project_temporal_mvs`, `temporal_candidates`) before it scans the
     top/left corner and the secondary rows. Which frames contribute, and with
     which distance ratio, follows the reference decoders exactly, including
     the "at most three, or two when the last frame is not the golden one"
     rule and the 8x8 band a vector is allowed to land in. `AV1GPU_TRACE_T=1`
     dumps the saved (`T`) and projected (`U`) grids in the shape a patched
     dav1d prints, and on the `gm6` fixture the saved grid of frame 1 (1191
     cells) and the projected grid of frame 2 (1041 cells) are cell for cell
     identical, block trace included.
   * *Skip mode*, done: a skipped block takes both references and both motion
     vectors from the pair of reference frames the frame header names, at the
     frame's vector precision, with no mode symbols, no residual and the
     average blend. The frame header derives the pair (`skip_mode_params`), the
     block layer reads the flag with its own two neighbour context, and the
     fixtures code it 1033 (`inter_intra`), 2220 (`lw_t2`), 2042 (`lw_t3`) and
     43 (`svt_av1_320x240`) times - bit exact, which is what pins the pair, the
     vector and the blend.
   * *Compound prediction.* The syntax (reference pair, `comp_group_idx`,
     `compound_idx`, wedge index and sign) plus the engine's masked blend; the
     kernels already implement the average and distance blends.
   * *OBMC and local warped motion*, both of which need neighbour motion
     vectors from the map that `mark_inter` already fills.
   * *Reference sample buffers and inter reconstruction*: the reference
     refresh in the decoder loop, the engine's inter jobs for these blocks, and
     `show_existing_frame` output. The syntax is there; nothing draws it yet.

   Two bugs the block level trace found on the way, worth recording because
   both are the kind that decodes plausibly and then desynchronises: the single
   reference tree's table is indexed `[context][step]` (the steps are not in
   coding order, so three second-level tests live in rows 2, 4 and 5), and
   filter intra, palette and the transform depth symbol are intra only - an
   inter block that reads them consumes symbols that belong to the next block.

   The first inter frame of `tests/data/svt_av1_320x240.ivf` now parses block
   for block against dav1d's own symbol trace up to the first block that codes
   a variable transform size, and four more of the same kind came out of that
   comparison:

   * a table that is copied once per motion vector component has to be copied
     *per row* - copying it as one block puts row 2 where row 1 of the first
     component belongs, which only shows up once a component's probabilities
     have adapted;
   * `mv_joint` counts from `MV_JOINT_HNZVZ`, so bit 0 is the horizontal
     component, while the two components' tables are indexed vertically first
     and a block that codes both reads the vertical one first;
   * OBMC's neighbour test asks whether an odd 4x4 offset along the edge holds
     an *inter* predicted block (`findoddzero` returns true on a zero entry of
     the intra map), not an intra one - inverted, the block skips a symbol;
   * inter/intra compound is only allowed up to 32x32, not 64x64 - and its
     neighbour scan must not see an inter/intra neighbour as a single reference
     one, which is what `MiRef::interintra` is for: the specification writes -1
     in `RefFrames[][][1]` for a single reference block and an inter/intra block
     keeps INTRA there, so comparing that field alone accepts a neighbour
     `find_warp_samples` and `find_matching_ref` have to skip;
   * `needs_interp_filter()` is false for a `LOCALWARP` block: reading a subpel
     filter there consumes the next block's first symbol, and the desync only
     shows up as a pixel difference a few frames later.
2. *A second intra frame to check against.* One reference picture proves a lot
   but it cannot cover filter intra, every transform shape or a 128x128
   superblock; a stream per feature class is cheap insurance and needs an
   encoder (or a hand-patched fixture) to produce.
3. *Restoration.* `loop_restoration` is currently rejected at the frame header;
   the fixture signals it off, so nothing below it has been exercised.

## 2. Subpel filters from the specification — done

`Subpel_Filters` (six rows: 8 tap regular/smooth/sharp, bilinear, and the 4 tap
variants), 1/16 pel precision, the two stage rounding of the specification and
the 1/8 pel motion vectors it is driven with. Inter prediction with a single
reference and with compound averaging is bit exact against dav1d.

## 3. Masked compound, warped motion, and OBMC — done

Wedge and difference weighted masks are per-sample blend weights, so they fit
the existing "one work item per output sample" model without new
synchronisation. Both are in `av1g_inter_px`: the wedge mask comes from the
`Wedge_Master` table (16 directions x corner offsets, selected per block shape
with the specification's flipSign rule), and the difference weighted mask is
`Clip3(0, 64, 38 + Round2(abs(pred0 - pred1), (BitDepth - 8) + InterPostRound) /
16)`. Two details cost a hunt each and are worth writing down:

* The difference mask is built on the **luma** grid even when the plane being
  predicted is chroma: 7.11.3.12 only runs for plane 0, and 7.11.3.14 then
  averages that array over the two or four luma samples under each chroma
  sample (with the specification's rounding, `(m0+m1+m2+m3+2)>>2`). A chroma
  job therefore has to *recompute* the two luma predictions - which is why the
  job carries the luma plane, the patch origin and the visible size.
* Those recomputed predictions have to go through the same warped-motion path
  the luma plane took. `useWarp` is decided per prediction region, so an 8x8
  luma region can be warped while the 4x4 chroma region under it is not: the
  job carries both decisions (`warp` for this plane, `mask_warp` for the luma
  region) and the difference mask follows the luma one. Mixing the two up is a
  one-level error on a handful of chroma samples, which is what `w_cu2.ivf`
  kept showing.

Global warped motion (7.11.3.5, the block warp process with the setup shear of
7.11.3.6 and the resolve divisor of 7.11.3.7) is implemented as two shears in
`av1g_warp_tap`: the horizontal pass builds the intermediate rows at 16 times
the sample precision with a phase that drifts by alpha/beta across the section,
and the vertical pass brings them back with gamma/delta. `Warped_Filters` is
appended to the same table the subpel path uploads, so a backend still transfers
one table. Every fixture whose global motion is TRANSLATION or absent decodes
bit exactly, and so do the ROTZOOM ones (`w_cu2`, `w_cu4`) now.

Local warped motion (7.11.3.8) is the other half of the warp story: instead of a
model the frame header codes, the model is fitted to the block's *neighbours*, so
it has to be derived while the frame is parsed. `find_warp_samples` walks the top
and then the left edge of the block and collects up to
`LEAST_SQUARES_SAMPLES_MAX` (8) neighbours that use the block's reference frame
and are not compound, each as the centre of its block and that centre moved by
its vector, in 1/8 luma samples. A neighbour whose vector disagrees with the
block's by more than `Clip3(16, 112, max(Block_Width, Block_Height))` is not part
of the same motion, so it is left out of the fit; if none at all qualifies, the
first one scanned is used on its own and the fit degenerates to a translation.
The fit itself (`frontend/warp.hpp`) is the specification's 2x2 normal equations
solve over `ls_product(a, b) = ((a * b) >> 2) + (a + b)`, including the +8/+4
bias terms that absorb the model's constant column. Its determinant is a product
of two squared sums and does not fit 32 bits, so the divisor of 7.11.3.7 is
resolved at 64 and the divisions use 64 bit products. The result takes exactly
the same road a global model does - setup shear, then the same kernel - and a
model that fails the shear check is `LocalValid = 0`: the block is then predicted
by its vector.

Three traps, one of which cost a hunt:

* Motion vector components are stored **x first** everywhere in this decoder,
  while the specification writes the row component first. Both the sample
  positions and the fit must use the specification's order.
* `useWarp` is decided **per plane**; only the fit is luma only. A chroma region
  of the same block is warped with the same parameters whenever it is at least
  8x8. Leaving chroma translational is a one-level error on a few dozen chroma
  samples per frame - invisible in PSNR, and exactly what the first stream built
  for this showed.
* The "vector too far from the block's" test skips a sample, it does not end the
  scan: the specification only stops after eight candidates.

No fixture in this repository codes LOCALWARP (`motion_mode` is `SIMPLE` in all
of them), so this was developed against streams from libaom: 320x240, 30 frames,
`--cpu-used=2 --crf=30 --lag-in-frames=25 --enable-restoration=0 --tiles=1x1`,
which codes about 60 local-warp blocks over those 30 frames. Two such streams
(the ROTZOOM clip behind `w_cu2`, and the same clip at `--cpu-used=3`) decode bit
exactly against dav1d, 30 of 30 frames each.

OBMC is implemented (mask table, per-strip prediction, the above-then-left pass
order) and verified on real streams: the two clips above re-encoded with OBMC
left enabled decode bit exactly against dav1d, 30 of 30 frames each, and are now
in `tests/data/` with a hash based regression test. Two things in the candidate
scan had to be right for that, and both were wrong at first:

* The block's job carries the strips to blend, and the array held four. The scan
  takes at most `min(4, log2(size))` neighbours on the top edge and as many on
  the left, and a strip advances by `Clip3(2, neighbour_size, 16)` 4x4 units, so
  a 32x32 block legitimately produces three strips above and three from the left
  - eight in total, which is what `Av1gInterJob.obmc[8]` holds. Four was enough
  for the small blocks the synthetic tests use, which is why nothing caught it.
* `Clip3(2, 16, W)` and `Clip3(2, W, 16)` are not the same function. The first
  clamps the neighbour's width into 2..16, so every neighbour wider than 16 mi
  advanced by 16 and every narrower one advanced by its width - accidentally
  right for narrow neighbours, wrong for wide ones, and a wrong *step* also moves
  the strip's origin, so the blend covered the wrong samples.

Both mistakes produce a difference of a few dozen samples per frame, which reads
like a rounding difference in the blend itself; that is the trap this took a hunt
to get out of.

The inter path is now complete. Inter/intra compound came last: the intra half of a block is predicted
over the block's full size, from the reconstructed neighbours of the frame being
decoded, and blended with the inter half at the taper (`II_V`, `II_H`,
`II_SMOOTH`, flat `II_DC`) or the wedge mask of a `wedge_interintra` block, with
a sign of zero. `tests/data/inter_intra.ivf` pins it: 23 such blocks over 110
frames, the taper and the wedge, every frame bit exact against dav1d and through
the device kernels (see `docs/ACCURACY.md`).

The intra tools are complete: palette (see `tests/data/palette_sc.ivf`) and
intra block copy (see `tests/data/intrabc.ivf`), both bit exact against dav1d and
through the device kernels. Intrabc is the tool that predicts from inside the
frame the decoder is still building, so it is the one place where the wavefront
schedule has to wait for a rectangle of samples rather than for the neighbouring
edges: the block's vector is clipped to the part of the tile that already exists
(7.11.5), and the scheduler turns that rectangle into wave dependencies.

Of the tools AV1 defines, what is left is superres - refused by name rather
than misparsed - plus a film grain synthesis filter (section 4). Every inter and
intra prediction tool decodes bit exactly, and so does segmentation, which is
not a prediction tool but a per block quantiser and loop filter override with
three forced decisions attached (`tests/data/seg_altq.ivf`, see
`docs/ACCURACY.md`). The reference machinery is complete: short signalling,
frame ids, `frame_size_with_refs` and the pool invalidation rules of 5.9.2 are
implemented and pinned by tests (see `docs/ACCURACY.md`).

## 4. Spec-conformant film grain

Replace the synthetic template with the specification's AR filter and overlap
blending, and pick up the per-plane grain parameters from the bitstream.

## 5. Faster scheduling for intra frames

An intra frame currently needs a wave per block diagonal (~142 waves for
640×360), which is a lot of kernel launches. Options, in increasing order of
effort:

* reduce the grid cell to the smallest block size in the frame (cheap, already
  helps),
* decode all three planes through one dependency grid so chroma does not add a
  second diagonal pass (this is where most of the 142 comes from),
* a persistent kernel that walks the wavefront on the device (needs an atomics
  based scheduler; well worth it for 1080p+).

## 6. Zero-copy output, and being usable by a browser

`Decoder::to_rgb` converts on the device, but the result is copied back to host
memory. For a browser to pick the decoder up transparently there is a layer
missing on top of the library, and it is the layer `compute-vp9` (a VP9 port of
the same idea, Vulkan compute shaders for cards without a fixed function VP9
decoder) already built for VA-API:

* **A VA-API driver plugin.** ~950 lines in `compute-vp9/src/vaapi/`: the
  entrypoints Chrome and Firefox call (`vaQueryConfigProfiles`,
  `vaCreateSurfaces2`, `vaBeginPicture`/`vaRenderPicture`/`vaEndPicture`,
  `vaExportSurfaceHandle`), with `VAProfileAV1Profile0` in place of the VP9
  profiles. Chrome listed that driver as "hardware accelerated" even though the
  decode is compute shaders, which is exactly the effect we want.
* **NV12 pack straight into the surface.** A VA surface is NV12 with a pitch of
  its own, so the last step is planar YUV in, interleaved UV out, on the device.
  That is a small kernel (their `vp9_nv12_pack.comp.glsl` is 59 lines) and it
  removes the CPU copy that would otherwise give back the CPU time we saved.
* **DMA-BUF export** (`VK_EXT_external_memory_dma_buf`; for us the OpenCL
  equivalents) so the compositor imports the frame with no copy at all.
* **A pipelined decode API.** `compute-vp9` keeps three frames in flight and
  offers a non-blocking `cvp9_get_frame` next to the blocking one. Our engine is
  synchronous per frame; a submission queue of that shape is what lets the host
  parse frame N+1 while the device reconstructs frame N, which is the only way
  the CPU share stays where the benchmarks say it should be.

Order matters: inter frames first (a browser cannot use an intra-only decoder),
then this. None of it touches the kernels.

## 7. Bit depth and chroma coverage

10/12 bit already flows through the kernels (samples are 16 bit, `maxv` is
derived from the bit depth) but is untested; 4:4:4 is supported by the layout
code and untested. Both need tests and a real stream to verify against.

## 8. Verification on a real GPU

Everything in this repository is checkable without a device except the driver
interface itself, and that is where a port to another machine usually breaks:
device queries, buffer sizes, the layouts the kernels declare, launch shapes, and
how much CPU time is left over while the card works. `tests/gpu_verify.py` (and
`docs/GPU.md`) is that check: synthetic parity against the CPU reference, then the
fixture streams decoded through the GPU backend with the fallback disabled.

As of this writing it has run on OpenCL — on PoCL, a CPU device, which is enough
to prove the plumbing and the arithmetic but says nothing about speed — and not at
all on CUDA, which has had no NVIDIA device here. The first run of that shape is
what the numbers in this file are waiting for: which frame sizes play in real
time, what the CPU share looks like next to it, and whether the wave scheduling in
section 5 is the right thing to attack first.
