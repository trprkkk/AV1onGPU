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
Inter frames read a `RefPool` the caller keeps up to date, so reference sizes and
skip mode are derived, not guessed. Features that are not implemented (superres,
intra block copy, restoration, global motion, short reference signalling, frame
ids, film grain data) are reported with a message instead of being misparsed.
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

   * the inter frame preamble: skip mode (read, then refused), skip, `cdef`,
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
   * *Skip mode.* A skipped block takes both references and both motion vectors
     from the pair of reference frames the frame header names; its context is
     already tracked (`skip_mode_map_`).
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
   * inter/intra compound is only allowed up to 32x32, not 64x64.
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

## 3. Masked compound, warped motion, and OBMC's first fixture

Wedge and difference weighted masks are the two inter tools still missing from
`av1gpu_inter_px`. Both are per-sample blend weights, so they fit the existing
"one work item per output sample" model without new synchronisation: the wedge
tables are `Wedge_Master` (16 directions x 8 corner offsets per block shape) and
the difference mask is two sums over the two predictors against a 1/64 threshold
in `preds[][0]`, both spelled out in the specification's sections 7.11.3.11 and
7.11.3.12.

The order in which a decoder meets the remaining inter tools, measured on plain
libaom encodes with every tool at its default, is masked compound, then warped
motion, then inter/intra compound.

Warped motion is now the one that shows: `w_cu2.ivf` codes global motion of type
ROTZOOM from its first inter frame on, and a translator that ignores the affine
part diverges in every following frame (luma included), while every fixture
whose global motion is TRANSLATION or absent decodes bit exactly. OBMC and warped motion share the neighbour
search, and the OBMC blend is already implemented (mask table, per strip
prediction, the two passes, in the shared kernel) - but none of the fixtures
built so far code `motion_mode` as anything but `SIMPLE`, so the only thing
proven about it today is that it costs nothing when unused. A fixture that
actually uses OBMC needs libaom's speed settings raised until the encoder picks
it, and is worth building before the code is trusted.

Intrabc and palette are the remaining intra tools, and they only appear on
screen content streams.

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
