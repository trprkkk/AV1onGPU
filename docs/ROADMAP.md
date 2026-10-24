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

**Started: the default probability tables.** `src/frontend/cdf.hpp` holds the
specification's default tables for the partition symbol (all five block levels,
all four contexts) and for skip, in the inverse-CDF layout the range decoder
consumes. `tests/test_entropy.cpp` decodes the first symbol of the fixture's
key frame out of those tables and requires dav1d's value, so the table
convention (32768 - x, counter after the probabilities) is pinned against a real
stream rather than only against the specification text. What is still missing is
everything *above* the symbol reader: the default tables for the rest of the
syntax, and the syntax layer that turns symbols into `FrameSyntax`.

**Next, in order:**

1. *CDF tables and the tile syntax.* Finish the default tables (mode info,
   transform and coefficient syntax), then the superblock/partition loop that
   consumes them, plus `exit_symbol` and the end-of-frame CDF update. The trace
   of the fixture is the oracle for each layer: every symbol must come out of
   the tables and in the order dav1d decoded it.
2. *Tile and partition tree.* Superblock loop, partition symbols, block sizes,
   transform selection, and the coefficient syntax with its scan tables. This is
   the bulk of the work and the point where the front-end first emits a
   `FrameSyntax` the engine can decode.
3. *Reference frame management in the engine.* The front-end already tracks the
   eight slots for header parsing; the decoder needs the actual sample buffers,
   reference scaling and `show_existing_frame` output.
4. *Optional:* a libaom / dav1d hook that emits syntax from an already
   conformance-tested parser. Useful for testing the engine against real
   streams long before our own parser is complete.

Schedule: one `SyntaxBuilder` class, no changes to the backends.

## 2. Subpel filters from the specification

Currently Lanczos-windowed sinc at 1/8 pel. Real streams need the spec's four
16-phase filter sets, and 1/16 pel motion vector precision. This is a contained
change in `src/core/tables.cpp` (`FilterTable`) plus the MV scaling in the
engine; the kernels already read the taps from a table.

## 3. Overlapped block motion compensation and masked compound

Wedge/difference masks and OBMC are the two inter tools missing from
`av1gpu_inter_px`. Both are per-sample blend weights, so they fit the existing
"one work item per output sample" model without new synchronisation.

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

## 6. Zero-copy output

`Decoder::to_rgb` converts on the device, but the result is copied back to host
memory. Binding the OpenCL/CUDA buffer to a texture / `cl_khr_gl_sharing`
surface would let a player present it directly.

## 7. Bit depth and chroma coverage

10/12 bit already flows through the kernels (samples are 16 bit, `maxv` is
derived from the bit depth) but is untested; 4:4:4 is supported by the layout
code and untested. Both need tests and a real stream to verify against.
