# Roadmap

The library stops at the entropy decoder boundary on purpose: it consumes
reconstruction syntax and produces pictures. Everything below is future work in
the order it unlocks the most.

## 1. A front-end that produces `FrameSyntax`

This is the one thing standing between the library and playing real `.ivf` /
`.obu` files. The work is an AV1 parser (symbol decoder, partition tree,
reference frame management) that emits `av1gpu::FrameSyntax`; it does *not*
touch the GPU. Reasonable sources of the syntax:

* **libaom / dav1d hook.** Both decoders are structured so that the decode of a
  block can be intercepted after parsing. Emitting syntax there is the least
  code and gives a conformance-tested parser for free.
* **A self-contained parser.** More work, but it would make the project
  dependency free, which matters for the embedded/old-machine audience.

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
