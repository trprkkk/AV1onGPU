/*
 * AV1onGPU - synthetic content generator
 *
 * There is no AV1 front-end in this tree yet (see docs/ROADMAP.md), so the
 * demos, benchmarks and tests need a way to produce real FrameSyntax. This
 * helper is a tiny closed loop encoder: it builds a moving test picture,
 * predicts it (intra DC for the key frame, motion compensated for the following
 * ones), transforms and quantises the residual, and reconstructs exactly the way
 * the decoder will. The result is a valid syntax tree that exercises every stage
 * of the pipeline, with a known source picture to measure PSNR against.
 *
 * It is deliberately not a conforming AV1 encoder: the point is to drive the GPU
 * pipeline with realistic work, not to produce a bitstream.
 *
 * Copyright (c) 2026 AV1onGPU contributors. BSD-3-Clause.
 */
#ifndef AV1GPU_SYNTHETIC_HPP
#define AV1GPU_SYNTHETIC_HPP

#include "av1gpu/av1gpu.hpp"

namespace av1gpu {

struct SyntheticConfig {
  int width = 352;
  int height = 288;
  int bit_depth = 8;
  bool chroma_420 = true;

  int block_size = 16;     /* luma block (16 or 32)                        */
  int transform_size = 8;  /* transform inside the block                    */
  int qstep = 10;          /* uniform quantiser step, 1 = lossless-ish      */
  int search_radius = 8;   /* motion search radius in luma samples          */
  int deblock_level = 20;  /* 0 disables the deblocking edges               */
  bool cdef = false;
  bool grain = false;
  int seed = 1;
};

/* Renders the source picture for `frame_index`. */
void synthetic_source(const SyntheticConfig& cfg, int frame_index, Frame* out);

/* Encodes `frame_index`.
 *
 *   reference  previously reconstructed frame, or null for a key frame
 *   source     optional, receives the source picture
 *   recon      optional, receives the encoder's own reconstruction
 *
 * The returned syntax has reference_slots == {0} when inter blocks are present,
 * so the caller has to fill in the device slot it wants to use. */
FrameSyntax synthetic_encode(const SyntheticConfig& cfg, int frame_index,
                             const Frame* reference, Frame* source, Frame* recon);

} /* namespace av1gpu */

#endif /* AV1GPU_SYNTHETIC_HPP */
