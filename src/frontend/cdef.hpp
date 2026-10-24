/*
 * AV1onGPU - the CDEF block list a frame's syntax implies
 *
 * CDEF derings the picture in fixed eight by eight blocks of luma samples (four
 * by four of chroma at 4:2:0), and every one of them filters with the
 * parameters of the 64x64 block it sits in. The index into those parameters is
 * the cdef_idx syntax the tile decoder reads, and a 64x64 that was never
 * claimed by a block that is not skipped keeps idx -1: it is not filtered at
 * all, only copied.
 *
 * A block whose four 4x4 cells are all skipped is not filtered either, and a
 * plane whose strength is zero has nothing to do. What is left is the list of
 * blocks the engine has to run.
 *
 * Everything the filter still needs is per block: the direction (from the luma,
 * which is why the job carries the co-located luma block even for chroma) and
 * the strength that the block's own variance earns. Both depend on the
 * deblocked samples rather than on the bitstream, so they stay in the kernel -
 * this file only decides which blocks filter, with which parameters.
 */
#ifndef AV1GPU_FRONTEND_CDEF_HPP
#define AV1GPU_FRONTEND_CDEF_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "frontend/block_size.hpp"
#include "frontend/headers.hpp"
#include "frontend/tile_decoder.hpp"

namespace av1gpu {
namespace frontend {

/* The primary strength is the top four bits of the six the header codes, the
 * secondary the bottom two - where 3 means 4, which is how the specification
 * spends the unused value. */
inline int cdef_primary_strength(int coded) { return coded >> 2; }
inline int cdef_secondary_strength(int coded) {
  const int sec = coded & 3;
  return sec == 3 ? 4 : sec;
}

/*
 * Fills out->cdef with the blocks the CDEF process filters, one entry per eight
 * by eight luma block and plane. `tile` supplies the cdef indices and the skip
 * map; `frame` the strengths and the damping.
 */
inline bool plan_cdef(const SequenceHeader& seq, const FrameHeader& frame,
                      const TileDecodeResult& tile, FrameSyntax* out,
                      std::string* error) {
  (void)error;
  if (!seq.enable_cdef) return true;

  const int strength_count = 1 << frame.cdef.bits;
  bool any = false;
  for (int i = 0; i < strength_count; ++i) {
    if (frame.cdef.y_strength[i] || frame.cdef.uv_strength[i]) any = true;
  }
  if (!any) return true;

  const int mi_cols = (frame.frame_width + 3) >> 2;
  const int mi_rows = (frame.height + 3) >> 2;
  const int ss_x = seq.ss_x, ss_y = seq.ss_y;
  const int planes = seq.monochrome ? 1 : 3;
  const int width = frame.frame_width, height = frame.height;

  /* Skips per 4x4 cell, which is what the block process looks at. */
  std::vector<uint8_t> skip(static_cast<size_t>(mi_cols) * mi_rows, 0);
  for (const TileBlock& block : tile.blocks) {
    const int bw4 = block_width4(block.size);
    const int bh4 = block_height4(block.size);
    for (int y = 0; y < bh4; ++y) {
      for (int x = 0; x < bw4; ++x) {
        const int row = block.mi_row + y, col = block.mi_col + x;
        if (row >= mi_rows || col >= mi_cols) continue;
        skip[static_cast<size_t>(row) * mi_cols + col] = block.skip ? 1 : 0;
      }
    }
  }

  const int cdef_cols = tile.cdef_cols;
  const int cdef_rows = tile.cdef_rows;
  if (cdef_cols <= 0 || cdef_rows <= 0 ||
      tile.cdef_idx.size() != static_cast<size_t>(cdef_cols) * cdef_rows) {
    /* no index was read at all: the frame was coded without CDEF syntax */
    return true;
  }

  const int damping = frame.cdef.damping;

  for (int row = 0; row < mi_rows; row += 2) {
    for (int col = 0; col < mi_cols; col += 2) {
      const int cell_row = row >> 4, cell_col = col >> 4;
      const int idx =
          tile.cdef_idx[static_cast<size_t>(cell_row) * cdef_cols + cell_col];
      if (idx < 0 || idx >= strength_count) continue;

      /* a block the encoder skipped completely has nothing to dering */
      bool skipped = true;
      for (int y = 0; y < 2 && skipped; ++y) {
        for (int x = 0; x < 2; ++x) {
          const int r = row + y, c = col + x;
          if (r < mi_rows && c < mi_cols &&
              !skip[static_cast<size_t>(r) * mi_cols + c]) {
            skipped = false;
            break;
          }
        }
      }
      if (skipped) continue;

      const int x = col * 4, y = row * 4;
      if (x >= width || y >= height) continue;

      const int pri_y = cdef_primary_strength(frame.cdef.y_strength[idx]);
      const int sec_y = cdef_secondary_strength(frame.cdef.y_strength[idx]);
      if (pri_y || sec_y) {
        CdefBlock block;
        block.plane = 0;
        block.x = x;
        block.y = y;
        block.pri_strength = pri_y;
        block.sec_strength = sec_y;
        block.damping = damping;
        block.dir = -1; /* the kernel finds it, as a decoder must */
        out->cdef.push_back(block);
      }

      if (planes < 3) continue;
      const int pri_uv = cdef_primary_strength(frame.cdef.uv_strength[idx]);
      const int sec_uv = cdef_secondary_strength(frame.cdef.uv_strength[idx]);
      if (!pri_uv && !sec_uv) continue;
      for (int plane = 1; plane < 3; ++plane) {
        CdefBlock block;
        block.plane = plane;
        block.x = x >> ss_x;
        block.y = y >> ss_y;
        block.pri_strength = pri_uv;
        block.sec_strength = sec_uv;
        block.damping = damping;
        block.dir = -1;
        out->cdef.push_back(block);
      }
    }
  }
  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_CDEF_HPP */
