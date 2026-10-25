/*
 * AV1onGPU - the deblocking edge list a frame's syntax implies
 *
 * The engine does not decide *where* to deblock: it applies the edges it is
 * given. Working them out is a decoding step of its own, because an edge is not
 * a fixed grid line. Four separate conditions decide whether an edge is
 * filtered and how strongly:
 *
 *   - on screen and not the frame border (nothing to filter against outside),
 *   - a transform block edge (the filter never crosses one),
 *   - either a prediction block edge, or a block that has no residual, or an
 *     intra block (a skipped inter block is only filtered along prediction
 *     block boundaries),
 *   - the level that the frame header gives for this plane, pass, segment,
 *     reference frame and mode, after the delta-LF syntax.
 *
 * The width of the filter follows from the transform sizes on both sides, and
 * chroma caps it lower than luma so that a filter cannot reach into the next
 * transform block of the other plane. Everything here follows the
 * specification's edge loop filter process (7.14.2) and its helpers, so that
 * the same rules that a reference decoder applies end up in the job list.
 *
 * Edges are emitted per 4x4 position, i.e. one job per edge of four samples,
 * with the level and filter size of that position. The vertical pass has to run
 * before the horizontal one, which is how the engine already schedules it.
 */
#ifndef AV1GPU_FRONTEND_DEBLOCK_HPP
#define AV1GPU_FRONTEND_DEBLOCK_HPP

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "frontend/block_size.hpp"
#include "frontend/headers.hpp"
#include "frontend/tile_decoder.hpp"
#include "frontend/tx_size.hpp"

namespace av1gpu {
namespace frontend {

/* MAX_LOOP_FILTER: levels and the delta-LF syntax are clipped to this. */
constexpr int kMaxLoopFilter = 63;

inline int clip3(int lo, int v, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* The per-4x4 view of a decoded tile that the loop filter reads. Everything is
 * on the luma mi grid except the transform sizes, which live on each plane's
 * own 4x4 grid (that is how the specification addresses LoopfilterTxSizes). */
struct DeblockGrid {
  int mi_cols = 0;
  int mi_rows = 0;
  int cols[3] = {0, 0, 0}; /* per plane 4x4 grid */
  int rows[3] = {0, 0, 0};

  std::vector<BlockSize> size;  /* MiSizes */
  std::vector<uint8_t> skip;    /* Skips */
  std::vector<uint8_t> segment; /* SegmentIds */
  /* RefFrames[..][0] and YModes: the level of an inter edge depends on which
   * reference the block predicts from and on whether its mode is one that codes
   * a motion vector difference, which is what the specification's modeType
   * distinguishes. */
  std::vector<uint8_t> ref_frame;
  std::vector<uint8_t> mode;
  std::vector<int8_t> delta_lf; /* DeltaLFs, four per cell */
  std::vector<TxSize> tx[3];    /* LoopfilterTxSizes per plane */

  int cell(int row, int col) const { return row * mi_cols + col; }
  int tx_cell(int plane, int row, int col) const { return row * cols[plane] + col; }
};

/* seg_feature_active_idx: a feature only applies when segmentation is on and
 * the segment enables it. */
inline bool seg_feature_active(const FrameHeader& frame, int segment, int feature) {
  return frame.segmentation.enabled && frame.segmentation.feature_enabled[segment][feature];
}

/* loop_filter_level[i], with the specification's index: 0 is luma vertical, 1
 * luma horizontal, 2 chroma U, 3 chroma V. */
inline int loop_filter_level(const FrameHeader& frame, int index) {
  switch (index) {
    case 0: return frame.loop_filter.level_y[0];
    case 1: return frame.loop_filter.level_y[1];
    case 2: return frame.loop_filter.level_u;
    default: return frame.loop_filter.level_v;
  }
}

/* The adaptive filter strength selection process (7.14.5): the plane's level,
 * the segment's alternative level, then the reference frame and mode deltas,
 * scaled by how strong the level already is. */
inline int deblock_level_selection(const FrameHeader& frame, const DeblockGrid& grid, int row,
                                   int col, int plane, int pass) {
  const int index = plane == 0 ? pass : plane + 1;
  const int cell = grid.cell(row, col);
  const int segment = grid.segment[static_cast<size_t>(cell)];
  const int ref = grid.ref_frame[static_cast<size_t>(cell)];
  const int mode = grid.mode[static_cast<size_t>(cell)];
  const int delta_lf = grid.delta_lf[static_cast<size_t>(cell) * 4 +
                                    (frame.delta_lf_multi ? index : 0)];
  int level = clip3(0, delta_lf + loop_filter_level(frame, index), kMaxLoopFilter);
  const int feature = kSegAltLfYV + index;
  if (seg_feature_active(frame, segment, feature))
    level = clip3(0, level + frame.segmentation.feature_data[segment][feature], kMaxLoopFilter);
  if (frame.loop_filter.mode_ref_delta_enabled) {
    /* modeType is 1 for an inter mode that codes a motion vector difference -
     * NEARESTMV and friends - and 0 for an intra block, GLOBALMV and
     * GLOBAL_GLOBALMV. An intra block only ever pays the intra reference
     * delta. */
    const int mode_type =
        (ref > kRefIntra && mode != kModeGlobalMv && mode != kModeGlobalGlobalMv) ? 1 : 0;
    const int n_shift = level >> 5;
    if (ref <= kRefIntra) {
      level += frame.loop_filter.ref_deltas[0] << n_shift;
    } else {
      level += (frame.loop_filter.ref_deltas[ref] << n_shift) +
               (frame.loop_filter.mode_deltas[mode_type] << n_shift);
    }
    level = clip3(0, level, kMaxLoopFilter);
  }
  return level;
}

/* Builds every deblocking edge of one intra frame. Returns false only when the
 * tile cannot be filtered at all (no reconstruction would make it worse). */
inline bool plan_deblock(const SequenceHeader& seq, const FrameHeader& frame,
                         const TileDecodeResult& tile, FrameSyntax* out,
                         std::string* error) {
  std::string local;
  if (!error) error = &local;
  const int width = frame.width, height = frame.height;
  const int ss_x = (!seq.monochrome && seq.ss_x) ? 1 : 0;
  const int ss_y = (!seq.monochrome && seq.ss_y) ? 1 : 0;
  const int planes = seq.monochrome ? 1 : 3;
  const int mi_cols = (width + 3) >> 2;
  const int mi_rows = (height + 3) >> 2;

  DeblockGrid grid;
  grid.mi_cols = mi_cols;
  grid.mi_rows = mi_rows;
  const size_t luma_cells = static_cast<size_t>(mi_cols) * mi_rows;
  grid.size.assign(luma_cells, kBlock4x4);
  grid.skip.assign(luma_cells, 0);
  grid.segment.assign(luma_cells, 0);
  grid.ref_frame.assign(luma_cells, kRefIntra);
  grid.mode.assign(luma_cells, 0);
  grid.delta_lf.assign(luma_cells * 4, 0);
  for (int p = 0; p < 3; ++p) {
    const int sub_x = p > 0 ? ss_x : 0;
    const int sub_y = p > 0 ? ss_y : 0;
    grid.cols[p] = (mi_cols + (1 << sub_x) - 1) >> sub_x;
    grid.rows[p] = (mi_rows + (1 << sub_y) - 1) >> sub_y;
    grid.tx[p].assign(static_cast<size_t>(grid.cols[p]) * grid.rows[p], TxSize{0, 0});
  }

  /* MiSizes, Skips, SegmentIds, YModes and the transform sizes, cell by cell. */
  for (const TileBlock& block : tile.blocks) {
    const int bw4 = block_width4(block.size);
    const int bh4 = block_height4(block.size);
    for (int y = 0; y < bh4; ++y) {
      for (int x = 0; x < bw4; ++x) {
        const int row = block.mi_row + y, col = block.mi_col + x;
        if (row >= mi_rows || col >= mi_cols) continue;
        const size_t cell = static_cast<size_t>(row) * mi_cols + col;
        grid.size[cell] = block.size;
        grid.skip[cell] = block.skip ? 1 : 0;
        grid.segment[cell] = static_cast<uint8_t>(block.segment_id);
        grid.ref_frame[cell] = static_cast<uint8_t>(block.is_inter ? block.ref_frame[0] : 0);
        grid.mode[cell] = static_cast<uint8_t>(block.y_mode);
        for (int i = 0; i < 4; ++i)
          grid.delta_lf[cell * 4 + i] = static_cast<int8_t>(block.delta_lf[i]);
      }
    }
    /* A block with chroma that is coded as a sub 8x8 block does not own its
     * chroma; has_chroma says which block does, so the chroma transform sizes
     * land on exactly the cells that carry them. */
    for (const TileTransform& transform : block.transforms) {
      const int p = transform.plane;
      if (p < 0 || p >= planes) continue;
      const TxSize tx = transform.size;
      if (tx.lw < 0 || tx.lh < 0) continue;
      const int x0 = transform.x >> 2, y0 = transform.y >> 2;
      for (int y = y0; y < y0 + tx.h4() && y < grid.rows[p]; ++y) {
        for (int x = x0; x < x0 + tx.w4() && x < grid.cols[p]; ++x) {
          if (x >= 0 && y >= 0) grid.tx[p][static_cast<size_t>(y) * grid.cols[p] + x] = tx;
        }
      }
    }
  }

  /* ---- the edge loop ---------------------------------------------------- */
  for (int plane = 0; plane < planes; ++plane) {
    /* The specification only walks a chroma plane that has a level at all. */
    if (plane > 0 && loop_filter_level(frame, plane + 1) == 0) continue;
    const int sub_x = plane > 0 ? ss_x : 0;
    const int sub_y = plane > 0 ? ss_y : 0;
    const int plane_w = (width + (1 << sub_x) - 1) >> sub_x;
    const int plane_h = (height + (1 << sub_y) - 1) >> sub_y;
    const bool chroma = plane > 0;

    for (int pass = 0; pass < 2; ++pass) {
      const bool vertical = pass == 0;
      const int row_step = chroma ? (1 << sub_y) : 1;
      const int col_step = chroma ? (1 << sub_x) : 1;
      for (int row = 0; row < mi_rows; row += row_step) {
        for (int col = 0; col < mi_cols; col += col_step) {
          /* the plane's cell: the specification keeps row/col on the luma grid
           * and skips the cells the plane does not have */
          const int prow = row | sub_y, pcol = col | sub_x;
          if (prow >= mi_rows || pcol >= mi_cols) continue;
          const int x = col * 4, y = row * 4;
          if (x >= width || y >= height) continue;
          if (vertical ? x == 0 : y == 0) continue;
          const int x_p = x >> sub_x, y_p = y >> sub_y;
          if (x_p >= plane_w || y_p >= plane_h) continue;

          const int prev_row = prow - (vertical ? 0 : (1 << sub_y));
          const int prev_col = pcol - (vertical ? (1 << sub_x) : 0);
          if (prev_row < 0 || prev_col < 0) continue;

          const size_t cell = static_cast<size_t>(prow) * mi_cols + pcol;
          const BlockSize mi_size = grid.size[cell];
          const BlockSize plane_size = plane_residual_size(mi_size, plane, ss_x != 0, ss_y != 0);
          const TxSize tx = grid.tx[plane][static_cast<size_t>(prow >> sub_y) * grid.cols[plane] +
                                                  (pcol >> sub_x)];
          const TxSize prev_tx =
              grid.tx[plane][static_cast<size_t>(prev_row >> sub_y) * grid.cols[plane] +
                              (prev_col >> sub_x)];
          if (tx.lw < 0 || prev_tx.lw < 0) continue;

          /* isBlockEdge: the edge is on a prediction block boundary. A size the
           * specification's table has no entry for can only come from
           * asymmetric chroma subsampling, where the residual is 4x4 anyway. */
          bool is_block_edge;
          if (plane_size >= kBlockSizeCount) {
            is_block_edge = true;
          } else {
            const int block_w = block_width4(plane_size) * 4;
            const int block_h = block_height4(plane_size) * 4;
            is_block_edge = vertical ? (x_p % block_w == 0) : (y_p % block_h == 0);
          }
          const int tx_w = tx.width(), tx_h = tx.height();
          const bool is_tx_edge = vertical ? (x_p % tx_w == 0) : (y_p % tx_h == 0);
          if (!is_tx_edge) continue;
          /* applyFilter: a transform edge, and either a prediction block edge,
           * a block without residual, or an intra block. A skipped inter block
           * is therefore only filtered where the prediction block changes, not
           * at the transform boundaries inside it. */
          const bool is_intra_block = grid.ref_frame[cell] <= kRefIntra;
          const bool has_residual = grid.skip[cell] == 0;
          if (!(is_block_edge || has_residual || is_intra_block)) continue;

          /* The filter size process: the smaller transform on the two sides of
           * the edge caps the filter, 16 in luma and 8 in chroma. */
          const int base_size = vertical ? std::min(prev_tx.width(), tx_w)
                                         : std::min(prev_tx.height(), tx_h);
          const int filter_size = std::min(chroma ? 8 : 16, base_size);

          int level = deblock_level_selection(frame, grid, prow, pcol, plane, pass);
          if (level == 0) {
            level = deblock_level_selection(frame, grid, prev_row, prev_col, plane, pass);
            if (level == 0) continue;
          }

          /* One job per 4x4 position: the four samples along the edge share the
           * level, and the last group of a frame whose size is not a multiple of
           * four is shortened rather than allowed to run past the plane. */
          const int along = vertical ? y_p : x_p;
          const int along_end = vertical ? plane_h : plane_w;
          const int length = std::min(4, along_end - along);
          if (length <= 0) continue;
          DeblockEdge edge;
          edge.plane = plane;
          /* x/y address the sample on the far side of the edge (q0) */
          edge.x = x_p;
          edge.y = y_p;
          edge.vertical = vertical;
          edge.length = length;
          edge.level = level;
          edge.filter_size = filter_size;
          out->deblock.push_back(edge);
        }
      }
    }
  }

  out->sharpness = frame.loop_filter.sharpness;
  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_DEBLOCK_HPP */
