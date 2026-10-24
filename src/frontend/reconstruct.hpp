/*
 * AV1onGPU - from decoded tile syntax to a frame the engine can reconstruct
 *
 * The tile decoder produces syntax: blocks, modes, transform sizes, transform
 * types and quantised levels. The engine consumes a FrameSyntax: one
 * prediction block per transform block, plus the coefficient pool. This file
 * is the join between them, and it is where the specification's *bookkeeping*
 * lives - the parts that are not arithmetic but decide which arithmetic runs:
 *
 *   - One prediction block per *transform* block. The specification predicts
 *     and reconstructs one transform block at a time, so the second transform
 *     block of a 16x16 block sees the first one's output as its left
 *     neighbour. Planning one block per coding block instead would still give
 *     the right answer for DC, but not for any directional prediction that
 *     reads the neighbouring samples.
 *
 *   - haveLeft/haveAbove: inside the tile, and for chroma whether this luma
 *     block has chroma at all. A top level 4x4 luma block in a 4:2:0 frame
 *     shares its chroma with the block below it, so its chroma neighbour is
 *     two luma blocks away.
 *
 *   - haveAboveRight/haveBelowLeft: *not* the same kind of thing. They come
 *     from BlockDecoded, a per-plane 4x4 map of what has actually been
 *     reconstructed, with a one-block border primed at each superblock. At a
 *     superblock boundary the samples above-right exist in the frame but have
 *     not been decoded yet, and the prediction must not read them.
 *
 *   - filterType for the intra edge filter: 1 when the neighbour above or to
 *     the left used one of the smooth modes. That is a property of the
 *     neighbours, so it needs a mode map.
 *
 *   - MaxLumaW/MaxLumaH for chroma from luma, which are the luma extent of the
 *     transform block being predicted.
 *
 * Only intra frames are planned. Inter frames need reference frames, motion
 * vectors and the compound syntax, which the front end does not read yet.
 */
#ifndef AV1GPU_FRONTEND_RECONSTRUCT_HPP
#define AV1GPU_FRONTEND_RECONSTRUCT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "frontend/block_size.hpp"
#include "frontend/coef_model.hpp"
#include "frontend/deblock.hpp"
#include "frontend/dequant.hpp"
#include "frontend/headers.hpp"
#include "frontend/tile_decoder.hpp"
#include "frontend/tx_size.hpp"

namespace av1gpu {
namespace frontend {

/* coef_tables' transform type numbers are the specification's Tx_Type numbers,
 * and av1gpu::TransformType is the specification's numbering too, so this is a
 * cast with a name: it exists so that a future renumbering on either side turns
 * into a compile error or a visible default, not a silent corruption. */
inline TransformType to_transform_type(int tx_type) {
  switch (tx_type) {
    case coef_tables::kDctDct: return TransformType::DctDct;
    case coef_tables::kAdstDct: return TransformType::AdstDct;
    case coef_tables::kDctAdst: return TransformType::DctAdst;
    case coef_tables::kAdstAdst: return TransformType::AdstAdst;
    case coef_tables::kFlipadstDct: return TransformType::FlipadstDct;
    case coef_tables::kDctFlipadst: return TransformType::DctFlipadst;
    case coef_tables::kFlipadstFlipadst: return TransformType::FlipadstFlipadst;
    case coef_tables::kAdstFlipadst: return TransformType::AdstFlipadst;
    case coef_tables::kFlipadstAdst: return TransformType::FlipadstAdst;
    case coef_tables::kIdtx: return TransformType::Idtx;
    case coef_tables::kVdct: return TransformType::VDct;
    case coef_tables::kHdct: return TransformType::HDct;
    case coef_tables::kVadst: return TransformType::VAdst;
    case coef_tables::kVflipadst: return TransformType::VFlipadst;
    case coef_tables::kHadst: return TransformType::HAdst;
    case coef_tables::kHflipadst: return TransformType::HFlipadst;
    default: return TransformType::DctDct;
  }
}

/* The intra mode numbers are the specification's, and PredictionMode is the
 * specification's numbering with chroma-from-luma appended, so the mapping is
 * the identity for luma modes. */
inline PredictionMode to_prediction_mode(int mode) {
  if (mode == kModeChromaFromLuma) return PredictionMode::ChromaFromLuma;
  if (mode < 0 || mode >= kModeLumaCount) return PredictionMode::Dc;
  return static_cast<PredictionMode>(mode);
}

inline FilterIntraMode to_filter_intra_mode(int mode) {
  return mode >= 0 && mode <= 4 ? static_cast<FilterIntraMode>(mode)
                                : FilterIntraMode::Dc;
}

/* A per-4x4 map with a one-cell border on every side. BlockDecoded is exactly
 * that, and the border is how the specification answers "is the sample
 * above-right available?" at a superblock edge without special casing it. */
struct PlaneGrid {
  int cols = 0; /* real cells: 0 .. cols-1 */
  int rows = 0;
  std::vector<uint8_t> cells;

  void init(int c, int r) {
    cols = c < 1 ? 1 : c;
    rows = r < 1 ? 1 : r;
    cells.assign(static_cast<size_t>(cols + 2) * static_cast<size_t>(rows + 2), 0);
  }
  /* Clamped to the border ring, so a caller that runs off the end reads the
   * border instead of memory. Only the border is ever read past the edge. */
  int slot(int x, int y) const {
    const int cx = x < -1 ? -1 : (x > cols ? cols : x);
    const int cy = y < -1 ? -1 : (y > rows ? rows : y);
    return (cy + 1) * (cols + 2) + (cx + 1);
  }
  uint8_t& at(int x, int y) { return cells[static_cast<size_t>(slot(x, y))]; }
  uint8_t at(int x, int y) const { return cells[static_cast<size_t>(slot(x, y))]; }
  void fill(int x0, int y0, int x1, int y1, uint8_t value) {
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) at(x, y) = value;
  }
};

/* Everything the planner carries while it walks a tile. */
struct PlanState {
  int mi_cols = 0; /* luma 4x4 units, rounded up to a whole superblock */
  int mi_rows = 0;
  int sb4 = 16; /* superblock size in luma 4x4 units: 16 or 32 */
  bool ss_x = false;
  bool ss_y = false;

  PlaneGrid decoded[3]; /* BlockDecoded, in plane 4x4 units */
  PlaneGrid y_mode;     /* YModes on the luma mi grid */
  PlaneGrid uv_mode;    /* UVModes on the luma mi grid */

  int sb_col = -1; /* the superblock whose BlockDecoded this is */
  int sb_row = -1;
};

/* clear_block_decoded_flags: at the start of each superblock the map is reset,
 * except for the border cells that face the rest of the frame - they answer
 * "is there anything decoded above or to the left of this superblock?". The
 * cell to the left of the bottom row is cleared explicitly, because the block
 * below-left of the last row of a superblock may exist but is not decoded. */
inline void clear_block_decoded(PlanState* st, int r4, int c4) {
  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = plane > 0 && st->ss_x ? 1 : 0;
    const int sub_y = plane > 0 && st->ss_y ? 1 : 0;
    const int sb_w4 = (st->mi_cols - c4) >> sub_x;
    const int sb_h4 = (st->mi_rows - r4) >> sub_y;
    const int base_x = c4 >> sub_x, base_y = r4 >> sub_y;
    const int span_x = st->sb4 >> sub_x, span_y = st->sb4 >> sub_y;
    for (int y = -1; y <= span_y; ++y) {
      for (int x = -1; x <= span_x; ++x) {
        uint8_t value = 0;
        if (y < 0 && x < sb_w4) value = 1;
        else if (x < 0 && y < sb_h4) value = 1;
        st->decoded[plane].at(base_x + x, base_y + y) = value;
      }
    }
    st->decoded[plane].at(base_x - 1, base_y + span_y) = 0;
  }
  st->sb_col = c4;
  st->sb_row = r4;
}

/* is_smooth, on the luma mi grid: did the block covering this mi unit use one
 * of the smooth prediction modes? */
inline int is_smooth_at(const PlanState& st, int r4, int c4, int plane) {
  const int mode = plane == 0 ? st.y_mode.at(c4, r4) : st.uv_mode.at(c4, r4);
  return mode == kModeSmooth || mode == kModeSmoothVertical || mode == kModeSmoothHorizontal;
}

/* get_filter_type: 1 when the block above or to the left is smooth. For chroma
 * the neighbour is looked up on the chroma grid, which for a subsampled plane
 * is not the block's own mi position: the block's chroma is shared with its
 * luma neighbour, so the neighbour's position has to be adjusted. */
inline int intra_filter_type(const PlanState& st, int mi_row, int mi_col, int plane,
                             bool avail_u, bool avail_l) {
  int above_smooth = 0, left_smooth = 0;
  if (avail_u) {
    int r = mi_row - 1, c = mi_col;
    if (plane > 0) {
      if (st.ss_x && !(mi_col & 1)) ++c;
      if (st.ss_y && (mi_row & 1)) --r;
    }
    above_smooth = is_smooth_at(st, r, c, plane);
  }
  if (avail_l) {
    int r = mi_row, c = mi_col - 1;
    if (plane > 0) {
      if (st.ss_x && (mi_col & 1)) --c;
      if (st.ss_y && !(mi_row & 1)) ++r;
    }
    left_smooth = is_smooth_at(st, r, c, plane);
  }
  return above_smooth || left_smooth;
}

/* Builds the FrameSyntax of one intra frame from what the tile decoder and the
 * dequantiser produced. Returns false with a message when the tile contains
 * syntax this stage cannot express. */
inline bool plan_from_tile(const SequenceHeader& seq, const FrameHeader& frame,
                           const TileDecodeResult& tile, const DequantisedTile& quant,
                           FrameSyntax* out, std::string* error) {
  std::string local;
  if (!error) error = &local;
  error->clear();
  if (!out) {
    *error = "reconstruct: no output";
    return false;
  }
  *out = FrameSyntax();

  const int width = frame.width;
  const int height = frame.height;
  if (width <= 0 || height <= 0) {
    *error = "reconstruct: the frame has no size";
    return false;
  }

  const bool ss_x = !seq.monochrome && seq.ss_x;
  const bool ss_y = !seq.monochrome && seq.ss_y;

  out->width = width;
  out->height = height;
  out->bit_depth = seq.bit_depth;
  out->ss_x = ss_x ? 1 : 0;
  out->ss_y = ss_y ? 1 : 0;
  out->base_q_idx = frame.base_q_idx;
  out->sharpness = frame.loop_filter.sharpness;
  out->enable_intra_edge_filter = seq.enable_intra_edge_filter;

  PlanState st;
  st.sb4 = seq.use_128x128_superblock ? 32 : 16;
  st.ss_x = ss_x;
  st.ss_y = ss_y;
  /* MiCols / MiRows are the frame size rounded up to a whole 4x4 sample, and
   * then to a whole superblock: BlockDecoded is addressed relative to
   * superblocks, so the map has to cover the padding of the last one. */
  const int mi_cols = (width + 3) >> 2;
  const int mi_rows = (height + 3) >> 2;
  st.mi_cols = ((mi_cols + st.sb4 - 1) / st.sb4) * st.sb4;
  st.mi_rows = ((mi_rows + st.sb4 - 1) / st.sb4) * st.sb4;

  for (int plane = 0; plane < 3; ++plane) {
    if (plane > 0 && seq.monochrome) break;
    const int sub_x = plane > 0 && ss_x ? 1 : 0;
    const int sub_y = plane > 0 && ss_y ? 1 : 0;
    st.decoded[plane].init(st.mi_cols >> sub_x, st.mi_rows >> sub_y);
  }
  st.y_mode.init(st.mi_cols, st.mi_rows);
  st.uv_mode.init(st.mi_cols, st.mi_rows);

  out->coefficients.assign(quant.coefficients.begin(), quant.coefficients.end());
  out->blocks.reserve(tile.blocks.size());

  const bool lossless = frame.coded_lossless;
  size_t next_offset = 0;

  for (const TileBlock& block : tile.blocks) {
    const int bw4 = block_width4(block.size);
    const int bh4 = block_height4(block.size);
    if (block.mi_col >= mi_cols || block.mi_row >= mi_rows) continue;

    const int sb_c = block.mi_col & ~(st.sb4 - 1);
    const int sb_r = block.mi_row & ~(st.sb4 - 1);
    if (sb_c != st.sb_col || sb_r != st.sb_row) clear_block_decoded(&st, sb_r, sb_c);

    /* YModes / UVModes are written when the mode info is parsed, i.e. before
     * the residual: a later block's edge filter type reads them, and so does a
     * later block's directional prediction through the map. */
    for (int y = 0; y < bh4; ++y) {
      for (int x = 0; x < bw4; ++x) {
        st.y_mode.at(block.mi_col + x, block.mi_row + y) = static_cast<uint8_t>(block.y_mode);
        st.uv_mode.at(block.mi_col + x, block.mi_row + y) = static_cast<uint8_t>(block.uv_mode);
      }
    }

    const bool avail_u = block.have_above;
    const bool avail_l = block.have_left;
    /* AvailUChroma / AvailLChroma: a block whose luma width or height is one
     * 4x4 does not own its chroma, so the chroma neighbour is two mi units
     * away. has_chroma is false for such a block when it has none. */
    bool avail_u_chroma = avail_u, avail_l_chroma = avail_l;
    if (!block.has_chroma) {
      avail_u_chroma = false;
      avail_l_chroma = false;
    } else {
      if (ss_y && bh4 == 1) avail_u_chroma = avail_u && block.mi_row >= 2;
      if (ss_x && bw4 == 1) avail_l_chroma = avail_l && block.mi_col >= 2;
    }

    for (const TileTransform& transform : block.transforms) {
      const int plane = transform.plane;
      if (plane > 0 && seq.monochrome) continue;
      const int sub_x = plane > 0 && ss_x ? 1 : 0;
      const int sub_y = plane > 0 && ss_y ? 1 : 0;
      const TxSize tx = transform.size;
      if (tx.lw < 0 || tx.lh < 0) {
        *error = "reconstruct: undefined transform size";
        return false;
      }

      const int start_x = transform.x; /* plane samples */
      const int start_y = transform.y;
      const int max_x = (st.mi_cols * 4) >> sub_x;
      const int max_y = (st.mi_rows * 4) >> sub_y;
      if (start_x >= max_x || start_y >= max_y) continue;

      /* row/col are the specification's: the transform block's start in luma
       * mi units, which is also the index base of BlockDecoded. */
      const int col = (start_x << sub_x) >> 2;
      const int row = (start_y << sub_y) >> 2;
      const int step_x = tx.w4(), step_y = tx.h4();
      const int grid_x = col >> sub_x, grid_y = row >> sub_y;

      const int block_x = (block.mi_col >> sub_x) * 4;
      const int block_y = (block.mi_row >> sub_y) * 4;

      BlockSyntax bs;
      bs.plane = plane;
      bs.x = start_x;
      bs.y = start_y;
      bs.w = tx.width();
      bs.h = tx.height();
      bs.bit_depth = seq.bit_depth;
      bs.is_intra = true;
      bs.lossless = lossless;

      /* Chroma from luma is a prediction mode of its own: the engine has to
       * know, both to run the CfL prediction and to make the block wait for
       * the co-located luma. It carries the alpha separately. */
      const bool is_cfl = plane > 0 && block.uv_mode == kModeChromaFromLuma;
      bs.mode = to_prediction_mode(plane == 0 ? block.y_mode : block.uv_mode);
      bs.angle_delta = plane == 0 ? block.y_angle : (is_cfl ? 0 : block.uv_angle);
      if (plane == 0 && block.filter_intra) {
        bs.use_filter_intra = true;
        bs.filter_intra_mode = to_filter_intra_mode(block.filter_intra_mode);
      }
      if (is_cfl) bs.cfl_alpha = plane == 1 ? block.cfl_alpha[0] : block.cfl_alpha[1];

      bs.have_left = (plane == 0 ? avail_l : avail_l_chroma) || start_x > block_x;
      bs.have_above = (plane == 0 ? avail_u : avail_u_chroma) || start_y > block_y;
      bs.have_above_right = st.decoded[plane].at(grid_x + step_x, grid_y - 1) ? 1 : 0;
      bs.have_below_left = st.decoded[plane].at(grid_x - 1, grid_y + step_y) ? 1 : 0;
      bs.filter_type = intra_filter_type(st, block.mi_row, block.mi_col, plane,
                                         plane == 0 ? avail_u : avail_u_chroma,
                                         plane == 0 ? avail_l : avail_l_chroma);

      TransformBlock tb;
      tb.x = 0;
      tb.y = 0;
      tb.w = tx.width();
      tb.h = tx.height();
      tb.eob = transform.eob;
      tb.tx_type = to_transform_type(transform.tx_type);
      if (next_offset < quant.offsets.size()) {
        tb.coef_offset = static_cast<int>(quant.offsets[next_offset]);
      }
      ++next_offset;
      bs.transforms.push_back(tb);
      out->blocks.push_back(bs);

      /* The transform block counts as decoded once it is reconstructed, and
       * that is what the neighbours' above-right / below-left questions read. */
      st.decoded[plane].fill(grid_x, grid_y, grid_x + step_x, grid_y + step_y, 1);
    }
  }

  /* The loop filter's edges come from the block grid, not from the transform
   * blocks: whether an edge is filtered depends on the transform sizes on both
   * sides of it and on the block that owns it. */
  if (!plan_deblock(seq, frame, tile, out, error)) return false;

  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_RECONSTRUCT_HPP */
