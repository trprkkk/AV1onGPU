#include "frontend/tile_decoder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "frontend/coef_model.hpp"
#include "frontend/coef_tables.hpp"
#include "frontend/range_decoder.hpp"

namespace av1gpu {
namespace frontend {
namespace {

/* Intra_Mode_Context: the eight "closest" modes around DC/vertical/horizontal
 * are grouped into the context alphabet of the key frame y mode symbol. */
constexpr int8_t kIntraModeContext[kModeLumaCount] = {0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0};

/* AV1GPU_TRACE=1 dumps one line per parsed block, in the same shape a patched
 * dav1d prints for the same stream, so the two decoders can be diffed line by
 * line. A wrong probability or a context read from the wrong neighbour shows up
 * as the first line that differs, which is far easier to chase than a PSNR.
 * This is a debugging aid: it is off unless the variable is set, costs one
 * branch per block, and never runs on the GPU paths. */
bool trace_enabled() {
  static const bool enabled = std::getenv("AV1GPU_TRACE") != nullptr;
  return enabled;
}

void trace_block(const TileBlock& block) {
  if (block.is_inter) {
    std::fprintf(stderr,
                 "B %d %d %d %d cs=%d bs=%d bp=%d intra=0 skip=%d skipmode=%d seg=%d "
                 "refs=%d,%d mvs=%d,%d,%d,%d mode=%d mm=%d ct=%d drl=%d f=%d,%d "
                 "wedge=%d sign=%d iitype=%d iimode=%d ytx=%d q=%d\n",
                 block.mi_col, block.mi_row, block_width4(block.size),
                 block_height4(block.size), static_cast<int>(block.size),
                 static_cast<int>(block.size), static_cast<int>(block.partition),
                 block.skip ? 1 : 0, block.skip_mode ? 1 : 0, block.segment_id,
                 block.ref_frame[0], block.ref_frame[1], block.mv[0][0], block.mv[0][1],
                 block.mv[1][0], block.mv[1][1], block.inter_mode, block.motion_mode,
                 block.compound_type, block.drl_index, block.interp_filter,
                 block.interp_filter, 0, 0, block.interintra ? 1 : 0, block.interintra_mode,
                 tx_size_index(block.tx_size), block.qindex);
    return;
  }
  std::fprintf(stderr, "B %d %d %d %d cs=%d bs=%d bp=%d intra=1 skip=%d seg=%d y=%d uv=%d "
               "ya=%d uva=%d cfl=%d,%d fi=%d fim=%d pal=%d,%d tx=%d q=%d\n",
               block.mi_col, block.mi_row, block_width4(block.size),
               block_height4(block.size), static_cast<int>(block.size),
               static_cast<int>(block.size), static_cast<int>(block.partition),
               block.skip ? 1 : 0, block.segment_id, block.y_mode, block.uv_mode,
               block.y_angle, block.uv_angle, block.cfl_alpha[0], block.cfl_alpha[1],
               block.filter_intra ? 1 : 0, block.filter_intra_mode, block.palette_size[0],
               block.palette_size[1], tx_size_index(block.tx_size), block.qindex);
}

ParseStatus fail(std::string* error, const char* message) {
  if (error) *error = message;
  return ParseStatus::Invalid;
}

/* The partition symbol is not read at the frame edges: when there is no room to
 * the right, the only choices left are a horizontal split or a further split,
 * and the specification derives a two symbol CDF for it from the partition
 * table. With an inverse cumulative table, the probability of a group of
 * symbols is the difference between the table entries bracketing the group, so
 * these are interval sums.
 *
 * `no_room_to_the_right` gathers every partition that is not a horizontal
 * split; `no_room_below` gathers every partition that is not a vertical split.
 * Symbols no longer reachable (the four T shapes at a frame edge, and the 4:1
 * splits inside 128x128 blocks) are folded in exactly as the reference
 * decoders do. */
unsigned gather_split_probability(const uint16_t* cdf, int block_level, bool horizontal) {
  unsigned total;
  if (horizontal) {
    /* not a horizontal split: vertical, split, and the four T shapes */
    total = cdf[kPartitionVertical - 1] - cdf[kPartitionHorizontalA];
    total += cdf[kPartitionVerticalA - 1];
    if (block_level != 0) total += cdf[kPartitionVertical4 - 1] - cdf[kPartitionVerticalB];
  } else {
    /* not a vertical split: horizontal, split, and the four T shapes */
    total = cdf[kPartitionHorizontal - 1] - cdf[kPartitionHorizontal];
    total += cdf[kPartitionSplit - 1] - cdf[kPartitionVerticalA];
    if (block_level != 0) total += cdf[kPartitionHorizontal4 - 1] - cdf[kPartitionHorizontal4];
  }
  return total;
}

/* --- coefficient contexts ------------------------------------------------
 *
 * The specification keeps two arrays per plane and per direction, all indexed
 * by 4x4 unit on that plane's own grid: how large the last coded block's levels
 * were (`level`, used to predict whether a coefficient is zero) and which way
 * its DC pointed (`dc`, used to predict the next DC sign). They are the only
 * state the coefficient syntax shares between blocks, and they are updated after
 * every transform block, empty or not.
 */
struct CoefPlaneContext {
  std::vector<uint8_t> above_level;
  std::vector<uint8_t> above_dc;
  std::vector<uint8_t> left_level;
  std::vector<uint8_t> left_dc;
};

/* The coefficient raster of one transform block: the adjusted size in
 * coefficients, which is the grid the syntax indexes positions on. A 64x64
 * transform is a 32x32 raster here; the syntax never codes the other three
 * quarters. */
struct CoefGrid {
  int bwl = 2; /* log2 of the width */
  int width = 4;
  int height = 4;

  int area() const { return width * height; }
  int row(int pos) const { return pos >> bwl; }
  int col(int pos) const { return pos & (width - 1); }
  bool contains(int row_index, int col_index) const {
    return row_index >= 0 && col_index >= 0 && row_index < height && col_index < width;
  }
};

CoefGrid coef_grid(TxSize tx) {
  CoefGrid grid;
  const TxSize adjusted = adjusted_tx_size(tx);
  if (adjusted.lw >= 0) {
    grid.bwl = 2 + adjusted.lw;
    grid.width = 1 << grid.bwl;
    grid.height = 1 << (2 + adjusted.lh);
  }
  return grid;
}

/* get_coeff_base_ctx: the context the base level of a coefficient is coded
 * with. The five neighbours the transform class points at decide how likely a
 * non-zero level is, so their magnitudes select the context; a vertical-only
 * transform looks up the column, a horizontal-only one along the row.
 *
 * The last coefficient of a block is a special case (`is_eob`): it is always
 * non-zero, so the context instead says how far into the block it sits, and the
 * result is offset past the 2D contexts into the four end-of-block ones. */
int coeff_base_context(const CoefGrid& grid, int tx_class, const int16_t* levels, int pos,
                       TxSize tx, int scan_index, bool is_eob) {
  if (is_eob) {
    const int area = grid.area();
    if (scan_index == 0) return coef_tables::kSigCoefContexts - 4;
    if (scan_index <= area / 8) return coef_tables::kSigCoefContexts - 3;
    if (scan_index <= area / 4) return coef_tables::kSigCoefContexts - 2;
    return coef_tables::kSigCoefContexts - 1;
  }

  const int row = grid.row(pos), col = grid.col(pos);
  int mag = 0;
  for (int idx = 0; idx < coef_tables::kSigRefDiffOffsetNum; ++idx) {
    const int ref_row = row + coef_tables::kSigRefDiffOffset[tx_class][idx][0];
    const int ref_col = col + coef_tables::kSigRefDiffOffset[tx_class][idx][1];
    if (grid.contains(ref_row, ref_col)) {
      mag += std::min<int>(levels[ref_row * grid.width + ref_col], 3);
    }
  }
  const int ctx = std::min((mag + 1) >> 1, 4);
  if (tx_class == coef_tables::kTxClass2d) {
    if (row == 0 && col == 0) return 0;
    const int index = tx_size_index(tx);
    if (index < 0) return 0;
    return ctx + coef_tables::kCoeffBaseCtxOffset[index][std::min(row, 4)][std::min(col, 4)];
  }
  const int along = tx_class == coef_tables::kTxClassVert ? row : col;
  return ctx + coef_tables::kCoeffBasePosCtxOffset[std::min(along, 2)];
}

/* get_coeff_br_ctx: the context for the levels above the base range. Same idea,
 * three neighbours instead of five, and a magnitude capped at the largest value
 * the base tokens can express plus one. */
int coeff_br_context(const CoefGrid& grid, int tx_class, const int16_t* levels, int pos) {
  const int row = grid.row(pos), col = grid.col(pos);
  int mag = 0;
  for (int idx = 0; idx < 3; ++idx) {
    const int ref_row = row + coef_tables::kMagRefOffset[tx_class][idx][0];
    const int ref_col = col + coef_tables::kMagRefOffset[tx_class][idx][1];
    if (grid.contains(ref_row, ref_col)) {
      mag += std::min<int>(levels[ref_row * grid.width + ref_col],
                           coef_tables::kCoeffBaseRange + coef_tables::kNumBaseLevels + 1);
    }
  }
  mag = std::min((mag + 1) >> 1, 6);
  if (pos == 0) return mag;
  if (tx_class == coef_tables::kTxClass2d) return (row < 2 && col < 2) ? mag + 7 : mag + 14;
  if (tx_class == coef_tables::kTxClassHoriz) return col == 0 ? mag + 7 : mag + 14;
  return row == 0 ? mag + 7 : mag + 14;
}

/* The mv_joint symbol is MV_JOINT_HNZVZ first, so bit 0 of the joint value
 * codes the horizontal component and bit 1 the vertical one. The two
 * components have their own probability tables: index 0 is the vertical one,
 * which is also the one a block that codes both components reads first. */
constexpr int kMvVertical = 0;
constexpr int kMvHorizontal = 1;

/* Filter_Intra_Mode_To_Intra_Dir: a filter intra mode stands in for a luma
 * direction when the transform type alphabet is selected. */
constexpr int8_t kFilterIntraToIntraDir[5] = {kModeDc, kModeVertical, kModeHorizontal,
                                              kModeHorizontalDown, kModeDc};

class TileParser {
 public:
  TileParser(const SequenceHeader& seq, const FrameHeader& frame, const RefPool* refs,
             const uint8_t* data, size_t size, TileDecodeResult* out, std::string* error)
      : seq_(seq), frame_(frame), refs_(refs), data_(data), size_(size), out_(out),
        error_(error), trace_(std::getenv("AV1GPU_TRACE") != nullptr) {}

  ParseStatus run() {
    if (!frame_.tiles.col_start_sb.empty()) {
      /* one tile: its extent is the frame */
      tile_col_start_ = 0;
      tile_col_end_ = mi_cols();
      tile_row_start_ = 0;
      tile_row_end_ = mi_rows();
    }
    const int sb4 = seq_.use_128x128_superblock ? 32 : 16;
    const BlockSize sb_size = seq_.use_128x128_superblock ? kBlock128x128 : kBlock64x64;
    for (int r = tile_row_start_; r < tile_row_end_; r += sb4) {
      for (int c = tile_col_start_; c < tile_col_end_; c += sb4) {
        read_deltas_ = frame_.delta_q_present;
        const ParseStatus status = decode_partition(r, c, sb_size);
        if (status != ParseStatus::Ok) return status;
      }
    }
    out_->complete = true;
    return ParseStatus::Ok;
  }

  size_t bytes_used() const { return decoder_.bytes_used(); }
  bool overrun() const { return decoder_.overrun(); }
  bool failed() const { return decoder_.failed(); }
  const std::vector<int8_t>& cdef_index() const { return cdef_idx_; }

 private:
  int mi_cols() const { return (frame_.frame_width + 3) >> 2; }
  int mi_rows() const { return (frame_.height + 3) >> 2; }
  int index(int row, int col) const { return row * mi_cols() + col; }

  /* Records that the tile reached syntax this layer does not read, and why. The
   * block is still reported by the caller, so the reason always names a block
   * that is in the output. */
  ParseStatus unsupported(const char* reason) {
    out_->stop_reason = reason;
    if (error_) *error_ = reason;
    return ParseStatus::Unsupported;
  }

  /* The range decoder's position after the block, for comparing against another
   * decoder: reading the same bits with the same probabilities leaves both in
   * the same state, so the first block whose state differs is where the parse
   * went wrong - and it is the *previous* block's syntax that did it. */
  void trace_state(const TileBlock& block) {
    if (!trace_enabled()) return;
    const RangeDecoder::State state = decoder_.state();
    std::fprintf(stderr, "S %d %d %llu %u %d\n", block.mi_col, block.mi_row,
                 static_cast<unsigned long long>(state.window), state.range, state.count);
  }

  /* One line per transform block, in the shape the patched dav1d prints, with
   * the range decoder's state after it: the state is what makes a desync
   * visible at the block that caused it rather than at the next mode symbol. */
  void trace_coef(int plane, int x4, int y4, TxSize tx, int txtp, int eob) {
    if (!trace_enabled()) return;
    const RangeDecoder::State state = decoder_.state();
    std::fprintf(stderr, "C %d %d plane=%d tx=%d txtp=%d eob=%d dif=%llu rng=%u cnt=%d\n", x4,
                 y4, plane, tx_size_index(adjusted_tx_size(tx)), txtp, eob,
                 static_cast<unsigned long long>(state.window), state.range, state.count);
  }

  void trace_point(int row, int col, const char* tag) {
    if (!trace_enabled()) return;
    const RangeDecoder::State state = decoder_.state();
    std::fprintf(stderr, "M %s %d %d %llu %u %d\n", tag, col, row,
                 static_cast<unsigned long long>(state.window), state.range, state.count);
  }

  ParseStatus stop_at_residual(const TileBlock& block, const char* reason) {
    if (trace_enabled()) trace_block(block);
    out_->blocks.push_back(block);
    ++out_->blocks_needing_residuals;
    return unsupported(reason);
  }

  /* --- neighbour contexts ------------------------------------------------ */

  bool have_top(int row) const { return row > tile_row_start_; }
  bool have_left(int col) const { return col > tile_col_start_; }

  /* partition: how the neighbours were split, as the specification defines it -
   * a neighbour that is smaller than this block pushes towards a split. */
  int partition_context(int row, int col, int level) {
    const int bsl = 5 - level; /* log2 of the parent's side in 4x4 units */
    int above = 0, left = 0;
    if (have_top(row)) above = kBlockWidthLog2[mi_size_[index(row - 1, col)]] < bsl;
    if (have_left(col)) left = kBlockHeightLog2[mi_size_[index(row, col - 1)]] < bsl;
    return left * 2 + (above ? 1 : 0);
  }

  /* skip: how many of the two neighbours were skipped. */
  /* skip mode of the neighbours, as the skip mode context counts it */
  int mi_skip_mode(int row, int col) const { return skip_mode_map_[index(row, col)]; }
  void mi_skip_mode_set(int row, int col, int bw4, int bh4) {
    for (int y = 0; y < bh4 && row + y < mi_rows(); ++y)
      for (int x = 0; x < bw4 && col + x < mi_cols(); ++x)
        skip_mode_map_[index(row + y, col + x)] = 1;
  }

  int skip_context(int row, int col) {
    int ctx = 0;
    if (have_top(row)) ctx += skip_map_[index(row - 1, col)];
    if (have_left(col)) ctx += skip_map_[index(row, col - 1)];
    return ctx;
  }

  void mark_block(int row, int col, int w4, int h4, BlockSize size, bool skip,
                  int y_mode) {
    for (int y = 0; y < h4 && row + y < mi_rows(); ++y) {
      for (int x = 0; x < w4 && col + x < mi_cols(); ++x) {
        mi_size_[index(row + y, col + x)] = static_cast<uint8_t>(size);
        y_mode_map_[index(row + y, col + x)] = static_cast<uint8_t>(y_mode);
        if (skip) {
          skip_map_[index(row + y, col + x)] = 1;
        }
      }
    }
  }

  /* InterTxSizes: the transform every 4x4 cell of the frame was coded with.
   * An intra block fills its whole area with the one transform it chose; an
   * inter block that splits fills each leaf with its own size, and the cells of
   * a block that does not split with the block's largest transform. The
   * transform split tree reads it back through get_above_tx_width and
   * get_left_tx_height, which is why it has to be per cell and not per edge.
   *
   * The values are log2 of the side in 4x4 units, uncapped: a 128x128 block
   * that does not split stores 5, so that "the neighbour above is at least as
   * wide as this transform" can still answer no against a 64x64 transform. */
  std::vector<uint8_t> tx_cell_lw_;
  std::vector<uint8_t> tx_cell_lh_;
  void set_cell_tx(int row, int col, int w4, int h4, int lw, int lh) {
    for (int y = 0; y < h4 && row + y < mi_rows(); ++y)
      for (int x = 0; x < w4 && col + x < mi_cols(); ++x)
        tx_cell_lw_[index(row + y, col + x)] = static_cast<uint8_t>(lw),
        tx_cell_lh_[index(row + y, col + x)] = static_cast<uint8_t>(lh);
  }
  int cell_tx_width(int row, int col) const {
    return 4 << tx_cell_lw_[index(row, col)];
  }
  int cell_tx_height(int row, int col) const {
    return 4 << tx_cell_lh_[index(row, col)];
  }

  /* The transform the block uses, which the next block's tx_depth context
   * reads: one value per 4x4 column above and one per row to the left.
   *
   * The *whole block* takes the value, not just the area the transform covers:
   * InterTxSizes is written over bw4 x bh4 by read_block_tx_size, and the
   * neighbour lookup reads the entry directly above or to the left, which may
   * be outside a small transform at the block's far edge. A 64x64 block with a
   * 32x32 transform would otherwise leave its lower half at the initial 0 and
   * make the block below use the wrong tx_depth context.*/
  void mark_tx(int row, int col, int w4, int h4, TxSize tx) {
    for (int x = 0; x < w4 && col + x < mi_cols(); ++x) above_tx_lw_[col + x] = tx.lw;
    for (int y = 0; y < h4 && row + y < mi_rows(); ++y) left_tx_lh_[row + y] = tx.lh;
  }

  /* --- inter mode info --------------------------------------------------- */

  /* One 4x4 cell of the motion reference map: what the specification writes
   * into its RefMv map, and what the candidate scan of the *next* block reads.
   * Everything here is a property of the whole block, so the block's area is
   * filled with the same entry. */
  struct MiRef {
    uint8_t intra = 1;                 /* 1: intra predicted (or intrabc)   */
    uint8_t ref[2] = {kRefIntra, kRefIntra};
    uint8_t inter_mode = kModeNearestMv;
    int16_t mv[2][2] = {{0, 0}, {0, 0}};
    /* dav1d's `mf` bit 0: a global motion vector that should be used in place
     * of `mv` when this block is matched as a candidate. */
    uint8_t global_subpel = 0;
    /* dav1d's `mf` bit 1: the block coded a new motion vector. */
    uint8_t coded_newmv = 0;
    /* the interpolation filter this block used, for the filter context */
    uint8_t filter = 0;
  };

  struct MvCand {
    int x = 0;
    int y = 0;
    int weight = 0;
  };

  const MiRef& mi_at(int row, int col) const { return mi_ref_[index(row, col)]; }
  BlockSize block_size_at(int row, int col) const {
    return static_cast<BlockSize>(mi_size_[index(row, col)]);
  }

  /* Above-right availability as BlockDecoded reports it: blocks are coded in
   * raster order inside a superblock, so the sample past a block's right edge
   * on the row above has been decoded when it is in the same superblock (to
   * the right of us is *not* enough) or in the row of superblocks above. */
  bool above_right_available(int row, int col, int bw4) const {
    if (row <= tile_row_start_) return false;
    const int ar_col = col + bw4;
    if (ar_col >= tile_col_end_) return false;
    const int sb4 = seq_.use_128x128_superblock ? 32 : 16;
    const int above_row = row - 1;
    if (above_row / sb4 == row / sb4) return ar_col / sb4 == col / sb4;
    return true;
  }

  /* get_gmv_2d: the motion vector a global motion model produces for the
   * centre of this block, in 1/8 luma samples. The specification's own two
   * branches disagree about the order of the translation parameters - the
   * affine branch reads gm_params[0] as the horizontal shift, the translation
   * branch as the vertical one - and both reference decoders reproduce that,
   * so this does too. */
  void global_mv(const GlobalMotion& gm, int mi_row, int mi_col, int bw4, int bh4, int* out_x,
                 int* out_y) const {
    *out_x = 0;
    *out_y = 0;
    if (gm.identity()) return;
    const int hp = frame_.allow_high_precision_mv ? 1 : 0;
    if (gm.translation_only()) {
      *out_y = gm.params[0] >> (kWarpmodelPrecBits - 3);
      *out_x = gm.params[1] >> (kWarpmodelPrecBits - 3);
      return;
    }
    const int x = mi_col * 4 + bw4 * 2 - 1;
    const int y = mi_row * 4 + bh4 * 2 - 1;
    const int32_t* p = gm.params;
    const int xc = (p[2] - (1 << kWarpmodelPrecBits)) * x + p[3] * y + p[0];
    const int yc = (p[4] * x) + (p[5] - (1 << kWarpmodelPrecBits)) * y + p[1];
    const int shift = kWarpmodelPrecBits - (3 - !hp);
    const int round = (1 << shift) >> 1;
    const int sign_x = xc < 0 ? -1 : 1, sign_y = yc < 0 ? -1 : 1;
    *out_x = sign_x * ((((xc < 0 ? -xc : xc) + round) >> shift) << !hp);
    *out_y = sign_y * ((((yc < 0 ? -yc : yc) + round) >> shift) << !hp);
  }

  /* lower_mv_precision / dav1d's fix_mv_precision */
  void lower_mv_precision(int* x, int* y) const {
    if (frame_.force_integer_mv) {
      *x = (*x - (*x >> 15) + 3) & ~7U;
      *y = (*y - (*y >> 15) + 3) & ~7U;
    } else if (!frame_.allow_high_precision_mv) {
      *x = (*x - (*x >> 15)) & ~1U;
      *y = (*y - (*y >> 15)) & ~1U;
    }
  }

  /* The reference frames a block can name are resolved through the pool: the
   * candidate scan needs to know the reference *types*, not the slots. */
  int ref_type_of(int ref_idx) const { return ref_idx + 1; }
  const GlobalMotion& global_motion_of(int ref_type) const {
    return frame_.global_motion[ref_type];
  }
  /* --- candidate scan ---------------------------------------------------- */

  /* add_spatial_candidate for a single-reference block: the candidates are the
   * neighbours that name the same reference, and a neighbour that used global
   * motion contributes the vector the model gives *this* block, not its own. */
  void add_spatial_candidate(MvCand* stack, int* count, int weight, int row, int col,
                             int ref0, int gmv_x, int gmv_y, bool gmv_valid,
                             bool* have_newmv, bool* have_refmv) {
    if (row < 0 || row >= mi_rows() || col < 0 || col >= mi_cols()) return;
    const MiRef& cand = mi_at(row, col);
    if (cand.intra) return;
    for (int n = 0; n < 2; ++n) {
      const int cand_ref = cand.ref[n];
      if (cand_ref <= kRefIntra) break;
      if (cand_ref != ref0) continue;
      int cx = cand.mv[n][0], cy = cand.mv[n][1];
      if (cand.global_subpel && gmv_valid) {
        cx = gmv_x;
        cy = gmv_y;
      }
      *have_refmv = true;
      *have_newmv = *have_newmv || cand.coded_newmv;
      for (int m = 0; m < *count; ++m) {
        if (stack[m].x == cx && stack[m].y == cy) {
          stack[m].weight += weight;
          return;
        }
      }
      if (*count < 8) {
        stack[*count].x = cx;
        stack[*count].y = cy;
        stack[*count].weight = weight;
        ++*count;
      }
      return;
    }
  }

  /* scan_row: walk the row above from the block's left edge to its right edge,
   * adding one candidate per *block* encountered (a neighbour narrower than us
   * contributes several, a wider one only once). */
  int scan_row(MvCand* stack, int* count, int ref0, int gmv_x, int gmv_y, bool gmv_valid,
               int row, int base_col, int bw4, int w4, int max_rows, int step,
               bool* have_newmv, bool* have_refmv) {
    int col = base_col;
    const int first_w4 = block_width4(block_size_at(row, col));
    const int first_h4 = block_height4(block_size_at(row, col));
    int len = std::max(step, std::min(bw4, first_w4));
    if (bw4 <= first_w4) {
      const int weight = bw4 == 1 ? 2 : std::max(2, std::min(2 * max_rows, first_h4));
      add_spatial_candidate(stack, count, len * weight, row, col, ref0, gmv_x, gmv_y,
                            gmv_valid, have_newmv, have_refmv);
      return weight >> 1;
    }
    for (int x = 0;;) {
      add_spatial_candidate(stack, count, len * 2, row, col, ref0, gmv_x, gmv_y, gmv_valid,
                            have_newmv, have_refmv);
      x += len;
      if (x >= w4) return 1;
      col = base_col + x;
      const int cand_w4 = block_width4(block_size_at(row, col));
      len = std::max(step, cand_w4);
    }
  }

  int scan_col(MvCand* stack, int* count, int ref0, int gmv_x, int gmv_y, bool gmv_valid,
               int base_row, int col, int bh4, int h4, int max_cols, int step,
               bool* have_newmv, bool* have_refmv) {
    int row = base_row;
    const int first_h4 = block_height4(block_size_at(row, col));
    const int first_w4 = block_width4(block_size_at(row, col));
    int len = std::max(step, std::min(bh4, first_h4));
    if (bh4 <= first_h4) {
      const int weight = bh4 == 1 ? 2 : std::max(2, std::min(2 * max_cols, first_w4));
      add_spatial_candidate(stack, count, len * weight, row, col, ref0, gmv_x, gmv_y,
                            gmv_valid, have_newmv, have_refmv);
      return weight >> 1;
    }
    for (int y = 0;;) {
      add_spatial_candidate(stack, count, len * 2, row, col, ref0, gmv_x, gmv_y, gmv_valid,
                            have_newmv, have_refmv);
      y += len;
      if (y >= h4) return 1;
      row = base_row + y;
      const int cand_h4 = block_height4(block_size_at(row, col));
      len = std::max(step, cand_h4);
    }
  }

  /* find_ref_mvs, single reference case. The stack is the specification's
   * list of candidate motion vectors; `ctx` packs the contexts the mode syntax
   * needs (newmv_ctx, globalmv_ctx, refmv_ctx in that bit layout). */
  void find_ref_mvs(MvCand* stack, int* count, int* ctx, int ref0, BlockSize size, int mi_row,
                    int mi_col) {
    *count = 0;
    const int bw4 = block_width4(size), bh4 = block_height4(size);
    const int w4 = std::min(std::min(bw4, 16), tile_col_end_ - mi_col);
    const int h4 = std::min(std::min(bh4, 16), tile_row_end_ - mi_row);
    int gmv_x = 0, gmv_y = 0;
    global_mv(global_motion_of(ref0), mi_row, mi_col, bw4, bh4, &gmv_x, &gmv_y);
    const bool gmv_valid = global_motion_of(ref0).type > kGmTranslation;

    bool have_newmv = false, have_col_mvs = false, have_row_mvs = false;
    unsigned max_rows = 0, n_rows = ~0U, max_cols = 0, n_cols = ~0U;
    const bool top_avail = mi_row > tile_row_start_;
    const bool left_avail = mi_col > tile_col_start_;
    if (top_avail) {
      max_rows = std::min(static_cast<unsigned>((mi_row - tile_row_start_ + 1) >> 1),
                          2u + (bh4 > 1 ? 1u : 0u));
      n_rows = scan_row(stack, count, ref0, gmv_x, gmv_y, gmv_valid, mi_row - 1, mi_col, bw4,
                        w4, static_cast<int>(max_rows), bw4 >= 16 ? 4 : 1, &have_newmv,
                        &have_row_mvs);
    }
    if (left_avail) {
      max_cols = std::min(static_cast<unsigned>((mi_col - tile_col_start_ + 1) >> 1),
                          2u + (bw4 > 1 ? 1u : 0u));
      n_cols = scan_col(stack, count, ref0, gmv_x, gmv_y, gmv_valid, mi_row, mi_col - 1, bh4,
                        h4, static_cast<int>(max_cols), bh4 >= 16 ? 4 : 1, &have_newmv,
                        &have_col_mvs);
    }
    /* top/right */
    if (n_rows != ~0U && above_right_available(mi_row, mi_col, bw4) &&
        std::max(bw4, bh4) <= 16 && mi_col + bw4 < tile_col_end_) {
      add_spatial_candidate(stack, count, 4, mi_row - 1, mi_col + bw4, ref0, gmv_x, gmv_y,
                            gmv_valid, &have_newmv, &have_row_mvs);
    }

    const int nearest_match = (have_col_mvs ? 1 : 0) + (have_row_mvs ? 1 : 0);
    const int nearest_count = *count;
    for (int n = 0; n < nearest_count; ++n) stack[n].weight += 640;

    /* temporal candidates need the reference frames' own motion vectors; they
     * only exist when this frame uses reference frame MVs and the reference is
     * an inter frame, which the caller rejects before getting here. */
    /* The temporal candidate section starts the global motion context at the
     * frame's use_ref_frame_mvs flag and only ever lowers it: a temporal
     * candidate whose vector matches the global one leaves it alone, one that
     * differs clears it. A frame that refers only to intra frames has no
     * temporal candidates at all, so the flag survives as it is, which is why
     * this is not simply zero when the projection is empty. */
    int globalmv_ctx = frame_.use_ref_frame_mvs ? 1 : 0;

    /* top/left: the diagonal neighbour, when both scans ran */
    if (top_avail && left_avail) {
      bool dummy = false;
      add_spatial_candidate(stack, count, 4, mi_row - 1, mi_col - 1, ref0, gmv_x, gmv_y,
                            gmv_valid, &dummy, &have_row_mvs);
    }

    /* the "secondary" rows and columns, two and three blocks away, which the
     * specification scans at 8x8 resolution */
    for (int n = 2; n <= 3; ++n) {
      if (static_cast<unsigned>(n) > n_rows && static_cast<unsigned>(n) <= max_rows) {
        bool dummy = false;
        n_rows += scan_row(stack, count, ref0, gmv_x, gmv_y, gmv_valid, mi_row - 2 * n + 1,
                           mi_col | 1, bw4, w4, 1 + static_cast<int>(max_rows) - n,
                           bw4 >= 16 ? 4 : 2, &dummy, &have_row_mvs);
      }
      if (static_cast<unsigned>(n) > n_cols && static_cast<unsigned>(n) <= max_cols) {
        bool dummy = false;
        n_cols += scan_col(stack, count, ref0, gmv_x, gmv_y, gmv_valid, mi_row | 1,
                           (mi_col - n * 2 + 1) | 1, bh4, h4,
                           1 + static_cast<int>(max_cols) - n, bh4 >= 16 ? 4 : 2, &dummy,
                           &have_col_mvs);
      }
    }

    const int ref_match_count = (have_col_mvs ? 1 : 0) + (have_row_mvs ? 1 : 0);
    int refmv_ctx = 0, newmv_ctx = 0;
    switch (nearest_match) {
      case 0:
        refmv_ctx = std::min(2, ref_match_count);
        newmv_ctx = ref_match_count > 0 ? 1 : 0;
        break;
      case 1:
        refmv_ctx = std::min(ref_match_count * 3, 4);
        newmv_ctx = 3 - (have_newmv ? 1 : 0);
        break;
      default:
        refmv_ctx = 5;
        newmv_ctx = 5 - (have_newmv ? 1 : 0);
        break;
    }

    /* sort nearest and secondary groups separately, by descending weight */
    int len = nearest_count;
    while (len) {
      int last = 0;
      for (int n = 1; n < len; ++n) {
        if (stack[n - 1].weight < stack[n].weight) {
          const MvCand tmp = stack[n - 1];
          stack[n - 1] = stack[n];
          stack[n] = tmp;
          last = n;
        }
      }
      len = last;
    }
    len = *count;
    while (len > nearest_count) {
      int last = nearest_count;
      for (int n = nearest_count + 1; n < len; ++n) {
        if (stack[n - 1].weight < stack[n].weight) {
          const MvCand tmp = stack[n - 1];
          stack[n - 1] = stack[n];
          stack[n] = tmp;
          last = n;
        }
      }
      len = last;
    }

    /* clamping */
    if (*count) {
      const int left = -(mi_col + bw4 + 4) * 4 * 8;
      const int right = (mi_cols() - mi_col + 4) * 4 * 8;
      const int top = -(mi_row + bh4 + 4) * 4 * 8;
      const int bottom = (mi_rows() - mi_row + 4) * 4 * 8;
      for (int n = 0; n < *count; ++n) {
        stack[n].x = std::max(left, std::min(right, stack[n].x));
        stack[n].y = std::max(top, std::min(bottom, stack[n].y));
      }
    }

    /* A stack shorter than two is filled with the global motion vector, but the
     * count is what the mode syntax tests: an empty stack still means "no
     * candidate", which suppresses the dynamic reference list index bits. */
    for (int n = *count; n < 2; ++n) {
      stack[n].x = gmv_x;
      stack[n].y = gmv_y;
      stack[n].weight = 2;
    }

    *ctx = (refmv_ctx << 4) | (globalmv_ctx << 3) | newmv_ctx;
  }

  /* get_drl_context */
  int drl_context(const MvCand* stack, int idx) const {
    if (stack[idx].weight >= 640) return stack[idx + 1].weight < 640 ? 1 : 0;
    return stack[idx + 1].weight < 640 ? 2 : 0;
  }

  /* read_mv_residual: the difference from the predicted vector, coded as a
   * joint symbol then one component diff per coded component. */
  void read_mv_residual(int* x, int* y, int mv_prec) {
    const int joint = static_cast<int>(decoder_.read_symbol(cdf_.inter.mv_joint, 3));
    if (trace_) {
      const RangeDecoder::State state = decoder_.state();
      std::fprintf(stderr, "R joint=%d prec=%d dif=%llu rng=%u cnt=%d\n", joint, mv_prec,
                   static_cast<unsigned long long>(state.window), state.range, state.count);
    }
    /* vertical first: that is the order the specification's tables assume */
    if (joint & 2) *y += read_mv_component(kMvVertical, mv_prec);
    if (joint & 1) *x += read_mv_component(kMvHorizontal, mv_prec);
  }

  /* read_mv_component_diff: one component of a coded motion vector
   * difference. `comp` selects the tables (the two components have their own,
   * because their statistics differ); `mv_prec` is 1 with high precision
   * motion vectors, 0 without and -1 when the frame forces integer vectors. */
  int read_mv_component(int comp, int mv_prec) {
    const int sign = decoder_.read_symbol(cdf_.inter.mv_sign[comp], 1) != 0;
    const int cls = static_cast<int>(decoder_.read_symbol(cdf_.inter.mv_class[comp], 10));
    int up = 0, fp = 3, hp = 1;
    if (cls == 0) {
      up = decoder_.read_symbol(cdf_.inter.mv_class0_bit[comp], 1) != 0;
      if (mv_prec >= 0) {
        fp = static_cast<int>(decoder_.read_symbol(cdf_.inter.mv_class0_fr[comp][up], 3));
        if (mv_prec > 0) hp = decoder_.read_symbol(cdf_.inter.mv_class0_hp[comp], 1) != 0;
      }
    } else {
      up = 1 << cls;
      for (int n = 0; n < cls; ++n) {
        up |= (decoder_.read_symbol(cdf_.inter.mv_bit[n][comp], 1) != 0) << n;
      }
      if (mv_prec >= 0) {
        fp = static_cast<int>(decoder_.read_symbol(cdf_.inter.mv_fr[comp], 3));
        if (mv_prec > 0) hp = decoder_.read_symbol(cdf_.inter.mv_hp[comp], 1) != 0;
      }
    }
    const int diff = ((up << 3) | (fp << 1) | hp) + 1;
    if (trace_) {
      const RangeDecoder::State state = decoder_.state();
      std::fprintf(stderr,
                   "R sign=%d cl=%d up=%d fp=%d hp=%d prec=%d diff=%d dif=%llu rng=%u cnt=%d\n",
                   sign, cls, up, fp, hp, mv_prec, sign ? -diff : diff,
                   static_cast<unsigned long long>(state.window), state.range, state.count);
    }
    return sign ? -diff : diff;
  }

  /* --- contexts the inter syntax reads from its neighbours ---------------- */

  int intra_context(int row, int col) {
    const bool top = have_top(row), left = have_left(col);
    const int a = top ? (mi_at(row - 1, col).intra ? 1 : 0) : 0;
    const int l = left ? (mi_at(row, col - 1).intra ? 1 : 0) : 0;
    if (left) {
      if (top) {
        const int ctx = l + a;
        return ctx + (ctx == 2 ? 1 : 0);
      }
      return l * 2;
    }
    return top ? a * 2 : 0;
  }

  /* The reference context helpers all count, over the two neighbours, how many
   * named a reference of each kind. `kind` selects which references count. */
  int ref_context(int row, int col, int kind) {
    int count[4] = {0, 0, 0, 0};
    for (int pass = 0; pass < 2; ++pass) {
      const bool top = pass == 0;
      if (top ? !have_top(row) : !have_left(col)) continue;
      const MiRef& n = top ? mi_at(row - 1, col) : mi_at(row, col - 1);
      if (n.intra) continue;
      for (int r = 0; r < 2; ++r) {
        const int ref = n.ref[r];
        if (ref <= kRefIntra) continue;
        switch (kind) {
          case 0: count[ref >= kRefBwdref ? 1 : 0]++; break;              /* av1_get_ref_ctx */
          case 1: if (ref <= kRefGolden) count[ref - 1]++; break;         /* fwd ref   */
          case 2: if (ref < kRefLast3) count[ref - 1]++; break;           /* fwd ref 1 */
          case 3:
            if (ref == kRefLast3 || ref == kRefGolden) count[ref - kRefLast3]++;
            break;                                                        /* fwd ref 2 */
          case 4: if (ref >= kRefBwdref) count[ref - kRefBwdref]++; break; /* bwd ref   */
          case 5:
            if (ref >= kRefBwdref && ref <= kRefAltref2) count[ref - kRefBwdref]++;
            break;                                                        /* bwd ref 1 */
          default: break;
        }
      }
    }
    switch (kind) {
      case 0:
        return count[0] == count[1] ? 1 : (count[0] < count[1] ? 0 : 2);
      case 1:
        count[0] += count[1];
        count[2] += count[3];
        return count[0] == count[2] ? 1 : (count[0] < count[2] ? 0 : 2);
      case 2:
        return count[0] == count[1] ? 1 : (count[0] < count[1] ? 0 : 2);
      case 3:
        return count[0] == count[1] ? 1 : (count[0] < count[1] ? 0 : 2);
      case 4:
        count[1] += count[0];
        return count[2] == count[1] ? 1 : (count[1] < count[2] ? 0 : 2);
      case 5:
        return count[0] == count[1] ? 1 : (count[0] < count[1] ? 0 : 2);
      default:
        return 0;
    }
  }

  /* get_filter_ctx: 0 when both neighbours used the same filter for this
   * reference, 3 when they disagree, and the neighbour's filter when only one
   * of them used this reference at all. */
  int filter_context(int row, int col, bool comp, int ref) {
    const int kNone = 3;
    int a_filter = kNone, l_filter = kNone;
    if (have_top(row)) {
      const MiRef& n = mi_at(row - 1, col);
      if (!n.intra && (n.ref[0] == ref || n.ref[1] == ref)) a_filter = n.filter;
    }
    if (have_left(col)) {
      const MiRef& n = mi_at(row, col - 1);
      if (!n.intra && (n.ref[0] == ref || n.ref[1] == ref)) l_filter = n.filter;
    }
    const int base = (comp ? 4 : 0);
    if (a_filter == l_filter) return base + a_filter;
    if (a_filter == kNone) return base + l_filter;
    if (l_filter == kNone) return base + a_filter;
    return base + kNone;
  }


  /* --- inter mode info --------------------------------------------------- */

  /* Size_Group: the y mode of a block inside an inter frame is coded against
   * one of four tables, chosen by how large the block is. */
  int ymode_size_group(BlockSize size) const {
    const int w4 = block_width4(size), h4 = block_height4(size);
    const int lw = kBlockWidthLog2[size], lh = kBlockHeightLog2[size];
    (void)w4;
    (void)h4;
    return std::min(3, std::min(lw, lh));
  }

  /* get_comp_ctx, for frames that allow compound prediction. Only the parts
   * that can be reached count: a block whose neighbours were all single
   * reference has no compound information to share. */
  int comp_context(int row, int col) {
    const bool top = have_top(row), left = have_left(col);
    const MiRef* a = top ? &mi_at(row - 1, col) : nullptr;
    const MiRef* l = left ? &mi_at(row, col - 1) : nullptr;
    /* an intra neighbour stores the reference "-1", which counts as a
     * backward reference for these contexts (dav1d's unsigned comparison) */
    const auto is_bwd = [](const MiRef& n) { return n.intra || n.ref[0] >= kRefBwdref; };
    if (top && left) return is_bwd(*l) == is_bwd(*a) ? 0 : 1;
    if (top) return is_bwd(*a) ? 1 : 0;
    if (left) return is_bwd(*l) ? 1 : 0;
    return 1;
  }

  /* findoddzero: is there an inter predicted block at an odd 4x4 offset
   * inside this block's span on that edge? OBMC overlaps the neighbours it
   * finds there, and only inter predicted ones can be overlapped, so this is
   * the test that decides whether the block codes a motion mode at all. The
   * odd offsets are the ones a 4x4 aligned blend can reach. */
  bool find_odd_inter(bool top, int row, int col, int bw4, int bh4) const {
    const int len = top ? (bw4 >> 1) : (bh4 >> 1);
    for (int n = 0; n < len; ++n) {
      const int offset = n * 2;
      const MiRef& n_ref = top ? mi_at(row - 1, col + 1 + offset) : mi_at(row + 1 + offset, col - 1);
      if (!n_ref.intra) return true;
    }
    return false;
  }

  /* find_matching_ref: which neighbours above and to the left use this block's
   * reference frame, as a bitmask over 4x4 units. The mask decides whether
   * local warped motion may be used, which is why it is read here rather than
   * with the motion mode itself. */
  void find_matching_ref(int row, int col, int bw4, int bh4, int w4, int h4, bool left_avail,
                         bool top_avail, int ref, uint64_t masks[2]) {
    masks[0] = masks[1] = 0;
    int count = 0;
    bool have_topleft = top_avail && left_avail;
    bool have_topright = std::max(bw4, bh4) < 32 && top_avail && row > tile_row_start_ &&
                         col + bw4 < tile_col_end_ && above_right_available(row, col, bw4);
    const auto matches = [&](int r, int c) {
      if (r < 0 || c < 0 || r >= mi_rows() || c >= mi_cols()) return false;
      const MiRef& n = mi_at(r, c);
      return !n.intra && n.ref[0] == ref && n.ref[1] == kRefIntra;
    };
    if (top_avail) {
      if (matches(row - 1, col)) {
        masks[0] |= 1;
        count = 1;
      }
      int aw4 = block_width4(block_size_at(row - 1, col));
      if (aw4 >= bw4) {
        const int off = col & (aw4 - 1);
        if (off) have_topleft = false;
        if (aw4 - off > bw4) have_topright = false;
      } else {
        uint64_t mask = 1u << aw4;
        int c = col;
        for (int x = aw4; x < w4; x += aw4) {
          c += aw4;
          if (matches(row - 1, c)) {
            masks[0] |= mask;
            if (++count >= 8) return;
          }
          aw4 = block_width4(block_size_at(row - 1, std::min(c, mi_cols() - 1)));
          mask <<= aw4;
        }
      }
    }
    if (left_avail) {
      if (matches(row, col - 1)) {
        masks[1] |= 1;
        if (++count >= 8) return;
      }
      int lh4 = block_height4(block_size_at(row, col - 1));
      if (lh4 >= bh4) {
        if (row & (lh4 - 1)) have_topleft = false;
      } else {
        uint64_t mask = 1u << lh4;
        int r = row;
        for (int y = lh4; y < h4; y += lh4) {
          r += lh4;
          if (matches(r, col - 1)) {
            masks[1] |= mask;
            if (++count >= 8) return;
          }
          lh4 = block_height4(block_size_at(std::min(r, mi_rows() - 1), col - 1));
          mask <<= lh4;
        }
      }
    }
    if (have_topleft && matches(row - 1, col - 1)) {
      masks[1] |= 1ULL << 32;
      if (++count >= 8) return;
    }
    if (have_topright && matches(row - 1, col + bw4)) {
      masks[0] |= 1ULL << 32;
    }
  }

  /* The rest of intra mode info that an intra block of an *inter* frame shares
   * with a key frame: the chroma mode, its angle and chroma from luma. */
  ParseStatus read_chroma_mode(TileBlock* block, BlockSize size, bool has_chroma) {
    const int bw4 = block_width4(size), bh4 = block_height4(size);
    const bool ss_x = !seq_.monochrome && seq_.ss_x, ss_y = !seq_.monochrome && seq_.ss_y;
    const int log2_w = kBlockWidthLog2[size], log2_h = kBlockHeightLog2[size];
    if (!has_chroma) return ParseStatus::Ok;
    const bool cfl_allowed =
        !frame_.coded_lossless
            ? std::max(block_width4(size), block_height4(size)) <= 8
            : ((bw4 + ss_x) >> ss_x) == 1 && ((bh4 + ss_y) >> ss_y) == 1;
    const int nsymbols = kModeChromaCount - 1 - (cfl_allowed ? 0 : 1);
    block->uv_mode = static_cast<int>(
        decoder_.read_symbol(cdf_.mode.uv_mode[cfl_allowed ? 1 : 0][block->y_mode], nsymbols));
    if (block->uv_mode > nsymbols) return corrupt("uv_mode");
    if (block->uv_mode == kModeChromaFromLuma) {
      const int sign = static_cast<int>(decoder_.read_symbol(cdf_.mode.cfl_sign, 7)) + 1;
      const int sign_u = sign / 3, sign_v = sign % 3;
      if (sign_u) {
        const int ctx = (sign_u == 2) * 3 + sign_v;
        block->cfl_alpha[0] =
            static_cast<int>(decoder_.read_symbol(cdf_.mode.cfl_alpha[ctx], 15)) + 1;
        if (sign_u == 1) block->cfl_alpha[0] = -block->cfl_alpha[0];
      }
      if (sign_v) {
        const int ctx = (sign_v == 2) * 3 + sign_u;
        block->cfl_alpha[1] =
            static_cast<int>(decoder_.read_symbol(cdf_.mode.cfl_alpha[ctx], 15)) + 1;
        if (sign_v == 1) block->cfl_alpha[1] = -block->cfl_alpha[1];
      }
    } else if (log2_w + log2_h >= 2 && block->uv_mode >= kModeVertical &&
               block->uv_mode <= kModeVerticalLeft) {
      const int value = static_cast<int>(
          decoder_.read_symbol(cdf_.mode.angle_delta[block->uv_mode - kModeVertical], 6));
      block->uv_angle = value - 3;
    }
    return ParseStatus::Ok;
  }

  /* intra_block_mode_info: the y mode of a block that is intra predicted,
   * which is coded against the neighbour modes in a key frame and against a
   * block size group in an inter frame. */
  ParseStatus read_intra_block_mode(TileBlock* block, BlockSize size, int row, int col,
                                    bool intra_only) {
    if (intra_only) {
      const int above_mode =
          have_top(row) ? y_mode_map_[index(row - 1, col)] : static_cast<int>(kModeDc);
      const int left_mode =
          have_left(col) ? y_mode_map_[index(row, col - 1)] : static_cast<int>(kModeDc);
      block->y_mode = static_cast<int>(decoder_.read_symbol(
          cdf_.mode.kfym[kIntraModeContext[above_mode]][kIntraModeContext[left_mode]],
          kModeLumaCount - 1));
      if (block->y_mode >= kModeLumaCount) return corrupt("intra_frame_y_mode");
    } else {
      block->y_mode = static_cast<int>(decoder_.read_symbol(
          cdf_.inter.y_mode[ymode_size_group(size)], kModeLumaCount - 1));
      if (block->y_mode >= kModeLumaCount) return corrupt("y_mode");
    }
    trace_point(row, col, "ymode");

    const int log2_w = kBlockWidthLog2[size], log2_h = kBlockHeightLog2[size];
    if (log2_w + log2_h >= 2 && block->y_mode >= kModeVertical &&
        block->y_mode <= kModeVerticalLeft) {
      const int value = static_cast<int>(
          decoder_.read_symbol(cdf_.mode.angle_delta[block->y_mode - kModeVertical], 6));
      block->y_angle = value - 3;
    }
    trace_point(row, col, "yangle");
    return ParseStatus::Ok;
  }

  /* inter_frame_mode_info: everything an inter frame codes for a block after
   * skip and the superblock deltas. It ends either with an intra block (which
   * continues with the chroma mode) or with a single reference inter block
   * with its motion vector and interpolation filter. */
  ParseStatus read_inter_block_mode(TileBlock* block, BlockSize size, int row, int col,
                                    bool has_chroma) {
    const int bw4 = block_width4(size), bh4 = block_height4(size);
    const bool top_avail = have_top(row), left_avail = have_left(col);

    /* is_inter */
    block->is_inter = decoder_.read_symbol(cdf_.inter.is_inter[intra_context(row, col)], 1) != 0;
    if (!block->is_inter) {
      const ParseStatus status = read_intra_block_mode(block, size, row, col, false);
      if (status != ParseStatus::Ok) return status;
      return read_chroma_mode(block, size, has_chroma);
    }

    /* compound or single reference */
    if (frame_.reference_select && std::min(bw4, bh4) > 1) {
      const int ctx = comp_context(row, col);
      if (decoder_.read_symbol(cdf_.inter.comp_mode[ctx], 1) != 0) {
        return unsupported("tile: compound inter prediction is not implemented");
      }
    }

    /* read_ref_frames: the reference of a single reference block, as a binary
     * tree over the seven reference types */
    /* The table's first index is the context and the second the step of the
     * tree, which is not the order the steps are coded in: the three
     * second-level tests live in different rows. */
    const int ctx1 = ref_context(row, col, 0);
    if (decoder_.read_symbol(cdf_.inter.single_ref[ctx1][0], 1)) {
      const int ctx2 = ref_context(row, col, 4);
      if (decoder_.read_symbol(cdf_.inter.single_ref[ctx2][1], 1)) {
        block->ref_frame[0] = kRefAltref;
      } else {
        const int ctx3 = ref_context(row, col, 5);
        block->ref_frame[0] =
            decoder_.read_symbol(cdf_.inter.single_ref[ctx3][5], 1) ? kRefBwdref : kRefAltref2;
      }
    } else {
      const int ctx2 = ref_context(row, col, 1);
      if (decoder_.read_symbol(cdf_.inter.single_ref[ctx2][2], 1)) {
        const int ctx3 = ref_context(row, col, 3);
        block->ref_frame[0] = decoder_.read_symbol(cdf_.inter.single_ref[ctx3][4], 1) ? kRefLast3
                                                                                      : kRefLast2;
      } else {
        const int ctx3 = ref_context(row, col, 2);
        block->ref_frame[0] =
            decoder_.read_symbol(cdf_.inter.single_ref[ctx3][3], 1) ? kRefLast2 : kRefLast;
      }
    }
    block->ref_frame[1] = kRefIntra;

    /* the candidate list, then the mode that picks from it */
    MvCand stack[8];
    int count = 0, ctx = 0;
    find_ref_mvs(stack, &count, &ctx, block->ref_frame[0], size, row, col);
    if (trace_) {
      std::fprintf(stderr, "V %d %d n=%d ctx=%d s0=%d,%d s1=%d,%d w0=%d w1=%d\n", col, row,
                   count, ctx, stack[0].x, stack[0].y, stack[1].x, stack[1].y, stack[0].weight,
                   stack[1].weight);
    }

    const GlobalMotion& gm = global_motion_of(block->ref_frame[0]);
    int mv_x = 0, mv_y = 0;
    bool has_subpel_filter = true;
    const int mv_prec = (frame_.allow_high_precision_mv ? 1 : 0) - (frame_.force_integer_mv ? 1 : 0);

    if (decoder_.read_symbol(cdf_.inter.new_mv[ctx & 7], 1)) {
      /* not a new vector: global motion, nearest or near */
      if (!decoder_.read_symbol(cdf_.inter.zero_mv[(ctx >> 3) & 1], 1)) {
        block->inter_mode = kModeGlobalMv;
        global_mv(gm, row, col, bw4, bh4, &mv_x, &mv_y);
        has_subpel_filter = std::min(bw4, bh4) == 1 || gm.translation_only();
      } else {
        has_subpel_filter = true;
        if (decoder_.read_symbol(cdf_.inter.ref_mv[(ctx >> 4) & 15], 1)) {
          block->inter_mode = kModeNearMv;
          block->drl_index = 1;
          if (count > 2) {
            block->drl_index +=
                decoder_.read_symbol(cdf_.inter.drl_mode[drl_context(stack, 1)], 1);
            if (block->drl_index == 2 && count > 3) {
              block->drl_index +=
                  decoder_.read_symbol(cdf_.inter.drl_mode[drl_context(stack, 2)], 1);
            }
          }
        } else {
          block->inter_mode = kModeNearestMv;
          block->drl_index = 0;
        }
        mv_x = stack[block->drl_index].x;
        mv_y = stack[block->drl_index].y;
        if (block->drl_index < 2) lower_mv_precision(&mv_x, &mv_y);
      }
    } else {
      has_subpel_filter = true;
      block->inter_mode = kModeNewMv;
      block->drl_index = 0;
      if (count > 1) {
        block->drl_index += decoder_.read_symbol(cdf_.inter.drl_mode[drl_context(stack, 0)], 1);
        if (block->drl_index == 1 && count > 2) {
          block->drl_index += decoder_.read_symbol(cdf_.inter.drl_mode[drl_context(stack, 1)], 1);
        }
      }
      if (count > 1) {
        mv_x = stack[block->drl_index].x;
        mv_y = stack[block->drl_index].y;
      } else {
        mv_x = stack[0].x;
        mv_y = stack[0].y;
        lower_mv_precision(&mv_x, &mv_y);
      }
      read_mv_residual(&mv_x, &mv_y, mv_prec);
    }

    /* inter/intra compound: allowed for most block sizes, but this decoder
     * predicts with one reference only */
    if (seq_.enable_interintra_compound && std::min(bw4, bh4) >= 2 &&
        std::max(bw4, bh4) <= 8) {
      const int group = ymode_size_group(size);
      if (decoder_.read_symbol(cdf_.inter.inter_intra[group], 1) != 0) {
        return unsupported("tile: inter/intra compound prediction is not implemented");
      }
    }

    /* motion mode: simple translation, overlapped block motion compensation, or
     * local warped motion. Both of the latter need prediction work this
     * decoder does not do yet, but the symbol still decides the stream. */
    block->motion_mode = kMotionTranslation;
    if (frame_.is_motion_mode_switchable && std::min(bw4, bh4) >= 2 &&
        !(block->inter_mode == kModeGlobalMv && gm.type > kGmTranslation) &&
        ((left_avail && find_odd_inter(false, row, col, bw4, bh4)) ||
         (top_avail && find_odd_inter(true, row, col, bw4, bh4)))) {
      uint64_t masks[2] = {0, 0};
      find_matching_ref(row, col, bw4, bh4, std::min(bw4, 16), std::min(bh4, 16), left_avail,
                        top_avail, block->ref_frame[0], masks);
      const bool allow_warp = !frame_.force_integer_mv && frame_.allow_warped_motion &&
                              (masks[0] | masks[1]);
      if (allow_warp) {
        block->motion_mode = static_cast<int>(decoder_.read_symbol(cdf_.inter.motion_mode[size], 2));
      } else {
        block->motion_mode =
            decoder_.read_symbol(cdf_.inter.use_obmc[size], 1) ? kMotionObmc : kMotionTranslation;
      }
      /* Both of the switchable modes only change prediction, never the syntax:
       * OBMC re-derives its vectors from the neighbours, warped motion derives
       * an affine model from them. The block is therefore parsed and counted;
       * the reconstruction stage predicts it as plain translation until it
       * implements them, which is why the counter exists. */
      if (block->motion_mode == kMotionObmc) ++out_->obmc_blocks;
      if (block->motion_mode == kMotionWarped) ++out_->warped_blocks;
    }

    /* the subpel filter, if this frame switches them */
    int filter0 = frame_.interpolation_filter, filter1 = filter0;
    if (frame_.is_filter_switchable) {
      if (has_subpel_filter) {
        const bool comp = false;
        const int fctx = filter_context(row, col, comp, block->ref_frame[0]);
        filter0 = static_cast<int>(decoder_.read_symbol(cdf_.inter.interp_filter[fctx], 2));
        filter1 = filter0;
        if (seq_.enable_dual_filter) {
          const int fctx2 = filter_context(row, col, comp, block->ref_frame[0]);
          filter1 =
              static_cast<int>(decoder_.read_symbol(cdf_.inter.interp_filter[8 + fctx2], 2));
        }
      } else {
        filter0 = filter1 = 0;
      }
    }
    /* the engine has one filter per block; a dual filter pair is only a
     * problem when its two halves differ */
    if (filter0 != filter1) return unsupported("tile: dual interpolation filters");
    block->interp_filter = filter0;
    block->mv[0][0] = mv_x;
    block->mv[0][1] = mv_y;
    return ParseStatus::Ok;
  }

  /* Remember the block in the motion reference map the next block's candidate
   * scan reads. */
  void mark_inter(int row, int col, BlockSize size, const TileBlock& block, int filter) {
    const int bw4 = block_width4(size), bh4 = block_height4(size);
    MiRef entry;
    entry.intra = block.is_inter ? 0 : 1;
    entry.ref[0] = static_cast<uint8_t>(block.ref_frame[0]);
    entry.ref[1] = static_cast<uint8_t>(block.ref_frame[1]);
    entry.inter_mode = static_cast<uint8_t>(block.inter_mode);
    entry.mv[0][0] = static_cast<int16_t>(block.mv[0][0]);
    entry.mv[0][1] = static_cast<int16_t>(block.mv[0][1]);
    entry.mv[1][0] = static_cast<int16_t>(block.mv[1][0]);
    entry.mv[1][1] = static_cast<int16_t>(block.mv[1][1]);
    entry.global_subpel =
        block.is_inter && block.inter_mode == kModeGlobalMv && std::min(bw4, bh4) >= 2;
    entry.coded_newmv = block.is_inter && block.inter_mode == kModeNewMv;
    entry.filter = static_cast<uint8_t>(filter);
    for (int y = 0; y < bh4 && row + y < mi_rows(); ++y)
      for (int x = 0; x < bw4 && col + x < mi_cols(); ++x) mi_ref_[index(row + y, col + x)] = entry;
  }

  /* read_var_tx_size: the transform split tree an inter block codes instead of
   * the single tx depth symbol an intra block reads. Each node either stops -
   * a leaf whose size this function writes into InterTxSizes - or splits into
   * the quadrants Split_Tx_Size describes.
   *
   * The context of the split symbol counts two things: whether this node's
   * square class is already below the block's largest transform (which tells
   * the tree where it is), and whether the neighbours' transforms are narrower
   * than this node. `cat` packs the first into the row of the table and
   * `above + left` the second into the column, which is the same information
   * the specification's single flat context carries. */
  void read_var_tx_size(int row, int col, TxSize tx, int depth, BlockSize size) {
    if (row >= mi_rows() || col >= mi_cols()) return;
    const int w4 = tx.w4(), h4 = tx.h4();
    bool split = false;
    if (tx != TxSize{0, 0} && depth < 2) {
      /* the block's largest transform, as a square class */
      const int block_class = std::min(
          4, std::max(static_cast<int>(kBlockWidthLog2[size]),
                      static_cast<int>(kBlockHeightLog2[size])));
      const int above = (row > tile_row_start_ ? cell_tx_width(row - 1, col) : 64) < tx.width();
      const int left = (col > tile_col_start_ ? cell_tx_height(row, col - 1) : 64) < tx.height();
      const int cat = (tx_square_class(tx) != block_class ? 1 : 0) + 2 * (4 - block_class);
      if (cat < 0 || cat > 6) return;
      split = decoder_.read_symbol(cdf_.mode.txpart[cat][above + left], 1) != 0;
    }
    if (!split) {
      set_cell_tx(row, col, w4, h4, tx.lw, tx.lh);
      return;
    }
    const TxSize sub = split_tx_size(tx);
    for (int y = 0; y < h4; y += sub.h4())
      for (int x = 0; x < w4; x += sub.w4())
        read_var_tx_size(row + y, col + x, sub, depth + 1, size);
  }

  ParseStatus decode_partition(int row, int col, BlockSize size) {
    const int cols = mi_cols(), rows = mi_rows();
    if (row >= rows || col >= cols) return ParseStatus::Ok;

    const int half = block_width4(size) / 2;
    const bool have_rows = (row + half) < rows;
    const bool have_cols = (col + half) < cols;

    PartitionType partition = kPartitionNone;
    const int level = block_level_of(size);
    if (size < kBlock8x8) {
      partition = kPartitionNone;
    } else if (have_rows && have_cols) {
      const int ctx = partition_context(row, col, level);
      /* the alphabet shrinks with the level: 128x128 blocks cannot use the
       * 4:1 splits and 8x8 blocks only have the four legacy types */
      const int nsymbols = kPartitionSymbols[level];
      partition = static_cast<PartitionType>(
          decoder_.read_symbol(cdf_.partition.row(level, ctx), nsymbols));
      if (trace_enabled()) {
        const RangeDecoder::State state = decoder_.state();
        std::fprintf(stderr, "P %d %d bl=%d ctx=%d bp=%d %llu %u %d\n", col, row, level, ctx,
                     static_cast<int>(partition),
                     static_cast<unsigned long long>(state.window), state.range, state.count);
      }
      if (static_cast<int>(partition) > nsymbols) return corrupt("partition");
    } else if (have_cols) {
      const int ctx = partition_context(row, col, level);
      const unsigned probability = gather_split_probability(
          cdf_.partition.row(level, ctx), level, /*horizontal=*/true);
      partition = decoder_.read_bool(probability) ? kPartitionSplit : kPartitionHorizontal;
    } else if (have_rows) {
      const int ctx = partition_context(row, col, level);
      const unsigned probability = gather_split_probability(
          cdf_.partition.row(level, ctx), level, /*horizontal=*/false);
      partition = decoder_.read_bool(probability) ? kPartitionSplit : kPartitionVertical;
    } else {
      partition = kPartitionSplit;
    }

    const BlockSize sub = partition_subsize(partition, size);
    const BlockSize split = partition_subsize(kPartitionSplit, size);
    ParseStatus status = ParseStatus::Ok;
    switch (partition) {
      case kPartitionNone:
        return decode_block(row, col, sub, partition);
      case kPartitionHorizontal:
        status = decode_block(row, col, sub, partition);
        if (status == ParseStatus::Ok && have_rows) status = decode_block(row + half, col, sub, partition);
        return status;
      case kPartitionVertical:
        status = decode_block(row, col, sub, partition);
        if (status == ParseStatus::Ok && have_cols) status = decode_block(row, col + half, sub, partition);
        return status;
      case kPartitionSplit:
        status = decode_partition(row, col, sub);
        if (status == ParseStatus::Ok) status = decode_partition(row, col + half, sub);
        if (status == ParseStatus::Ok) status = decode_partition(row + half, col, sub);
        if (status == ParseStatus::Ok) status = decode_partition(row + half, col + half, sub);
        return status;
      case kPartitionHorizontalA:
        status = decode_block(row, col, split, partition);
        if (status == ParseStatus::Ok) status = decode_block(row, col + half, split, partition);
        if (status == ParseStatus::Ok) status = decode_block(row + half, col, sub, partition);
        return status;
      case kPartitionHorizontalB:
        status = decode_block(row, col, sub, partition);
        if (status == ParseStatus::Ok) status = decode_block(row + half, col, split, partition);
        if (status == ParseStatus::Ok)
          status = decode_block(row + half, col + half, split, partition);
        return status;
      case kPartitionVerticalA:
        status = decode_block(row, col, split, partition);
        if (status == ParseStatus::Ok) status = decode_block(row + half, col, split, partition);
        if (status == ParseStatus::Ok) status = decode_block(row, col + half, sub, partition);
        return status;
      case kPartitionVerticalB:
        status = decode_block(row, col, sub, partition);
        if (status == ParseStatus::Ok) status = decode_block(row, col + half, split, partition);
        if (status == ParseStatus::Ok)
          status = decode_block(row + half, col + half, split, partition);
        return status;
      case kPartitionHorizontal4: {
        const int quarter = half / 2;
        for (int i = 0; i < 4; ++i) {
          if (i == 3 && !(row + quarter * 3 < rows)) break;
          status = decode_block(row + quarter * i, col, sub, partition);
          if (status != ParseStatus::Ok) return status;
        }
        return status;
      }
      case kPartitionVertical4: {
        const int quarter = half / 2;
        for (int i = 0; i < 4; ++i) {
          if (i == 3 && !(col + quarter * 3 < cols)) break;
          status = decode_block(row, col + quarter * i, sub, partition);
          if (status != ParseStatus::Ok) return status;
        }
        return status;
      }
      default:
        return fail(error_, "tile: unknown partition type");
    }
  }

  ParseStatus decode_block(int row, int col, BlockSize size, PartitionType partition) {
    TileBlock block;
    block.mi_row = row;
    block.mi_col = col;
    block.size = size;
    block.partition = partition;
    block.block_level = block_level_of(size);

    const int bw4 = block_width4(size), bh4 = block_height4(size);
    const bool ss_x = !seq_.monochrome && seq_.ss_x, ss_y = !seq_.monochrome && seq_.ss_y;
    const bool has_chroma = !seq_.monochrome &&
                            !(bh4 == 1 && ss_y && (row & 1) == 0) &&
                            !(bw4 == 1 && ss_x && (col & 1) == 0);
    block.has_chroma = has_chroma;
    block.have_above = have_top(row);
    block.have_left = have_left(col);

    if (frame_.segmentation_enabled) {
      return fail(error_, "tile: segment id syntax is not implemented");
    }

    /* read_skip_mode, then read_skip. A frame that allows skip mode codes a
     * flag before the skip flag; a block that uses it is skipped and takes its
     * whole mode info from the two reference frames the frame header named,
     * which this decoder does not predict yet. */
    const bool intra_frame =
        frame_.type == FrameType::Key || frame_.type == FrameType::IntraOnly;
    block.skip_mode = false;
    if (!intra_frame && frame_.skip_mode_allowed && std::min(bw4, bh4) > 1) {
      const int smctx = (have_top(row) ? mi_skip_mode(row - 1, col) : 0) +
                        (have_left(col) ? mi_skip_mode(row, col - 1) : 0);
      block.skip_mode = decoder_.read_symbol(cdf_.inter.skip_mode[smctx], 1) != 0;
    }
    if (block.skip_mode) {
      block.skip = true;
      trace_point(row, col, "skip");
      return unsupported("tile: skip mode is not implemented");
    }
    {
      const int sctx = skip_context(row, col);
      block.skip = decoder_.read_symbol(cdf_.skip.row(sctx), 1) != 0;
    }
    trace_point(row, col, "skip");

    /* read_cdef: the first block of a 64x64 that is not skipped carries the
     * index the whole 64x64 filters with; a 64x64 that only holds skipped
     * blocks keeps -1 and is not filtered at all. */
    if (!block.skip && seq_.enable_cdef && !frame_.coded_lossless &&
        !frame_.allow_intrabc) {
      const int cdef_unit = 16; /* 64x64 in 4x4 units */
      const int cr = row / cdef_unit;
      const int cc = col / cdef_unit;
      const int cdef_cols = (mi_cols() + cdef_unit - 1) / cdef_unit;
      const int cdef_rows = (mi_rows() + cdef_unit - 1) / cdef_unit;
      if (cr < cdef_rows && cc < cdef_cols &&
          cdef_idx_[static_cast<size_t>(cr) * cdef_cols + cc] < 0) {
        const int idx = static_cast<int>(decoder_.read_literal(frame_.cdef.bits));
        const int w4 = block_width4(size), h4 = block_height4(size);
        /* a 128x128 superblock covers four 64x64 cells; every cell it reaches
         * takes the same index */
        const int row_start = cr * cdef_unit;
        const int col_start = cc * cdef_unit;
        for (int i = row_start; i < row_start + h4; i += cdef_unit) {
          for (int j = col_start; j < col_start + w4; j += cdef_unit) {
            const int cell_r = i / cdef_unit;
            const int cell_c = j / cdef_unit;
            if (cell_r < cdef_rows && cell_c < cdef_cols)
              cdef_idx_[static_cast<size_t>(cell_r) * cdef_cols + cell_c] =
                  static_cast<int8_t>(idx);
          }
        }
      }
    }

    /* read_delta_qindex / read_delta_lf: at the start of a superblock */
    trace_point(row, col, "cdef");
    const BlockSize sb_size = seq_.use_128x128_superblock ? kBlock128x128 : kBlock64x64;
    const int sb4 = seq_.use_128x128_superblock ? 32 : 16;
    const bool at_sb_start = ((row - tile_row_start_) % sb4) == 0 &&
                             ((col - tile_col_start_) % sb4) == 0;
    if (read_deltas_ && at_sb_start) {
      if (!(size == sb_size && block.skip)) {
        block.qindex = current_q_;
        const int abs = static_cast<int>(decoder_.read_symbol(cdf_.mode.delta_q, 3));
        int delta = abs;
        if (delta == 3) {
          const int rem = static_cast<int>(decoder_.read_literal(3)) + 1;
          delta = static_cast<int>(decoder_.read_literal(rem)) + (1 << rem) + 1;
        }
        if (delta != 0) {
          if (decoder_.read_bit()) delta = -delta;
          delta <<= frame_.delta_q_res;
          current_q_ = std::min(255, std::max(1, current_q_ + delta));
        }
        if (frame_.delta_lf_present) {
          const int count = frame_.delta_lf_multi ? (seq_.monochrome ? 2 : 4) : 1;
          for (int i = 0; i < count; ++i) {
            const int row_index = i + (frame_.delta_lf_multi ? 1 : 0);
            int lf_abs = static_cast<int>(decoder_.read_symbol(cdf_.mode.delta_lf[row_index], 3));
            if (lf_abs == 3) {
              const int rem = static_cast<int>(decoder_.read_literal(3)) + 1;
              lf_abs = static_cast<int>(decoder_.read_literal(rem)) + (1 << rem) + 1;
            }
            if (lf_abs != 0) {
              const int sign = decoder_.read_bit() ? -1 : 1;
              /* DeltaLF is a running value, clipped like the specification says */
              const int value = delta_lf_[i] + (sign * lf_abs << frame_.delta_lf_res);
              delta_lf_[i] = std::max(-63, std::min(63, value));
            }
          }
        }
      }
    }

    /* the frame's deltas are coded once per superblock, by its first block */
    read_deltas_ = false;
    trace_point(row, col, "delta");

    /* intra_frame_mode_info or inter_frame_mode_info: a key frame always codes
     * an intra block, an inter frame codes whether this block is intra and then
     * either the intra modes or the reference, motion vector and filter. */
    const int log2_w = kBlockWidthLog2[size], log2_h = kBlockHeightLog2[size];
    if (intra_frame) {
      const ParseStatus mode_status = read_intra_block_mode(&block, size, row, col, true);
      if (mode_status != ParseStatus::Ok) return mode_status;
      const ParseStatus chroma_status = read_chroma_mode(&block, size, has_chroma);
      if (chroma_status != ParseStatus::Ok) return chroma_status;
      block.is_inter = false;
      block.ref_frame[0] = kRefIntra;
      block.ref_frame[1] = kRefIntra;
    } else {
      const ParseStatus mode_status = read_inter_block_mode(&block, size, row, col, has_chroma);
      if (mode_status != ParseStatus::Ok) return mode_status;
    }

    /* palette_mode_info: only for frames that allow screen content tools */
    trace_point(row, col, "uv");
    if (!block.is_inter && frame_.allow_screen_content_tools && size >= kBlock8x8 &&
        block_width4(size) * 4 <= 64 && block_height4(size) * 4 <= 64) {
      return stop_at_residual(block, "tile: palette mode syntax is not implemented");
    }

    /* filter_intra_mode_info */
    trace_point(row, col, "pal");
    if (!block.is_inter && block.y_mode == kModeDc && block.palette_size[0] == 0 &&
        std::max(log2_w, log2_h) <= 3 && seq_.enable_filter_intra) {
      block.filter_intra = decoder_.read_symbol(cdf_.mode.use_filter_intra[size], 1) != 0;
      if (block.filter_intra) {
        block.filter_intra_mode = static_cast<int>(decoder_.read_symbol(cdf_.mode.filter_intra, 4));
      }
    }

    /* read_tx_size: the block codes its transform size when the frame allows a
     * choice; an inter block can then split it further (var-tx), which the
     * residual layer reads as part of the transform tree. */
    trace_point(row, col, "fi");
    TxSize tx = max_tx_size(bw4, bh4);
    /* the tx_depth context, for the trace only: -1 when no symbol is coded */
    int tx_size_row = -1, tx_depth_context = -1;
    if (!block.is_inter) {
      if (!frame_.coded_lossless && size > kBlock4x4 && frame_.tx_mode_select) {
        const int cls = tx_square_class(tx);
        const int depth_ctx =
            (above_tx_lw_[col] >= tx.lw ? 1 : 0) + (left_tx_lh_[row] >= tx.lh ? 1 : 0);
        tx_size_row = cls;
        tx_depth_context = depth_ctx;
        const int depth = static_cast<int>(
            decoder_.read_symbol(cdf_.mode.txsz[cls - 1][depth_ctx], std::min(cls, 2)));
        for (int i = 0; i < depth; ++i) tx = split_tx_size(tx);
      }
      set_cell_tx(row, col, bw4, bh4, tx.lw, tx.lh);
    } else if (block.skip || !frame_.tx_mode_select || frame_.coded_lossless) {
      /* A skipped inter block carries no residual, and a frame that does not
       * switch transform sizes codes none: the block keeps its largest. */
      if (!block.skip && frame_.coded_lossless) tx = TxSize{0, 0};
      set_cell_tx(row, col, bw4, bh4, tx.lw, tx.lh);
    } else if (tx.lw > 0 || tx.lh > 0) {
      /* Variable transform size: one split tree per largest transform the
       * block holds, each root the block's own largest transform. The leaves
       * are what the residual layer reads, one transform block each, so the
       * block no longer has a single transform size. */
      for (int y = 0; y < bh4; y += tx.h4())
        for (int x = 0; x < bw4; x += tx.w4())
          read_var_tx_size(row + y, col + x, tx, 0, size);
    } else {
      set_cell_tx(row, col, bw4, bh4, tx.lw, tx.lh);
    }
    block.tx_size = tx;
    {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "tx%d/%d", tx_size_row, tx_depth_context);
      trace_point(row, col, tag);
    }
    block.qindex = current_q_;

    for (int i = 0; i < 4; ++i) block.delta_lf[i] = delta_lf_[i];

    mark_block(row, col, bw4, bh4, size, block.skip,
               block.is_inter ? block.inter_mode : block.y_mode);
    mark_tx(row, col, bw4, bh4, tx);
    if (block.skip_mode) mi_skip_mode_set(row, col, bw4, bh4);
    if (!intra_frame) mark_inter(row, col, size, block, block.interp_filter);
    if (block.is_inter) {
      /* the uv mode of an inter block is not coded: the specification sets it
       * to DC for the purposes of the neighbour contexts */
      block.uv_mode = kModeDc;
    }

    if (block.skip) reset_coef_context(row, col, bw4, bh4, 1 + (has_chroma ? 2 : 0));
    {
      const ParseStatus status = read_residuals(&block, !block.skip);
      if (status != ParseStatus::Ok) {
        if (trace_enabled()) trace_block(block);
        trace_state(block);
        out_->blocks.push_back(block);
        ++out_->blocks_needing_residuals;
        return status;
      }
    }
    if (trace_enabled()) trace_block(block);
    trace_state(block);
    out_->blocks.push_back(block);
    return ParseStatus::Ok;
  }

  /* --- coefficients ------------------------------------------------------ */

  int sub_x() const { return (!seq_.monochrome && seq_.ss_x) ? 1 : 0; }
  int sub_y() const { return (!seq_.monochrome && seq_.ss_y) ? 1 : 0; }

  /* Size of a plane's 4x4 grid: the frame's mi grid, shrunk for chroma. This is
   * how much state the plane's contexts need; the syntax reads at most
   * MiCols >> subsampling_x of it. */
  int plane_cols4(int plane) const {
    const int sub = plane > 0 ? sub_x() : 0;
    return ((mi_cols() + (1 << sub) - 1) >> sub) + 2;
  }
  int plane_rows4(int plane) const {
    const int sub = plane > 0 ? sub_y() : 0;
    return ((mi_rows() + (1 << sub) - 1) >> sub) + 2;
  }

  /* maxX4 / maxY4 in the specification's context arithmetic. */
  int plane_max_x4(int plane) const { return mi_cols() >> (plane > 0 ? sub_x() : 0); }
  int plane_max_y4(int plane) const { return mi_rows() >> (plane > 0 ? sub_y() : 0); }

  /* all_zero: is this transform block empty? The context asks how strong the
   * neighbouring coefficients were - a block next to a busy one is unlikely to
   * be empty - with a special case for a block that is the whole residual of
   * its block size, which has no neighbours to speak of and is usually empty. */
  int all_zero_context(const TileBlock& block, int plane, int x4, int y4, TxSize tx) {
    const int max_x4 = plane_max_x4(plane), max_y4 = plane_max_y4(plane);
    const int w4 = tx.w4(), h4 = tx.h4();
    const BlockSize residual = plane_residual_size(block.size, plane, sub_x() != 0, sub_y() != 0);
    if (residual >= kBlockSizeCount) return 0;
    const int bw = block_width4(residual) * 4, bh = block_height4(residual) * 4;

    CoefPlaneContext& ctx = coef_[plane];
    if (plane == 0) {
      int top = 0, left = 0;
      for (int k = 0; k < w4 && x4 + k < max_x4; ++k) {
        top = std::max<int>(top, ctx.above_level[x4 + k]);
      }
      for (int k = 0; k < h4 && y4 + k < max_y4; ++k) {
        left = std::max<int>(left, ctx.left_level[y4 + k]);
      }
      top = std::min(top, 255);
      left = std::min(left, 255);
      const int w = tx.width(), h = tx.height();
      if (bw == w && bh == h) return 0;
      if (top == 0 && left == 0) return 1;
      if (top == 0 || left == 0) return 2 + (std::max(top, left) > 3 ? 1 : 0);
      if (std::max(top, left) <= 3) return 4;
      if (std::min(top, left) <= 3) return 5;
      return 6;
    }

    int above = 0, left = 0;
    for (int k = 0; k < w4 && x4 + k < max_x4; ++k) {
      above |= ctx.above_level[x4 + k];
      above |= ctx.above_dc[x4 + k];
    }
    for (int k = 0; k < h4 && y4 + k < max_y4; ++k) {
      left |= ctx.left_level[y4 + k];
      left |= ctx.left_dc[y4 + k];
    }
    int out = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0) + 7;
    if (bw * bh > tx.width() * tx.height()) out += 3;
    return out;
  }

  /* dc_sign: which way the neighbouring DCs leaned, so that a block in a
   * smooth gradient is likely to continue in the same direction. */
  int dc_sign_context(int plane, int x4, int y4, TxSize tx) {
    const int max_x4 = plane_max_x4(plane), max_y4 = plane_max_y4(plane);
    CoefPlaneContext& ctx = coef_[plane];
    int dc_sign = 0;
    for (int k = 0; k < tx.w4() && x4 + k < max_x4; ++k) {
      const int sign = ctx.above_dc[x4 + k];
      if (sign == 1) dc_sign--;
      else if (sign == 2) dc_sign++;
    }
    for (int k = 0; k < tx.h4() && y4 + k < max_y4; ++k) {
      const int sign = ctx.left_dc[y4 + k];
      if (sign == 1) dc_sign--;
      else if (sign == 2) dc_sign++;
    }
    return dc_sign < 0 ? 1 : (dc_sign > 0 ? 2 : 0);
  }

  /* A block that is skipped carries no residual at all, and the specification
   * clears the coefficient contexts over the whole block's area rather than
   * letting a neighbour's levels leak into the next block (reset_block_context).
   * All planes, with the chroma bounds subsampled like the plane's own grid. */
  void reset_coef_context(int row, int col, int bw4, int bh4, int planes) {
    for (int plane = 0; plane < planes; ++plane) {
      const int subx = plane > 0 ? sub_x() : 0;
      const int suby = plane > 0 ? sub_y() : 0;
      CoefPlaneContext& ctx = coef_[plane];
      const int end_x = (col + bw4) >> subx, end_y = (row + bh4) >> suby;
      for (int x = col >> subx; x < end_x && x < static_cast<int>(ctx.above_level.size()); ++x) {
        ctx.above_level[x] = 0;
        ctx.above_dc[x] = 0;
      }
      for (int y = row >> suby; y < end_y && y < static_cast<int>(ctx.left_level.size()); ++y) {
        ctx.left_level[y] = 0;
        ctx.left_dc[y] = 0;
      }
    }
  }

  /* Writes the level and DC contexts of a transform block over the area it
   * covers. */
  void set_coef_context(int plane, int x4, int y4, TxSize tx, int level, int dc_category) {
    const int max_x4 = plane_max_x4(plane), max_y4 = plane_max_y4(plane);
    CoefPlaneContext& ctx = coef_[plane];
    for (int k = 0; k < tx.w4() && x4 + k < max_x4; ++k) {
      ctx.above_level[x4 + k] = static_cast<uint8_t>(level);
      ctx.above_dc[x4 + k] = static_cast<uint8_t>(dc_category);
    }
    for (int k = 0; k < tx.h4() && y4 + k < max_y4; ++k) {
      ctx.left_level[y4 + k] = static_cast<uint8_t>(level);
      ctx.left_dc[y4 + k] = static_cast<uint8_t>(dc_category);
    }
  }

  /* The luma transform type map: transform_type() writes the chosen type over
   * every 4x4 unit of the transform, and chroma (in an inter frame) later reads
   * the luma type of the 4x4 unit above it. */
  void set_tx_type(int x4, int y4, TxSize tx, int tx_type) {
    for (int y = 0; y < tx.h4() && y4 + y < mi_rows(); ++y) {
      for (int x = 0; x < tx.w4() && x4 + x < mi_cols(); ++x) {
        tx_type_map_[index(y4 + y, x4 + x)] = static_cast<uint8_t>(tx_type);
      }
    }
  }

  int tx_type_at(int x4, int y4) const {
    if (x4 < 0 || y4 < 0 || x4 >= mi_cols() || y4 >= mi_rows()) return coef_tables::kDctDct;
    return tx_type_map_[index(y4, x4)];
  }

  /* transform_type: the luma transform type, when the size allows a choice and
   * the frame is not lossless (a zero quantiser implies DCT and saves the
   * symbol). */
  int read_tx_type(TileBlock* block, int x4, int y4, TxSize tx, int set) {
    int tx_type = coef_tables::kDctDct;
    if (set != coef_tables::kTxSetDctOnly && frame_.base_q_idx > 0) {
      const int dir = block->filter_intra
                          ? kFilterIntraToIntraDir[std::min(block->filter_intra_mode, 4)]
                          : block->y_mode;
      const int row = tx_sqr(tx);
      uint16_t* cdf = nullptr;
      if (set == coef_tables::kTxSetIntra1) {
        if (row < 2) cdf = cdf_.mode.txtp_intra1[row][dir];
      } else if (set == coef_tables::kTxSetIntra2) {
        if (row < 3) cdf = cdf_.mode.txtp_intra2[row][dir];
      }
      if (!cdf) return coef_tables::kDctDct;
      const int symbol = static_cast<int>(
          decoder_.read_symbol(cdf, tx_type_symbol_count(set, /*is_inter=*/false) - 1));
      tx_type = tx_type_from_symbol(set, symbol, /*is_inter=*/false);
    }
    set_tx_type(x4, y4, tx, tx_type);
    return tx_type;
  }

  /* compute_tx_type for the luma plane of an inter block: three tables, picked
   * by the transform's square classes - a 32 point transform (or a frame with
   * the reduced set) codes one bit that chooses between DCT and identity,
   * 16 point transforms have their own twelve symbol alphabet and everything
   * smaller shares a sixteen symbol one. A 64 point transform has no symbol at
   * all: its set is DCT only. */
  int inter_luma_tx_type(TileBlock* block, int x4, int y4, TxSize tx) {
    const int set = tx_set(tx, /*is_inter=*/true, frame_.reduced_tx_set);
    int tx_type = coef_tables::kDctDct;
    if (set != coef_tables::kTxSetDctOnly && frame_.base_q_idx > 0) {
      const int min_class = tx_min_class(tx);
      uint16_t* cdf = nullptr;
      if (set == coef_tables::kTxSetInter1 && min_class < 2) {
        cdf = cdf_.inter.txtp_inter1[min_class];
      } else if (set == coef_tables::kTxSetInter2) {
        cdf = cdf_.inter.txtp_inter2;
      } else if (set == coef_tables::kTxSetInter3 && min_class < 4) {
        cdf = cdf_.inter.txtp_inter3[min_class];
      }
      if (!cdf) return coef_tables::kDctDct;
      const int symbol = static_cast<int>(
          decoder_.read_symbol(cdf, tx_type_symbol_count(set, /*is_inter=*/true) - 1));
      tx_type = tx_type_from_symbol(set, symbol, /*is_inter=*/true);
    }
    set_tx_type(x4, y4, tx, tx_type);
    return tx_type;
  }

  /* compute_tx_type for a chroma plane of an inter block: the luma transform
   * type is inherited, except that a 32 point chroma transform can only be DCT
   * or identity, and a 16 point one cannot carry the one dimensional ADST and
   * flip variants. `luma_type` is the type of the luma transform that covers
   * the same samples - the map `set_tx_type` fills. */
  int inter_chroma_tx_type(TxSize tx, int x4, int y4) const {
    const int lx = x4 << sub_x(), ly = y4 << sub_y();
    const int luma_type = tx_type_map_[index(ly, lx)];
    if (tx_square_class(tx) == 3) {
      return luma_type == coef_tables::kIdtx ? coef_tables::kIdtx : coef_tables::kDctDct;
    }
    if (tx_min_class(tx) == 2 &&
        (luma_type == coef_tables::kHflipadst || luma_type == coef_tables::kVflipadst ||
         luma_type == coef_tables::kHadst || luma_type == coef_tables::kVadst)) {
      return coef_tables::kDctDct;
    }
    return luma_type;
  }

  /* compute_tx_type for a chroma plane of an intra frame: derived from the
   * chroma mode, and only if the size's transform set admits it. */
  int chroma_tx_type(const TileBlock& block, TxSize tx) const {
    const int implied = tx_type_from_uv_mode(block.uv_mode);
    const int set = tx_set(tx, /*is_inter=*/false, frame_.reduced_tx_set);
    return is_tx_type_in_set(set, implied, /*is_inter=*/false) ? implied : coef_tables::kDctDct;
  }

  /* The transform blocks of one block, in coding order: for each 64x64 chunk of
   * it, all luma transform blocks then both chroma planes.
   *
   * A block with no coefficient syntax at all (skip) still *has* this structure
   * - the specification calls residual() for every block, and a skipped block
   * simply reaches transform_block with skip set, so it is predicted one
   * transform block at a time and reconstruct_add is not called. Predicting a
   * skipped block as a single rectangle instead would change its intra
   * prediction wherever the transform tiling is finer than the block, so the
   * emit below is shared and `read_coefficients` only decides whether symbols
   * are consumed. */
  ParseStatus read_residuals(TileBlock* block, bool read_coefficients) {
    const int bw4 = block_width4(block->size), bh4 = block_height4(block->size);
    const int width_chunks = std::max(1, (bw4 * 4) >> 6);
    const int height_chunks = std::max(1, (bh4 * 4) >> 6);
    const BlockSize chunk_size =
        (width_chunks > 1 || height_chunks > 1) ? kBlock64x64 : block->size;
    const bool lossless = frame_.coded_lossless;
    const int planes = 1 + (block->has_chroma ? 2 : 0);

    for (int chunk_y = 0; chunk_y < height_chunks; ++chunk_y) {
      for (int chunk_x = 0; chunk_x < width_chunks; ++chunk_x) {
        for (int plane = 0; plane < planes; ++plane) {
          const int sx = plane > 0 ? sub_x() : 0;
          const int sy = plane > 0 ? sub_y() : 0;
          const TxSize tx =
              lossless ? TxSize{0, 0}
                       : plane_tx_size(block->size, plane, block->tx_size, sub_x() != 0,
                                       sub_y() != 0);
          if (tx.lw < 0) {
            return unsupported("tile: this block's chroma residual size is undefined");
          }
          const BlockSize plane_size =
              plane_residual_size(chunk_size, plane, sub_x() != 0, sub_y() != 0);
          if (plane_size >= kBlockSizeCount) {
            return unsupported("tile: this block's chroma residual size is undefined");
          }
          /* An inter block's luma residual follows its transform tree, which
           * can give each quadrant its own size; everything else - every
           * chroma plane, and luma of an intra block - is uniform. */
          if (!lossless && block->is_inter && plane == 0) {
            const int cx4 = block->mi_col + (chunk_x << 4);
            const int cy4 = block->mi_row + (chunk_y << 4);
            const int cw4 = std::min(16, bw4 - (chunk_x << 4));
            const int ch4 = std::min(16, bh4 - (chunk_y << 4));
            const ParseStatus status = read_luma_transform_tree(
                block, cx4, cy4, cw4, ch4, read_coefficients);
            if (status != ParseStatus::Ok) return status;
            continue;
          }
          const int num_w4 = block_width4(plane_size), num_h4 = block_height4(plane_size);
          const int step_x = tx.w4(), step_y = tx.h4();
          const int block_x = (block->mi_col >> sx) * 4;
          const int block_y = (block->mi_row >> sy) * 4;
          const int max_x = (mi_cols() * 4) >> sx;
          const int max_y = (mi_rows() * 4) >> sy;

          for (int y = 0; y < num_h4; y += step_y) {
            for (int x = 0; x < num_w4; x += step_x) {
              const int start_x = block_x + 4 * (x + ((chunk_x << 4) >> sx));
              const int start_y = block_y + 4 * (y + ((chunk_y << 4) >> sy));
              if (start_x >= max_x || start_y >= max_y) continue; /* off the frame */
              if (!read_coefficients) {
                emit_empty_transform(block, plane, start_x, start_y, tx);
                continue;
              }
              const ParseStatus status = read_transform_block(
                  block, plane, start_x >> 2, start_y >> 2, tx, lossless);
              if (status != ParseStatus::Ok) return status;
            }
          }
        }
      }
    }
    return ParseStatus::Ok;
  }

  /* A transform block of a skipped block: it costs no coefficients, but the
   * prediction and residual stages still expect an entry per transform. */
  void emit_empty_transform(TileBlock* block, int plane, int x, int y, TxSize tx) {
    if (plane == 0) set_tx_type(x >> 2, y >> 2, tx, coef_tables::kDctDct);
    TileTransform empty;
    empty.plane = plane;
    empty.size = tx;
    empty.x = x;
    empty.y = y;
    empty.qindex = block->qindex;
    empty.tx_type = coef_tables::kDctDct;
    empty.eob = 0;
    empty.level_offset = out_->levels.size();
    block->transforms.push_back(empty);
    ++out_->transform_blocks;
  }

  /* find_tx_size: the transform that covers exactly this many 4x4 blocks. The
   * residual walk only asks for sizes that come out of halving the block's
   * largest transform, so both sides are powers of two. */
  static TxSize find_tx_size(int w4, int h4) {
    int lw = 0, lh = 0;
    while ((1 << lw) < w4) ++lw;
    while ((1 << lh) < h4) ++lh;
    return TxSize{lw, lh};
  }

  /* transform_tree: how an inter block's luma residual is cut into transform
   * blocks. The transform sizes are already known - read_block_tx_size read the
   * split tree - so this walks InterTxSizes: an area that fits inside the
   * transform its top left cell names is one transform block, anything larger
   * is halved along its longer side until it does. */
  ParseStatus read_luma_transform_tree(TileBlock* block, int x4, int y4, int w4, int h4,
                                       bool read_coefficients) {
    if (x4 >= mi_cols() || y4 >= mi_rows()) return ParseStatus::Ok;
    const int lw = tx_cell_lw_[index(y4, x4)], lh = tx_cell_lh_[index(y4, x4)];
    if (w4 <= (1 << lw) && h4 <= (1 << lh)) {
      const TxSize tx = find_tx_size(w4, h4);
      if (read_coefficients) return read_transform_block(block, 0, x4, y4, tx, false);
      emit_empty_transform(block, 0, x4 * 4, y4 * 4, tx);
      return ParseStatus::Ok;
    }
    ParseStatus status = ParseStatus::Ok;
    if (w4 > h4) {
      status = read_luma_transform_tree(block, x4, y4, w4 / 2, h4, read_coefficients);
      if (status == ParseStatus::Ok)
        status = read_luma_transform_tree(block, x4 + w4 / 2, y4, w4 / 2, h4, read_coefficients);
    } else if (w4 < h4) {
      status = read_luma_transform_tree(block, x4, y4, w4, h4 / 2, read_coefficients);
      if (status == ParseStatus::Ok)
        status = read_luma_transform_tree(block, x4, y4 + h4 / 2, w4, h4 / 2, read_coefficients);
    } else {
      status = read_luma_transform_tree(block, x4, y4, w4 / 2, h4 / 2, read_coefficients);
      if (status == ParseStatus::Ok)
        status = read_luma_transform_tree(block, x4 + w4 / 2, y4, w4 / 2, h4 / 2,
                                          read_coefficients);
      if (status == ParseStatus::Ok)
        status = read_luma_transform_tree(block, x4, y4 + h4 / 2, w4 / 2, h4 / 2,
                                          read_coefficients);
      if (status == ParseStatus::Ok)
        status = read_luma_transform_tree(block, x4 + w4 / 2, y4 + h4 / 2, w4 / 2, h4 / 2,
                                          read_coefficients);
    }
    return status;
  }

  /* One transform block's coefficients. */
  ParseStatus read_transform_block(TileBlock* block, int plane, int x4, int y4, TxSize tx,
                                   bool lossless) {
    const bool chroma = plane > 0;
    const int chroma_index = chroma ? 1 : 0;
    const int tx_ctx = tx_size_context(tx);

    TileTransform transform;
    transform.plane = plane;
    transform.size = tx;
    transform.x = x4 * 4;
    transform.y = y4 * 4;
    transform.qindex = block->qindex;
    transform.tx_type = coef_tables::kDctDct;
    transform.level_offset = out_->levels.size();

    const int zero_ctx = all_zero_context(*block, plane, x4, y4, tx);
    if (zero_ctx < 0 || zero_ctx > 12) return corrupt("all_zero");
    if (trace_) {
      std::fprintf(stderr, "cf plane=%d x4=%d y4=%d tx=%dx%d blk=%dx%d@(%d,%d) ctx=%d/%d q=%d\n",
                   plane, x4, y4, tx.width(), tx.height(), block_width4(block->size) * 4,
                   block_height4(block->size) * 4, block->mi_col, block->mi_row, tx_ctx,
                   zero_ctx, block->qindex);
    }
    if (decoder_.read_symbol(cdf_.coef.skip[tx_ctx][zero_ctx], 1)) {
      /* empty: the transform kernels still run, they just have nothing to add */
      if (plane == 0) set_tx_type(x4, y4, tx, coef_tables::kDctDct);
      set_coef_context(plane, x4, y4, tx, 0, 0);
      if (trace_) {
        std::fprintf(stderr, "cf plane=%d x4=%d y4=%d tx=%dx%d zero\n", plane, x4, y4,
                     tx.width(), tx.height());
      }
      trace_coef(plane, x4, y4, tx, coef_tables::kDctDct, 0);
      block->transforms.push_back(transform);
      ++out_->transform_blocks;
      return ParseStatus::Ok;
    }

    /* the transform type */
    int plane_tx_type = coef_tables::kDctDct;
    if (lossless) {
      if (plane == 0) set_tx_type(x4, y4, tx, coef_tables::kDctDct);
    } else if (plane == 0) {
      plane_tx_type = block->is_inter
                          ? inter_luma_tx_type(block, x4, y4, tx)
                          : read_tx_type(block, x4, y4, tx,
                                         tx_set(tx, false, frame_.reduced_tx_set));
    } else {
      plane_tx_type = block->is_inter ? inter_chroma_tx_type(tx, x4, y4)
                                      : chroma_tx_type(*block, tx);
    }
    transform.tx_type = plane_tx_type;
    const int tx_class = tx_class_of(plane_tx_type);
    const int is_1d = tx_class != coef_tables::kTxClass2d ? 1 : 0;

    /* where the last non-zero coefficient is */
    const CoefGrid grid = coef_grid(tx);
    const ScanOrder scan = coef_scan(tx, plane_tx_type);
    const int size_class = eob_size_class(tx);
    uint16_t* eob_cdf = nullptr;
    switch (size_class) {
      case 0: eob_cdf = cdf_.coef.eob_bin_16[chroma_index][is_1d]; break;
      case 1: eob_cdf = cdf_.coef.eob_bin_32[chroma_index][is_1d]; break;
      case 2: eob_cdf = cdf_.coef.eob_bin_64[chroma_index][is_1d]; break;
      case 3: eob_cdf = cdf_.coef.eob_bin_128[chroma_index][is_1d]; break;
      case 4: eob_cdf = cdf_.coef.eob_bin_256[chroma_index][is_1d]; break;
      case 5: eob_cdf = cdf_.coef.eob_bin_512[chroma_index]; break;
      default: eob_cdf = cdf_.coef.eob_bin_1024[chroma_index]; break;
    }
    const int eob_symbol = static_cast<int>(decoder_.read_symbol(eob_cdf, 4 + size_class));
    const int eob_pt = eob_symbol + 1;
    int eob = eob_pt < 2 ? eob_pt : ((1 << (eob_pt - 2)) + 1);
    int eob_shift = std::max(-1, eob_pt - 3);
    if (eob_shift >= 0) {
      const int extra_row = eob_pt - 3;
      if (extra_row < 0 || extra_row > 8) return corrupt("eob_extra");
      if (decoder_.read_symbol(cdf_.coef.eob_hi_bit[tx_ctx][chroma_index][extra_row], 1)) {
        eob += 1 << eob_shift;
      }
      const int tail_bits = std::max(0, eob_pt - 2);
      for (int i = 1; i < tail_bits; ++i) {
        eob_shift = tail_bits - 1 - i;
        if (decoder_.read_bit()) eob += 1 << eob_shift;
      }
    }
    if (eob <= 0 || eob > grid.area()) return corrupt("eob");

    /* the coefficient levels, last one first: it is never zero, so it is coded
     * with its own alphabet, and everything before it is coded as 0, 1, 2 or
     * "more", with the "more" extended by further tokens. */
    int16_t quant[1024] = {0};
    uint16_t (*base_cdf)[4] = cdf_.coef.base_tok[tx_ctx][chroma_index];
    uint16_t (*br_cdf)[4] = cdf_.coef.br_tok[std::min(tx_ctx, 3)][chroma_index];
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan.at(c);
      if (pos < 0) return corrupt("scan");
      int level;
      if (c == eob - 1) {
        const int ctx = coeff_base_context(grid, tx_class, quant, pos, tx, c, /*is_eob=*/true) -
                        coef_tables::kSigCoefContexts + coef_tables::kSigCoefContextsEob;
        if (ctx < 0 || ctx >= coef_tables::kSigCoefContextsEob) return corrupt("coeff_base_eob");
        level = static_cast<int>(decoder_.read_symbol(
                    cdf_.coef.eob_base_tok[tx_ctx][chroma_index][ctx],
                    2 /* 0, 1 or "more" */)) + 1;
      } else {
        const int ctx = coeff_base_context(grid, tx_class, quant, pos, tx, c, /*is_eob=*/false);
        if (ctx < 0 || ctx > 40) return corrupt("coeff_base");
        level = static_cast<int>(decoder_.read_symbol(base_cdf[ctx], 3));
      }
      if (level > coef_tables::kNumBaseLevels) {
        for (int idx = 0; idx < coef_tables::kCoeffBaseRange / (coef_tables::kBrCdfSize - 1);
             ++idx) {
          const int ctx = coeff_br_context(grid, tx_class, quant, pos);
          if (ctx < 0 || ctx > 20) return corrupt("coeff_br");
          const int br = static_cast<int>(decoder_.read_symbol(br_cdf[ctx], 3));
          level += br;
          if (br < coef_tables::kBrCdfSize - 1) break;
        }
      }
      quant[pos] = static_cast<int16_t>(level & 0xffff);
    }

    /* signs, and the coefficients too large for the base range, in scan order
     * from the DC outwards */
    const int dc_ctx = dc_sign_context(plane, x4, y4, tx);
    int cul_level = 0;
    int dc_category = 0;
    for (int c = 0; c < eob; ++c) {
      const int pos = scan.at(c);
      int sign = 0;
      if (quant[pos] != 0) {
        if (c == 0) {
          sign = static_cast<int>(decoder_.read_symbol(cdf_.coef.dc_sign[chroma_index][dc_ctx], 1));
        } else {
          sign = static_cast<int>(decoder_.read_bit());
        }
      }
      if (quant[pos] > coef_tables::kNumBaseLevels + coef_tables::kCoeffBaseRange) {
        int length = 0;
        while (decoder_.read_bit() == 0) {
          ++length;
          if (length > 20) return corrupt("golomb");
        }
        ++length;
        int value = 1;
        for (int i = length - 2; i >= 0; --i) {
          value = (value << 1) | static_cast<int>(decoder_.read_bit());
        }
        quant[pos] = static_cast<int16_t>(value + coef_tables::kCoeffBaseRange +
                                          coef_tables::kNumBaseLevels);
      }
      if (pos == 0 && quant[pos] > 0) dc_category = sign ? 1 : 2;
      cul_level += quant[pos];
      const int16_t level = static_cast<int16_t>(quant[pos]);
      out_->levels.push_back(static_cast<int16_t>(sign ? -level : level));
    }
    cul_level = std::min(63, cul_level);
    set_coef_context(plane, x4, y4, tx, cul_level, dc_category);

    transform.eob = eob;
    if (trace_) {
      std::fprintf(stderr, "cf plane=%d x4=%d y4=%d tx=%dx%d txtp=%d eob=%d\n", plane, x4, y4,
                   tx.width(), tx.height(), plane_tx_type, eob);
    }
    trace_coef(plane, x4, y4, tx, plane_tx_type, eob);
    out_->coefficients += static_cast<size_t>(eob);
    block->transforms.push_back(transform);
    ++out_->transform_blocks;
    return ParseStatus::Ok;
  }

  ParseStatus corrupt(const char* what) {
    if (error_) {
      *error_ = std::string("tile: ") + what + " symbol is outside its alphabet";
    }
    return ParseStatus::Invalid;
  }

  const SequenceHeader& seq_;
  const FrameHeader& frame_;
  const RefPool* refs_;
  const uint8_t* data_;
  size_t size_;
  TileDecodeResult* out_;
  std::string* error_;

  RangeDecoder decoder_;
  CdfContext cdf_;

  int tile_col_start_ = 0, tile_col_end_ = 0, tile_row_start_ = 0, tile_row_end_ = 0;
  int current_q_ = 0;
  /* DeltaLF[0..3]: a running per tile value, updated by the delta_lf syntax at
   * a superblock start and copied into every block of that superblock. */
  int delta_lf_[4] = {0, 0, 0, 0};
  bool read_deltas_ = false;

  std::vector<MiRef> mi_ref_;
  std::vector<uint8_t> mi_size_;
  std::vector<uint8_t> skip_map_;
  std::vector<uint8_t> skip_mode_map_;
  /* cdef_idx per 64x64, -1 until a block that is not skipped reads it */
  std::vector<int8_t> cdef_idx_;
  std::vector<uint8_t> y_mode_map_;
  std::vector<int8_t> above_tx_lw_;
  std::vector<int8_t> left_tx_lh_;
  /* luma transform type per 4x4 unit, as transform_type() writes it */
  std::vector<uint8_t> tx_type_map_;
  /* coefficient level and DC statistics, per plane */
  CoefPlaneContext coef_[3];
  /* AV1GPU_TRACE: one line per transform block on stderr, for debugging */
  bool trace_ = false;

 public:
  void init_state() {
    const int cols = mi_cols(), rows = mi_rows();
    mi_size_.assign(static_cast<size_t>(cols) * rows, kBlock4x4);
    mi_ref_.assign(static_cast<size_t>(cols) * rows, MiRef());
    skip_map_.assign(static_cast<size_t>(cols) * rows, 0);
    skip_mode_map_.assign(static_cast<size_t>(cols) * rows, 0);
    y_mode_map_.assign(static_cast<size_t>(cols) * rows, kModeDc);
    above_tx_lw_.assign(static_cast<size_t>(cols), 0);
    left_tx_lh_.assign(static_cast<size_t>(rows), 0);
    tx_cell_lw_.assign(static_cast<size_t>(cols) * rows, 0);
    tx_cell_lh_.assign(static_cast<size_t>(cols) * rows, 0);
    tx_type_map_.assign(static_cast<size_t>(cols) * rows, coef_tables::kDctDct);
    const int cdef_cols = (cols + 15) >> 4;
    const int cdef_rows = (rows + 15) >> 4;
    cdef_idx_.assign(static_cast<size_t>(cdef_cols) * cdef_rows, -1);
    out_->cdef_cols = cdef_cols;
    out_->cdef_rows = cdef_rows;
    for (int plane = 0; plane < 3; ++plane) {
      const size_t plane_cols = static_cast<size_t>(plane_cols4(plane));
      const size_t plane_rows = static_cast<size_t>(plane_rows4(plane));
      coef_[plane].above_level.assign(plane_cols, 0);
      coef_[plane].above_dc.assign(plane_cols, 0);
      coef_[plane].left_level.assign(plane_rows, 0);
      coef_[plane].left_dc.assign(plane_rows, 0);
    }
    init_default_cdf(&cdf_, frame_.base_q_idx);
    current_q_ = frame_.base_q_idx;
    for (int i = 0; i < 4; ++i) delta_lf_[i] = 0;
    decoder_.init(data_, size_, !frame_.disable_cdf_update);
  }
};

} /* namespace */

ParseStatus decode_intra_tile(const SequenceHeader& seq, const FrameHeader& frame,
                              const uint8_t* data, size_t size, TileDecodeResult* out,
                              std::string* error) {
  if (error) error->clear();
  if (!out) return fail(error, "tile: no output");
  *out = TileDecodeResult();
  if (!data || size == 0) return fail(error, "tile: no tile data");
  if (frame.type != FrameType::Key && frame.type != FrameType::IntraOnly) {
    if (error) *error = "tile: only key and intra-only frames are decoded so far";
    return ParseStatus::Unsupported;
  }
  if (frame.tiles.cols != 1 || frame.tiles.rows != 1) {
    if (error) *error = "tile: multi-tile frames are not decoded so far";
    return ParseStatus::Unsupported;
  }

  TileParser parser(seq, frame, nullptr, data, size, out, error);
  parser.init_state();
  const ParseStatus status = parser.run();
  /* the index grid is only read once the tile is done: it is per 64x64, not per
   * block, and a tile that failed half way through has no plans to feed */
  out->cdef_idx = parser.cdef_index();
  out->bytes_used = parser.bytes_used();
  out->overrun = parser.overrun();
  if (status == ParseStatus::Ok && parser.failed()) {
    return fail(error, "tile: the range coder lost synchronisation");
  }
  return status;
}

ParseStatus decode_tile(const SequenceHeader& seq, const FrameHeader& frame,
                        const RefPool* refs, const uint8_t* data, size_t size,
                        TileDecodeResult* out, std::string* error) {
  if (error) error->clear();
  if (!out) return fail(error, "tile: no output");
  *out = TileDecodeResult();
  if (!data || size == 0) return fail(error, "tile: no tile data");
  const bool intra_frame = frame.type == FrameType::Key || frame.type == FrameType::IntraOnly;
  if (!intra_frame) {
    if (!refs) {
      if (error) *error = "tile: an inter frame needs its reference frames";
      return ParseStatus::Unsupported;
    }
    /* Temporal motion vector candidates come from the motion vectors the
     * reference frames stored. An intra reference frame has none - every cell
     * of its projected map is a no-op - so a frame whose references are all
     * intra can be decoded without implementing the projection. */
    if (frame.use_ref_frame_mvs) {
      bool have_inter_reference = false;
      for (int i = 0; i < 7; ++i) {
        const int slot = frame.ref_frame_idx[i];
        if (slot < 0 || slot > 7) continue;
        const RefSlot& reference = refs->slots[slot];
        if (reference.valid && reference.type != FrameType::Key &&
            reference.type != FrameType::IntraOnly) {
          have_inter_reference = true;
        }
      }
      if (have_inter_reference) {
        if (error) *error = "tile: projected motion vectors are not implemented";
        return ParseStatus::Unsupported;
      }
    }
  }
  if (frame.tiles.cols != 1 || frame.tiles.rows != 1) {
    if (error) *error = "tile: multi-tile frames are not decoded so far";
    return ParseStatus::Unsupported;
  }

  TileParser parser(seq, frame, refs, data, size, out, error);
  parser.init_state();
  const ParseStatus status = parser.run();
  out->cdef_idx = parser.cdef_index();
  out->bytes_used = parser.bytes_used();
  out->overrun = parser.overrun();
  if (status == ParseStatus::Ok && parser.failed()) {
    return fail(error, "tile: the range coder lost synchronisation");
  }
  return status;
}

} /* namespace frontend */
} /* namespace av1gpu */
