#include "frontend/tile_decoder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "frontend/coef_model.hpp"
#include "frontend/coef_tables.hpp"
#include "frontend/range_decoder.hpp"
#include "frontend/warp.hpp"

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

/* AV1GPU_TRACE_CDF=1 adds the partition context derivation and the whole CDF
 * row to that trace. Finding the first block where two decoders disagree needs
 * it, but it is far too much output for every other use of the trace. */
/* The frame this decoder is currently parsing, counted from zero. Only the
 * debug probes read it, and only so a dump can be attributed to a frame. */
int g_frame_ordinal = -1;

bool cdf_trace_enabled() {
  static const bool enabled = std::getenv("AV1GPU_TRACE_CDF") != nullptr;
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
                 block.compound_type, block.drl_index, block.interp_filter[0],
                 block.interp_filter[1], block.wedge_index, block.mask_sign,
                 block.interintra ? 1 : 0, block.interintra_mode,
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

/* Sign extended distance between two order hints, `a - b`: positive when `a` is
 * the later of the two, wrapped into the sequence header's bit width. Zero when
 * the stream does not carry order hints at all, which is also what disables
 * temporal motion vector prediction. */
int order_hint_delta(const SequenceHeader& seq, int a, int b) {
  if (!seq.enable_order_hint) return 0;
  int diff = a - b;
  const int m = 1 << (seq.order_hint_n_bits - 1);
  return (diff & (m - 1)) - (diff & m);
}

/* AV1GPU_TRACE_T=1 dumps the temporal motion vector grids: what a frame saved
 * ("T" lines) and what it projected onto itself ("U" lines), in the shape the
 * reference decoder prints, so the two projections can be diffed cell by cell.
 * The projection is integer arithmetic with a sign convention per reference
 * type, which is exactly the kind of thing that decodes plausibly and is wrong
 * for a handful of blocks. */
bool trace_tmvs() {
  static const bool enabled = std::getenv("AV1GPU_TRACE_T") != nullptr;
  return enabled;
}

/* INVALID_REF2CUR: a contributing reference frame whose distance to this one is
 * too large to be projected (the order hint wrapped around). */
constexpr int kInvalidRef2Cur = -32;

/* Wedge_Bits: how many bits a block size's wedge parameters carry. Zero means
 * the size has no wedge mask at all, and a masked compound block of that size
 * has to use the difference weighted mask instead. */
constexpr int8_t kWedgeBits[kBlockSizeCount] = {0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0,
                                                0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0};

/* The wedge context: block sizes grouped by aspect ratio, which is the row of
 * every wedge table the syntax reads. Sizes without a wedge keep 0, which is
 * what makes an out of range lookup impossible rather than merely unlikely. */
constexpr int8_t kWedgeContext[kBlockSizeCount] = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0};

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
    if (frame_.type == FrameType::Inter || frame_.type == FrameType::Switch) {
      auto store = std::make_shared<RefMvs>();
      save_temporal_mvs(store.get());
      out_->ref_mvs = std::move(store);
    }
    /* The tables this frame passes on: the ones it adapted when it may update
     * the context at its end, otherwise the ones it started from. A frame that
     * updates nothing leaves its references with exactly the tables they
     * already had, rather than the ones it used and abandoned.
     *
     * The adapted tables lose their adaptation counters on the way out, and the
     * two tables a key frame alone reads (the key frame mode table and the
     * intra block copy one) keep the values the frame inherited: a later frame
     * starts from the probabilities this one reached but adapts them at the
     * rate a fresh context adapts. */
    auto published = std::make_shared<CdfContext>(
        frame_.refresh_frame_context ? cdf_ : start_cdf_);
    if (frame_.refresh_frame_context) {
      std::memcpy(published->mode.kfym, start_cdf_.mode.kfym, sizeof(published->mode.kfym));
      std::memcpy(published->mode.intrabc, start_cdf_.mode.intrabc,
                  sizeof(published->mode.intrabc));
      reset_cdf_counts(published.get());
    }
    out_->cdf = std::move(published);
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
    int above_sz = -1, left_sz = -1;
    if (have_top(row)) {
      above_sz = mi_size_[index(row - 1, col)];
      above = kBlockWidthLog2[above_sz] < bsl;
    }
    if (have_left(col)) {
      left_sz = mi_size_[index(row, col - 1)];
      left = kBlockHeightLog2[left_sz] < bsl;
    }
    if (cdf_trace_enabled())
      std::fprintf(stderr, "PC %d %d bsl=%d above_sz=%d left_sz=%d above=%d left=%d\n", col, row,
                   bsl, above_sz, left_sz, above, left);
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
    /* the interpolation filter this block used, per direction, for the filter
     * context */
    uint8_t filter[2] = {0, 0};
    /* how this block blended its two predictions, which several of the
     * compound contexts read from their neighbours */
    uint8_t comp_type = kCompoundNone;
  };

  /* One candidate in the mv stack: a vector (or, for a compound block, a vector
   * pair) and the weight the scan accumulated for it. */
  struct MvCand {
    int mv[2][2] = {{0, 0}, {0, 0}};
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

  /* --- projected (temporal) motion vectors --------------------------------
   *
   * `use_ref_frame_mvs` lets a block borrow motion vectors from the frames it
   * refers to: each of those frames saved, per 8x8 cell, the vector of the block
   * covering it, and this frame projects every saved vector forward along its
   * own trajectory onto the cell it would reach here. The projection of a whole
   * frame is cheap and depends on nothing but the reference frames, so it is
   * done once, before the first block, and the candidate scan then reads it like
   * a spatial neighbour.
   *
   * Two things make this fiddly. The projection is a rational scaling by the
   * ratio of two "distance in time" values, and it has to land in the *same*
   * 8x8 row band it started in, so cells only move within a band plus one
   * above and below the band's left/right edges. And which reference frames
   * contribute is not simply "the ones this frame refers to": the specification
   * names at most three (or two, when one of them is the last frame), preferring
   * the ones closest in time, and each contributes with its own reference
   * distance, which is what the `mfmv_` arrays hold.
   */

  static int apply_sign(int magnitude, int sign_source) {
    return sign_source < 0 ? -magnitude : magnitude;
  }

  /* The frame's temporal candidate setup: which reference frames contribute
   * (`mfmv_ref_`), how far each is from this frame (`mfmv_ref2cur_`) and, for
   * the vectors they stored, how far the frame *they* stored them from is
   * (`mfmv_ref2ref_`). `pocdiff_` is this frame's distance to each of its own
   * seven references, which is the numerator of every projection. */
  void setup_temporal_mvs() {
    const int iw8 = (frame_.frame_width + 7) >> 3;
    const int ih8 = (frame_.height + 7) >> 3;
    proj_cols_ = iw8;
    proj_.assign(static_cast<size_t>(iw8) * ih8, TemporalMv());
    mfmv_count_ = 0;
    if (!refs_) return;

    for (int i = 0; i < 7; ++i) {
      const int to_ref = order_hint_delta(seq_, frame_.order_hint, frame_.ref_order_hint[i]);
      pocdiff_[i] = std::max(-31, std::min(31, to_ref));
      /* a vector that points at a frame before this one can be projected onto a
       * frame after it; the other direction is not usable */
      const int from_ref = order_hint_delta(seq_, frame_.ref_order_hint[i], frame_.order_hint);
      mfmv_sign_[i] = from_ref < 0;
      /* and a reference *after* this frame flips the sign of the vectors a
       * neighbour stored for it, which the extended candidate scan does */
      sign_bias_[i] = from_ref > 0;
    }
    if (!frame_.use_ref_frame_mvs || !seq_.enable_order_hint) return;

    /* The frames whose saved grid this frame can read: an inter frame of the
     * same size, in the slot its reference type names. */
    const RefMvs* stored[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    int stored_poc[7][7] = {};
    for (int i = 0; i < 7; ++i) {
      const int slot_index = frame_.ref_frame_idx[i];
      if (slot_index < 0 || slot_index > 7) continue;
      const RefSlot& slot = refs_->slots[slot_index];
      for (int r = 0; r < 7; ++r) stored_poc[i][r] = slot.ref_poc[r];
      if (slot.ref_mvs && slot.ref_mvs->cols == iw8 && slot.ref_mvs->rows == ih8) {
        stored[i] = slot.ref_mvs.get();
      }
    }

    for (int n = 0; n < 3; ++n) mfmv_stored_[n] = nullptr;
    const auto relative = [this](int a, int b) { return order_hint_delta(seq_, a, b); };

    int total = 2;
    if (stored[kRefIdxLast] && stored_poc[kRefIdxLast][kRefIdxAltref] !=
                                   frame_.ref_order_hint[kRefIdxGolden]) {
      /* the last frame was predicted from something other than the frame this
       * one calls golden, so the two are distinct points in time: it is worth a
       * candidate of its own */
      mfmv_ref_[mfmv_count_++] = kRefIdxLast;
      total = 3;
    }
    if (stored[kRefIdxBwdref] &&
        relative(frame_.ref_order_hint[kRefIdxBwdref], frame_.order_hint) > 0)
      mfmv_ref_[mfmv_count_++] = kRefIdxBwdref;
    if (stored[kRefIdxAltref2] &&
        relative(frame_.ref_order_hint[kRefIdxAltref2], frame_.order_hint) > 0)
      mfmv_ref_[mfmv_count_++] = kRefIdxAltref2;
    if (mfmv_count_ < total && stored[kRefIdxAltref] &&
        relative(frame_.ref_order_hint[kRefIdxAltref], frame_.order_hint) > 0)
      mfmv_ref_[mfmv_count_++] = kRefIdxAltref;
    if (mfmv_count_ < total && stored[kRefIdxLast2]) mfmv_ref_[mfmv_count_++] = kRefIdxLast2;

    if (trace_tmvs()) {
      std::fprintf(stderr, "TS per=%d n=%d hint=%d refs=", pocdiff_[0], mfmv_count_,
                   frame_.order_hint);
      for (int i = 0; i < 7; ++i) std::fprintf(stderr, " %d", frame_.ref_order_hint[i]);
      std::fprintf(stderr, "\n");
    }
    for (int n = 0; n < mfmv_count_; ++n) {
      const int ref = mfmv_ref_[n];
      mfmv_stored_[n] = stored[ref];
      const int ref_hint = frame_.ref_order_hint[ref];
      const int distance = relative(ref_hint, frame_.order_hint);
      if (std::abs(distance) > 31) {
        mfmv_ref2cur_[n] = kInvalidRef2Cur;
        continue;
      }
      /* A backward reference (bwd, altref2, altref) is stored with the sign the
       * specification gives it; the forward ones are negated. */
      mfmv_ref2cur_[n] = ref < kRefIdxBwdref ? -distance : distance;
      for (int m = 0; m < 7; ++m) {
        const int their_distance = relative(ref_hint, stored_poc[ref][m]);
        /* a distance the wrong way round, or too large to be representable, is
         * stored as zero, which means "this vector cannot be projected" */
        mfmv_ref2ref_[n][m] =
            static_cast<uint8_t>(their_distance <= 0 || their_distance > 31 ? 0 : their_distance);
      }
    }
    if (trace_tmvs()) {
      for (int n = 0; n < mfmv_count_; ++n) {
        std::fprintf(stderr, "TA n=%d ref=%d ref2cur=%d ref2ref=", n, mfmv_ref_[n],
                     mfmv_ref2cur_[n]);
        for (int m = 0; m < 7; ++m) std::fprintf(stderr, " %d", mfmv_ref2ref_[n][m]);
        std::fprintf(stderr, "\n");
      }
    }
    project_temporal_mvs(iw8, ih8);
  }

  /* load_tmvs: project every stored vector of the contributing reference frames
   * onto this frame's grid. Later contributors overwrite earlier ones, which is
   * the specification's way of preferring the closer frames (they come later in
   * `mfmv_ref_`). */
  void project_temporal_mvs(int iw8, int ih8) {
    for (int n = 0; n < mfmv_count_; ++n) {
      const RefMvs* stored = mfmv_stored_[n];
      if (!stored || mfmv_ref2cur_[n] == kInvalidRef2Cur) continue;
      /* the reference type the stored vector points at is inverted for the
       * forward references, and the projection follows the same sign */
      const int ref_sign = mfmv_ref_[n] - kRefIdxBwdref;
      for (int y = 0; y < ih8; ++y) {
        /* a vector only reaches the same 8x8 band of rows it started in */
        const int band = y & ~7;
        const int band_end = std::min(band + 8, ih8);
        for (int x = 0; x < iw8; ++x) {
          const TemporalMv& cell = stored->at(x, y);
          if (!cell.valid()) continue;
          const int ref2ref = mfmv_ref2ref_[n][cell.ref - 1];
          if (!ref2ref) continue;
          const int offset_x = project_mv_component(cell.x, mfmv_ref2cur_[n], ref2ref);
          const int offset_y = project_mv_component(cell.y, mfmv_ref2cur_[n], ref2ref);
          /* 8x8 cells: the stored vector is in 1/8 sample units, the projection
           * divides by 64 to land on one */
          const int pos_x = x + apply_sign(std::abs(offset_x) >> 6, offset_x ^ ref_sign);
          const int pos_y = y + apply_sign(std::abs(offset_y) >> 6, offset_y ^ ref_sign);
          if (pos_y < band || pos_y >= band_end) continue;
          /* and it only spreads one cell to either side of its own band */
          const int x_band = x & ~7;
          if (pos_x < std::max(x_band - 8, 0) || pos_x >= std::min(x_band + 16, iw8)) continue;
          TemporalMv& dst = proj_[static_cast<size_t>(pos_y) * proj_cols_ + pos_x];
          dst.x = cell.x;
          dst.y = cell.y;
          dst.ref = static_cast<uint8_t>(ref2ref);
        }
      }
    }
    if (trace_tmvs()) {
      for (int y = 0; y < ih8; ++y) {
        for (int x = 0; x < iw8; ++x) {
          const TemporalMv& cell = projected_at(x, y);
          if (cell.valid()) {
            std::fprintf(stderr, "U %d %d %d %d %d\n", x, y, cell.ref, cell.x, cell.y);
          }
        }
      }
    }
  }

  const TemporalMv& projected_at(int col, int row) const {
    static const TemporalMv kEmpty;
    if (col < 0 || row < 0 || col >= proj_cols_) return kEmpty;
    if (static_cast<size_t>(row) * proj_cols_ + col >= proj_.size()) return kEmpty;
    return proj_[static_cast<size_t>(row) * proj_cols_ + col];
  }

  /* Two cells belong to the same "run" when they would contribute the same
   * candidate: the scan counts identical neighbours instead of visiting one
   * candidate per cell. Cells with no vector compare equal to each other
   * whatever they contain, because adding them is a no-op either way. */
  static bool same_temporal_cell(const TemporalMv& a, const TemporalMv& b) {
    if (a.valid() != b.valid()) return false;
    if (!a.valid()) return true;
    return a.ref == b.ref && a.x == b.x && a.y == b.y;
  }

  /* add_temporal_candidate: one projected vector, from a run of `weight / 2`
   * cells. The first call of the whole scan also settles `globalmv_ctx`, the
   * context bit that says whether the temporal prediction already agrees with
   * the global motion model (in which case the far/near modes are cheap).
   *
   * A compound block projects the same stored vector twice, once with the
   * distance to each of its references: the stored vector points from the
   * reference frame to *its* reference, so the two projections land on this
   * frame along two different directions. */
  void add_temporal_candidate(MvCand* stack, int* count, const TemporalMv& cell, int ref0,
                              int ref1, int* globalmv_ctx, int gmv_x, int gmv_y, int weight,
                              int origin_row, int origin_col) {
    if (!cell.valid()) return;
    MvCand entry;
    entry.weight = weight;
    entry.mv[0][0] = project_mv_component(cell.x, pocdiff_[ref0 - 1], cell.ref);
    entry.mv[0][1] = project_mv_component(cell.y, pocdiff_[ref0 - 1], cell.ref);
    lower_mv_precision(&entry.mv[0][0], &entry.mv[0][1]);
    if (ref1 == kRefIntra) {
      if (globalmv_ctx) {
        *globalmv_ctx = (std::abs(entry.mv[0][0] - gmv_x) | std::abs(entry.mv[0][1] - gmv_y)) >= 16;
        if (trace_enabled())
          std::fprintf(stderr,
                       "GMV fo=%d %d %d mv=%d,%d gmv=%d,%d res=%d raw=%d,%d rr=%d pd=%d,%d\n",
                       g_frame_ordinal, origin_col, origin_row, entry.mv[0][1], entry.mv[0][0],
                       gmv_y, gmv_x, *globalmv_ctx, cell.y, cell.x, cell.ref,
                       pocdiff_[ref0 - 1], pocdiff_[cell.ref - 1]);
      }
    } else {
      entry.mv[1][0] = project_mv_component(cell.x, pocdiff_[ref1 - 1], cell.ref);
      entry.mv[1][1] = project_mv_component(cell.y, pocdiff_[ref1 - 1], cell.ref);
      lower_mv_precision(&entry.mv[1][0], &entry.mv[1][1]);
    }
    const int last = *count;
    for (int n = 0; n < last; ++n) {
      if (stack[n].mv[0][0] != entry.mv[0][0] || stack[n].mv[0][1] != entry.mv[0][1]) continue;
      if (ref1 != kRefIntra &&
          (stack[n].mv[1][0] != entry.mv[1][0] || stack[n].mv[1][1] != entry.mv[1][1]))
        continue;
      stack[n].weight += weight;
      return;
    }
    if (last < 8) {
      stack[last] = entry;
      *count = last + 1;
    }
  }

  /* The temporal part of find_ref_mvs: the block's cells, the small block edge
   * cases, and the resulting global motion context. */
  int temporal_candidates(MvCand* stack, int* count, int ref0, int ref1, int bw4, int bh4, int w4,
                          int h4, int mi_row, int mi_col, int gmv_x, int gmv_y) {
    int globalmv_ctx = frame_.use_ref_frame_mvs ? 1 : 0;
    if (mfmv_count_ == 0) return globalmv_ctx;

    const int bx8 = mi_col >> 1, by8 = mi_row >> 1;
    const int step_h = bw4 >= 16 ? 2 : 1, step_v = bh4 >= 16 ? 2 : 1;
    const int w8 = std::min((w4 + 1) >> 1, 8), h8 = std::min((h4 + 1) >> 1, 8);

    /* The scan walks the cells in runs, starting with the one at the block's own
     * origin (which the traversal below skips) and counting the run instead of
     * adding the same candidate over and over. */
    TemporalMv run = projected_at(bx8, by8);
    int run_len = 1;
    int* run_gmv_ctx = &globalmv_ctx;
    for (int y = 0; y < h8; y += step_v) {
      for (int x = y ? 0 : step_h; x < w8; x += step_h) {
        const TemporalMv& cell = projected_at(bx8 + x, by8 + y);
        if (same_temporal_cell(cell, run)) {
          ++run_len;
          continue;
        }
        add_temporal_candidate(stack, count, run, ref0, ref1, run_gmv_ctx, gmv_x, gmv_y,
                               2 * run_len, mi_row, mi_col);
        run_gmv_ctx = nullptr;
        run = cell;
        run_len = 1;
      }
    }
    add_temporal_candidate(stack, count, run, ref0, ref1, run_gmv_ctx, gmv_x, gmv_y, 2 * run_len, mi_row, mi_col);

    /* Small blocks also look at the cells just past their bottom right corner,
     * which are the ones a neighbouring block's own scan would not reach. */
    if (std::min(bw4, bh4) >= 2 && std::max(bw4, bh4) < 16) {
      const int bw8 = bw4 >> 1, bh8 = bh4 >> 1;
      const bool has_bottom = by8 + bh8 < std::min(tile_row_end_ >> 1, (by8 & ~7) + 8);
      if (has_bottom && bx8 - 1 >= std::max(tile_col_start_ >> 1, bx8 & ~7)) {
        add_temporal_candidate(stack, count, projected_at(bx8 - 1, by8 + bh8), ref0, ref1,
                               nullptr, 0, 0, 2, mi_row, mi_col);
      }
      if (bx8 + bw8 < std::min(tile_col_end_ >> 1, (bx8 & ~7) + 8)) {
        if (has_bottom) {
          add_temporal_candidate(stack, count, projected_at(bx8 + bw8, by8 + bh8), ref0, ref1,
                                 nullptr, 0, 0, 2, mi_row, mi_col);
        }
        if (by8 + bh8 - 1 < std::min(tile_row_end_ >> 1, (by8 & ~7) + 8)) {
          add_temporal_candidate(stack, count, projected_at(bx8 + bw8, by8 + bh8 - 1), ref0,
                                 ref1, nullptr, 0, 0, 2, mi_row, mi_col);
        }
      }
    }
    return globalmv_ctx;
  }

  /* save_tmvs: what this frame leaves behind for the frames that refer to it.
   *
   * One record per 8x8 cell, taken from the block covering the cell's bottom
   * right 4x4 sample - that is the sample the specification uses, because it is
   * the one a block's neighbours can see. A block only contributes a vector
   * that points *backwards* in time (which is what makes it projectable onto a
   * later frame) and that is small enough to be a real prediction rather than a
   * scroll; the whole width of the block is filled with the same record. */
  void save_temporal_mvs(RefMvs* out) const {
    const int iw8 = (frame_.frame_width + 7) >> 3;
    const int ih8 = (frame_.height + 7) >> 3;
    out->resize(iw8, ih8);
    for (int y = 0; y < ih8; ++y) {
      const int row = std::min(y * 2 + 1, mi_rows() - 1);
      int x = 0;
      while (x < iw8) {
        const int col = std::min(x * 2 + 1, mi_cols() - 1);
        const int bw8 = (block_width4(block_size_at(row, col)) + 1) >> 1;
        TemporalMv cell;
        const MiRef& mi = mi_at(row, col);
        if (!mi.intra) {
          for (int n = 1; n >= 0; --n) {
            const int ref = mi.ref[n];
            if (ref <= kRefIntra || !mfmv_sign_[ref - 1]) continue;
            if ((std::abs(mi.mv[n][0]) | std::abs(mi.mv[n][1])) >= 4096) continue;
            cell.x = mi.mv[n][0];
            cell.y = mi.mv[n][1];
            cell.ref = static_cast<uint8_t>(ref);
            break;
          }
        }
        for (int n = 0; n < bw8 && x < iw8; ++n, ++x) out->at(x, y) = cell;
      }
    }
    if (trace_tmvs()) {
      for (int y = 0; y < ih8; ++y)
        for (int x = 0; x < iw8; ++x)
          if (out->at(x, y).valid())
            std::fprintf(stderr, "T %d %d %d %d %d\n", x, y, out->at(x, y).ref,
                         out->at(x, y).x, out->at(x, y).y);
    }
  }

  /* --- candidate scan ----------------------------------------------------
   *
   * find_ref_mvs collects the motion vectors a block may inherit: the ones its
   * neighbours above and to the left used for the same reference (or the same
   * reference *pair*, for a compound block), plus the vectors the reference
   * frames projected onto this one. Each candidate carries a weight that says
   * how much of the block's edge it covers and how often it was seen, and the
   * stack is sorted by that weight, so the most likely vector comes first.
   *
   * The nearest group and the "secondary" one are weighted and sorted
   * separately: the direct neighbours get a large bonus, so they always precede
   * the projected and secondary candidates, but within their own group the
   * weight decides.
   */

  /* One scan, for one reference pair. `ref[1] == kRefIntra` marks a single
   * reference block: the specification writes -1 there, and that value is what
   * selects the single reference form of every test below. */
  struct MvScan {
    int origin_row = 0;   /* the block the scan belongs to, for the traces */
    int origin_col = 0;
    MvCand* stack = nullptr;
    int* count = nullptr;
    int ref[2] = {kRefIntra, kRefIntra};
    int gmv[2][2] = {{0, 0}, {0, 0}}; /* the global motion vector per reference */
    bool gmv_valid[2] = {false, false}; /* ... and whether a match substitutes it */
    bool have_newmv = false;
  };

  /* add_spatial_candidate: one neighbour's contribution. A single reference
   * block only looks at the neighbour's first references that name the same
   * reference; a compound block needs the exact pair, in the same order, which
   * is what makes a compound candidate much rarer. A neighbour that used global
   * motion contributes the vector the model gives *this* block instead of its
   * own. */
  void add_spatial_candidate(MvScan& scan, int weight, int row, int col, bool* have_refmv) {
    if (row < 0 || row >= mi_rows() || col < 0 || col >= mi_cols()) return;
    const MiRef& cand = mi_at(row, col);
    if (cand.intra) return;
    if (trace_enabled())
      std::fprintf(stderr, "SC fo=%d %d %d nb=%d,%d bs=%d refs=%d,%d mv=%d,%d,%d,%d mf=%d w=%d\n",
                   g_frame_ordinal, scan.origin_col, scan.origin_row, col, row,
                   static_cast<int>(block_size_at(row, col)), cand.ref[0],
                   cand.ref[1] ? cand.ref[1] : -1, cand.mv[0][1], cand.mv[0][0],
                   cand.mv[1][1], cand.mv[1][0], cand.global_subpel + 2 * cand.coded_newmv,
                   weight);
    const bool compound = scan.ref[1] != kRefIntra;
    if (!compound) {
      if (cand.ref[0] != scan.ref[0] && cand.ref[1] != scan.ref[0]) return;
    } else if (cand.ref[0] != scan.ref[0] || cand.ref[1] != scan.ref[1]) {
      return;
    }

    MvCand entry;
    entry.weight = weight;
    for (int n = 0; n < (compound ? 2 : 1); ++n) {
      int slot = n;
      if (!compound && cand.ref[0] != scan.ref[0]) slot = 1;
      const bool substitute = cand.global_subpel && scan.gmv_valid[n];
      entry.mv[n][0] = substitute ? scan.gmv[n][0] : cand.mv[slot][0];
      entry.mv[n][1] = substitute ? scan.gmv[n][1] : cand.mv[slot][1];
    }

    *have_refmv = true;
    scan.have_newmv = scan.have_newmv || cand.coded_newmv;
    const int last = *scan.count;
    for (int m = 0; m < last; ++m) {
      if (scan.stack[m].mv[0][0] != entry.mv[0][0] || scan.stack[m].mv[0][1] != entry.mv[0][1])
        continue;
      if (compound && (scan.stack[m].mv[1][0] != entry.mv[1][0] ||
                       scan.stack[m].mv[1][1] != entry.mv[1][1]))
        continue;
      scan.stack[m].weight += weight;
      return;
    }
    if (last < 8) {
      scan.stack[last] = entry;
      *scan.count = last + 1;
    }
  }

  /* scan_row: walk the row above from the block's left edge to its right edge,
   * adding one candidate per *block* encountered (a neighbour narrower than us
   * contributes several, a wider one only once). */
  int scan_row(MvScan& scan, int row, int base_col, int bw4, int w4, int max_rows, int step,
               bool& have_refmv) {
    int col = base_col;
    const int first_w4 = block_width4(block_size_at(row, col));
    const int first_h4 = block_height4(block_size_at(row, col));
    int len = std::max(step, std::min(bw4, first_w4));
    if (bw4 <= first_w4) {
      const int weight = bw4 == 1 ? 2 : std::max(2, std::min(2 * max_rows, first_h4));
      add_spatial_candidate(scan, len * weight, row, col, &have_refmv);
      return weight >> 1;
    }
    for (int x = 0;;) {
      add_spatial_candidate(scan, len * 2, row, col, &have_refmv);
      x += len;
      if (x >= w4) return 1;
      col = base_col + x;
      const int cand_w4 = block_width4(block_size_at(row, col));
      len = std::max(step, cand_w4);
    }
  }

  int scan_col(MvScan& scan, int base_row, int col, int bh4, int h4, int max_cols, int step,
               bool& have_refmv) {
    int row = base_row;
    const int first_h4 = block_height4(block_size_at(row, col));
    const int first_w4 = block_width4(block_size_at(row, col));
    int len = std::max(step, std::min(bh4, first_h4));
    if (bh4 <= first_h4) {
      const int weight = bh4 == 1 ? 2 : std::max(2, std::min(2 * max_cols, first_w4));
      add_spatial_candidate(scan, len * weight, row, col, &have_refmv);
      return weight >> 1;
    }
    for (int y = 0;;) {
      add_spatial_candidate(scan, len * 2, row, col, &have_refmv);
      y += len;
      if (y >= h4) return 1;
      row = base_row + y;
      const int cand_h4 = block_height4(block_size_at(row, col));
      len = std::max(step, cand_h4);
    }
  }

  /* add_single_extended_candidate: a neighbour that named a *different*
   * reference still holds a usable vector, mirrored if the two references are
   * on opposite sides of this frame. It only fills a stack that has fewer than
   * two candidates, so it never outranks a real match. */
  void add_single_extended_candidate(MvCand* stack, int* count, int row, int col, int ref0) {
    if (row < 0 || row >= mi_rows() || col < 0 || col >= mi_cols()) return;
    const MiRef& cand = mi_at(row, col);
    if (cand.intra) return;
    for (int n = 0; n < 2; ++n) {
      const int cand_ref = cand.ref[n];
      if (cand_ref <= kRefIntra) break;
      int cx = cand.mv[n][0], cy = cand.mv[n][1];
      if (sign_bias_[ref0 - 1] != sign_bias_[cand_ref - 1]) {
        cx = -cx;
        cy = -cy;
      }
      const int last = *count;
      int m = 0;
      for (; m < last; ++m)
        if (stack[m].mv[0][0] == cx && stack[m].mv[0][1] == cy) break;
      if (m == last) {
        stack[m].mv[0][0] = cx;
        stack[m].mv[0][1] = cy;
        stack[m].weight = 2;
        *count = last + 1;
      }
    }
  }

  /* add_compound_extended_candidate: the same idea for a compound block, but the
   * neighbour's vector has to be assigned to whichever of our two references it
   * can serve. `same` collects the vectors that already name one of our
   * references (component 0 for reference 0, component 1 for reference 1) and
   * `same + 2` the ones that name another reference and therefore need
   * mirroring - those fill the other component. */
  void add_compound_extended_candidate(MvCand* same, int* same_count, int row, int col, int ref0,
                                       int ref1) {
    MvCand* diff = &same[2];
    int* diff_count = &same_count[2];
    if (row < 0 || row >= mi_rows() || col < 0 || col >= mi_cols()) return;
    const MiRef& cand = mi_at(row, col);
    if (cand.intra) return;
    for (int n = 0; n < 2; ++n) {
      const int cand_ref = cand.ref[n];
      if (cand_ref <= kRefIntra) break;
      int cx = cand.mv[n][0], cy = cand.mv[n][1];
      const bool flip = sign_bias_[cand_ref - 1];
      if (cand_ref == ref0) {
        if (same_count[0] < 2) {
          same[same_count[0]].mv[0][0] = cx;
          same[same_count[0]].mv[0][1] = cy;
          ++same_count[0];
        }
        if (diff_count[1] < 2) {
          if (flip != (sign_bias_[ref1 - 1] != 0)) {
            cx = -cx;
            cy = -cy;
          }
          diff[diff_count[1]].mv[1][0] = cx;
          diff[diff_count[1]].mv[1][1] = cy;
          ++diff_count[1];
        }
      } else if (cand_ref == ref1) {
        if (same_count[1] < 2) {
          same[same_count[1]].mv[1][0] = cx;
          same[same_count[1]].mv[1][1] = cy;
          ++same_count[1];
        }
        if (diff_count[0] < 2) {
          if (flip != (sign_bias_[ref0 - 1] != 0)) {
            cx = -cx;
            cy = -cy;
          }
          diff[diff_count[0]].mv[0][0] = cx;
          diff[diff_count[0]].mv[0][1] = cy;
          ++diff_count[0];
        }
      } else {
        int ix = -cx, iy = -cy;
        if (diff_count[0] < 2) {
          const bool mirror = flip != (sign_bias_[ref0 - 1] != 0);
          diff[diff_count[0]].mv[0][0] = mirror ? ix : cx;
          diff[diff_count[0]].mv[0][1] = mirror ? iy : cy;
          ++diff_count[0];
        }
        if (diff_count[1] < 2) {
          const bool mirror = flip != (sign_bias_[ref1 - 1] != 0);
          diff[diff_count[1]].mv[1][0] = mirror ? ix : cx;
          diff[diff_count[1]].mv[1][1] = mirror ? iy : cy;
          ++diff_count[1];
        }
      }
    }
  }

  /* find_ref_mvs: the specification's candidate scan. `ctx` packs the three
   * contexts the mode syntax needs (newmv_ctx, globalmv_ctx, refmv_ctx in that
   * bit layout). A single reference block passes `ref1 == kRefIntra`. */
  void find_ref_mvs(MvCand* stack, int* count, int* ctx, int ref0, int ref1, BlockSize size,
                    int mi_row, int mi_col) {
    *count = 0;
    const int bw4 = block_width4(size), bh4 = block_height4(size);
    const int w4 = std::min(std::min(bw4, 16), tile_col_end_ - mi_col);
    const int h4 = std::min(std::min(bh4, 16), tile_row_end_ - mi_row);
    const bool compound = ref1 != kRefIntra;

    MvScan scan;
    scan.origin_row = mi_row;
    scan.origin_col = mi_col;
    scan.stack = stack;
    scan.count = count;
    scan.ref[0] = ref0;
    scan.ref[1] = ref1;
    for (int n = 0; n < (compound ? 2 : 1); ++n) {
      const GlobalMotion& gm = global_motion_of(scan.ref[n]);
      global_mv(gm, mi_row, mi_col, bw4, bh4, &scan.gmv[n][0], &scan.gmv[n][1]);
      /* the model's vector only stands in for a neighbour's when the model is
       * more than a translation: a translated neighbour already carries the
       * same vector explicitly */
      scan.gmv_valid[n] = gm.type > kGmTranslation;
    }

    bool have_col_mvs = false, have_row_mvs = false;
    unsigned max_rows = 0, n_rows = ~0U, max_cols = 0, n_cols = ~0U;
    const bool top_avail = mi_row > tile_row_start_;
    const bool left_avail = mi_col > tile_col_start_;
    if (top_avail) {
      max_rows = std::min(static_cast<unsigned>((mi_row - tile_row_start_ + 1) >> 1),
                          2u + (bh4 > 1 ? 1u : 0u));
      n_rows = scan_row(scan, mi_row - 1, mi_col, bw4, w4, static_cast<int>(max_rows),
                        bw4 >= 16 ? 4 : 1, have_row_mvs);
    }
    if (left_avail) {
      max_cols = std::min(static_cast<unsigned>((mi_col - tile_col_start_ + 1) >> 1),
                          2u + (bw4 > 1 ? 1u : 0u));
      n_cols = scan_col(scan, mi_row, mi_col - 1, bh4, h4, static_cast<int>(max_cols),
                        bh4 >= 16 ? 4 : 1, have_col_mvs);
    }
    /* top/right */
    if (n_rows != ~0U && above_right_available(mi_row, mi_col, bw4) &&
        std::max(bw4, bh4) <= 16 && mi_col + bw4 < tile_col_end_) {
      add_spatial_candidate(scan, 4, mi_row - 1, mi_col + bw4, &have_row_mvs);
    }

    const int nearest_match = (have_col_mvs ? 1 : 0) + (have_row_mvs ? 1 : 0);
    const int nearest_count = *count;
    for (int n = 0; n < nearest_count; ++n) stack[n].weight += 640;

    /* Projected candidates: the vectors the reference frames stored, moved onto
     * this frame's grid. They come after the nearest group and before the
     * top/left corner, because the corner and the secondary rows and columns
     * belong to the "secondary" group.
     *
     * The global motion context starts at the frame's use_ref_frame_mvs flag and
     * the first projected candidate may clear it: a vector that agrees with the
     * global motion model leaves it alone, one that disagrees clears it. A frame
     * that refers only to intra frames has no projected candidates at all, so
     * the flag survives as it is, which is why this is not simply zero when the
     * projection is empty. */
    const int globalmv_ctx = temporal_candidates(stack, count, ref0, ref1, bw4, bh4, w4, h4,
                                                 mi_row, mi_col, scan.gmv[0][0], scan.gmv[0][1]);

    /* top/left: the diagonal neighbour, when both scans ran. It adds a
     * candidate and counts as a reference match, but its vector is not one the
     * new/global modes can name, so it must not claim a new vector was coded. */
    if (top_avail && left_avail) {
      const bool newmv_before = scan.have_newmv;
      add_spatial_candidate(scan, 4, mi_row - 1, mi_col - 1, &have_row_mvs);
      scan.have_newmv = newmv_before;
    }

    /* the "secondary" rows and columns, two and three blocks away, which the
     * specification scans at 8x8 resolution */
    for (int n = 2; n <= 3; ++n) {
      /* a secondary scan adds candidates and counts as a reference match, but it
       * must not claim the neighbours coded a new vector: only the direct
       * neighbours decide that half of the mode context */
      const bool newmv_before = scan.have_newmv;
      if (static_cast<unsigned>(n) > n_rows && static_cast<unsigned>(n) <= max_rows) {
        n_rows += scan_row(scan, (mi_row - 2 * n + 1) | 1, mi_col | 1, bw4, w4,
                           1 + static_cast<int>(max_rows) - n, bw4 >= 16 ? 4 : 2, have_row_mvs);
      }
      if (static_cast<unsigned>(n) > n_cols && static_cast<unsigned>(n) <= max_cols) {
        n_cols += scan_col(scan, mi_row | 1, (mi_col - n * 2 + 1) | 1, bh4, h4,
                           1 + static_cast<int>(max_cols) - n, bh4 >= 16 ? 4 : 2, have_col_mvs);
      }
      scan.have_newmv = newmv_before;
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
        newmv_ctx = 3 - (scan.have_newmv ? 1 : 0);
        break;
      default:
        refmv_ctx = 5;
        newmv_ctx = 5 - (scan.have_newmv ? 1 : 0);
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

    /* A compound scan always has two candidates: the mode syntax needs a vector
     * for each reference, so a short stack is completed from the neighbours that
     * named some other reference (mirrored onto ours) and, failing that, from
     * the global motion vectors. */
    if (compound) {
      if (*count < 2) {
        const int sz4 = std::min(w4, h4);
        MvCand* same = &stack[*count];
        int same_count[4] = {0, 0, 0, 0};
        if (n_rows != ~0U) {
          for (int x = 0; x < sz4;) {
            const int at = std::min(mi_col + x, mi_cols() - 1);
            add_compound_extended_candidate(same, same_count, mi_row - 1, at, ref0, ref1);
            x += block_width4(block_size_at(mi_row - 1, at));
          }
        }
        if (n_cols != ~0U) {
          for (int y = 0; y < sz4;) {
            const int at = std::min(mi_row + y, mi_rows() - 1);
            add_compound_extended_candidate(same, same_count, at, mi_col - 1, ref0, ref1);
            y += block_height4(block_size_at(at, mi_col - 1));
          }
        }
        MvCand* diff = &same[2];
        const int* diff_count = &same_count[2];
        for (int n = 0; n < 2; ++n) {
          int m = same_count[n];
          if (m >= 2) continue;
          const int l = diff_count[n];
          if (l) {
            same[m].mv[n][0] = diff[0].mv[n][0];
            same[m].mv[n][1] = diff[0].mv[n][1];
            if (++m == 2) continue;
            if (l == 2) {
              same[1].mv[n][0] = diff[1].mv[n][0];
              same[1].mv[n][1] = diff[1].mv[n][1];
              continue;
            }
          }
          do {
            same[m].mv[n][0] = scan.gmv[n][0];
            same[m].mv[n][1] = scan.gmv[n][1];
          } while (++m < 2);
        }

        /* If the one candidate we had was the same as the first extended one,
         * keep the second extended one instead: the two would otherwise be the
         * same vector twice, which is exactly what the mode syntax must not
         * see. */
        const int base = *count;
        if (base == 1 && stack[0].mv[0][0] == same[0].mv[0][0] &&
            stack[0].mv[0][1] == same[0].mv[0][1] && stack[0].mv[1][0] == same[0].mv[1][0] &&
            stack[0].mv[1][1] == same[0].mv[1][1]) {
          for (int n = 0; n < 2; ++n) {
            stack[1].mv[n][0] = same[1].mv[n][0];
            stack[1].mv[n][1] = same[1].mv[n][1];
          }
        }
        for (int n = base; n < 2; ++n) stack[n].weight = 2;
        *count = 2;
      }

      /* clamping, both components of every candidate */
      const int left = -(mi_col + bw4 + 4) * 4 * 8;
      const int right = (mi_cols() - mi_col + 4) * 4 * 8;
      const int top = -(mi_row + bh4 + 4) * 4 * 8;
      const int bottom = (mi_rows() - mi_row + 4) * 4 * 8;
      for (int n = 0; n < *count; ++n) {
        for (int m = 0; m < 2; ++m) {
          stack[n].mv[m][0] = std::max(left, std::min(right, stack[n].mv[m][0]));
          stack[n].mv[m][1] = std::max(top, std::min(bottom, stack[n].mv[m][1]));
        }
      }

      /* Compound_Mode_Ctx_Map: the compound mode has eight contexts, built from
       * the same scan statistics but far more coarsely than the single
       * reference modes use them. */
      switch (refmv_ctx >> 1) {
        case 0: *ctx = std::min(newmv_ctx, 1); break;
        case 1: *ctx = 1 + std::min(newmv_ctx, 3); break;
        default: *ctx = std::max(4, std::min(7, 3 + newmv_ctx)); break;
      }
      return;
    } else if (*count < 2 && ref0 > kRefIntra) {
      const int sz4 = std::min(w4, h4);
      if (n_rows != ~0U) {
        for (int x = 0; x < sz4 && *count < 2;) {
          const int at = mi_col + x;
          add_single_extended_candidate(stack, count, mi_row - 1, at, ref0);
          x += block_width4(block_size_at(mi_row - 1, std::min(at, mi_cols() - 1)));
        }
      }
      if (n_cols != ~0U) {
        for (int y = 0; y < sz4 && *count < 2;) {
          const int at = mi_row + y;
          add_single_extended_candidate(stack, count, at, mi_col - 1, ref0);
          y += block_height4(block_size_at(std::min(at, mi_rows() - 1), mi_col - 1));
        }
      }
    }

    /* clamping */
    if (*count) {
      const int left = -(mi_col + bw4 + 4) * 4 * 8;
      const int right = (mi_cols() - mi_col + 4) * 4 * 8;
      const int top = -(mi_row + bh4 + 4) * 4 * 8;
      const int bottom = (mi_rows() - mi_row + 4) * 4 * 8;
      for (int n = 0; n < *count; ++n) {
        stack[n].mv[0][0] = std::max(left, std::min(right, stack[n].mv[0][0]));
        stack[n].mv[0][1] = std::max(top, std::min(bottom, stack[n].mv[0][1]));
      }
    }

    /* A stack shorter than two is filled with the global motion vector, but the
     * count is what the mode syntax tests: an empty stack still means "no
     * candidate", which suppresses the dynamic reference list index bits. */
    for (int n = *count; n < 2; ++n) {
      stack[n].mv[0][0] = scan.gmv[0][0];
      stack[n].mv[0][1] = scan.gmv[0][1];
      stack[n].weight = 2;
    }

    *ctx = (refmv_ctx << 4) | (globalmv_ctx << 3) | newmv_ctx;
  }

  void trace_stack(int col, int row, int count, const MvCand* stack) const {
    if (!trace_enabled()) return;
    std::fprintf(stderr, "ST X %d %d n=%d", col, row, count);
    for (int i = 0; i < count && i < 8; ++i)
      std::fprintf(stderr, " [%d %d,%d w=%d]", i, stack[i].mv[0][1], stack[i].mv[0][0],
                   stack[i].weight);
    std::fprintf(stderr, "\n");
  }

  void trace_drl(int col, int row, int idx, int ctx, int count, const MvCand* stack) const {
    if (!trace_enabled()) return;
    std::fprintf(stderr, "DRL %d %d idx=%d ctx=%d n=%d w0=%d w1=%d cdf=%u,%u\n", col, row, idx, ctx,
                 count, stack[0].weight, stack[1].weight, cdf_.inter.drl_mode[ctx][0],
                 cdf_.inter.drl_mode[ctx][1]);
  }

  /* get_drl_context */
  int drl_context(const MvCand* stack, int idx) const {
    if (stack[idx].weight >= 640) return stack[idx + 1].weight < 640 ? 1 : 0;
    return stack[idx + 1].weight < 640 ? 2 : 0;
  }

  /* One dynamic reference list bit: the context comes from how strong the two
   * candidates at `idx` and `idx + 1` are, and the trace prints both the
   * context and the row it lands on, which is what a wrong context shows up as. */
  int read_drl_bit(int col, int row, int idx, int count, const MvCand* stack) {
    const int ctx = drl_context(stack, idx);
    trace_drl(col, row, idx, ctx, count, stack);
    return static_cast<int>(decoder_.read_symbol(cdf_.inter.drl_mode[ctx], 1));
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

  /* Which references a reference context counts. The names are the
   * specification's; several of them are aliases of each other, which is why
   * the compound syntax can ask the same question under two names. */
  enum RefKind {
    kRefKindAll = 0,      /* av1_get_ref_ctx, uni_p: forward against backward */
    kRefKindForward,      /* fwd_ref, fwd_ref_3: LAST+LAST2 against LAST3+GOLDEN */
    kRefKindForward1,     /* fwd_ref_1, uni_p1: LAST against LAST2 */
    kRefKindForward2,     /* fwd_ref_2, uni_p2: LAST3 against GOLDEN */
    kRefKindBackward,     /* bwd_ref, bwd_ref_2: the three backward references */
    kRefKindBackward1,    /* bwd_ref_1: BWDREF against ALTREF2 */
    kRefKindUniP1,        /* uni_p1: LAST2 against LAST3 + GOLDEN */
  };

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
          case 6:
            if (ref >= kRefLast2 && ref <= kRefGolden) count[ref - kRefLast2]++;
            break;                                                        /* uni p1    */
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
      case 6:
        count[1] += count[2];
        return count[0] == count[1] ? 1 : (count[0] < count[1] ? 0 : 2);
      default:
        return 0;
    }
  }

  /* get_filter_ctx: 0 when both neighbours used the same filter for this
   * reference, 3 when they disagree, and the neighbour's filter when only one
   * of them used this reference at all. `dir` picks which of the two filters a
   * dual filter frame is coding, and `comp` moves on to the compound half of the
   * table, where the neighbours' statistics are the compound ones. */
  int filter_context(int row, int col, bool comp, int dir, int ref) {
    const int kNone = 3;
    int a_filter = kNone, l_filter = kNone;
    if (have_top(row)) {
      const MiRef& n = mi_at(row - 1, col);
      if (n.ref[0] == ref || n.ref[1] == ref) a_filter = n.filter[dir];
    }
    if (have_left(col)) {
      const MiRef& n = mi_at(row, col - 1);
      if (n.ref[0] == ref || n.ref[1] == ref) l_filter = n.filter[dir];
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

  /* A neighbour that pointed at a backward reference. An intra neighbour is
   * *not* one: the reference it would have pointed at does not exist. The
   * compound flag's context is the one place where intra is scored as if it
   * were backward (see neighbour_is_backward_or_intra below), so the two tests
   * are kept apart. */
  static bool neighbour_is_backward(const MiRef& n) { return n.ref[0] >= kRefBwdref; }

  /* The test the two mixed cases of get_comp_ctx use: with one compound and one
   * single reference neighbour, an intra neighbour counts as backward, because
   * the compound block it abuts cannot itself have used two directions. */
  static bool neighbour_is_backward_or_intra(const MiRef& n) {
    return n.intra || neighbour_is_backward(n);
  }

  /* A neighbour that coded a second reference, which is what "compound" means
   * for a context: an intra neighbour stores none. */
  static bool neighbour_is_compound(const MiRef& n) { return !n.intra && n.ref[1] > kRefIntra; }

  /* get_comp_ctx: how likely the two neighbours make a compound block, which
   * is what the compound flag's context says. A neighbour that was compound
   * itself is the strongest signal, then one that at least pointed a reference
   * backwards (the two directions of a compound block have to differ). */
  int comp_context(int row, int col) {
    const bool top = have_top(row), left = have_left(col);
    const MiRef* a = top ? &mi_at(row - 1, col) : nullptr;
    const MiRef* l = left ? &mi_at(row, col - 1) : nullptr;
    if (a && l) {
      if (neighbour_is_compound(*a)) {
        if (neighbour_is_compound(*l)) return 4;
        return 2 + (neighbour_is_backward_or_intra(*l) ? 1 : 0);
      }
      if (neighbour_is_compound(*l)) return 2 + (neighbour_is_backward_or_intra(*a) ? 1 : 0);
      return neighbour_is_backward(*l) == neighbour_is_backward(*a) ? 0 : 1;
    }
    if (a) return neighbour_is_compound(*a) ? 3 : (neighbour_is_backward(*a) ? 1 : 0);
    if (l) return neighbour_is_compound(*l) ? 3 : (neighbour_is_backward(*l) ? 1 : 0);
    return 1;
  }

  /* has_uni_comp: a compound neighbour whose two references are both forward or
   * both backward, which is the case the compound reference syntax codes with a
   * shorter tree. */
  static bool has_uni_comp(const MiRef& n) {
    return (n.ref[0] < kRefBwdref) == (n.ref[1] < kRefBwdref);
  }

  /* get_comp_dir_ctx: bidirectionality of the two neighbours. A single
   * reference neighbour says nothing about the direction, so it is scored
   * differently from a compound one - the whole point of the context is to
   * predict whether this block uses a forward and a backward reference or two
   * of the same direction. */
  int comp_dir_context(int row, int col) {
    const bool top = have_top(row), left = have_left(col);
    const MiRef* a = top ? &mi_at(row - 1, col) : nullptr;
    const MiRef* l = left ? &mi_at(row, col - 1) : nullptr;
    const auto is_comp = [](const MiRef* n) { return n && !n->intra && n->comp_type != kCompoundNone; };
    if (a && l) {
      const bool a_intra = a->intra, l_intra = l->intra;
      if (a_intra && l_intra) return 2;
      if (a_intra || l_intra) {
        const MiRef& edge = a_intra ? *l : *a;
        if (!is_comp(&edge)) return 2;
        return 1 + (has_uni_comp(edge) ? 2 : 0);
      }
      const bool a_comp = is_comp(a), l_comp = is_comp(l);
      if (!a_comp && !l_comp) {
        return 1 + (((a->ref[0] >= kRefBwdref) == (l->ref[0] >= kRefBwdref)) ? 2 : 0);
      }
      if (!a_comp || !l_comp) {
        const MiRef& edge = a_comp ? *a : *l;
        if (!has_uni_comp(edge)) return 1;
        return 3 + (((a->ref[0] >= kRefBwdref) == (l->ref[0] >= kRefBwdref)) ? 1 : 0);
      }
      const bool a_uni = has_uni_comp(*a), l_uni = has_uni_comp(*l);
      if (!a_uni && !l_uni) return 0;
      if (!a_uni || !l_uni) return 2;
      return 3 + ((a->ref[0] == kRefBwdref) == (l->ref[0] == kRefBwdref) ? 1 : 0);
    }
    const MiRef* edge = l ? l : a;
    if (!edge) return 2;
    if (edge->intra) return 2;
    if (edge->comp_type == kCompoundNone) return 2;
    return has_uni_comp(*edge) ? 4 : 0;
  }

  /* get_jnt_comp_ctx: whether the two references are equally far from this
   * frame, which decides whether the distance weighted blend is even
   * interesting, plus the neighbours' use of it. */
  int jnt_comp_context(int row, int col, int ref0, int ref1) {
    const int d0 = std::abs(order_hint_delta(seq_, frame_.ref_order_hint[ref0 - 1],
                                             frame_.order_hint));
    const int d1 = std::abs(order_hint_delta(seq_, frame_.order_hint,
                                             frame_.ref_order_hint[ref1 - 1]));
    const int a_ctx = (have_top(row) && !mi_at(row - 1, col).intra &&
                       (mi_at(row - 1, col).comp_type >= kCompoundAverage ||
                        mi_at(row - 1, col).ref[0] == kRefAltref))
                          ? 1
                          : 0;
    const int l_ctx = (have_left(col) && !mi_at(row, col - 1).intra &&
                       (mi_at(row, col - 1).comp_type >= kCompoundAverage ||
                        mi_at(row, col - 1).ref[0] == kRefAltref))
                          ? 1
                          : 0;
    return 3 * (d0 == d1 ? 1 : 0) + a_ctx + l_ctx;
  }

  /* get_mask_comp_ctx: whether the neighbours used a masked blend. */
  int mask_comp_context(int row, int col) {
    const auto side = [](const MiRef& n) {
      if (n.intra) return 0;
      if (n.comp_type >= kCompoundDiffwtd) return 1;
      return n.ref[0] == kRefAltref ? 3 : 0;
    };
    const int a_ctx = have_top(row) ? side(mi_at(row - 1, col)) : 0;
    const int l_ctx = have_left(col) ? side(mi_at(row, col - 1)) : 0;
    return std::min(a_ctx + l_ctx, 5);
  }

  /* Wedge_Bits: the number of wedge bits a block size can carry, as the
   * reference decoders tabulate it. Only the nine square-ish sizes have a wedge
   * mask at all; the other sizes have none and therefore code the difference
   * weighted mask instead. */
  int wedge_bits(BlockSize size) const { return kWedgeBits[size]; }

  /* The wedge direction context: the wedge table is indexed by this number,
   * which is the block size grouped by aspect ratio. */
  int wedge_context(BlockSize size) const { return kWedgeContext[size]; }

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

  /* find_warp_samples (7.11.3.8): the neighbours the local warp is fitted to.
   * A neighbour counts when it uses this block's reference frame, is not
   * compound, and was decoded in this frame - a block's grid entry is written
   * when its mode info is read, so an entry still marked intra is either an
   * intra block or a block this frame has not reached, and neither is a usable
   * sample.
   *
   * Each sample is the centre of the neighbour's block and that centre moved by
   * the neighbour's vector, both at 1/8 luma sample precision, which is the
   * form the fit consumes. The scan walks the top edge and then the left edge,
   * one neighbour block at a time rather than one 4x4 at a time: a neighbour
   * wider (or taller) than this block contributes a single sample, a narrower
   * one contributes one per block along the edge.
   *
   * The vector distance gate is what makes the fit robust: a neighbour whose
   * vector disagrees with this block's by more than the block's size allows is
   * not part of the same motion, so it is left out of the fit. It does not end
   * the scan - the specification only stops looking once
   * LEAST_SQUARES_SAMPLES_MAX candidates have been seen. A block with no usable
   * neighbour at all still uses the first candidate scanned: the fit then has a
   * single point and produces a translation.
   *
   * Returns NumSamples. */
  int find_warp_samples(int row, int col, BlockSize size, int ref, int mv_x, int mv_y, bool avail_u,
                        bool avail_l, int cand[kWarpSamplesMax][4]) {
    const int w4 = block_width4(size), h4 = block_height4(size);
    const int threshold = clip3_local(16, std::max(w4, h4) * 4, 112);
    int samples = 0;
    int scanned = 0;
    int first[4] = {0, 0, 0, 0};
    bool have_first = false;

    const auto add_sample = [&](int delta_row, int delta_col) {
      if (scanned >= kWarpSamplesMax) return;
      const int mv_row = row + delta_row;
      const int mv_col = col + delta_col;
      if (mv_row < 0 || mv_row >= mi_rows() || mv_col < 0 || mv_col >= mi_cols()) return;
      const MiRef& edge = mi_at(mv_row, mv_col);
      if (edge.intra || edge.ref[0] != ref || edge.ref[1] != kRefIntra) return;
      const BlockSize cand_size = block_size_at(mv_row, mv_col);
      const int cand_w4 = block_width4(cand_size), cand_h4 = block_height4(cand_size);
      /* the candidate's own top left: when the neighbour is larger than this
       * block, the cell the scan reached is not the start of it */
      const int cand_row = mv_row & ~(cand_h4 - 1);
      const int cand_col = mv_col & ~(cand_w4 - 1);
      const int mid_y = cand_row * 4 + cand_h4 * 2 - 1;
      const int mid_x = cand_col * 4 + cand_w4 * 2 - 1;
      const MiRef& cand_ref = mi_at(cand_row, cand_col);
      const int mv_diff_row = std::abs(static_cast<int>(cand_ref.mv[0][1]) - mv_y);
      const int mv_diff_col = std::abs(static_cast<int>(cand_ref.mv[0][0]) - mv_x);
      const bool valid = (mv_diff_row + mv_diff_col) <= threshold;
      const int sample[4] = {mid_y * 8, mid_x * 8, mid_y * 8 + cand_ref.mv[0][1],
                             mid_x * 8 + cand_ref.mv[0][0]};
      ++scanned;
      if (!have_first) {
        for (int j = 0; j < 4; ++j) first[j] = sample[j];
        have_first = true;
      }
      if (!valid) return;
      for (int j = 0; j < 4; ++j) cand[samples][j] = sample[j];
      ++samples;
    };

    /* doTopLeft/doTopRight: whether the top left and the top right corner of
     * this block is a whole neighbour block's corner as well. A neighbour that
     * straddles the corner is already sampled above or to the left, so its
     * vector must not be counted twice. */
    bool do_top_left = true;
    bool do_top_right = true;
    if (avail_u) {
      const int src_w = block_width4(block_size_at(row - 1, col));
      if (w4 <= src_w) {
        const int col_offset = -(col & (src_w - 1));
        if (col_offset < 0) do_top_left = false;
        if (col_offset + src_w > w4) do_top_right = false;
        add_sample(-1, 0);
      } else {
        const int limit = std::min(w4, mi_cols() - col);
        for (int i = 0; i < limit && scanned < kWarpSamplesMax;) {
          const int src_w2 = block_width4(block_size_at(row - 1, col + i));
          add_sample(-1, i);
          i += std::min(w4, src_w2);
        }
      }
    }
    if (avail_l) {
      const int src_h = block_height4(block_size_at(row, col - 1));
      if (h4 <= src_h) {
        const int row_offset = -(row & (src_h - 1));
        if (row_offset < 0) do_top_left = false;
        add_sample(0, -1);
      } else {
        const int limit = std::min(h4, mi_rows() - row);
        for (int i = 0; i < limit && scanned < kWarpSamplesMax;) {
          const int src_h2 = block_height4(block_size_at(row + i, col - 1));
          add_sample(i, -1);
          i += std::min(h4, src_h2);
        }
      }
    }
    if (do_top_left) add_sample(-1, -1);
    /* the neighbour past the top right corner is only reachable when the block
     * is small enough for one block to cover its right edge */
    if (do_top_right && std::max(w4, h4) <= 16) add_sample(-1, w4);
    if (samples == 0 && have_first) {
      for (int j = 0; j < 4; ++j) cand[0][j] = first[j];
      samples = 1;
    }
    return samples;
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
   * continues with the chroma mode) or with an inter block carrying one or two
   * references, their motion vectors, the blend that combines them and the
   * interpolation filter.
   *
   * A skip mode block is the special case the frame header sets up: it is always
   * inter, always compound, takes both references from the header, uses the
   * nearest candidate pair and codes nothing but its presence. */
  ParseStatus read_inter_block_mode(TileBlock* block, BlockSize size, int row, int col,
                                    bool has_chroma) {
    const int bw4 = block_width4(size), bh4 = block_height4(size);
    const bool top_avail = have_top(row), left_avail = have_left(col);
    const bool compound_allowed = std::min(bw4, bh4) > 1;
    const int mv_prec = (frame_.allow_high_precision_mv ? 1 : 0) - (frame_.force_integer_mv ? 1 : 0);

    /* is_inter: a skip mode block is inter by definition and codes no flag */
    if (block->skip_mode) {
      block->is_inter = true;
    } else {
      block->is_inter = decoder_.read_symbol(cdf_.inter.is_inter[intra_context(row, col)], 1) != 0;
    }
    if (!block->is_inter) {
      const ParseStatus status = read_intra_block_mode(block, size, row, col, false);
      if (status != ParseStatus::Ok) return status;
      return read_chroma_mode(block, size, has_chroma);
    }

    /* read_ref_frames: one reference or a pair, and which pair */
    int ref0 = kRefIntra, ref1 = kRefIntra;
    bool compound = false;
    if (block->skip_mode) {
      compound = true;
      ref0 = frame_.skip_mode_frame[0];
      ref1 = frame_.skip_mode_frame[1];
    } else {
      if (frame_.reference_select && compound_allowed) {
        const int ctx = comp_context(row, col);
        compound = decoder_.read_symbol(cdf_.inter.comp_mode[ctx], 1) != 0;
      }
      if (compound) {
        /* the pair is coded as a direction ("bidirectional" or not) and then one
         * tree per reference over the four forward and the three backward
         * references */
        const int dir_ctx = comp_dir_context(row, col);
        if (decoder_.read_symbol(cdf_.inter.comp_ref_type[dir_ctx], 1)) {
          /* bidirectional: a forward reference, then a backward one */
          if (decoder_.read_symbol(
                  cdf_.inter.comp_ref[ref_context(row, col, kRefKindForward)][0], 1)) {
            ref0 = 2 + (decoder_.read_symbol(
                            cdf_.inter.comp_ref[ref_context(row, col, kRefKindForward2)][2], 1)
                            ? 1
                            : 0);
          } else {
            ref0 = decoder_.read_symbol(
                       cdf_.inter.comp_ref[ref_context(row, col, kRefKindForward1)][1], 1)
                       ? 1
                       : 0;
          }
          ref0 += 1;
          if (decoder_.read_symbol(
                  cdf_.inter.comp_bwd_ref[ref_context(row, col, kRefKindBackward)][0], 1)) {
            ref1 = kRefAltref;
          } else {
            ref1 = kRefBwdref + (decoder_.read_symbol(
                                     cdf_.inter.comp_bwd_ref[ref_context(row, col,
                                                                         kRefKindBackward1)][1],
                                     1)
                                     ? 1
                                     : 0);
          }
        } else if (decoder_.read_symbol(
                       cdf_.inter.uni_comp_ref[ref_context(row, col, kRefKindAll)][0], 1)) {
          /* unidirectional: BWDREF and ALTREF, or LAST with one of the rest */
          ref0 = kRefBwdref;
          ref1 = kRefAltref;
        } else {
          ref0 = kRefLast;
          /* uni_comp_ref_p1 asks a different question than comp_ref_p1: LAST2
           * against LAST3 and GOLDEN together, not LAST against LAST2. */
          int index = decoder_.read_symbol(
              cdf_.inter.uni_comp_ref[ref_context(row, col, kRefKindUniP1)][1], 1);
          if (index) {
            index += decoder_.read_symbol(
                cdf_.inter.uni_comp_ref[ref_context(row, col, kRefKindForward2)][2], 1);
          }
          ref1 = kRefLast2 + index;
        }
      } else {
        /* read_ref_frames: the reference of a single reference block, as a
         * binary tree over the seven reference types. The table's first index is
         * the context and the second the step of the tree, which is not the
         * order the steps are coded in: the three second-level tests live in
         * different rows. */
        const auto ref_bit = [&](int kind, int step) {
          return decoder_.read_symbol(cdf_.inter.single_ref[ref_context(row, col, kind)][step], 1);
        };
        if (ref_bit(kRefKindAll, 0)) {
          /* the backward half: ALTREF, or BWDREF against ALTREF2 */
          if (ref_bit(kRefKindBackward, 1)) {
            ref0 = kRefAltref;
          } else {
            ref0 = ref_bit(kRefKindBackward1, 5) ? kRefAltref2 : kRefBwdref;
          }
        } else {
          /* the forward half: {LAST3, GOLDEN} or {LAST, LAST2} */
          if (ref_bit(kRefKindForward, 2)) {
            ref0 = ref_bit(kRefKindForward2, 4) ? kRefGolden : kRefLast3;
          } else {
            ref0 = ref_bit(kRefKindForward1, 3) ? kRefLast2 : kRefLast;
          }
        }
        ref1 = kRefIntra;
      }
    }
    block->ref_frame[0] = ref0;
    block->ref_frame[1] = ref1;

    /* the candidate list, then the mode that picks from it */
    MvCand stack[8];
    int count = 0, ctx = 0;
    find_ref_mvs(stack, &count, &ctx, ref0, ref1, size, row, col);
    trace_stack(col, row, count, stack);
    if (trace_) {
      std::fprintf(stderr, "V %d %d n=%d ctx=%d s0=%d,%d s1=%d,%d w0=%d w1=%d\n", col, row,
                   count, ctx, stack[0].mv[0][0], stack[0].mv[0][1], stack[1].mv[0][0],
                   stack[1].mv[0][1], stack[0].weight, stack[1].weight);
    }

    const GlobalMotion& gm0 = global_motion_of(ref0);
    int mv0x = 0, mv0y = 0, mv1x = 0, mv1y = 0;
    bool has_subpel_filter = true;
    block->drl_index = 0;

    if (block->skip_mode) {
      /* no mode, no residual: the nearest pair, at the frame's precision */
      block->inter_mode = kModeNearestNearestMv;
      mv0x = stack[0].mv[0][0];
      mv0y = stack[0].mv[0][1];
      mv1x = stack[0].mv[1][0];
      mv1y = stack[0].mv[1][1];
      lower_mv_precision(&mv0x, &mv0y);
      lower_mv_precision(&mv1x, &mv1y);
      has_subpel_filter = false;
    } else if (compound) {
      const int symbol = static_cast<int>(decoder_.read_symbol(cdf_.inter.compound_mode[ctx], 7));
      if (trace_) {
        std::fprintf(stderr, "R comp=%d ctx=%d n=%d\n", symbol, ctx, count);
      }
      block->inter_mode = kModeNearestNearestMv + symbol;
      const CompoundModePair& pair = kCompoundModePairs[symbol];
      /* the dynamic reference list index: a new vector starts at the nearest
       * candidate, a near one at the second */
      if (block->inter_mode == kModeNewNewMv) {
        if (count > 1) {
          block->drl_index +=
              read_drl_bit(col, row, 0, count, stack);
          if (block->drl_index == 1 && count > 2) {
            block->drl_index +=
                read_drl_bit(col, row, 1, count, stack);
          }
        }
      } else if (pair.mode[0] == kModeNearMv || pair.mode[1] == kModeNearMv) {
        block->drl_index = 1;
        if (count > 2) {
          block->drl_index +=
              read_drl_bit(col, row, 1, count, stack);
          if (block->drl_index == 2 && count > 3) {
            block->drl_index +=
                read_drl_bit(col, row, 2, count, stack);
          }
        }
      }
      for (int n = 0; n < 2; ++n) {
        int mx = 0, my = 0;
        switch (pair.mode[n]) {
          case kModeNearMv:
          case kModeNearestMv:
            mx = stack[block->drl_index].mv[n][0];
            my = stack[block->drl_index].mv[n][1];
            lower_mv_precision(&mx, &my);
            break;
          case kModeGlobalMv:
            global_mv(global_motion_of(n ? ref1 : ref0), row, col, bw4, bh4, &mx, &my);
            break;
          default: /* kModeNewMv */
            mx = stack[block->drl_index].mv[n][0];
            my = stack[block->drl_index].mv[n][1];
            read_mv_residual(&mx, &my, mv_prec);
            break;
        }
        if (n == 0) {
          mv0x = mx;
          mv0y = my;
        } else {
          mv1x = mx;
          mv1y = my;
        }
      }
      has_subpel_filter = std::min(bw4, bh4) == 1 || block->inter_mode != kModeGlobalGlobalMv;
    } else {
      if (decoder_.read_symbol(cdf_.inter.new_mv[ctx & 7], 1)) {
        /* not a new vector: global motion, nearest or near */
        if (!decoder_.read_symbol(cdf_.inter.zero_mv[(ctx >> 3) & 1], 1)) {
          block->inter_mode = kModeGlobalMv;
          global_mv(gm0, row, col, bw4, bh4, &mv0x, &mv0y);
          has_subpel_filter = std::min(bw4, bh4) == 1 || gm0.translation_only();
        } else {
          if (decoder_.read_symbol(cdf_.inter.ref_mv[(ctx >> 4) & 15], 1)) {
            block->inter_mode = kModeNearMv;
            block->drl_index = 1;
            if (count > 2) {
              block->drl_index +=
                  read_drl_bit(col, row, 1, count, stack);
              if (block->drl_index == 2 && count > 3) {
                block->drl_index +=
                    read_drl_bit(col, row, 2, count, stack);
              }
            }
          } else {
            block->inter_mode = kModeNearestMv;
            block->drl_index = 0;
          }
          mv0x = stack[block->drl_index].mv[0][0];
          mv0y = stack[block->drl_index].mv[0][1];
          if (block->drl_index < 2) lower_mv_precision(&mv0x, &mv0y);
        }
      } else {
        block->inter_mode = kModeNewMv;
        block->drl_index = 0;
        if (count > 1) {
          block->drl_index += read_drl_bit(col, row, 0, count, stack);
          if (block->drl_index == 1 && count > 2) {
            block->drl_index +=
                read_drl_bit(col, row, 1, count, stack);
          }
        }
        if (count > 1) {
          mv0x = stack[block->drl_index].mv[0][0];
          mv0y = stack[block->drl_index].mv[0][1];
        } else {
          mv0x = stack[0].mv[0][0];
          mv0y = stack[0].mv[0][1];
          lower_mv_precision(&mv0x, &mv0y);
        }
        read_mv_residual(&mv0x, &mv0y, mv_prec);
      }
    }

    /* read_interintra_mode: a single reference block whose prediction is blended
     * with an intra one. Only the sizes from 8x8 to 32x32 can, and only when the
     * block is neither a skip mode block nor already a compound one - the
     * reference frame pair of a compound block leaves no room for the intra
     * one.
     *
     * The intra half is not reconstructed yet, but its syntax is short and
     * reading it is what keeps the range decoder in step: the inter half is
     * predicted as usual and the intra half is simply missing from the frame. */
    if (!block->skip_mode && seq_.enable_interintra_compound && !compound && size >= kBlock8x8 &&
        size <= kBlock32x32) {
      const int group = ymode_size_group(size) - 1;
      if (decoder_.read_symbol(cdf_.inter.inter_intra[group], 1) != 0) {
        block->interintra = true;
        block->interintra_mode =
            static_cast<int>(decoder_.read_symbol(cdf_.inter.inter_intra_mode[group], 3));
        block->interintra_wedge =
            decoder_.read_symbol(cdf_.inter.wedge_inter_intra[size], 1) != 0;
        if (block->interintra_wedge) {
          block->wedge_index =
              static_cast<int>(decoder_.read_symbol(cdf_.inter.wedge_index[size], 15));
        }
        /* the pair is the reference and the intra frame from now on, which is
         * what the syntax set RefFrame[1] to and what the block carries */
        ref1 = kRefIntra;
        block->ref_frame[1] = kRefIntra;
      }
    }

    /* motion mode: simple translation, overlapped block motion compensation, or
     * local warped motion. Both of the latter need prediction work this decoder
     * does not do yet, but the symbol still decides the stream. */
    block->motion_mode = kMotionTranslation;
    if (!block->skip_mode && !compound && !block->interintra &&
        frame_.is_motion_mode_switchable && std::min(bw4, bh4) >= 2 &&
        !(!frame_.force_integer_mv && block->inter_mode == kModeGlobalMv &&
          gm0.type > kGmTranslation) &&
        ((left_avail && find_odd_inter(false, row, col, bw4, bh4)) ||
         (top_avail && find_odd_inter(true, row, col, bw4, bh4)))) {
      uint64_t masks[2] = {0, 0};
      find_matching_ref(row, col, bw4, bh4, std::min(bw4, 16), std::min(bh4, 16), left_avail,
                        top_avail, ref0, masks);
      const bool allow_warp = !frame_.force_integer_mv && frame_.allow_warped_motion &&
                              (masks[0] | masks[1]);
      if (trace_enabled()) {
        const RangeDecoder::State mm_state = decoder_.state();
        std::fprintf(stderr, "MM %d %d bs=%d warp=%d obmc_cdf=%u,%u dif=%llu rng=%u cnt=%d\n", col,
                     row, static_cast<int>(size), allow_warp ? 1 : 0,
                     cdf_.inter.use_obmc[size][0], cdf_.inter.use_obmc[size][1],
                     static_cast<unsigned long long>(mm_state.window), mm_state.range,
                     mm_state.count);
      }
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
      if (block->motion_mode == kMotionWarped) {
        ++out_->warped_blocks;
        /* Local warped motion (7.11.3.8): the model is fitted to the neighbours
         * of this block, so it is derived here, while the motion vector grid is
         * still the one the block saw. A block with no usable neighbour leaves
         * the fit unsolvable, which is the specification's LocalValid = 0: the
         * block is then predicted by its own vector. */
        const int mid_x = col * 4 + bw4 * 2 - 1;
        const int mid_y = row * 4 + bh4 * 2 - 1;
        int cand[kWarpSamplesMax][4] = {};
        const int samples = find_warp_samples(row, col, size, ref0, mv0x, mv0y, top_avail,
                                              left_avail, cand);
        block->local_warp_valid = samples > 0 && warp_estimation(cand, samples, mid_x, mid_y, mv0x,
                                                                 mv0y, block->local_warp);
        if (trace_enabled()) {
          std::fprintf(stderr, "LW %d %d n=%d valid=%d p=%d,%d,%d,%d,%d,%d\n", col, row, samples,
                       block->local_warp_valid ? 1 : 0, block->local_warp[0], block->local_warp[1],
                       block->local_warp[2], block->local_warp[3], block->local_warp[4],
                       block->local_warp[5]);
        }
      }
    }

    /* read_compound_type: nothing to code for a single reference block, and a
     * skip mode block always averages */
    if (block->skip_mode) {
      block->compound_type = kCompoundAverage;
    } else if (!compound) {
      block->compound_type = kCompoundNone;
    } else {
      bool masked = false;
      if (seq_.enable_masked_compound) {
        const int mask_ctx = mask_comp_context(row, col);
        masked = decoder_.read_symbol(cdf_.inter.comp_group_idx[mask_ctx], 1) != 0;
        if (trace_) {
          const RangeDecoder::State st = decoder_.state();
          std::fprintf(stderr, "MK mask %d %d %llu %u %d\n", mask_ctx, masked ? 1 : 0,
                       static_cast<unsigned long long>(st.window), st.range, st.count);
        }
      }
      if (!masked) {
        if (seq_.enable_jnt_comp) {
          const int jnt_ctx = jnt_comp_context(row, col, ref0, ref1);
          block->compound_type = kCompoundWeightedAverage +
                                 (decoder_.read_symbol(cdf_.inter.compound_idx[jnt_ctx], 1) ? 1
                                                                                            : 0);
          if (trace_) {
            const RangeDecoder::State st = decoder_.state();
            std::fprintf(stderr, "MK jnt %d %d %llu %u %d\n", jnt_ctx,
                         block->compound_type == kCompoundAverage ? 1 : 0,
                         static_cast<unsigned long long>(st.window), st.range, st.count);
          }
        } else {
          block->compound_type = kCompoundAverage;
        }
      } else {
        if (wedge_bits(size) > 0) {
          /* masked, with a mask shape to choose: wedge or difference weighted */
          block->compound_type =
              kCompoundWedge -
              (decoder_.read_symbol(cdf_.inter.compound_type[size], 1) ? 1 : 0);
          if (block->compound_type == kCompoundWedge) {
            block->wedge_index = static_cast<int>(
                decoder_.read_symbol(cdf_.inter.wedge_index[size], 15));
          }
        } else {
          block->compound_type = kCompoundDiffwtd;
        }
        /* the mask sign for a wedge, the mask type for a difference weighted
         * mask: one literal bit either way */
        block->mask_sign = static_cast<int>(decoder_.read_bit());
      }
    }

    /* the subpel filter, if this frame switches them */
    int filter[2] = {frame_.interpolation_filter, frame_.interpolation_filter};
    if (frame_.is_filter_switchable) {
      if (has_subpel_filter) {
        const bool comp = block->compound_type != kCompoundNone;
        const int fctx = filter_context(row, col, comp, 0, ref0);
        filter[0] = static_cast<int>(decoder_.read_symbol(cdf_.inter.interp_filter[fctx], 2));
        filter[1] = filter[0];
        if (trace_) {
          const RangeDecoder::State st = decoder_.state();
          std::fprintf(stderr, "MK f0 %d %d %llu %u %d\n", fctx, filter[0],
                       static_cast<unsigned long long>(st.window), st.range, st.count);
        }
        if (seq_.enable_dual_filter) {
          const int fctx2 = filter_context(row, col, comp, 1, ref0);
          filter[1] =
              static_cast<int>(decoder_.read_symbol(cdf_.inter.interp_filter[8 + fctx2], 2));
          if (trace_) {
            const RangeDecoder::State st = decoder_.state();
            std::fprintf(stderr, "MK f1 %d %d %llu %u %d\n", fctx2, filter[1],
                         static_cast<unsigned long long>(st.window), st.range, st.count);
          }
        }
      } else {
        filter[0] = filter[1] = 0;
      }
    }
    block->interp_filter[0] = filter[0];
    block->interp_filter[1] = filter[1];
    block->mv[0][0] = mv0x;
    block->mv[0][1] = mv0y;
    block->mv[1][0] = mv1x;
    block->mv[1][1] = mv1y;
    return ParseStatus::Ok;
  }

  /* Remember the block in the motion reference map the next block's candidate
   * scan reads. */
  void mark_inter(int row, int col, BlockSize size, const TileBlock& block, int filter0,
                  int filter1) {
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
    /* The two bits a candidate scan reads off a neighbour: whether its vector
     * should be replaced by the global motion model's, and whether it was coded
     * as a new one. A compound block answers both for the pair as a whole, and
     * its global mode counts even for small blocks (the pair syntax leaves no
     * other way to code two global vectors). */
    const int mode = block.inter_mode;
    const bool inter = block.is_inter;
    const bool compound_mode = inter && mode >= kModeNearestNearestMv;
    entry.global_subpel = inter && (compound_mode ? mode == kModeGlobalGlobalMv
                                                  : (mode == kModeGlobalMv &&
                                                     std::min(bw4, bh4) >= 2));
    entry.coded_newmv = inter && (mode == kModeNewMv || mode == kModeNearestNewMv ||
                                  mode == kModeNewNearestMv || mode == kModeNearNewMv ||
                                  mode == kModeNewNearMv || mode == kModeNewNewMv);
    entry.filter[0] = static_cast<uint8_t>(filter0);
    entry.filter[1] = static_cast<uint8_t>(filter1);
    entry.comp_type = static_cast<uint8_t>(block.is_inter ? block.compound_type : kCompoundNone);
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
      if (trace_) {
        const RangeDecoder::State state = decoder_.state();
        std::fprintf(stderr, "XT %d %d tx=%dx%d depth=%d cat=%d al=%d split=%d dif=%llu rng=%u cnt=%d\n",
                     col, row, tx.w4(), tx.h4(), depth, cat, above + left, split ? 1 : 0,
                     static_cast<unsigned long long>(state.window), state.range, state.count);
      }
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
      uint16_t* const row_cdf = cdf_.partition.row(level, ctx);
      const uint16_t cdf_before[4] = {row_cdf[0], row_cdf[1], row_cdf[2], row_cdf[3]};
      const uint16_t cdf_counter_before = row_cdf[nsymbols];
      partition = static_cast<PartitionType>(decoder_.read_symbol(row_cdf, nsymbols));
      if (trace_enabled()) {
        const RangeDecoder::State state = decoder_.state();
        std::fprintf(stderr,
                     "P %d %d bl=%d ctx=%d bp=%d %llu %u %d cdf=%u,%u,%u,%u n=%u\n", col, row,
                     level, ctx, static_cast<int>(partition),
                     static_cast<unsigned long long>(state.window), state.range, state.count,
                     cdf_before[0], cdf_before[1], cdf_before[2], cdf_before[3],
                     cdf_counter_before);
        if (cdf_trace_enabled()) {
          std::fprintf(stderr, "PA");
          for (int z = 0; z <= nsymbols; ++z) std::fprintf(stderr, ",%d", row_cdf[z]);
          std::fprintf(stderr, "\n");
        }
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
    decoder_.reset_symbol_hash();
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
     * flag before the skip flag; a block that uses it is always skipped, always
     * inter and always compound, so it codes neither of the two flags that
     * would say so. */
    const bool intra_frame =
        frame_.type == FrameType::Key || frame_.type == FrameType::IntraOnly;
    block.skip_mode = false;
    if (!intra_frame && frame_.skip_mode_present && std::min(bw4, bh4) > 1) {
      const int smctx = (have_top(row) ? mi_skip_mode(row - 1, col) : 0) +
                        (have_left(col) ? mi_skip_mode(row, col - 1) : 0);
      block.skip_mode = decoder_.read_symbol(cdf_.inter.skip_mode[smctx], 1) != 0;
    }
    if (block.skip_mode) {
      block.skip = true;
    } else {
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
    if (trace_enabled()) std::fprintf(stderr, "BF %d %d\n", col, row);

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
        /* An unavailable neighbour contributes zero, and the comparison is in
         * pixels: a 4 wide transform must still lose against "no neighbour".
         * The side is kept in 4x4 units, where a 4 pixel side is 0, so absence
         * has to be -1 rather than 0 to compare the same way. */
        const int above_lw = row > tile_row_start_ ? above_tx_lw_[col] : -1;
        const int left_lh = col > tile_col_start_ ? left_tx_lh_[row] : -1;
        const int depth_ctx = (above_lw >= tx.lw ? 1 : 0) + (left_lh >= tx.lh ? 1 : 0);
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
    if (!intra_frame) mark_inter(row, col, size, block, block.interp_filter[0],
                                 block.interp_filter[1]);
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
    if (trace_enabled()) {
      trace_block(block);
      std::fprintf(stderr, "H %d %d %d %d %u\n", col, row, block_width4(size),
                   block_height4(size), decoder_.symbol_hash());
    }
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
  int inter_chroma_tx_type(const TileBlock& block, TxSize tx, int x4, int y4) const {
    /* The luma position is the chroma one scaled back up, but a chroma sample
     * can sit half a luma sample outside its block (the block origin is not
     * always even), and then the cell belongs to the neighbour. The
     * specification clamps to the block origin for exactly that reason. */
    const int lx = std::max(block.mi_col, x4 << sub_x());
    const int ly = std::max(block.mi_row, y4 << sub_y());
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
      plane_tx_type = block->is_inter ? inter_chroma_tx_type(*block, tx, x4, y4)
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
  /* the tables the frame started from, see run() */
  CdfContext start_cdf_;

  int tile_col_start_ = 0, tile_col_end_ = 0, tile_row_start_ = 0, tile_row_end_ = 0;
  int current_q_ = 0;
  /* DeltaLF[0..3]: a running per tile value, updated by the delta_lf syntax at
   * a superblock start and copied into every block of that superblock. */
  int delta_lf_[4] = {0, 0, 0, 0};
  bool read_deltas_ = false;

  /* Projected motion vectors: this frame's 8x8 grid, the distances the
   * projection scales by, and the frames that contributed to it. */
  std::vector<TemporalMv> proj_;
  int proj_cols_ = 0;
  int pocdiff_[7] = {0, 0, 0, 0, 0, 0, 0};
  bool mfmv_sign_[7] = {false, false, false, false, false, false, false};
  /* sign_bias: whether a reference frame is *after* this one, which inverts the
   * sign of the vectors neighbours stored that point at it */
  bool sign_bias_[7] = {false, false, false, false, false, false, false};
  int mfmv_ref_[3] = {0, 0, 0};
  int mfmv_ref2cur_[3] = {0, 0, 0};
  uint8_t mfmv_ref2ref_[3][7] = {};
  const RefMvs* mfmv_stored_[3] = {nullptr, nullptr, nullptr};
  int mfmv_count_ = 0;

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
    setup_temporal_mvs();
    /* The frame's starting probabilities: its primary reference's tables when it
     * names one, the specification's defaults otherwise. A frame that resets
     * (error resilient, or a key frame) always falls back to the defaults. */
    const CdfContext* inherited = nullptr;
    if (frame_.primary_ref_frame != kPrimaryRefNone && refs_) {
      const int slot_index = frame_.ref_frame_idx[frame_.primary_ref_frame];
      if (slot_index >= 0 && slot_index < 8) {
        inherited = refs_->slots[slot_index].cdf.get();
      }
    }
    if (inherited) {
      cdf_ = *inherited;
    } else {
      init_default_cdf(&cdf_, frame_.base_q_idx);
    }
    /* the tables the frame started from: a frame that may not update the
     * context at its end hands these on instead of the adapted ones */
    start_cdf_ = cdf_;
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
  }
  if (frame.tiles.cols != 1 || frame.tiles.rows != 1) {
    if (error) *error = "tile: multi-tile frames are not decoded so far";
    return ParseStatus::Unsupported;
  }

  ++g_frame_ordinal;
  if (trace_enabled()) {
    std::fprintf(stderr, "FR %d oh=%d type=%d show=%d\n", g_frame_ordinal, frame.order_hint,
                 static_cast<int>(frame.type), frame.show_frame ? 1 : 0);
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
