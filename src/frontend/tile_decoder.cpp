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

/* Filter_Intra_Mode_To_Intra_Dir: a filter intra mode stands in for a luma
 * direction when the transform type alphabet is selected. */
constexpr int8_t kFilterIntraToIntraDir[5] = {kModeDc, kModeVertical, kModeHorizontal,
                                              kModeHorizontalUp, kModeDc};

class TileParser {
 public:
  TileParser(const SequenceHeader& seq, const FrameHeader& frame, const uint8_t* data,
             size_t size, TileDecodeResult* out, std::string* error)
      : seq_(seq), frame_(frame), data_(data), size_(size), out_(out), error_(error) {}

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

  ParseStatus stop_at_residual(const TileBlock& block, const char* reason) {
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

  /* The transform the block uses, which the next block's tx_depth context
   * reads: one value per 4x4 column above and one per row to the left. */
  void mark_tx(int row, int col, TxSize tx) {
    for (int x = 0; x < tx.w4() && col + x < mi_cols(); ++x) above_tx_lw_[col + x] = tx.lw;
    for (int y = 0; y < tx.h4() && row + y < mi_rows(); ++y) left_tx_lh_[row + y] = tx.lh;
  }

  /* --- syntax ------------------------------------------------------------ */

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
      if (static_cast<int>(partition) >= nsymbols) return corrupt("partition");
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

    /* read_skip */
    const int sctx = skip_context(row, col);
    block.skip = decoder_.read_symbol(cdf_.skip.row(sctx), 1) != 0;

    /* read_cdef: one index per 64x64 (or per 32x32 quadrant with 128x128
     * superblocks) when the frame codes them at all */
    if (!block.skip && frame_.cdef.bits > 0) {
      const int cdef_unit = 16; /* 64x64 */
      if ((row % cdef_unit) == 0 && (col % cdef_unit) == 0) {
        decoder_.read_literal(frame_.cdef.bits);
      }
    }

    /* read_delta_qindex / read_delta_lf: at the start of a superblock */
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
            if (lf_abs != 0) decoder_.read_bit(); /* the sign is applied by the filters */
          }
        }
      }
    }

    /* the frame's deltas are coded once per superblock, by its first block */
    read_deltas_ = false;

    /* intra_frame_mode_info: the frame is intra, so the y mode is coded against
     * the neighbour modes and the block is always intra predicted */
    const int above_mode =
        have_top(row) ? y_mode_map_[index(row - 1, col)] : static_cast<int>(kModeDc);
    const int left_mode =
        have_left(col) ? y_mode_map_[index(row, col - 1)] : static_cast<int>(kModeDc);
    block.y_mode = static_cast<int>(decoder_.read_symbol(
        cdf_.mode.kfym[kIntraModeContext[above_mode]][kIntraModeContext[left_mode]],
        kModeLumaCount - 1));
    if (block.y_mode >= kModeLumaCount) return corrupt("intra_frame_y_mode");

    /* intra_angle_info_y */
    const int log2_w = kBlockWidthLog2[size], log2_h = kBlockHeightLog2[size];
    if (log2_w + log2_h >= 2 && block.y_mode >= kModeVertical &&
        block.y_mode <= kModeVerticalLeft) {
      const int value = static_cast<int>(
          decoder_.read_symbol(cdf_.mode.angle_delta[block.y_mode - kModeVertical], 6));
      block.y_angle = value - 3;
    }

    if (has_chroma) {
      /* chroma from luma is allowed for blocks small enough to have a
       * 4x4 chroma transform */
      const bool cfl_allowed = !frame_.coded_lossless
                                   ? std::max(block_width4(size), block_height4(size)) <= 8
                                   : ((bw4 + ss_x) >> ss_x) == 1 && ((bh4 + ss_y) >> ss_y) == 1;
      const int nsymbols = kModeChromaCount - 1 - (cfl_allowed ? 0 : 1);
      block.uv_mode = static_cast<int>(
          decoder_.read_symbol(cdf_.mode.uv_mode[cfl_allowed ? 1 : 0][block.y_mode], nsymbols));
      if (block.uv_mode >= nsymbols) return corrupt("uv_mode");
      if (block.uv_mode == kModeChromaFromLuma) {
        const int sign = static_cast<int>(decoder_.read_symbol(cdf_.mode.cfl_sign, 7)) + 1;
        const int sign_u = sign / 3, sign_v = sign % 3;
        if (sign_u) {
          const int ctx = (sign_u == 2) * 3 + sign_v;
          block.cfl_alpha[0] =
              static_cast<int>(decoder_.read_symbol(cdf_.mode.cfl_alpha[ctx], 15)) + 1;
          if (sign_u == 1) block.cfl_alpha[0] = -block.cfl_alpha[0];
        }
        if (sign_v) {
          const int ctx = (sign_v == 2) * 3 + sign_u;
          block.cfl_alpha[1] =
              static_cast<int>(decoder_.read_symbol(cdf_.mode.cfl_alpha[ctx], 15)) + 1;
          if (sign_v == 1) block.cfl_alpha[1] = -block.cfl_alpha[1];
        }
      } else if (log2_w + log2_h >= 2 && block.uv_mode >= kModeVertical &&
                 block.uv_mode <= kModeVerticalLeft) {
        const int value = static_cast<int>(
            decoder_.read_symbol(cdf_.mode.angle_delta[block.uv_mode - kModeVertical], 6));
        block.uv_angle = value - 3;
      }
    }

    /* palette_mode_info: only for frames that allow screen content tools */
    if (frame_.allow_screen_content_tools && size >= kBlock8x8 &&
        block_width4(size) * 4 <= 64 && block_height4(size) * 4 <= 64) {
      return stop_at_residual(block, "tile: palette mode syntax is not implemented");
    }

    /* filter_intra_mode_info */
    if (block.y_mode == kModeDc && block.palette_size[0] == 0 &&
        std::max(log2_w, log2_h) <= 3 && seq_.enable_filter_intra) {
      block.filter_intra = decoder_.read_symbol(cdf_.mode.use_filter_intra[size], 1) != 0;
      if (block.filter_intra) {
        block.filter_intra_mode = static_cast<int>(decoder_.read_symbol(cdf_.mode.filter_intra, 4));
      }
    }

    /* read_tx_size: an intra block always codes its transform size when the
     * frame allows a choice */
    TxSize tx = max_tx_size(bw4, bh4);
    if (!frame_.coded_lossless && size > kBlock4x4 && frame_.tx_mode_select) {
      const int cls = tx_square_class(tx);
      const int depth_ctx = (above_tx_lw_[col] >= tx.lw ? 1 : 0) + (left_tx_lh_[row] >= tx.lh ? 1 : 0);
      const int depth = static_cast<int>(
          decoder_.read_symbol(cdf_.mode.txsz[cls - 1][depth_ctx], std::min(cls, 2)));
      for (int i = 0; i < depth; ++i) tx = split_tx_size(tx);
    }
    block.tx_size = tx;
    block.qindex = current_q_;

    mark_block(row, col, bw4, bh4, size, block.skip, block.y_mode);
    mark_tx(row, col, tx);

    if (!block.skip) {
      const ParseStatus status = read_residuals(&block);
      if (status != ParseStatus::Ok) {
        out_->blocks.push_back(block);
        ++out_->blocks_needing_residuals;
        return status;
      }
    }
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

  /* compute_tx_type for a chroma plane of an intra frame: derived from the
   * chroma mode, and only if the size's transform set admits it. */
  int chroma_tx_type(const TileBlock& block, TxSize tx) const {
    const int implied = tx_type_from_uv_mode(block.uv_mode);
    const int set = tx_set(tx, /*is_inter=*/false, frame_.reduced_tx_set);
    return is_tx_type_in_set(set, implied, /*is_inter=*/false) ? implied : coef_tables::kDctDct;
  }

  /* Residual syntax of one block: for each 64x64 chunk of it, all luma
   * transform blocks then both chroma planes, which is the order the encoder
   * writes them in. */
  ParseStatus read_residuals(TileBlock* block) {
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
    if (std::getenv("AV1GPU_TRACE")) {
      std::fprintf(stderr, "all_zero ctx=%d/%d prob=%u rng=%u q=%d\n", tx_ctx, zero_ctx,
                   cdf_.coef.skip[tx_ctx][zero_ctx][0], decoder_.state().range, block->qindex);
    }
    if (decoder_.read_symbol(cdf_.coef.skip[tx_ctx][zero_ctx], 1)) {
      /* empty: the transform kernels still run, they just have nothing to add */
      if (plane == 0) set_tx_type(x4, y4, tx, coef_tables::kDctDct);
      set_coef_context(plane, x4, y4, tx, 0, 0);
      if (std::getenv("AV1GPU_TRACE")) {
        std::fprintf(stderr, "cf plane=%d x4=%d y4=%d tx=%dx%d zero rng=%u\n", plane, x4, y4,
                     tx.width(), tx.height(), decoder_.state().range);
      }
      block->transforms.push_back(transform);
      ++out_->transform_blocks;
      return ParseStatus::Ok;
    }

    /* the transform type */
    int plane_tx_type = coef_tables::kDctDct;
    if (lossless) {
      if (plane == 0) set_tx_type(x4, y4, tx, coef_tables::kDctDct);
    } else if (plane == 0) {
      plane_tx_type =
          read_tx_type(block, x4, y4, tx, tx_set(tx, false, frame_.reduced_tx_set));
    } else {
      plane_tx_type = chroma_tx_type(*block, tx);
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
      if (pos < 0) {
        std::fprintf(stderr, "scan fail plane=%d tx=%dx%d c=%d eob=%d count=%d area=%d\n",
                     plane, tx.width(), tx.height(), c, eob, scan.count, grid.area());
        return corrupt("scan");
      }
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
    if (std::getenv("AV1GPU_TRACE")) {
      std::fprintf(stderr, "cf plane=%d x4=%d y4=%d tx=%dx%d txtp=%d eob=%d rng=%u\n", plane, x4,
                   y4, tx.width(), tx.height(), plane_tx_type, eob, decoder_.state().range);
    }
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
  const uint8_t* data_;
  size_t size_;
  TileDecodeResult* out_;
  std::string* error_;

  RangeDecoder decoder_;
  CdfContext cdf_;

  int tile_col_start_ = 0, tile_col_end_ = 0, tile_row_start_ = 0, tile_row_end_ = 0;
  int current_q_ = 0;
  bool read_deltas_ = false;

  std::vector<uint8_t> mi_size_;
  std::vector<uint8_t> skip_map_;
  std::vector<uint8_t> y_mode_map_;
  std::vector<int8_t> above_tx_lw_;
  std::vector<int8_t> left_tx_lh_;
  /* luma transform type per 4x4 unit, as transform_type() writes it */
  std::vector<uint8_t> tx_type_map_;
  /* coefficient level and DC statistics, per plane */
  CoefPlaneContext coef_[3];

 public:
  void init_state() {
    const int cols = mi_cols(), rows = mi_rows();
    mi_size_.assign(static_cast<size_t>(cols) * rows, kBlock4x4);
    skip_map_.assign(static_cast<size_t>(cols) * rows, 0);
    y_mode_map_.assign(static_cast<size_t>(cols) * rows, kModeDc);
    above_tx_lw_.assign(static_cast<size_t>(cols), 0);
    left_tx_lh_.assign(static_cast<size_t>(rows), 0);
    tx_type_map_.assign(static_cast<size_t>(cols) * rows, coef_tables::kDctDct);
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

  TileParser parser(seq, frame, data, size, out, error);
  parser.init_state();
  const ParseStatus status = parser.run();
  out->bytes_used = parser.bytes_used();
  out->overrun = parser.overrun();
  if (status == ParseStatus::Ok && parser.failed()) {
    return fail(error, "tile: the range coder lost synchronisation");
  }
  return status;
}

} /* namespace frontend */
} /* namespace av1gpu */
