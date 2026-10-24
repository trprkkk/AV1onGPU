#include "frontend/tile_decoder.hpp"

#include <algorithm>
#include <cstring>

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

  /* Stops at the first block whose residuals this layer does not read. Every
   * block is recorded first, so the caller can see exactly where the parser
   * stopped. */
  ParseStatus stop_at_residual(const TileBlock& block, const char* reason) {
    out_->blocks.push_back(block);
    ++out_->blocks_needing_residuals;
    out_->stop_reason = reason;
    return ParseStatus::Unsupported;
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
    block.skip = decoder_.read_bool(cdf_.skip.row(sctx)[0]) != 0;

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
      block.filter_intra = decoder_.read_bool(cdf_.mode.use_filter_intra[size][0]) != 0;
      if (block.filter_intra) {
        const int value = static_cast<int>(decoder_.read_symbol(cdf_.mode.filter_intra, 4));
        block.y_angle = value;
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
      return stop_at_residual(block, "tile: coefficient decoding is not implemented");
    }
    out_->blocks.push_back(block);
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

 public:
  void init_state() {
    const int cols = mi_cols(), rows = mi_rows();
    mi_size_.assign(static_cast<size_t>(cols) * rows, kBlock4x4);
    skip_map_.assign(static_cast<size_t>(cols) * rows, 0);
    y_mode_map_.assign(static_cast<size_t>(cols) * rows, kModeDc);
    above_tx_lw_.assign(static_cast<size_t>(cols), 0);
    left_tx_lh_.assign(static_cast<size_t>(rows), 0);
    init_default_cdf(&cdf_);
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
