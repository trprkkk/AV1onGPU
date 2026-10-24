/*
 * AV1onGPU - transform sizes
 *
 * The specification counts transform sizes in two different ways and both are
 * needed: as a rectangle (what the coefficient syntax and the transform
 * kernels use) and as a square class (what the CDF tables are indexed by). A
 * 16x32 transform, for instance, uses the 16x16 tables for its "min" and the
 * 32x32 tables for its "max", and the coefficient code addresses tables by
 * one or the other.
 *
 * Rectangles are stored as log2 of the size in 4x4 units, because that is what
 * every table in the syntax layer needs: the number of 4x4 blocks a transform
 * covers, its position on the 4x4 grid, and the "is this neighbour at least as
 * large as the maximum transform" comparisons all fall out of those two
 * numbers without another lookup table.
 */
#ifndef AV1GPU_FRONTEND_TX_SIZE_HPP
#define AV1GPU_FRONTEND_TX_SIZE_HPP

#include <cstdint>

#include "frontend/block_size.hpp"

namespace av1gpu {
namespace frontend {

/* A transform rectangle, in log2 of its width and height in 4x4 units: a 4x4
 * transform is {0, 0}, a 16x32 is {2, 3}. */
struct TxSize {
  int lw = 0;
  int lh = 0;

  int w4() const { return 1 << lw; }
  int h4() const { return 1 << lh; }
  int width() const { return w4() * 4; }
  int height() const { return h4() * 4; }
  bool operator==(const TxSize& other) const { return lw == other.lw && lh == other.lh; }
  bool operator!=(const TxSize& other) const { return !(*this == other); }
};

/* The square class: log2 of the larger side in 4x4 units, i.e. 0 for 4x4, 4
 * for 64x64. The coefficient CDF tables and the transform type tables index by
 * this, which is why a 16x32 and a 32x16 share entries. */
inline int tx_square_class(TxSize tx) { return tx.lw > tx.lh ? tx.lw : tx.lh; }

/* The square class of the *smaller* side, used to pick the transform type table
 * for rectangles. */
inline int tx_min_class(TxSize tx) { return tx.lw < tx.lh ? tx.lw : tx.lh; }

/* Splitting a transform once halves the longer side: 64x64 -> 32x64 and
 * 32x64 -> 32x32, which is what Split_Tx_Size does. */
inline TxSize split_tx_size(TxSize tx) {
  if (tx.lw > 0 && tx.lw >= tx.lh) return TxSize{tx.lw - 1, tx.lh};
  if (tx.lh > 0) return TxSize{tx.lw, tx.lh - 1};
  return tx;
}

/* Max_Tx_Size_Rect for a block of `w4` by `h4` 4x4 units: the block size with
 * each side capped at the 64x64 transform. */
inline TxSize max_tx_size(int w4, int h4) {
  int lw = 0, lh = 0;
  while ((1 << lw) < w4 && lw < 4) ++lw;
  while ((1 << lh) < h4 && lh < 4) ++lh;
  return TxSize{lw, lh};
}

/* Max_Tx_Depth: how many times the maximum transform has to be split to reach
 * 4x4, which is the number of bits the tx_depth symbol can carry. */
inline int max_tx_depth(TxSize max_tx) {
  const int longer = tx_square_class(max_tx);
  return longer < 4 ? longer : 4;
}

/* Subsampled_Size: the size of a chroma plane's residual for a block, indexed
 * by the block size, the horizontal and the vertical subsampling; entries are
 * BlockSize numbers and `kBlockSizeCount` marks the combinations the table
 * declares invalid; the specification never
 * looks those up, so they are a bug if they are ever reached. The values are the
 * specification's table, not a formula: a subsampled 8x32 block has a 4x16
 * chroma residual, not a 4x32 one, because the table caps the aspect ratio. */
constexpr uint8_t kSubsampledSize[kBlockSizeCount][2][2] = {
    {{0, 0}, {0, 0}},   /* 4x4 */
    {{1, 0}, {kBlockSizeCount, 0}},   /* 4x8 */
    {{2, kBlockSizeCount}, {0, 0}},   /* 8x4 */
    {{3, 2}, {1, 0}},   /* 8x8 */
    {{4, 3}, {kBlockSizeCount, 1}},   /* 8x16 */
    {{5, kBlockSizeCount}, {3, 2}},   /* 16x8 */
    {{6, 5}, {4, 3}},   /* 16x16 */
    {{7, 6}, {kBlockSizeCount, 4}},   /* 16x32 */
    {{8, kBlockSizeCount}, {6, 5}},   /* 32x16 */
    {{9, 8}, {7, 6}},   /* 32x32 */
    {{10, 9}, {kBlockSizeCount, 7}},  /* 32x64 */
    {{11, kBlockSizeCount}, {9, 8}},  /* 64x32 */
    {{12, 11}, {10, 9}},              /* 64x64 */
    {{13, 12}, {kBlockSizeCount, 10}}, /* 64x128 */
    {{14, kBlockSizeCount}, {12, 11}}, /* 128x64 */
    {{15, 14}, {13, 12}},             /* 128x128 */
    {{16, 1}, {kBlockSizeCount, 1}},  /* 4x16 */
    {{17, kBlockSizeCount}, {2, 2}},  /* 16x4 */
    {{18, 4}, {kBlockSizeCount, 16}}, /* 8x32 */
    {{19, kBlockSizeCount}, {5, 17}}, /* 32x8 */
    {{20, 7}, {kBlockSizeCount, 18}}, /* 16x64 */
    {{21, kBlockSizeCount}, {8, 19}}, /* 64x16 */
};

/* The block size the residual of `plane` is coded in, or kBlockSizeCount when
 * the specification's table has no entry for this combination. */
inline BlockSize plane_residual_size(BlockSize size, int plane, bool ss_x, bool ss_y) {
  if (size >= kBlockSizeCount) return kBlockSizeCount;
  const int sub_x = plane > 0 && ss_x ? 1 : 0;
  const int sub_y = plane > 0 && ss_y ? 1 : 0;
  return static_cast<BlockSize>(kSubsampledSize[size][sub_x][sub_y]);
}

/* Get_TX_Size: the transform one plane of a block uses. Luma uses the transform
 * size the block coded; chroma derives its own from the *block* size (not from
 * the luma transform), because a chroma plane of a subsampled block covers a
 * different area, and it is capped at 32x32 - the largest transform AV1 codes
 * for chroma. A 64 wide or high residual therefore becomes 32x32, 16x32 or
 * 32x16 depending on which side was long. */
inline TxSize plane_tx_size(BlockSize size, int plane, TxSize luma_tx, bool ss_x, bool ss_y) {
  if (plane == 0) return luma_tx;
  const BlockSize residual = plane_residual_size(size, plane, ss_x, ss_y);
  if (residual >= kBlockSizeCount) return TxSize{-1, -1}; /* not a valid size */
  TxSize uv = max_tx_size(block_width4(residual), block_height4(residual));
  if (uv.lw == 4 || uv.lh == 4) {
    if (uv.lw == 2) return TxSize{2, 3}; /* 16x32 */
    if (uv.lh == 2) return TxSize{3, 2}; /* 32x16 */
    return TxSize{3, 3};                 /* 32x32 */
  }
  return uv;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_TX_SIZE_HPP */
