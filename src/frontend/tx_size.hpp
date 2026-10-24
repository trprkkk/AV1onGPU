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

/* The transform used for a chroma plane of a block whose luma transform is
 * `luma`: chroma is subsampled, so the chroma transform is at most the luma
 * one and at most 32x32 (the largest chroma transform AV1 defines). */
inline TxSize chroma_tx_size(TxSize luma, bool ss_x, bool ss_y) {
  TxSize out{luma.lw - (ss_x ? 1 : 0), luma.lh - (ss_y ? 1 : 0)};
  if (out.lw < 0) out.lw = 0;
  if (out.lh < 0) out.lh = 0;
  if (out.lw > 3) out.lw = 3;
  if (out.lh > 3) out.lh = 3;
  return out;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_TX_SIZE_HPP */
