/*
 * AV1onGPU - how the coefficient syntax chooses its tables
 *
 * The coefficient syntax is a small symbol decoder wrapped in a lot of table
 * selection. Which scan order, which transform type alphabet, which context
 * row: all of it is derived from three things - the transform size, the plane,
 * and the neighbouring blocks' coefficient levels - and all of it has to match
 * the encoder exactly, because one wrong context row desynchronises the range
 * coder a symbol later and the whole frame is lost.
 *
 * This header holds the selection rules that are pure functions of the
 * bitstream state: transform size numbering, scan order, transform sets and
 * classes. The context arithmetic that reads decoded levels lives with the
 * parser, since it needs the coefficient arrays.
 *
 * Three sizes are in play at once and it matters which one is which:
 *
 *   the coded transform      what the inverse transform will use (up to 64x64)
 *   the adjusted transform   the same size capped at 32x32, which is the grid
 *                            the coefficient *positions* live on and the size
 *                            the context tables are indexed by
 *   the square class         the log2 of the adjusted size, which is how the
 *                            CDF tables are addressed
 *
 * A 64x64 transform, for instance, codes 1024 coefficients in a 32x32 raster
 * with a 32x32 scan; the remaining three quarters of the block are zero by
 * construction.
 */
#ifndef AV1GPU_FRONTEND_COEF_MODEL_HPP
#define AV1GPU_FRONTEND_COEF_MODEL_HPP

#include <cstdint>

#include "frontend/coef_tables.hpp"
#include "frontend/tx_size.hpp"

namespace av1gpu {
namespace frontend {

/* The specification's transform size number for a rectangle, which is what
 * every table in coef_tables.hpp is indexed by, or -1 for a size the
 * specification does not define. The table holds log2 of the size in pixels,
 * while TxSize holds log2 of it in 4x4 units, hence the 2. */
inline int tx_size_index(TxSize tx) {
  if (tx.lw < 0 || tx.lh < 0) return -1;
  for (int i = 0; i < coef_tables::kTxSizesAll; ++i) {
    if (coef_tables::kTxWidthLog2[i] == tx.lw + 2 &&
        coef_tables::kTxHeightLog2[i] == tx.lh + 2) {
      return i;
    }
  }
  return -1;
}

/* The rectangle of a specification transform size number. */
inline TxSize tx_size_from_index(int index) {
  if (index < 0 || index >= coef_tables::kTxSizesAll) return TxSize{-1, -1};
  return TxSize{coef_tables::kTxWidthLog2[index] - 2, coef_tables::kTxHeightLog2[index] - 2};
}

/* Adjusted_Tx_Size: the transform size the coefficient syntax works in, with
 * each side capped at 32. */
inline TxSize adjusted_tx_size(TxSize tx) {
  const int index = tx_size_index(tx);
  if (index < 0) return TxSize{-1, -1};
  return tx_size_from_index(coef_tables::kAdjustedTxSize[index]);
}

/* Tx_Size_Sqr and Tx_Size_Sqr_Up as log2 of the 4x4 unit count: the square
 * class of the shorter and of the longer side. These are the two numbers the
 * CDF tables are indexed by, and their average is the "transform size context"
 * the coefficient symbols use. */
inline int tx_sqr(TxSize tx) { return tx_min_class(tx); }
inline int tx_sqr_up(TxSize tx) { return tx_square_class(tx); }

/* txSzCtx in the specification: (Tx_Size_Sqr + Tx_Size_Sqr_Up + 1) >> 1, the
 * row of the coefficient CDF tables. A square transform has 0..4, a rectangle
 * lands on its longer side. */
inline int tx_size_context(TxSize tx) { return (tx_sqr(tx) + tx_sqr_up(tx) + 1) >> 1; }

/* The number of coefficients a transform block codes: the adjusted area, which
 * counts a 64x64 transform as 1024 coefficients and a 16x64 as 512. */
inline int coef_count(TxSize tx) {
  const TxSize adjusted = adjusted_tx_size(tx);
  if (adjusted.lw < 0) return 0;
  return (4 << adjusted.lw) * (4 << adjusted.lh);
}

/* Transform_Class: which neighbourhood the coefficient contexts look at. A 2D
 * transform reads the diagonal neighbours, a vertical-only one the column, a
 * horizontal-only one the row. */
inline int tx_class_of(int tx_type) {
  switch (tx_type) {
    case coef_tables::kVdct:
    case coef_tables::kVadst:
    case coef_tables::kVflipadst:
      return coef_tables::kTxClassVert;
    case coef_tables::kHdct:
    case coef_tables::kHadst:
    case coef_tables::kHflipadst:
      return coef_tables::kTxClassHoriz;
    default:
      return coef_tables::kTxClass2d;
  }
}

/* Get_TX_Set: which transform type alphabet this transform size can use. Zero
 * means DCT only (no type is coded at all); the intra sets are 1 and 2. */
inline int tx_set(TxSize tx, bool is_inter, bool reduced_tx_set) {
  const int sqr_up = tx_sqr_up(tx);
  if (sqr_up > 3) return coef_tables::kTxSetDctOnly; /* 64 wide or high */
  if (is_inter) {
    if (reduced_tx_set || sqr_up == 3) return coef_tables::kTxSetInter3;
    if (tx_sqr(tx) == 2) return coef_tables::kTxSetInter2;
    return coef_tables::kTxSetInter1;
  }
  if (sqr_up == 3) return coef_tables::kTxSetDctOnly;
  if (reduced_tx_set) return coef_tables::kTxSetIntra2;
  if (tx_sqr(tx) == 2) return coef_tables::kTxSetIntra2;
  return coef_tables::kTxSetIntra1;
}

/* Whether a transform type belongs to a set. */
inline bool is_tx_type_in_set(int set, int tx_type, bool is_inter) {
  if (tx_type < 0 || tx_type >= coef_tables::kTxTypes) return false;
  if (set < 0) return false;
  if (is_inter) {
    if (set > coef_tables::kTxSetInter3) return false;
    return coef_tables::kTxTypeInSetInter[set][tx_type] != 0;
  }
  if (set > coef_tables::kTxSetIntra2) return false;
  return coef_tables::kTxTypeInSetIntra[set][tx_type] != 0;
}

/* The transform type a coded symbol stands for. The symbol is an index into the
 * set's own ordering, not into TxType, which is why each set has an inversion
 * table. */
inline int tx_type_from_symbol(int set, int symbol, bool is_inter) {
  if (is_inter) {
    if (set == coef_tables::kTxSetInter1 && symbol >= 0 && symbol < 16) {
      return coef_tables::kTxTypeInterSet1[symbol];
    }
    if (set == coef_tables::kTxSetInter2 && symbol >= 0 && symbol < 12) {
      return coef_tables::kTxTypeInterSet2[symbol];
    }
    if (set == coef_tables::kTxSetInter3 && symbol >= 0 && symbol < 2) {
      return coef_tables::kTxTypeInterSet3[symbol];
    }
    return coef_tables::kDctDct;
  }
  if (set == coef_tables::kTxSetIntra1 && symbol >= 0 && symbol < 7) {
    return coef_tables::kTxTypeIntraSet1[symbol];
  }
  if (set == coef_tables::kTxSetIntra2 && symbol >= 0 && symbol < 5) {
    return coef_tables::kTxTypeIntraSet2[symbol];
  }
  return coef_tables::kDctDct;
}

/* How many symbols a set's alphabet has. The intra and inter sets share the
 * numbers 1 and 2 but not their alphabets, so this takes the frame type. */
inline int tx_type_symbol_count(int set, bool is_inter) {
  if (is_inter) {
    if (set == coef_tables::kTxSetInter1) return 16;
    if (set == coef_tables::kTxSetInter2) return 12;
    if (set == coef_tables::kTxSetInter3) return 2;
    return 1;
  }
  if (set == coef_tables::kTxSetIntra1) return 7;
  if (set == coef_tables::kTxSetIntra2) return 5;
  return 1;
}

/* One scan order: `positions[c]` is the coefficient position, in the adjusted
 * transform's raster, of the c'th coded coefficient. `count` is the number of
 * coefficients, so a scan is always a permutation of 0..count-1. */
struct ScanOrder {
  const uint16_t* positions = nullptr;
  int count = 0;

  /* The raster position of the c'th coefficient, or -1. */
  int at(int c) const { return c >= 0 && c < count ? positions[c] : -1; }
};

/* Get_Scan: the order the coefficients are coded in. It depends on the
 * transform *type*, not on the size: an identity or 2D transform scans
 * diagonally, a vertical-only one row by row, a horizontal-only one column by
 * column. Sizes whose transform set is DCT only can only ask for the diagonal
 * order, which is why the row and column orders exist for nine sizes only. */
inline ScanOrder coef_scan(TxSize tx, int tx_type) {
  ScanOrder scan;
  const TxSize adjusted = adjusted_tx_size(tx);
  if (adjusted.lw < 0) return scan;
  scan.count = (4 << adjusted.lw) * (4 << adjusted.lh);

  /* A 64x64 transform is scanned as a 32x32 one; 16x64 and 64x16 as 16x32 and
   * 32x16. That falls out of the adjusted size clamp above. */
  int lw = adjusted.lw, lh = adjusted.lh;
  if (lw > 3) lw = 3;
  if (lh > 3) lh = 3;

  const int tx_class = tx_class_of(tx_type);
  if (tx_class == coef_tables::kTxClassVert) {
    const int offset = coef_tables::row_scan_offset(lw, lh);
    if (offset >= 0) {
      scan.positions = coef_tables::kMrowScan + offset;
      return scan;
    }
  } else if (tx_class == coef_tables::kTxClassHoriz) {
    const int offset = coef_tables::col_scan_offset(lw, lh);
    if (offset >= 0) {
      scan.positions = coef_tables::kMcolScan + offset;
      return scan;
    }
  }
  const int offset = coef_tables::default_scan_offset(lw, lh);
  scan.positions = coef_tables::kDefaultScan + (offset >= 0 ? offset : 0);
  return scan;
}

/* The chroma transform type an intra mode implies (Mode_To_Txfm). The mode is
 * the specification's uv mode, so CFL is 13. */
inline int tx_type_from_uv_mode(int uv_mode) {
  if (uv_mode < 0 || uv_mode >= 14) return coef_tables::kDctDct;
  return coef_tables::kModeToTxfm[uv_mode];
}

/* The eob alphabet is picked by the sum of the adjusted width and height log2,
 * 0 for 4x4 up to 6 for 32x32. */
inline int eob_size_class(TxSize tx) {
  const TxSize adjusted = adjusted_tx_size(tx);
  if (adjusted.lw < 0) return 0;
  const int lw = adjusted.lw > 3 ? 3 : adjusted.lw;
  const int lh = adjusted.lh > 3 ? 3 : adjusted.lh;
  return lw + lh;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_COEF_MODEL_HPP */
