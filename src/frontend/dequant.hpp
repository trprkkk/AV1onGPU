/*
 * AV1onGPU - dequantisation
 *
 * A decoded coefficient is a level: a small signed integer that says *how much*
 * of a basis function the block contains, on a scale every block of the frame
 * shares. Turning it into a residual needs two multiplications per coefficient
 * and a size dependent shift, and the constants come from a table the
 * specification ships, indexed by the block's quantiser index.
 *
 * The quantiser index itself is per block, because AV1 lets a stream change it
 * per superblock (delta_q_present). Chroma then adds its own offsets to the
 * step size rather than to the index, which is why the three planes can share
 * an index and still have different step sizes.
 *
 * Quantiser matrices (using_qmatrix) are not supported: they scale every
 * coefficient by a value from a table the frame references, and no stream seen
 * so far uses them.
 *
 * The shift is the one part that is easy to get wrong: transforms larger than
 * 32x32 in either direction are coded with a scaled down set of coefficients,
 * so their dequantised values are divided by 2 (32x32 and its rectangles) or by
 * 4 (the 64x64 family) to bring them back to the sample scale the inverse
 * transform expects. That factor is also why a 64x64 transform only carries
 * 1024 coefficients: the other three quarters are not coded at all.
 *
 * Coefficients are stored dequantised and in raster order, which is what both
 * the GPU kernels and the CPU reference want to read: the entropy decoder
 * produces them in scan order, and the scan is a property of the transform
 * type, so undoing it belongs here, once per block, not in the transform.
 */
#ifndef AV1GPU_FRONTEND_DEQUANT_HPP
#define AV1GPU_FRONTEND_DEQUANT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "frontend/block_size.hpp"
#include "frontend/coef_model.hpp"
#include "frontend/headers.hpp"
#include "frontend/quant_tables.hpp"
#include "frontend/tile_decoder.hpp"
#include "frontend/tx_size.hpp"

namespace av1gpu {
namespace frontend {

/* dc_q / ac_q: the step size the specification's two lookup tables give for a
 * quantiser index, at this frame's bit depth. */
inline int dc_quant(int qindex, int bit_depth) {
  const int row = bit_depth >= 12 ? 2 : (bit_depth >= 10 ? 1 : 0);
  const int index = qindex < 0 ? 0 : (qindex > 255 ? 255 : qindex);
  return quant_tables::kDcQlookup[row][index];
}

inline int ac_quant(int qindex, int bit_depth) {
  const int row = bit_depth >= 12 ? 2 : (bit_depth >= 10 ? 1 : 0);
  const int index = qindex < 0 ? 0 : (qindex > 255 ? 255 : qindex);
  return quant_tables::kAcQlookup[row][index];
}

/* dqDenom: transforms of 32x32 and up are coded at a coarser scale and have to
 * be divided down when they are expanded again, by 2 for the 32x32 family and
 * by 4 for the 64x64 one. */
inline int dq_denominator(TxSize tx) {
  switch (tx_size_index(tx)) {
    case coef_tables::kTx32x32:
    case coef_tables::kTx16x32:
    case coef_tables::kTx32x16:
    case coef_tables::kTx16x64:
    case coef_tables::kTx64x16:
      return 2;
    case coef_tables::kTx64x64:
    case coef_tables::kTx32x64:
    case coef_tables::kTx64x32:
      return 4;
    default:
      return 1;
  }
}

/* One coefficient, dequantised: level * step, truncated to 24 bits, divided by
 * the transform's scale and clipped to what an inverse transform of this bit
 * depth can represent. */
inline int32_t dequantise_level(int level, int step, int denominator, int bit_depth) {
  const int64_t product = static_cast<int64_t>(level) * step;
  const int64_t magnitude = (product < 0 ? -product : product) & 0xFFFFFF;
  const int64_t scaled = magnitude / denominator;
  const int64_t value = product < 0 ? -scaled : scaled;
  const int64_t limit = 1 << (7 + bit_depth);
  if (value > limit - 1) return static_cast<int32_t>(limit - 1);
  if (value < -limit) return static_cast<int32_t>(-limit);
  return static_cast<int32_t>(value);
}

/* AV1 keeps one quantiser index per block - the one the delta_q syntax
 * adjusted - and offsets the *step* per plane, so all three planes read the
 * same index and add different deltas to it. */

/* Everything needed to dequantise one plane's coefficients. */
struct PlaneQuant {
  int dc_step = 0;
  int ac_step = 0;
  int denominator = 1;
};

inline PlaneQuant plane_quant(const FrameHeader& frame, int plane, int block_qindex, TxSize tx,
                              int bit_depth) {
  const int qindex = block_qindex;
  PlaneQuant quant;
  quant.denominator = dq_denominator(tx);
  const int dc_delta = plane == 0 ? frame.delta_q_y_dc
                                  : (plane == 1 ? frame.delta_q_u_dc : frame.delta_q_v_dc);
  const int ac_delta = plane == 0 ? 0 : (plane == 1 ? frame.delta_q_u_ac : frame.delta_q_v_ac);
  quant.dc_step = dc_quant(qindex + dc_delta, bit_depth);
  quant.ac_step = ac_quant(qindex + ac_delta, bit_depth);
  return quant;
}

/* A frame's coefficients, dequantised and in raster order.
 *
 * `coefficients` is one pool for the whole frame; each transform block owns
 * `width * height` consecutive entries, in the adjusted raster the entropy
 * decoder coded (a 64x64 transform owns 32x32 of them). `offsets` holds the
 * start of each block's run, in the same order as the TileTransform records
 * the tile produced, and `transforms` is how many there are. */
struct DequantisedTile {
  std::vector<int32_t> coefficients;
  std::vector<uint32_t> offsets;
  int bit_depth = 8;

  /* The raster of transform block `index`, or nullptr when it had none. */
  const int32_t* block(int index) const {
    if (index < 0 || index >= static_cast<int>(offsets.size())) return nullptr;
    return coefficients.data() + offsets[static_cast<size_t>(index)];
  }
};

/* Expands every transform block of a tile from scan order levels into
 * dequantised raster order. The levels of one block are contiguous in
 * TileDecodeResult::levels (TileTransform::level_offset, `eob` of them), and
 * they are in scan order, so the scan of the block's transform type decides
 * where each one lands. */
inline void dequantise_tile(const FrameHeader& frame, int bit_depth, const TileDecodeResult& tile,
                            DequantisedTile* out) {
  if (!out) return;
  *out = DequantisedTile();
  out->bit_depth = bit_depth;

  size_t total = 0;
  for (const TileBlock& block : tile.blocks) {
    for (const TileTransform& transform : block.transforms) {
      const TxSize adjusted = adjusted_tx_size(transform.size);
      if (adjusted.lw < 0) continue;
      total += static_cast<size_t>(4 << adjusted.lw) * static_cast<size_t>(4 << adjusted.lh);
    }
  }
  out->coefficients.assign(total, 0);
  out->offsets.reserve(total / 16 + 1);

  size_t offset = 0;
  for (const TileBlock& block : tile.blocks) {
    for (const TileTransform& transform : block.transforms) {
      const TxSize adjusted = adjusted_tx_size(transform.size);
      if (adjusted.lw < 0) continue;
      const int width = 4 << adjusted.lw, height = 4 << adjusted.lh;
      const int area = width * height;
      out->offsets.push_back(static_cast<uint32_t>(offset));

      if (transform.eob > 0) {
        const PlaneQuant quant = plane_quant(frame, transform.plane, transform.qindex,
                                             transform.size, bit_depth);
        const ScanOrder scan = coef_scan(adjusted, transform.tx_type);
        int32_t* raster = out->coefficients.data() + offset;
        const int count = transform.eob < scan.count ? transform.eob : scan.count;
        for (int i = 0; i < count; ++i) {
          const int pos = scan.at(i);
          if (pos < 0 || pos >= area) continue;
          const int level = tile.levels[transform.level_offset + static_cast<size_t>(i)];
          if (level == 0) continue;
          const int step = pos == 0 ? quant.dc_step : quant.ac_step;
          raster[pos] = dequantise_level(level, step, quant.denominator, out->bit_depth);
        }
      }
      offset += static_cast<size_t>(area);
    }
  }
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_DEQUANT_HPP */
