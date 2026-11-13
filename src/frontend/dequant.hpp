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
 * A stream may also request a quantiser matrix (using_qmatrix): every
 * coefficient is then scaled by a per-position factor from a table the frame
 * header picks (qm_y for luma, qm_u/qm_v for chroma). The factor is a Q5.5
 * number, applied as (step * factor + 16) >> 5, so a flat 32 is a no-op.
 * Matrices only apply to the two-dimensional transform types; identity and
 * the purely horizontal/vertical ones ignore them. A level of 15 means the
 * plane uses no matrix at all.
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
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "frontend/block_size.hpp"
#include "frontend/coef_model.hpp"
#include "frontend/headers.hpp"
#include "frontend/quant_tables.hpp"
#include "frontend/qm_tables.hpp"
#include "frontend/thread_pool.hpp"
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

/* The quantiser matrix of a plane at one transform size, or nullptr when the
 * frame uses none for this plane. `tx` is the adjusted size, so it is at most
 * 32x32 - larger transforms share the 32x32 matrix (or its transpose-class
 * rectangle relatives). The level comes from the frame header, 15 meaning
 * "no matrix". Returns the row-major matrix: entry `r * width + c` scales the
 * coefficient at row r, column c. */
inline const uint8_t* qm_matrix(const FrameHeader& frame, int plane, TxSize tx) {
  static const bool qm_disabled = std::getenv("AV1GPU_NO_QM") != nullptr;
  if (!frame.using_qmatrix || qm_disabled) return nullptr;
  const int level = plane == 0 ? frame.qm_y : (plane == 1 ? frame.qm_u : frame.qm_v);
  if (level >= 15) return nullptr;
  const int cls = plane == 0 ? 0 : 1;
  const int lw = tx.lw, lh = tx.lh;
  if (lw == lh) {
    switch (lw) {
      case 0: return tables::kQm4x4[level][cls];
      case 1: return tables::kQm8x8[level][cls];
      case 2: return tables::kQm16x16[level][cls];
      default: return tables::kQm32x32[level][cls];
    }
  }
  if (lw == 3 && lh == 2) return tables::kQm32x16[level][cls];
  if (lw == 2 && lh == 3) return tables::kQm16x32[level][cls];
  if (lw == 3 && lh == 1) return tables::kQm32x8[level][cls];
  if (lw == 1 && lh == 3) return tables::kQm8x32[level][cls];
  if (lw == 2 && lh == 0) return tables::kQm16x4[level][cls];
  if (lw == 0 && lh == 2) return tables::kQm4x16[level][cls];
  if (lw == 2 && lh == 1) return tables::kQm16x8[level][cls];
  if (lw == 1 && lh == 2) return tables::kQm8x16[level][cls];
  if (lw == 1 && lh == 0) return tables::kQm8x4[level][cls];
  return tables::kQm4x8[level][cls]; /* lw == 0 && lh == 1 */
}

/* A frame's coefficients, dequantised and in raster order.
 *
 * `coefficients` is one pool for the whole frame; each transform block owns
 * `width * height` consecutive entries, in the adjusted raster the entropy
 * decoder coded (a 64x64 transform owns 32x32 of them). `offsets` holds the
 * start of each block's run, in the same order as the TileTransform records
 * the tile produced, and `transforms` is how many there are. */
/* What a device side dequantiser needs for one transform beyond the raster
 * pool: the plane's steps, the transform's scale, and where its quantiser
 * matrix sits in DequantisedTile::qm_pool (-1 for none). The host computes
 * these once per transform here, so the kernel is the same arithmetic and
 * nothing else. */
struct DequantParam {
  int32_t dc_step = 0;
  int32_t ac_step = 0;
  int32_t denominator = 1;
  int32_t qm_off = -1;
};

struct DequantisedTile {
  std::vector<int32_t> coefficients;
  std::vector<uint32_t> offsets;
  /* Parallel to `offsets`: one entry per transform, in the same order. */
  std::vector<DequantParam> params;
  /* The distinct 256 byte quantiser matrices this tile's transforms use,
   * concatenated; DequantParam::qm_off indexes it. */
  std::vector<uint8_t> qm_pool;
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
 * where each one lands.
 *
 * `levels_only` stops after the per transform bookkeeping (the pool offsets
 * and the dequantiser parameters): the raster itself is left unbuilt. It is for
 * the caller whose backend expands the levels on the device, where the fill is
 * ~10 ms a frame of work whose result travels over the bus to be read by one
 * kernel. Everything the plan still needs - the offsets, the steps, the QM
 * pool and the levels - is produced either way. */
inline void dequantise_tile(const FrameHeader& frame, int bit_depth, const TileDecodeResult& tile,
                            DequantisedTile* out, bool levels_only = false) {
  if (!out) return;
  *out = DequantisedTile();
  out->bit_depth = bit_depth;

  /* One entry per transform block, collected once: the area of a transform
   * depends on nothing else, so the pool offsets are a prefix sum over this
   * list and the fill below can run over any range of it, on any thread,
   * without touching another entry's memory. */
  struct Entry {
    const TileTransform* transform;
    size_t area;
    size_t offset;
  };
  /* The QM matrices are shared between the transforms that use the same one
   * (a frame usually has one per plane): dedup by the table address, which is
   * exact because qm_matrix returns pointers into static tables, and by the
   * byte count (a 4x4 matrix and the head of an 8x8 one can hold the same
   * bytes). */
  struct QmEntry {
    const uint8_t* ptr;
    size_t size;
    int32_t off;
  };
  std::vector<QmEntry> qm_seen;
  /* The QM matrix a transform uses is a function of the plane and the adjusted
   * size alone (the level comes from the frame header), so the pool offset it
   * dedups to is too: build one 3 x 16 table per call and the per transform
   * work becomes an array read. The scan it replaces walked every matrix the
   * tile had already seen, once for each of a 4K frame's ~230k transforms. */
  int32_t qm_tab[3][16];
  for (int p = 0; p < 3; ++p)
    for (int i = 0; i < 16; ++i) qm_tab[p][i] = -1;
  size_t total = 0;
  out->offsets.reserve(static_cast<size_t>(tile.transform_blocks) + 16);
  /* levels_only: no raster is built, and nothing needs the entry list (the
   * offsets are just the running prefix sum), so the loop writes the offsets
   * and the parameters straight out instead of staging entries first. */
  const bool scan_entries = !levels_only;
  std::vector<Entry> entries;
  if (scan_entries) entries.reserve(static_cast<size_t>(tile.transform_blocks) + 16);
  for (const TileBlock& block : tile.blocks) {
    for (const TileTransform& transform : block.transforms) {
      const TxSize adjusted = adjusted_tx_size(transform.size);
      if (adjusted.lw < 0) continue;
      const size_t area =
          static_cast<size_t>(4 << adjusted.lw) * static_cast<size_t>(4 << adjusted.lh);
      if (scan_entries) entries.push_back({&transform, area, total});
      out->offsets.push_back(static_cast<uint32_t>(total));
      total += area;
      /* The per transform dequantiser, computed once here rather than in the
       * per coefficient loop below: a device side dequant reads these from
       * the job instead of recomputing them. */
      const PlaneQuant quant = plane_quant(frame, transform.plane, transform.qindex,
                                           transform.size, bit_depth);
      DequantParam param;
      param.dc_step = quant.dc_step;
      param.ac_step = quant.ac_step;
      param.denominator = quant.denominator;
      const uint8_t* qm = transform.tx_type < coef_tables::kIdtx
                              ? qm_matrix(frame, transform.plane, adjusted)
                              : nullptr;
      if (qm) {
        int32_t& cell = qm_tab[transform.plane][(adjusted.lw << 2) | adjusted.lh];
        if (cell >= 0) {
          param.qm_off = cell;
        } else {
          /* first use of this (plane, size): dedup it into the pool once */
          int32_t found = -1;
          for (const QmEntry& seen : qm_seen) {
            if (seen.ptr == qm && seen.size == area) {
              found = seen.off;
              break;
            }
          }
          if (found < 0) {
            found = static_cast<int32_t>(out->qm_pool.size());
            out->qm_pool.insert(out->qm_pool.end(), qm, qm + area);
            qm_seen.push_back({qm, area, found});
          }
          cell = found;
          param.qm_off = found;
        }
      }
      out->params.push_back(param);
    }
  }
  if (!levels_only) out->coefficients.assign(total, 0);
  out->offsets.resize(out->params.size());

  /* AV1GPU_DQDBG=x,y: print the steps of the transform at that position. The
   * variable is read once: getenv walks the environment block, which is not
   * something to do per coefficient block. */
  static const char* const dqdbg = std::getenv("AV1GPU_DQDBG");

  auto fill = [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const TileTransform& transform = *entries[i].transform;
      if (transform.eob <= 0) continue;
      const TxSize adjusted = adjusted_tx_size(transform.size);
      const int width = 4 << adjusted.lw, height = 4 << adjusted.lh;
      const int area = width * height;
      const size_t offset = entries[i].offset;

      const DequantParam& param = out->params[i];
      if (dqdbg && transform.plane == 0) {
        int qx = -1, qy = -1;
        if (std::sscanf(dqdbg, "%d,%d", &qx, &qy) == 2 &&
            transform.x <= qx && qx < transform.x + width &&
            transform.y <= qy && qy < transform.y + height)
          std::fprintf(stderr, "DQDBG x=%d y=%d w=%d h=%d qindex=%d dc=%d ac=%d den=%d "
                               "eob=%d tx_type=%d off=%zu\n",
                       transform.x, transform.y, width, height,
                       transform.qindex, param.dc_step, param.ac_step, param.denominator,
                       transform.eob, transform.tx_type, offset);
      }
      const ScanOrder scan = coef_scan(adjusted, transform.tx_type);
      /* The matrix applies only to the 2-D transform types, and never to
       * the identity (kIdtx and the 1-D types come after it). */
      const uint8_t* qm =
          param.qm_off >= 0 ? out->qm_pool.data() + param.qm_off : nullptr;
      int32_t* raster = out->coefficients.data() + offset;
      const int count = transform.eob < scan.count ? transform.eob : scan.count;
      for (int i = 0; i < count; ++i) {
        const int pos = scan.at(i);
        if (pos < 0 || pos >= area) continue;
        const int level = tile.levels[transform.level_offset + static_cast<size_t>(i)];
        if (level == 0) continue;
        int step = pos == 0 ? param.dc_step : param.ac_step;
        if (qm) step = (step * qm[pos] + 16) >> 5;
        raster[pos] = dequantise_level(level, step, param.denominator, out->bit_depth);
      }
    }
  };

  /* The fill is the frame's dequantisation cost, and every entry writes its
   * own run of the pool, so it splits across the shared worker pool without
   * any lock. Small frames stay on the calling thread. A backend that expands
   * the levels on the device skips it entirely: that kernel reads the same
   * levels and writes the same raster, on the device. */
  if (!levels_only) {
    out->coefficients.assign(total, 0);
    parallel_chunks(entries.size(), 8, [&](size_t, size_t b0, size_t e0) { fill(b0, e0); });
  }
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_DEQUANT_HPP */
