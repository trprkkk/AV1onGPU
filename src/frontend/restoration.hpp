/*
 * AV1onGPU - loop restoration (7.17), the plan half
 *
 * The units themselves are parsed by the tile decoder (5.11.46), because their
 * syntax lives in the tile payload and is arithmetic coded right after the
 * superblock's other symbols. All this stage does is hand the frame's unit
 * table to the reconstruction plan, together with the frame level parameters
 * the process needs: which planes are restored at all and how large a unit is.
 */
#ifndef AV1GPU_FRONTEND_RESTORATION_HPP
#define AV1GPU_FRONTEND_RESTORATION_HPP

#include <string>

#include "av1gpu/av1gpu.hpp"
#include "frontend/headers.hpp"
#include "frontend/tile_decoder.hpp"

namespace av1gpu {
namespace frontend {

inline bool plan_restoration(const SequenceHeader& seq, const FrameHeader& frame,
                             const TileDecodeResult& tile, FrameSyntax* out,
                             std::string* error) {
  std::string local;
  if (!error) error = &local;
  (void)seq;

  bool uses_lr = false;
  for (int plane = 0; plane < 3; ++plane) {
    out->restoration_type[plane] = frame.restoration_type[plane];
    out->restoration_unit_log2[plane] = frame.restoration_unit_log2[plane];
    out->restoration[plane] = tile.restoration[plane];
    if (frame.restoration_type[plane] != kRestoreNone) uses_lr = true;
    if (frame.restoration_type[plane] != kRestoreNone && out->restoration[plane].empty()) {
      *error = "loop restoration is on but the tile coded no units";
      return false;
    }
  }
  if (uses_lr) {
    /* The unit column count is derived, not stored: it is what the process
     * needs to index the row major table. */
    for (int plane = 0; plane < 3; ++plane) {
      if (frame.restoration_type[plane] == kRestoreNone) {
        out->restoration_unit_cols[plane] = 0;
        continue;
      }
      const int unit_size = 1 << frame.restoration_unit_log2[plane];
      const int sub_x = plane > 0 && seq.ss_x ? 1 : 0;
      const int width = plane > 0 ? (frame.upscaled_width + sub_x) >> sub_x
                                  : frame.upscaled_width;
      const int cols = std::max((width + (unit_size >> 1)) / unit_size, 1);
      out->restoration_unit_cols[plane] = cols;
    }
  }
  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_RESTORATION_HPP */
