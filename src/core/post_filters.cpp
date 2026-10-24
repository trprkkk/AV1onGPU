/*
 * AV1onGPU - host side loop filters
 *
 * An encoder that uses this project (including the synthetic one in
 * src/core/synthetic.cpp) has to reconstruct exactly like the decoder, loop
 * filters included, otherwise its reference frames drift and every following
 * inter frame is wrong. This helper runs the deblock / CDEF / grain stages on a
 * host frame with the same jobs the engine builds, so the two cannot disagree.
 */
#include <algorithm>
#include <cstring>

#include "core/backend.hpp"

namespace av1gpu {
namespace internal {

void apply_post_filters_host(Frame* frame, const FrameSyntax& syntax) {
  const int maxv = (1 << frame->bit_depth) - 1;

  auto at = [&](int plane, int x, int y) { return y * frame->stride[plane] + x; };

  /* ---- deblocking ------------------------------------------------------- */
  /* two ordered passes, vertical edges then horizontal ones, exactly like the
   * engine schedules them - a vertical and a horizontal edge cross */
  for (int pass = 0; pass < 2; ++pass) {
    const bool vertical = pass == 0;
    for (const DeblockEdge& e : syntax.deblock) {
      if (e.vertical != vertical) continue;
      Av1gDeblockJob job;
      std::memset(&job, 0, sizeof(job));
      const int plane = e.plane;
      job.p_off = e.vertical ? at(plane, e.x - 2, e.y) : at(plane, e.x, e.y - 2);
      job.stride = e.vertical ? 1 : frame->stride[plane];
      job.step = e.vertical ? frame->stride[plane] : 1;
      job.len = e.length;
      job.level = e.level;
      job.sharpness = syntax.sharpness;
      job.filter_len = e.filter_len;
      job.is_chroma = plane != 0;
      job.bitdepth = frame->bit_depth;
      av1g_deblock_job(&job, frame->plane(plane), maxv);
    }
  }

  /* ---- CDEF ------------------------------------------------------------- */
  if (!syntax.cdef.empty()) {
    Frame scratch = *frame;
    for (const CdefBlock& c : syntax.cdef) {
      Av1gCdefJob job;
      std::memset(&job, 0, sizeof(job));
      const int plane = c.plane;
      job.src_off = at(plane, c.x, c.y);
      job.src_stride = frame->stride[plane];
      job.dst_off = job.src_off;
      job.dst_stride = job.src_stride;
      job.pri_strength = c.pri_strength;
      job.sec_strength = c.sec_strength;
      job.damping = c.damping;
      job.dir = c.dir;
      job.bitdepth = frame->bit_depth;
      job.plane_w = frame->plane_width(plane);
      job.plane_h = frame->plane_height(plane);
      av1g_cdef_block(&job, scratch.plane(plane), frame->plane(plane), maxv);
    }
  }

  /* ---- film grain ------------------------------------------------------- */
  if (syntax.apply_grain && syntax.grain.template_size > 0) {
    const int grain_stride = std::max(64, syntax.grain.template_size);
    const int grain_h = grain_stride;
    std::vector<signed char> temp;
    tables::grain_template(syntax.grain.seed, grain_stride, grain_h, &temp);
    int lut[64] = {0};
    for (int i = 0; i < 32; ++i) {
      lut[i] = syntax.grain.lut[i];
      lut[32 + i] = syntax.grain.lut[i];
    }
    const int tile = 128;
    for (int plane = 0; plane < 3; ++plane) {
      const int pw = frame->plane_width(plane);
      const int ph = frame->plane_height(plane);
      for (int y = 0; y < ph; y += tile) {
        for (int x = 0; x < pw; x += tile) {
          Av1gGrainJob job;
          std::memset(&job, 0, sizeof(job));
          job.dst_off = at(plane, x, y);
          job.dst_stride = frame->stride[plane];
          job.x = x;
          job.y = y;
          job.w = std::min(tile, pw - x);
          job.h = std::min(tile, ph - y);
          job.bitdepth = frame->bit_depth;
          job.grain_stride = grain_stride;
          job.grain_h = grain_h;
          job.overlap = syntax.grain.overlap;
          av1g_grain_block(&job, frame->plane(plane), temp.data(), lut, maxv);
        }
      }
    }
  }
}

} /* namespace internal */
} /* namespace av1gpu */
