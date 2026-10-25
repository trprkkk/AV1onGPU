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

void apply_grain_host(Frame* frame, const FrameSyntax& syntax) {
  if (!syntax.grain.apply || frame->empty()) return;
  /* The sequential half of the synthesis, exactly as the plan build runs it;
   * the blend itself is the kernels' per sample map. */
  GrainNoise noise;
  build_grain(syntax.grain, frame->bit_depth, syntax.ss_x, syntax.ss_y,
              syntax.color.matrix == ColorMatrix::Identity, frame->width, frame->height, &noise);
  std::vector<Av1gGrainJob> chroma, luma;
  for (int plane = 0; plane < 3; ++plane) {
    append_grain_jobs(noise, plane, frame->plane_width(plane), frame->plane_height(plane), 0,
                      frame->stride[plane], 0, frame->stride[0], frame->width, syntax.ss_x,
                      syntax.ss_y, &chroma, &luma);
  }
  /* chroma first: it blends against the luma the picture has before any grain */
  for (const Av1gGrainJob& job : chroma) {
    av1g_grain_block(&job, frame->plane(job.plane), frame->plane(0), noise.noise.data(), noise.lut,
                     frame->max_value());
  }
  for (const Av1gGrainJob& job : luma) {
    av1g_grain_block(&job, frame->plane(0), frame->plane(0), noise.noise.data(), noise.lut,
                     frame->max_value());
  }
}

void apply_post_filters_host(Frame* frame, const FrameSyntax& syntax, bool grain) {
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
      job.filter_size = e.filter_size;
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
      const bool chroma = plane > 0;
      job.x = c.x;
      job.y = c.y;
      job.src_off = at(plane, c.x, c.y);
      job.src_stride = frame->stride[plane];
      job.dst_off = job.src_off;
      job.dst_stride = job.src_stride;
      job.luma_off = at(0, chroma ? c.x << syntax.ss_x : c.x,
                        chroma ? c.y << syntax.ss_y : c.y);
      job.luma_stride = frame->stride[0];
      job.pri_strength = c.pri_strength;
      job.sec_strength = c.sec_strength;
      job.damping = c.damping;
      job.dir = c.dir;
      job.uv = chroma ? 1 : 0;
      job.ss_x = syntax.ss_x;
      job.ss_y = syntax.ss_y;
      job.bitdepth = frame->bit_depth;
      job.plane_w = frame->plane_width(plane);
      job.plane_h = frame->plane_height(plane);
      /* the direction is read from the luma of the same snapshot the taps come
       * from, at the job's luma offset */
      av1g_cdef_block(&job, scratch.plane(plane), frame->plane(plane),
                      scratch.plane(0), maxv);
    }
  }

  if (grain) apply_grain_host(frame, syntax);
}

} /* namespace internal */
} /* namespace av1gpu */
