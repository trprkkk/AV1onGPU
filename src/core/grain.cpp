/*
 * AV1onGPU - the host half of the film grain synthesis filter
 *
 * Film grain (7.18.3) is the one loop filter that is not a local neighbourhood
 * operation. Its noise comes from an auto-regressive filter over a pseudo
 * random template - every sample depends on the ones already filtered to its
 * left and above - and the noise blocks are stitched together with weights that
 * depend on how far through the frame they are. Neither half parallelises
 * across samples, and both have to produce exactly the same bytes in every
 * backend.
 *
 * So they run here, once per frame, on the host: `build_grain()` produces the
 * grain planes, the stitched noise image and the three scaling tables, and the
 * kernels are left with the last step - scaling one noise sample by the curve
 * and adding it - which is a pure map over the picture and therefore runs
 * anywhere. The plan carries the noise image, so a backend never has to know
 * how it was made.
 */
#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/backend.hpp"

namespace av1gpu {
namespace internal {

namespace {

/* The public parameters in the kernels' layout. The kernels are compiled as C
 * (OpenCL, CUDA) and cannot see the C++ struct, but both hold the same values -
 * the signed AR coefficients, the point lists and the chroma coupling. */
void to_kernel_params(const FilmGrainParams& p, int bit_depth, Av1gGrainParams* k) {
  std::memset(k, 0, sizeof(*k));
  k->seed = p.seed;
  k->bitdepth = bit_depth;
  k->num_y_points = p.num_y_points;
  k->num_cb_points = p.num_cb_points;
  k->num_cr_points = p.num_cr_points;
  k->chroma_scaling_from_luma = p.chroma_scaling_from_luma ? 1 : 0;
  for (int i = 0; i < 16; ++i) {
    k->y_value[i] = p.point_y_value[i];
    k->y_scaling[i] = p.point_y_scaling[i];
    k->cb_value[i] = p.point_cb_value[i];
    k->cb_scaling[i] = p.point_cb_scaling[i];
    k->cr_value[i] = p.point_cr_value[i];
    k->cr_scaling[i] = p.point_cr_scaling[i];
  }
  k->grain_scaling_minus_8 = p.grain_scaling_minus_8;
  k->ar_coeff_lag = p.ar_coeff_lag;
  for (int i = 0; i < AV1G_GRAIN_POSITIONS; ++i) k->ar_coeffs_y[i] = p.ar_coeffs_y[i];
  for (int i = 0; i < AV1G_GRAIN_CHROMA_POSITIONS; ++i) {
    k->ar_coeffs_cb[i] = p.ar_coeffs_cb[i];
    k->ar_coeffs_cr[i] = p.ar_coeffs_cr[i];
  }
  k->ar_coeff_shift_minus_6 = p.ar_coeff_shift_minus_6;
  k->grain_scale_shift = p.grain_scale_shift;
  k->cb_mult = p.cb_mult;
  k->cb_luma_mult = p.cb_luma_mult;
  k->cb_offset = p.cb_offset;
  k->cr_mult = p.cr_mult;
  k->cr_luma_mult = p.cr_luma_mult;
  k->cr_offset = p.cr_offset;
  k->overlap_flag = p.overlap_flag ? 1 : 0;
  k->clip_to_restricted_range = p.clip_to_restricted_range ? 1 : 0;
}

} /* namespace */

void build_grain(const FilmGrainParams& params, int bit_depth, int ss_x, int ss_y,
                 bool identity_matrix, int luma_w, int luma_h, GrainNoise* out) {
  to_kernel_params(params, bit_depth, &out->params);
  const Av1gGrainParams& k = out->params;

  /* A frame whose parameters give chroma no grain at all (monochrome, or a 4:2:0
   * picture with no chroma points) builds no chroma noise: the blend would be a
   * no-op on those planes. */
  const bool chroma_grain =
      params.chroma_scaling_from_luma || params.num_cb_points > 0 || params.num_cr_points > 0;

  /* ---- generate the grain (7.18.3.3) -------------------------------- */
  const int chroma_w = (luma_w + ss_x) >> ss_x;
  const int chroma_h = (luma_h + ss_y) >> ss_y;
  std::vector<signed char> luma_grain(static_cast<size_t>(AV1G_GRAIN_LUMA_W) * AV1G_GRAIN_LUMA_H);
  std::vector<signed char> cb_grain;
  std::vector<signed char> cr_grain;
  if (chroma_grain) {
    cb_grain.assign(static_cast<size_t>(chroma_w) * chroma_h, 0);
    cr_grain.assign(static_cast<size_t>(chroma_w) * chroma_h, 0);
  }
  av1g_grain_generate(&k, luma_grain.data(), chroma_grain ? cb_grain.data() : nullptr,
                      chroma_grain ? cr_grain.data() : nullptr, chroma_w, chroma_h, ss_x, ss_y);

  /* ---- the noise image (7.18.3.5) ----------------------------------- */
  const int maxv = (1 << bit_depth) - 1;
  out->min_value = 0;
  out->max_luma = maxv;
  out->max_chroma = maxv;
  if (params.clip_to_restricted_range) {
    out->min_value = 16 << (bit_depth - 8);
    out->max_luma = 235 << (bit_depth - 8);
    out->max_chroma = identity_matrix ? out->max_luma : (240 << (bit_depth - 8));
  }

  const int planes = chroma_grain ? 3 : 1;
  size_t total = 0;
  for (int plane = 0; plane < planes; ++plane) {
    const int sub_x = plane ? ss_x : 0;
    const int sub_y = plane ? ss_y : 0;
    out->noise_off[plane] = static_cast<int>(total);
    out->noise_stride[plane] = (luma_w + sub_x) >> sub_x;
    total += static_cast<size_t>(out->noise_stride[plane]) * ((luma_h + sub_y) >> sub_y);
  }
  for (int plane = planes; plane < 3; ++plane) {
    out->noise_off[plane] = static_cast<int>(total);
    out->noise_stride[plane] = 0;
  }
  out->noise.assign(total, 0);

  for (int plane = 0; plane < planes; ++plane) {
    /* The stripes are a scratch buffer, not part of the picture: they hold the
     * grain blocks before they are blended into the noise image. */
    const size_t scratch_size = av1g_grain_scratch_size(luma_w, luma_h, ss_x, ss_y, plane);
    std::vector<signed char> scratch(scratch_size);
    const signed char* source = plane == 0 ? luma_grain.data()
                                           : (plane == 1 ? cb_grain.data() : cr_grain.data());
    av1g_grain_noise_plane(&k, source, plane, luma_w, luma_h, ss_x, ss_y,
                           out->noise.data() + out->noise_off[plane], scratch.data());
    av1g_grain_scaling_lut(&k, plane, out->lut + 256 * plane);
  }
  /* A monochrome frame still carries three tables; the ones it has no noise for
   * stay empty so a stray job could not read them. */
  for (int plane = planes; plane < 3; ++plane) {
    for (int i = 0; i < 256; ++i) out->lut[256 * plane + i] = 0;
  }
}

void append_grain_jobs(const GrainNoise& grain, int plane, int plane_w, int plane_h, int dst_off,
                       int dst_stride, int luma_off, int luma_stride, int luma_w, int ss_x,
                       int ss_y, std::vector<Av1gGrainJob>* chroma,
                       std::vector<Av1gGrainJob>* luma) {
  const Av1gGrainParams& k = grain.params;
  /* A plane the parameters give no strength has nothing to add: the noise image
   * is all zeros there and the specification skips the plane entirely. */
  if (plane == 0) {
    if (k.num_y_points <= 0) return;
  } else if (!k.chroma_scaling_from_luma &&
             (plane == 1 ? k.num_cb_points : k.num_cr_points) <= 0) {
    return;
  }
  /* The blend is a per sample map, so the plane is cut into square jobs; the
   * noise image is already stitched, which means a job never needs to look at
   * its neighbours and the jobs can run in any order. */
  const int tile = 128;
  for (int y = 0; y < plane_h; y += tile) {
    for (int x = 0; x < plane_w; x += tile) {
      Av1gGrainJob job;
      std::memset(&job, 0, sizeof(job));
      job.dst_off = dst_off + y * dst_stride + x;
      job.dst_stride = dst_stride;
      job.noise_off = grain.noise_off[plane];
      job.noise_stride = grain.noise_stride[plane];
      job.x = x;
      job.y = y;
      job.w = std::min(tile, plane_w - x);
      job.h = std::min(tile, plane_h - y);
      job.plane = plane;
      job.ss_x = plane ? ss_x : 0;
      job.ss_y = plane ? ss_y : 0;
      job.bitdepth = k.bitdepth;
      job.scaling_shift = k.grain_scaling_minus_8 + 8;
      job.num_y_points = k.num_y_points;
      job.num_cb_points = k.num_cb_points;
      job.num_cr_points = k.num_cr_points;
      job.chroma_scaling_from_luma = k.chroma_scaling_from_luma;
      job.cb_mult = k.cb_mult;
      job.cb_luma_mult = k.cb_luma_mult;
      job.cb_offset = k.cb_offset;
      job.cr_mult = k.cr_mult;
      job.cr_luma_mult = k.cr_luma_mult;
      job.cr_offset = k.cr_offset;
      job.min_value = grain.min_value;
      job.max_value = plane == 0 ? grain.max_luma : grain.max_chroma;
      job.luma_off = luma_off;
      job.luma_stride = luma_stride;
      job.luma_w = luma_w;
      if (plane == 0) {
        luma->push_back(job);
      } else {
        chroma->push_back(job);
      }
    }
  }
}

} /* namespace internal */
} /* namespace av1gpu */
