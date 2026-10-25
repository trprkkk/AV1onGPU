/* AV1onGPU - OpenCL entry points.
 *
 * The maths lives in av1gpu_kernels.h and is shared with the CUDA backend and
 * with the CPU reference path, so all three compute bit-identical pixels.  This
 * file only maps the flat job arrays produced by the frame planner onto an
 * NDRange: one work item per output element, and the owning job is found with a
 * binary search over a prefix-sum table (jobs have different element counts, so
 * a plain division would waste work items on the smaller blocks).
 *
 * Needs OpenCL 2.0 or newer for generic address spaces.
 */

#include "av1gpu_kernels.h"

#define AV1G_JOB_LOOKUP(job, elem)                                            \
  int _gid = (int)get_global_id(0);                                           \
  if (_gid >= total) return;                                                  \
  int lo = 0;                                                                 \
  int hi = njobs - 1;                                                         \
  while (lo < hi) {                                                           \
    int mid = (lo + hi) >> 1;                                                 \
    if (joff[mid + 1] <= _gid)                                                \
      lo = mid + 1;                                                           \
    else                                                                      \
      hi = mid;                                                               \
  }                                                                           \
  const int job = lo;                                                         \
  const int elem = _gid - joff[job]

/* ---------------------------------------------------------------- transform */
__kernel void av1g_k_itx_rows(__global const Av1gItxJob *jobs,
                              __global const int *joff, int njobs, int total,
                              __global const int *coef, __global int *tmp) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  if (job.eob <= 0) return;
  /* one work item per transform row: the element count per job is job.h */
  av1g_itx_row(&job, coef, tmp, elem);
}

__kernel void av1g_k_itx_cols(__global const Av1gItxJob *jobs,
                              __global const int *joff, int njobs, int total,
                              __global const int *tmp, __global av1g_px *dst,
                              int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  if (job.eob <= 0) return;
  /* one work item per transform column: the element count per job is job.w.
   * av1g_itx_col_add accumulates, so it must run exactly once per column. */
  av1g_itx_col_add(&job, tmp, dst, elem, maxv);
}

/* --------------------------------------------------------------- prediction */
/* One work item per block: intra prediction is a whole-block operation,
 * because the edge filtering and the recursive predictor look at more than one
 * sample around the position being predicted. */
__kernel void av1g_k_intra(__global const Av1gIntraJob *jobs, int njobs,
                           __global av1g_px *plane,
                           __global const unsigned char *palette, int maxv) {
  const int j = (int)get_global_id(0);
  if (j >= njobs) return;
  Av1gIntraJob job = jobs[j];
  av1g_intra_block(&job, plane, palette, maxv);
}

__kernel void av1g_k_inter(__global const Av1gInterJob *jobs,
                           __global const int *joff, int njobs, int total,
                           __global const av1g_px *cur,
                           __global const av1g_px *pool,
                           __global av1g_px *dst,
                           __global const int *filters, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gInterJob job = jobs[job_idx];
  /* every reference is an absolute offset into the pool; a job that names no
   * slot reads the picture being decoded */
  av1g_inter_px(&job, dst, cur, pool, filters, elem % job.w, elem / job.w,
                maxv);
}

__kernel void av1g_k_add_residual(__global const Av1gItxJob *jobs,
                                  __global const int *joff, int njobs,
                                  int total, __global const int *res,
                                  __global av1g_px *dst, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  av1g_add_residual_px(&job, res, dst, elem % job.w, elem / job.w, maxv);
}

/* ------------------------------------------------------------------ filters */
__kernel void av1g_k_deblock(__global const Av1gDeblockJob *jobs,
                             __global const int *joff, int njobs, int total,
                             __global av1g_px *plane, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gDeblockJob job = jobs[job_idx];
  /* elem is a group index; the group helper takes a sample offset */
  av1g_deblock_group(&job, plane, elem * 4, maxv);
}

__kernel void av1g_k_cdef(__global const Av1gCdefJob *jobs,
                          __global const int *joff, int njobs, int total,
                          __global const av1g_px *src, __global av1g_px *dst,
                          int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gCdefJob job = jobs[job_idx];
  /* one work item per block: the direction, the strength it earns and the
   * copy for a block with no strength are all per block, not per sample */
  /* src and dst are the picture; the luma of the same snapshot is where the
   * direction is read from, at the job's luma_off */
  av1g_cdef_block(&job, src, dst, src, maxv);
}

/* Film grain: one work item per sample. `luma` is the same picture (the chroma
 * blend reads the luma samples before any grain was added, which is why the
 * chroma jobs are launched first). */
__kernel void av1g_k_grain(__global const Av1gGrainJob *jobs,
                           __global const int *joff, int njobs, int total,
                           __global av1g_px *dst,
                           __global const av1g_px *luma,
                           __global const short *noise,
                           __global const int *lut, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gGrainJob job = jobs[job_idx];
  av1g_grain_px(&job, dst, luma, noise, lut, elem % job.w, elem / job.w, maxv);
}

/* Superres: one work item per output sample. The source and the target are two
 * different buffers (the coded picture and the upscaled one), so every sample
 * can be resampled independently. */
__kernel void av1g_k_upscale(__global const Av1gUpscaleJob *jobs,
                             __global const int *joff, int njobs, int total,
                             __global const av1g_px *src,
                             __global av1g_px *dst,
                             __global const int *filters, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gUpscaleJob job = jobs[job_idx];
  av1g_upscale_px(&job, src, dst, filters, elem % job.upscaled_w,
                  elem / job.upscaled_w, maxv);
}

/* Loop restoration: one work item per sample. Two source pictures - the CDEF
 * output and the deblocked picture before it - because the two lines either
 * side of a stripe come from the latter, and a separate destination because a
 * unit's filter reads samples its neighbours write. */
__kernel void av1g_k_lr(__global const Av1gLrJob *jobs,
                        __global const int *joff, int njobs, int total,
                        __global const av1g_px *cdef,
                        __global const av1g_px *curr,
                        __global av1g_px *dst, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gLrJob job = jobs[job_idx];
  av1g_lr_px(&job, cdef, curr, dst, elem % job.w, elem / job.w, maxv);
}

/* ------------------------------------------------------------------- output */
__kernel void av1g_k_color(__global const Av1gColorJob *jobs,
                           __global const int *joff, int njobs, int total,
                           __global const av1g_px *y_plane,
                           __global const av1g_px *u_plane,
                           __global const av1g_px *v_plane,
                           __global unsigned char *rgb) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gColorJob job = jobs[job_idx];
  av1g_color_px(&job, y_plane, u_plane, v_plane, rgb, elem % job.w,
                elem / job.w);
}

/* One work item per output byte, so a job's element count is
 * pitch * (h + ch) and the padding bytes of a row are written as well. */
__kernel void av1g_k_nv12(__global const Av1gNv12Job *jobs,
                          __global const int *joff, int njobs, int total,
                          __global const av1g_px *y_plane,
                          __global const av1g_px *u_plane,
                          __global const av1g_px *v_plane,
                          __global unsigned char *dst) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gNv12Job job = jobs[job_idx];
  dst[elem] = av1g_nv12_px(&job, y_plane, u_plane, v_plane, elem % job.pitch,
                           elem / job.pitch);
}
