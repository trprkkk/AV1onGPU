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
                              __global const int *coef, __global int *tmp,
                              __global const int *basis) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  if (job.eob <= 0) return;
  /* one work item per transform row: the element count per job is job.h */
  av1g_itx_row(&job, coef, tmp, basis, elem);
}

__kernel void av1g_k_itx_cols(__global const Av1gItxJob *jobs,
                              __global const int *joff, int njobs, int total,
                              __global const int *tmp, __global av1g_px *dst,
                              __global const int *basis, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  if (job.eob <= 0) return;
  /* one work item per transform column: the element count per job is job.w.
   * av1g_itx_col_add accumulates, so it must run exactly once per column. */
  av1g_itx_col_add(&job, tmp, dst, basis, elem, maxv);
}

/* --------------------------------------------------------------- prediction */
__kernel void av1g_k_intra_aux(__global const Av1gIntraJob *jobs, int njobs,
                               __global const av1g_px *plane,
                               __global Av1gIntraAux *aux) {
  const int j = (int)get_global_id(0);
  if (j >= njobs) return;
  Av1gIntraJob job = jobs[j];
  av1g_intra_aux(&job, plane, aux + j);
}

__kernel void av1g_k_intra(__global const Av1gIntraJob *jobs,
                           __global const int *joff, int njobs, int total,
                           __global av1g_px *plane,
                           __global const Av1gIntraAux *aux, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gIntraJob job = jobs[job_idx];
  av1g_intra_px(&job, plane, aux + job_idx, elem % job.w, elem / job.w, maxv);
}

__kernel void av1g_k_inter(__global const Av1gInterJob *jobs,
                           __global const int *joff, int njobs, int total,
                           __global const av1g_px *ref0,
                           __global const av1g_px *ref1,
                           __global av1g_px *dst,
                           __global const int *filters, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gInterJob job = jobs[job_idx];
  av1g_inter_px(&job, dst, ref0, ref1, filters, elem % job.w, elem / job.w,
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
  av1g_cdef_px(&job, src, dst, elem & (AV1G_CDEF_DIM - 1),
               elem / AV1G_CDEF_DIM, maxv);
}

__kernel void av1g_k_grain(__global const Av1gGrainJob *jobs,
                           __global const int *joff, int njobs, int total,
                           __global av1g_px *dst,
                           __global const signed char *grain,
                           __global const int *lut, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gGrainJob job = jobs[job_idx];
  av1g_grain_px(&job, dst, grain, lut, elem % job.w, elem / job.w, maxv);
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
