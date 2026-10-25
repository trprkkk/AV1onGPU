/* AV1onGPU - CUDA entry points.
 *
 * Same maths as the OpenCL file (kernels/av1gpu_kernels.h is shared with the
 * CPU reference path), compiled with NVRTC at run time.  One thread per output
 * element; the owning job is located with a binary search over the prefix-sum
 * table because jobs own different element counts.
 *
 * This is the interesting case for the project goal: it needs no AV1 fixed
 * function block, so any CUDA capable GPU (or an old NVIDIA card with only
 * basic compute) can decode with its shaders instead of a hardware engine.
 */

#include "av1gpu_kernels.h"

#define AV1G_GID ((int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x)

#define AV1G_JOB_LOOKUP(job, elem)                                            \
  const int _gid = AV1G_GID;                                                  \
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

extern "C" __global__ void av1g_k_itx_rows(const Av1gItxJob *jobs,
                                           const int *joff, int njobs, int total,
                                           const int *coef, int *tmp) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  if (job.eob <= 0) return;
  /* one thread per transform row (element count per job is job.h) */
  av1g_itx_row(&job, coef, tmp, elem);
}

extern "C" __global__ void av1g_k_itx_cols(const Av1gItxJob *jobs,
                                           const int *joff, int njobs, int total,
                                           const int *tmp, av1g_px *dst,
                                           int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  if (job.eob <= 0) return;
  /* one thread per transform column (element count per job is job.w):
   * av1g_itx_col_add accumulates, so it must run exactly once per column */
  av1g_itx_col_add(&job, tmp, dst, elem, maxv);
}

/* One thread per block: intra prediction is a whole-block operation, since the
 * edge filtering and the recursive predictor read well past one sample. */
extern "C" __global__ void av1g_k_intra(const Av1gIntraJob *jobs, int njobs,
                                        av1g_px *plane,
                                        const unsigned char *palette, int maxv) {
  const int j = AV1G_GID;
  if (j >= njobs) return;
  Av1gIntraJob job = jobs[j];
  av1g_intra_block(&job, plane, palette, maxv);
}

extern "C" __global__ void av1g_k_inter(const Av1gInterJob *jobs,
                                        const int *joff, int njobs, int total,
                                        const av1g_px *cur,
                                        const av1g_px *pool, av1g_px *dst,
                                        const int *filters, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gInterJob job = jobs[job_idx];
  /* every reference is an absolute offset into the pool; a job that names no
   * slot reads the picture being decoded */
  av1g_inter_px(&job, dst, cur, pool, filters, elem % job.w, elem / job.w,
                maxv);
}

extern "C" __global__ void av1g_k_add_residual(const Av1gItxJob *jobs,
                                               const int *joff, int njobs,
                                               int total, const int *res,
                                               av1g_px *dst, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gItxJob job = jobs[job_idx];
  av1g_add_residual_px(&job, res, dst, elem % job.w, elem / job.w, maxv);
}

extern "C" __global__ void av1g_k_deblock(const Av1gDeblockJob *jobs,
                                          const int *joff, int njobs, int total,
                                          av1g_px *plane, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gDeblockJob job = jobs[job_idx];
  /* elem is a group index; the group helper takes a sample offset */
  av1g_deblock_group(&job, plane, elem * 4, maxv);
}

extern "C" __global__ void av1g_k_cdef(const Av1gCdefJob *jobs, const int *joff,
                                       int njobs, int total,
                                       const av1g_px *src, av1g_px *dst,
                                       int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gCdefJob job = jobs[job_idx];
  /* one work item per block: the direction, the strength it earns and the
   * copy for a block with no strength are all per block, not per sample */
  /* src and dst are the picture; the luma of the same snapshot is where the
   * direction is read from, at the job's luma_off */
  av1g_cdef_block(&job, src, dst, src, maxv);
}

/* Superres: one thread per output sample, from the coded picture into the
 * upscaled one. */
extern "C" __global__ void av1g_k_upscale(const Av1gUpscaleJob *jobs,
                                          const int *joff, int njobs, int total,
                                          const av1g_px *src, av1g_px *dst,
                                          const int *filters, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gUpscaleJob job = jobs[job_idx];
  av1g_upscale_px(&job, src, dst, filters, elem % job.upscaled_w,
                  elem / job.upscaled_w, maxv);
}

/* Film grain: one thread per sample. `luma` is the same picture (the chroma
 * blend reads the luma samples before any grain was added, which is why the
 * chroma jobs are launched first). */
extern "C" __global__ void av1g_k_grain(const Av1gGrainJob *jobs,
                                        const int *joff, int njobs, int total,
                                        av1g_px *dst, const av1g_px *luma,
                                        const short *noise,
                                        const int *lut, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gGrainJob job = jobs[job_idx];
  av1g_grain_px(&job, dst, luma, noise, lut, elem % job.w, elem / job.w, maxv);
}

extern "C" __global__ void av1g_k_color(const Av1gColorJob *jobs,
                                        const int *joff, int njobs, int total,
                                        const av1g_px *y_plane,
                                        const av1g_px *u_plane,
                                        const av1g_px *v_plane,
                                        unsigned char *rgb) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  Av1gColorJob job = jobs[job_idx];
  av1g_color_px(&job, y_plane, u_plane, v_plane, rgb, elem % job.w,
                elem / job.w);
}
