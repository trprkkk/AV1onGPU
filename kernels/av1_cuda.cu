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

/* The job structs are large - the inter job is nearly a kilobyte - and a copy
 * of one lands in every thread's stack frame, which is local memory: the
 * kernel then writes those bytes once per work item and reads them back.
 * The jobs are read-only, so the kernels work on the array in place instead. */
#ifndef AV1G_INTRA_BLOCK_PER_JOB
#define AV1G_INTRA_BLOCK_PER_JOB 0
#endif

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
  const Av1gItxJob *job = &jobs[job_idx];
  if (job->eob <= 0) return;
  /* one thread per transform row (element count per job is job.h) */
  av1g_itx_row(job, coef, tmp, elem);
}

extern "C" __global__ void av1g_k_itx_cols(const Av1gItxJob *jobs,
                                           const int *joff, int njobs, int total,
                                           const int *tmp, av1g_px *dst,
                                           int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gItxJob *job = &jobs[job_idx];
  if (job->eob <= 0) return;
  /* one thread per transform column (element count per job is job.w):
   * av1g_itx_col_add accumulates, so it must run exactly once per column */
  av1g_itx_col_add(job, tmp, dst, elem, maxv);
}

/* One thread per row of a block: intra prediction of a row only reads the
 * block's edge (which every row thread gathers identically), so the rows of a
 * block run in parallel. The element count per job is job.h.
 *
 * The other mapping puts one thread block on every transform block and shares
 * the edge through shared memory: the edge preparation is then done once per
 * block instead of once per row, but a block of a few rows wastes most of its
 * threads, so which one is faster depends on the material. */
extern "C" __global__ void av1g_k_intra(const Av1gIntraJob *jobs,
                                        const int *joff, int njobs, int total,
                                        av1g_px *plane,
                                        const unsigned char *palette, int maxv) {
#if AV1G_INTRA_BLOCK_PER_JOB
  (void)joff;
  (void)total;
  const int job_idx = (int)blockIdx.x;
  if (job_idx >= njobs) return;
  const Av1gIntraJob *job = &jobs[job_idx];
  const int h = job->h;

  /* The two branches below are decided by the job alone, so every thread of
   * the block takes the same one: a barrier below may not be divergent. */
  if (job->palette) {
    /* Palette prediction: a table lookup per sample, no edge at all. */
    for (int r = (int)threadIdx.x; r < h; r += (int)blockDim.x)
      av1g_intra_palette_row(job, plane, palette, r);
    return;
  }
  if (job->mode == AV1G_PRED_CFL ||
      (job->filter_intra && job->w >= 4 && job->h >= 2)) {
    /* The recursive predictor and chroma from luma read the whole block. */
    if (threadIdx.x == 0) av1g_intra_block(job, plane, palette, maxv);
    return;
  }

  __shared__ int s_above[AV1G_INTRA_EDGE];
  __shared__ int s_left[AV1G_INTRA_EDGE];
  __shared__ int s_pangle;
  __shared__ int s_upsample_above;
  __shared__ int s_upsample_left;
  int *const above = s_above + AV1G_INTRA_EDGE_OFF;
  int *const left = s_left + AV1G_INTRA_EDGE_OFF;
  if (threadIdx.x == 0) {
    av1g_intra_prep(job, plane, above, left, maxv, &s_pangle, &s_upsample_above,
                    &s_upsample_left);
  }
  __syncthreads();
  for (int r = (int)threadIdx.x; r < h; r += (int)blockDim.x)
    av1g_intra_row_prep(job, plane, maxv, r, above, left, s_pangle, s_upsample_above,
                        s_upsample_left);
#else
  AV1G_JOB_LOOKUP(job_idx, row);
  const Av1gIntraJob *job = &jobs[job_idx];
  av1g_intra_row(job, plane, palette, maxv, row);
#endif
}

extern "C" __global__ void av1g_k_inter(const Av1gInterJob *jobs,
                                        const int *joff, int njobs, int total,
                                        const av1g_px *cur,
                                        const av1g_px *pool, av1g_px *dst,
                                        const int *filters, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gInterJob *job = &jobs[job_idx];
  /* every reference is an absolute offset into the pool; a job that names no
   * slot reads the picture being decoded */
  av1g_inter_px(job, dst, cur, pool, filters, elem % job->w, elem / job->w,
                maxv);
}

extern "C" __global__ void av1g_k_add_residual(const Av1gItxJob *jobs,
                                               const int *joff, int njobs,
                                               int total, const int *res,
                                               av1g_px *dst, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gItxJob *job = &jobs[job_idx];
  av1g_add_residual_px(job, res, dst, elem % job->w, elem / job->w, maxv);
}

extern "C" __global__ void av1g_k_deblock(const Av1gDeblockJob *jobs,
                                          const int *joff, int njobs, int total,
                                          av1g_px *plane, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gDeblockJob *job = &jobs[job_idx];
  /* elem is a group index; the group helper takes a sample offset */
  av1g_deblock_group(job, plane, elem * 4, maxv);
}

extern "C" __global__ void av1g_k_cdef(const Av1gCdefJob *jobs, const int *joff,
                                       int njobs, int total,
                                       const av1g_px *src, av1g_px *dst,
                                       int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gCdefJob *job = &jobs[job_idx];
  /* one work item per block: the direction, the strength it earns and the
   * copy for a block with no strength are all per block, not per sample */
  /* src and dst are the picture; the luma of the same snapshot is where the
   * direction is read from, at the job's luma_off */
  av1g_cdef_block(job, src, dst, src, maxv);
}

/* Superres: one thread per output sample, from the coded picture into the
 * upscaled one. */
extern "C" __global__ void av1g_k_upscale(const Av1gUpscaleJob *jobs,
                                          const int *joff, int njobs, int total,
                                          const av1g_px *src, av1g_px *dst,
                                          const int *filters, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gUpscaleJob *job = &jobs[job_idx];
  av1g_upscale_px(job, src, dst, filters, elem % job->upscaled_w,
                  elem / job->upscaled_w, maxv);
}

/* Loop restoration: one thread per sample, from the two source pictures (the
 * CDEF output and the deblocked picture before it) into the restored one. */
extern "C" __global__ void av1g_k_lr(const Av1gLrJob *jobs,
                                     const int *joff, int njobs, int total,
                                     const av1g_px *cdef, const av1g_px *curr,
                                     av1g_px *dst, int maxv) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gLrJob *job = &jobs[job_idx];
  av1g_lr_px(job, cdef, curr, dst, elem % job->w, elem / job->w, maxv);
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
  const Av1gGrainJob *job = &jobs[job_idx];
  av1g_grain_px(job, dst, luma, noise, lut, elem % job->w, elem / job->w, maxv);
}

extern "C" __global__ void av1g_k_color(const Av1gColorJob *jobs,
                                        const int *joff, int njobs, int total,
                                        const av1g_px *y_plane,
                                        const av1g_px *u_plane,
                                        const av1g_px *v_plane,
                                        unsigned char *rgb) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gColorJob *job = &jobs[job_idx];
  av1g_color_px(job, y_plane, u_plane, v_plane, rgb, elem % job->w,
                elem / job->w);
}

/* One thread per output byte; the padding bytes of a row are written too, so a
 * job owns pitch * (h + ch) of them. */
extern "C" __global__ void av1g_k_nv12(const Av1gNv12Job *jobs,
                                       const int *joff, int njobs, int total,
                                       const av1g_px *y_plane,
                                       const av1g_px *u_plane,
                                       const av1g_px *v_plane,
                                       unsigned char *dst) {
  AV1G_JOB_LOOKUP(job_idx, elem);
  const Av1gNv12Job *job = &jobs[job_idx];
  dst[elem] = av1g_nv12_px(job, y_plane, u_plane, v_plane, elem % job->pitch,
                           elem / job->pitch);
}

/* Combined per-wave kernel: intra -> itx_cols -> residual in ONE launch.
 * Phases separated by __syncthreads(). Reduces 3 launches/wave to 1. */
extern "C" __global__ void av1g_k_wave3(
    const Av1gIntraJob *intra_jobs, const int *intra_pref, int intra_njobs, int intra_total,
    const Av1gItxJob *itx_jobs, const int *itx_col_pref, int itx_njobs, int itx_total,
    const Av1gItxJob *res_jobs, const int *res_pref, int res_njobs, int res_total,
    av1g_px *plane, const unsigned char *palette,
    const int *tmp, const int *res_buf, int maxv) {
  for (int g = threadIdx.x; g < intra_total; g += blockDim.x) {
    int lo = 0, hi = intra_njobs - 1;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (intra_pref[mid + 1] <= g) lo = mid + 1; else hi = mid;
    }
    av1g_intra_row(&intra_jobs[lo], plane, palette, maxv, g - intra_pref[lo]);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < itx_total; g += blockDim.x) {
    int lo = 0, hi = itx_njobs - 1;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (itx_col_pref[mid + 1] <= g) lo = mid + 1; else hi = mid;
    }
    const Av1gItxJob *job = &itx_jobs[lo];
    if (job->eob > 0)
      av1g_itx_col_add(job, tmp, plane, g - itx_col_pref[lo], maxv);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < res_total; g += blockDim.x) {
    int lo = 0, hi = res_njobs - 1;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (res_pref[mid + 1] <= g) lo = mid + 1; else hi = mid;
    }
    const Av1gItxJob *job = &res_jobs[lo];
    int elem = g - res_pref[lo];
    av1g_add_residual_px(job, res_buf, plane, elem % job->w, elem / job->w, maxv);
  }
}

/* ------------------------------------------------------------------ */
/* Persistent wave kernel: processes all waves in ONE launch.          */
/*                                                                     */
/* Grid-stride design: grid = num_SMs blocks (all co-resident),        */
/* block = 1024 threads. Each thread strides over elements of each      */
/* wave. Between waves a software barrier (atomicAdd + spin with       */
/* nanosleep backoff) ensures wave N completes before wave N+1 starts. */
/*                                                                     */
/* bounds is packed: [intra_begin(nw+1), intra_end(nw+1),               */
/*                   itx_begin(nw+1), itx_end(nw+1),                   */
/*                   res_begin(nw+1), res_end(nw+1)]                   */
/* ------------------------------------------------------------------ */
extern "C" __global__ void av1g_k_waves(
    const Av1gIntraJob *intra_jobs, const int *intra_pref,
    const Av1gItxJob *itx_jobs, const int *itx_col_pref,
    const Av1gItxJob *res_jobs, const int *res_pref,
    const int *bounds,
    int n_waves, int *barrier,
    av1g_px *plane, const unsigned char *palette,
    const int *tmp, const int *res_buf, int maxv) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gstride = gridDim.x * blockDim.x;
  const int nblocks = gridDim.x;
  const int nw1 = n_waves + 1;
  const int *wib = bounds;
  const int *wie = bounds + nw1;
  const int *wtb = bounds + 2 * nw1;
  const int *wte = bounds + 3 * nw1;
  const int *wrb = bounds + 4 * nw1;
  const int *wre = bounds + 5 * nw1;

  for (int w = 0; w < n_waves; ++w) {
    /* ---- intra phase ---- */
    {
      const int jb = wib[w], je = wie[w];
      if (jb < je) {
        const int base = jb > 0 ? intra_pref[jb - 1] : 0;
        const int total = intra_pref[je] - base;
        const int njobs = je - jb;
        for (int g = gid; g < total; g += gstride) {
          int lo = 0, hi = njobs - 1;
          while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (intra_pref[jb + mid + 1] - base <= g) lo = mid + 1;
            else hi = mid;
          }
          const int job_idx = jb + lo;
          const int elem = g - (intra_pref[job_idx] - base);
          av1g_intra_row(&intra_jobs[job_idx], plane, palette, maxv, elem);
        }
      }
    }
    /* ---- itx_cols phase ---- */
    {
      const int jb = wtb[w], je = wte[w];
      if (jb < je) {
        const int base = jb > 0 ? itx_col_pref[jb - 1] : 0;
        const int total = itx_col_pref[je] - base;
        const int njobs = je - jb;
        for (int g = gid; g < total; g += gstride) {
          int lo = 0, hi = njobs - 1;
          while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (itx_col_pref[jb + mid + 1] - base <= g) lo = mid + 1;
            else hi = mid;
          }
          const int job_idx = jb + lo;
          const int elem = g - (itx_col_pref[job_idx] - base);
          const Av1gItxJob *job = &itx_jobs[job_idx];
          if (job->eob > 0)
            av1g_itx_col_add(job, tmp, plane, elem, maxv);
        }
      }
    }
    /* ---- residual phase ---- */
    {
      const int jb = wrb[w], je = wre[w];
      if (jb < je) {
        const int base = jb > 0 ? res_pref[jb - 1] : 0;
        const int total = res_pref[je] - base;
        const int njobs = je - jb;
        for (int g = gid; g < total; g += gstride) {
          int lo = 0, hi = njobs - 1;
          while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (res_pref[jb + mid + 1] - base <= g) lo = mid + 1;
            else hi = mid;
          }
          const int job_idx = jb + lo;
          const int elem = g - (res_pref[job_idx] - base);
          const Av1gItxJob *job = &res_jobs[job_idx];
          av1g_add_residual_px(job, res_buf, plane, elem % job->w, elem / job->w, maxv);
        }
      }
    }
    /* ---- software barrier between waves ---- */
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(barrier, 1);
      volatile int *vb = barrier;
      const int target = nblocks * (w + 1);
      while (*vb < target) {
        __nanosleep(50);
      }
    }
    __syncthreads();
  }
}

/* ---------------------------------------------------------------------------
 * Device side dequantisation: the entry point for av1g_dequant_one (the
 * maths, which the OpenCL entry point shares).
 *
 * The host path expands every transform's coded levels into the raster pool
 * and uploads the whole pool: at 4K that is ~25 MB a frame of PCIe traffic on
 * top of the ~10 ms the expansion costs the CPU. Here the frame's levels (the
 * coded size of the picture, a few MB) and the per transform dequantiser the
 * frontend precomputed go up instead, and the raster is written where the itx
 * kernels read it - their access pattern does not change.
 *
 * One thread per transform: the job carries level_offset, qindex, plane,
 * dc_step, ac_step, denominator and qm_off, so there is no prefix table to
 * build and no binary search to find the work.
 */
extern "C" __global__ void av1g_k_dequant(const Av1gItxJob *jobs, int njobs,
                                          const short *levels,
                                          const unsigned char *qm,
                                          const unsigned short *scan2d,
                                          const unsigned short *scanrow,
                                          const unsigned short *scancol,
                                          int *coef) {
  const int t = AV1G_GID;
  if (t >= njobs) return;
  av1g_dequant_one(&jobs[t], levels, qm, scan2d, scanrow, scancol, coef);
}
