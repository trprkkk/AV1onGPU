/*
 * AV1onGPU - device-portable AV1 reconstruction kernels
 *
 * This header is compiled three times:
 *   1. by the host C++ compiler  -> reference CPU backend (and unit tests)
 *   2. by the OpenCL compiler    -> kernels/av1_opencl.cl
 *   3. by NVRTC (CUDA C++)       -> kernels/av1_cuda.cu
 *
 * It therefore stays inside the intersection of C++17, OpenCL C 1.2 and
 * CUDA C++: no templates, no references, no overloading, no classes, no
 * standard library and no address space qualifiers.  All maths uses 32/64 bit
 * integer arithmetic so CPU, OpenCL and CUDA results are bit identical.
 *
 * Pixel storage is always 16 bit (`unsigned short`) regardless of the
 * sequence bit depth.  That keeps the device code branch free and lets 8 bit
 * and 10 bit streams share one kernel set.
 *
 * Every stage is expressed twice:
 *   av1g_xxx_px()   - one sample / one work item  (used by the GPU backends)
 *   av1g_xxx_block()- whole block loop           (used by the CPU backend and
 *                                                  by the unit tests)
 * Both call the same scalar code, so results are identical by construction.
 *
 * Copyright (c) 2026 AV1onGPU contributors. BSD-3-Clause.
 */
#ifndef AV1GPU_KERNELS_H
#define AV1GPU_KERNELS_H

/*
 * Address spaces: OpenCL C needs every pointer parameter to carry an address
 * space, and in OpenCL 1.2/2.0 an unqualified one means __private.  All large
 * buffers (planes, coefficient pools, tables) are global, so they are marked
 * with AV1G_GLOBAL.  Small scratch arrays stay private and never leave the
 * function that owns them.  The macro is empty for C++ and CUDA.
 */
#if defined(__OPENCL_VERSION__)
#define AV1G_FN static inline
#define AV1G_GLOBAL __global
#define AV1G_DEVICE 1
#elif defined(__CUDACC__)
#define AV1G_FN static __device__ __forceinline__
#define AV1G_GLOBAL
#define AV1G_DEVICE 1
#else
#define AV1G_FN static inline
#define AV1G_GLOBAL
#endif

/* whole-block helpers: used by the host reference path and the tests only */
#define AV1G_HOST_FN AV1G_FN

typedef unsigned short av1g_px; /* one plane sample, always 16 bit */

/* ------------------------------------------------------------------ */
/* scalar helpers                                                     */
/* ------------------------------------------------------------------ */

AV1G_FN int av1g_clip(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

AV1G_FN int av1g_abs(int v) { return v < 0 ? -v : v; }

AV1G_FN int av1g_clip_px(int v, int maxv) { return av1g_clip(v, 0, maxv); }

AV1G_FN int av1g_sign(int v) { return (v > 0) - (v < 0); }

/* round(x / 2^n) for any sign of x */
AV1G_FN int av1g_rshift_rnd(long long v, int n) {
  return (int)((v + (1LL << (n - 1))) >> n);
}

/* ------------------------------------------------------------------ */
/* job descriptors (identical layout on host and device)              */
/* ------------------------------------------------------------------ */

/* one 2D inverse transform followed by a residual add */
typedef struct {
  int coef_off;   /* offset (in ints) into the dequantised coefficient buffer */
  int tmp_off;    /* offset (in ints) into the intermediate buffer           */
  int dst_off;    /* offset of the top-left sample in the destination plane  */
  int dst_stride; /* destination plane stride in samples                     */
  int w, h;       /* transform size                                          */
  int row_basis;  /* offset (in ints) into the basis table, `w` rows         */
  int col_basis;  /* offset (in ints) into the basis table, `h` rows         */
  int eob;        /* coded coefficient count; 0 => the residual is zero      */
  int shift;      /* extra right shift after the second pass                 */
} Av1gItxJob;

/* per block scratch for intra prediction (computed by one thread) */
typedef struct {
  int dc;    /* DC predictor value                                      */
  int ac_dc; /* CFL: DC term of the luma AC contribution                */
  int avg;   /* CFL: average luma of the co-located block               */
} Av1gIntraAux;

/* one intra prediction block */
typedef struct {
  int dst_off;
  int dst_stride;
  int w, h;
  int mode;      /* AV1G_PRED_*                                          */
  int bitdepth;
  int have_left;
  int have_above;
  int cfl_alpha;    /* Q3 alpha for AV1G_PRED_CFL                          */
  int luma_off;     /* CFL: offset of the co-located luma samples          */
  int luma_stride;
  int ss_x, ss_y;   /* chroma subsampling shifts (0/1)                     */
} Av1gIntraJob;

/* one inter prediction block (single reference or compound) */
typedef struct {
  int dst_off;
  int dst_stride;
  int w, h;
  int ref0_off, ref0_stride; /* plane offset inside the reference pool   */
  int ref1_off, ref1_stride;
  int ref0_slot, ref1_slot;  /* pool slot index, resolved by the backend */
  int ref_x, ref_y;          /* patch position inside the reference plane */
  int ref_w, ref_h;          /* visible reference size, for clamping     */
  int border;                /* extra border samples available (0)       */
  int mv0_x, mv0_y;   /* 1/8 pel, already scaled to the current plane     */
  int mv1_x, mv1_y;
  int filt;           /* AV1G_FILT_*                                      */
  int compound;       /* AV1G_COMPOUND_*                                  */
  int w0, w1;         /* compound blend weights, sum == 64                */
  int bitdepth;
} Av1gInterJob;

/*
 * One deblocking edge segment.  `p_off` points at sample p1 of the first of
 * `len` (multiple of 4) sample groups; the edge runs along increasing
 * `stride` steps, i.e. p0 = p1 + stride and q0 = p1 + 2*stride.
 */
typedef struct {
  int p_off;      /* sample p1 of the first group                        */
  int stride;     /* step across the edge: p0 = p1 + stride, ...          */
  int step;       /* step along the edge between samples                 */
  int len;
  int level;      /* deblocking level for this edge (0 => skip)          */
  int sharpness;
  int filter_len; /* 4, 6 or 8 taps (14 is handled as 8)                 */
  int is_chroma;
  int bitdepth;
} Av1gDeblockJob;

/* one 8x8 CDEF block */
typedef struct {
  int src_off;
  int src_stride;
  int dst_off;
  int dst_stride;
  int pri_strength; /* 0 => copy src to dst                              */
  int sec_strength;
  int damping;
  int dir;          /* 0..7                                              */
  int bitdepth;
  int plane_w, plane_h; /* visible frame size, used to clamp the halo     */
} Av1gCdefJob;

/* CDEF works on 8x8 blocks */
#define AV1G_CDEF_DIM 8

/*
 * Film grain over a rectangular region.  `grain` is a signed pseudo-random
 * template of `grain_h` rows generated on the host, `lut` maps a luma range to
 * a Q6 gain.
 */
typedef struct {
  int dst_off, dst_stride;
  int x, y, w, h;
  int bitdepth;
  int grain_off;
  int grain_stride;
  int grain_h;
  int lut_off;
  int overlap; /* 0 none, 1 bottom edge only, 2 all edges               */
} Av1gGrainJob;

/* YUV -> RGB conversion of one rectangular region */
typedef struct {
  int dst_off;   /* offset into the interleaved RGB buffer              */
  int dst_stride;
  int w, h;
  int y_off, y_stride;
  int u_off, v_off, c_stride;
  int ss_x, ss_y;
  int m[8];      /* Q16: r=m0*Y+m1*V, g=m2*Y+m3*U+m4*V, b=m5*Y+m6*U */
  int o[3];      /* rounding + range offsets                            */
  int bitdepth;
} Av1gColorJob;

/* ------------------------------------------------------------------ */
/* enums shared by host and device                                    */
/* ------------------------------------------------------------------ */

#define AV1G_PRED_DC 0
#define AV1G_PRED_V 1
#define AV1G_PRED_H 2
#define AV1G_PRED_PAETH 3
#define AV1G_PRED_SMOOTH 4
#define AV1G_PRED_SMOOTH_V 5
#define AV1G_PRED_SMOOTH_H 6
#define AV1G_PRED_CFL 7

#define AV1G_FILT_REGULAR 0
#define AV1G_FILT_SMOOTH 1
#define AV1G_FILT_SHARP 2
#define AV1G_FILT_BILINEAR 3
#define AV1G_FILT_COUNT 4

#define AV1G_COMPOUND_NONE 0
#define AV1G_COMPOUND_AVG 1
#define AV1G_COMPOUND_DIST 2

/* ------------------------------------------------------------------ */
/* inverse transform                                                  */
/* ------------------------------------------------------------------ */

/*
 * 1-D inverse (synthesis) transform:
 *     out[i] = ( sum_k in[k] * B[i*n + k] + 2048 ) >> 12
 * B is the Q12 orthonormal synthesis matrix generated on the host from the
 * exact DCT-II / DST-IV definitions (see src/core/tables.cpp).
 */
AV1G_FN void av1g_itx_1d(const int *in, int *out, int n, AV1G_GLOBAL const int *basis) {
  /* `in` and `out` may alias (the column pass transforms in place), so the
   * result is accumulated in private storage before it is written back. */
  int tmp[64];
  int i;
  for (i = 0; i < n; ++i) {
    AV1G_GLOBAL const int *row = basis + i * n;
    long long acc = 0;
    int k;
    for (k = 0; k < n; ++k) acc += (long long)in[k] * (long long)row[k];
    tmp[i] = (int)((acc + 2048) >> 12);
  }
  for (i = 0; i < n; ++i) out[i] = tmp[i];
}

/* pass 1: transform coefficient row `y` along x into the scratch buffer */
AV1G_FN void av1g_itx_row(const Av1gItxJob *job, AV1G_GLOBAL const int *coef,
                         AV1G_GLOBAL int *tmp, AV1G_GLOBAL const int *basis, int y) {
  int src[64];
  int dst[64];
  int i;
  for (i = 0; i < job->w; ++i) src[i] = coef[job->coef_off + y * job->w + i];
  av1g_itx_1d(src, dst, job->w, basis + job->row_basis);
  for (i = 0; i < job->w; ++i) tmp[job->tmp_off + y * job->w + i] = dst[i];
}

/* pass 2: transform column `x` along y and add the residual to the plane */
AV1G_FN void av1g_itx_col_add(const Av1gItxJob *job, AV1G_GLOBAL const int *tmp,
                              AV1G_GLOBAL av1g_px *dst, AV1G_GLOBAL const int *basis,
                              int x, int maxv) {
  int col[64];
  int y;
  if (job->eob == 0) return;
  for (y = 0; y < job->h; ++y) col[y] = tmp[job->tmp_off + y * job->w + x];
  av1g_itx_1d(col, col, job->h, basis + job->col_basis);
  for (y = 0; y < job->h; ++y) {
    int r = col[y];
    AV1G_GLOBAL av1g_px *p = dst + job->dst_off + y * job->dst_stride + x;
    if (job->shift > 0) r = av1g_rshift_rnd((long long)r, job->shift);
    *p = (av1g_px)av1g_clip_px((int)*p + r, maxv);
  }
}

/* flat residual copy, used when a front-end already produced the residual */
AV1G_FN void av1g_add_residual_px(const Av1gItxJob *job, AV1G_GLOBAL const int *res,
                                  AV1G_GLOBAL av1g_px *dst, int x, int y, int maxv) {
  AV1G_GLOBAL av1g_px *p = dst + job->dst_off + y * job->dst_stride + x;
  *p = (av1g_px)av1g_clip_px((int)*p + res[job->coef_off + y * job->w + x], maxv);
}

/* ------------------------------------------------------------------ */
/* intra prediction                                                   */
/* ------------------------------------------------------------------ */

/* synthesised AV1 style smooth weights: a symmetric ramp over the block */
AV1G_FN int av1g_smooth_w(int i, int n) {
  return (int)(((long long)(i + 1) * 256) / (n + 1));
}

/* one thread per block computes the shared predictors */
AV1G_FN void av1g_intra_aux(const Av1gIntraJob *job, AV1G_GLOBAL const av1g_px *plane,
                            AV1G_GLOBAL Av1gIntraAux *aux) {
  const int w = job->w, h = job->h, stride = job->dst_stride;
  AV1G_GLOBAL const av1g_px *out = plane + job->dst_off;
  AV1G_GLOBAL const av1g_px *above = out - stride;
  AV1G_GLOBAL const av1g_px *left = out - 1;
  int x, y, sum = 0;

  if (job->have_above)
    for (x = 0; x < w; ++x) sum += (int)above[x];
  if (job->have_left)
    for (y = 0; y < h; ++y) sum += (int)left[y * stride];
  if (job->have_above && job->have_left)
    aux->dc = (sum + ((w + h) >> 1)) / (w + h);
  else if (job->have_above)
    aux->dc = (sum + (w >> 1)) / w;
  else if (job->have_left)
    aux->dc = (sum + (h >> 1)) / h;
  else
    aux->dc = 1 << (job->bitdepth - 1);

  aux->avg = 0;
  aux->ac_dc = 0;
  if (job->mode == AV1G_PRED_CFL) {
    const int lw = w << job->ss_x, lh = h << job->ss_y;
    const int n = lw * lh;
    int lx, ly;
    sum = 0;
    for (ly = 0; ly < lh; ++ly)
      for (lx = 0; lx < lw; ++lx)
        sum += (int)plane[job->luma_off + ly * job->luma_stride + lx];
    aux->avg = (sum + (n >> 1)) / n;
    {
      int dc_sum = 0;
      for (y = 0; y < h; ++y)
        for (x = 0; x < w; ++x)
          dc_sum += (int)plane[job->luma_off + (y << job->ss_y) * job->luma_stride +
                              (x << job->ss_x)] - aux->avg;
      aux->ac_dc = (dc_sum + ((w * h) >> 1)) / (w * h);
    }
  }
}

AV1G_FN void av1g_intra_px(const Av1gIntraJob *job, AV1G_GLOBAL av1g_px *plane,
                           AV1G_GLOBAL const Av1gIntraAux *aux, int x, int y, int maxv) {
  AV1G_GLOBAL av1g_px *out = plane + job->dst_off + y * job->dst_stride + x;
  AV1G_GLOBAL const av1g_px *above = plane + job->dst_off - job->dst_stride;
  AV1G_GLOBAL const av1g_px *left = plane + job->dst_off - 1;
  const int w = job->w, h = job->h, stride = job->dst_stride;
  /* edge blocks have no neighbours: fall back to the DC predictor */
  int mode = job->mode;
  if (!job->have_above && (mode == AV1G_PRED_V || mode == AV1G_PRED_SMOOTH_V ||
                           mode == AV1G_PRED_SMOOTH || mode == AV1G_PRED_PAETH))
    mode = AV1G_PRED_DC;
  if (!job->have_left && (mode == AV1G_PRED_H || mode == AV1G_PRED_SMOOTH_H ||
                          mode == AV1G_PRED_SMOOTH || mode == AV1G_PRED_PAETH))
    mode = AV1G_PRED_DC;
  const int base = (job->have_above && job->have_left) ? (int)above[-1] : aux->dc;

  switch (mode) {
    case AV1G_PRED_DC:
      *out = (av1g_px)aux->dc;
      break;
    case AV1G_PRED_V:
      *out = above[x];
      break;
    case AV1G_PRED_H:
      *out = left[y * stride];
      break;
    case AV1G_PRED_PAETH: {
      /* Paeth predictor: pick the neighbour closest to (left + above - base) */
      const int l = (int)left[y * stride];
      const int t = (int)above[x];
      const int p = l + t - base;
      const int dl = av1g_abs(p - l);
      const int dt = av1g_abs(p - t);
      const int db = av1g_abs(p - base);
      if (dl <= dt && dl <= db)
        *out = (av1g_px)l;
      else if (dt <= db)
        *out = (av1g_px)t;
      else
        *out = (av1g_px)base;
      break;
    }
    case AV1G_PRED_SMOOTH:
    case AV1G_PRED_SMOOTH_V:
    case AV1G_PRED_SMOOTH_H: {
      const int right = (int)above[w - 1];
      const int bottom = (int)left[(h - 1) * stride];
      const int wy = av1g_smooth_w(y, h);
      const int wx = av1g_smooth_w(x, w);
      const int t = (int)above[x];
      const int l = (int)left[y * stride];
      int pred;
      if (mode == AV1G_PRED_SMOOTH_V)
        pred = ((256 - wy) * t + wy * bottom + 128) >> 8;
      else if (mode == AV1G_PRED_SMOOTH_H)
        pred = ((256 - wx) * l + wx * right + 128) >> 8;
      else
        pred = (((256 - wx) * t + wx * right) + ((256 - wy) * l + wy * bottom) + 256) >> 9;
      *out = (av1g_px)av1g_clip_px(pred, maxv);
      break;
    }
    case AV1G_PRED_CFL: {
      const int v = (int)plane[job->luma_off + (y << job->ss_y) * job->luma_stride +
                               (x << job->ss_x)];
      const int ac = (v - aux->avg) - aux->ac_dc;
      *out = (av1g_px)av1g_clip_px(base + ((job->cfl_alpha * ac) >> 3), maxv);
      break;
    }
    default:
      *out = (av1g_px)(1 << (job->bitdepth - 1));
      break;
  }
}

#if !defined(AV1G_DEVICE)
AV1G_HOST_FN void av1g_intra_block(const Av1gIntraJob *job, av1g_px *plane,
                              Av1gIntraAux *aux, int maxv) {
  int x, y;
  av1g_intra_aux(job, plane, aux);
  for (y = 0; y < job->h; ++y)
    for (x = 0; x < job->w; ++x) av1g_intra_px(job, plane, aux, x, y, maxv);
}

#endif
/* ------------------------------------------------------------------ */
/* inter prediction                                                   */
/* ------------------------------------------------------------------ */

/*
 * Subpel filter tables are generated on the host and uploaded once:
 * layout filt -> 8 phases (1/8 pel) -> 8 taps, Q7, each row sums to 128.
 * Short filters (bilinear 2-tap, smooth 4-tap) are zero padded into the 8
 * tap slots so the device loop stays uniform.
 */
AV1G_FN int av1g_filt(AV1G_GLOBAL const int *tbl, int filt, int phase, int tap) {
  return tbl[filt * 8 * 8 + phase * 8 + tap];
}

/* fetch a reference sample with edge clamping */
AV1G_FN int av1g_fetch(AV1G_GLOBAL const av1g_px *p, int stride, int x, int y, int bx0,
                       int bx1, int by0, int by1) {
  if (x < bx0) x = bx0;
  if (x > bx1) x = bx1;
  if (y < by0) y = by0;
  if (y > by1) y = by1;
  return (int)p[y * stride + x];
}

AV1G_FN int av1g_inter_tap(const Av1gInterJob *job, AV1G_GLOBAL const av1g_px *ref,
                           int ref_off, int ref_stride, int x, int y, int mvx,
                           int mvy, AV1G_GLOBAL const int *filters) {
  const int xs = mvx >> 3, xf = mvx & 7;
  const int ys = mvy >> 3, yf = mvy & 7;
  const int bx0 = -job->border, bx1 = job->ref_w - 1 + job->border;
  const int by0 = -job->border, by1 = job->ref_h - 1 + job->border;
  /* absolute position of the patch origin inside the reference plane */
  const int ax = job->ref_x + x + xs;
  const int ay = job->ref_y + y + ys;
  int acc = 0, i, j;
  for (j = 0; j < 8; ++j) {
    int rowacc = 0;
    for (i = 0; i < 8; ++i) {
      const int s = av1g_fetch(ref + ref_off, ref_stride, ax - 3 + i, ay - 3 + j,
                               bx0, bx1, by0, by1);
      rowacc += s * av1g_filt(filters, job->filt, xf, i);
    }
    acc += ((rowacc + 64) >> 7) * av1g_filt(filters, job->filt, yf, j);
  }
  return (acc + 64) >> 7;
}

AV1G_FN void av1g_inter_px(const Av1gInterJob *job, AV1G_GLOBAL av1g_px *dst,
                           AV1G_GLOBAL const av1g_px *ref0,
                           AV1G_GLOBAL const av1g_px *ref1,
                           AV1G_GLOBAL const int *filters, int x, int y, int maxv) {
  const int v0 = av1g_inter_tap(job, ref0, job->ref0_off, job->ref0_stride, x, y,
                                job->mv0_x, job->mv0_y, filters);
  int acc = v0;
  if (job->compound != AV1G_COMPOUND_NONE) {
    const int v1 = av1g_inter_tap(job, ref1, job->ref1_off, job->ref1_stride, x, y,
                                  job->mv1_x, job->mv1_y, filters);
    if (job->compound == AV1G_COMPOUND_AVG)
      acc = (v0 + v1 + 1) >> 1;
    else
      acc = (job->w0 * v0 + job->w1 * v1 + 32) >> 6;
  }
  dst[job->dst_off + y * job->dst_stride + x] = (av1g_px)av1g_clip_px(acc, maxv);
}

#if !defined(AV1G_DEVICE)
AV1G_HOST_FN void av1g_inter_block(const Av1gInterJob *job, av1g_px *dst,
                              const av1g_px *ref0, const av1g_px *ref1,
                              const int *filters, int maxv) {
  int x, y;
  for (y = 0; y < job->h; ++y)
    for (x = 0; x < job->w; ++x)
      av1g_inter_px(job, dst, ref0, ref1, filters, x, y, maxv);
}

#endif
/* ------------------------------------------------------------------ */
/* deblocking filter                                                  */
/* ------------------------------------------------------------------ */

/* spec-style threshold derivation from level and sharpness */
AV1G_FN void av1g_thresholds(int level, int sharpness, int bitdepth, int *limit,
                             int *blimit, int *thresh) {
  const int shift = bitdepth - 8;
  const int base = (level >> 4) * 4 + (level & 15);
  int l = base - (((level & 15) == 0) ? 1 : 0);
  if (l < 1) l = 1;
  if (sharpness > 0 && l > 4) l -= (sharpness >> 2);
  *limit = (l + 2) << shift;
  *blimit = ((l + 2) * 3) << shift;
  *thresh = (l >> 2) << shift;
  if (*thresh < 1) *thresh = 1;
}

/*
 * Filters the samples of one group of four along an edge.  p1 is at
 * job->p_off, p0 one stride step later, q0 two, q1 three.
 */
AV1G_FN void av1g_deblock_group(const Av1gDeblockJob *job,
                                AV1G_GLOBAL av1g_px *plane, int g, int maxv) {
  const int stride = job->stride;
  int limit, blimit, thresh;
  int k;

  if (job->level == 0) return;
  av1g_thresholds(job->level, job->sharpness, job->bitdepth, &limit, &blimit, &thresh);

  for (k = 0; k < 4; ++k) {
    AV1G_GLOBAL av1g_px *base = plane + job->p_off + (g + k) * job->step;
    const int p1 = (int)base[0];
    const int p0 = (int)base[stride];
    const int q0 = (int)base[2 * stride];
    const int q1 = (int)base[3 * stride];
    const int p2 = (job->filter_len >= 6) ? (int)base[-stride] : p1;
    const int q2 = (job->filter_len >= 6) ? (int)base[4 * stride] : q1;
    int f, hev;

    /* skip flat areas and real image edges */
    if (av1g_abs(p0 - q0) * 2 + (av1g_abs(p1 - q1) >> 1) > blimit) continue;

    {
      const int f0 = av1g_clip(av1g_abs(p1 - q0), 0, 128);
      const int f1 = av1g_clip(av1g_abs(q1 - p0), 0, 128);
      f = av1g_clip(av1g_sign(p0 - q0) * ((f0 + f1 + 1) >> 1), -limit, limit);
    }
    hev = (av1g_abs(p1 - p0) > thresh) || (av1g_abs(q1 - q0) > thresh);

    if (job->filter_len == 4 || !hev) {
      /* narrow filter: only the two samples closest to the edge change */
      base[stride] = (av1g_px)av1g_clip_px(p0 - f, maxv);
      base[2 * stride] = (av1g_px)av1g_clip_px(q0 + f, maxv);
    } else {
      const int p0n = av1g_clip_px(p0 - f, maxv);
      const int q0n = av1g_clip_px(q0 + f, maxv);
      base[stride] = (av1g_px)p0n;
      base[2 * stride] = (av1g_px)q0n;
      if (job->filter_len >= 6) {
        const int mid = (p0n + q0n + 1) >> 1;
        const int hp = av1g_clip((p2 + mid - (p1 << 1)) >> 1, -limit, limit);
        const int hq = av1g_clip((q2 + mid - (q1 << 1)) >> 1, -limit, limit);
        base[0] = (av1g_px)av1g_clip_px(p1 + hp, maxv);
        base[3 * stride] = (av1g_px)av1g_clip_px(q1 + hq, maxv);
      }
    }
  }
}

#if !defined(AV1G_DEVICE)
AV1G_HOST_FN void av1g_deblock_job(const Av1gDeblockJob *job, av1g_px *plane, int maxv) {
  int g;
  for (g = 0; g < job->len; g += 4) av1g_deblock_group(job, plane, g, maxv);
}

#endif
/* ------------------------------------------------------------------ */
/* CDEF                                                               */
/* ------------------------------------------------------------------ */

AV1G_FN void av1g_cdef_px(const Av1gCdefJob *job, AV1G_GLOBAL const av1g_px *src,
                          AV1G_GLOBAL av1g_px *dst, int x, int y, int maxv) {
  /* direction vectors (dx, dy) for the 8 CDEF directions */
  const int dirx[8] = {1, 1, 1, -1, -1, -1, 0, 0};
  const int diry[8] = {1, 0, -1, -1, 0, 1, 1, -1};
  const int pri_taps[2] = {4, 2};
  const int sec_taps[2] = {2, 1};
  const int px = job->dst_off % job->dst_stride + x; /* frame coordinates */
  const int py = job->dst_off / job->dst_stride + y;
  const int src_base = job->src_off - job->dst_off; /* same frame, shifted */
  const int center = (int)src[job->src_off + y * job->src_stride + x];
  int sum = 0, count = 0, lo = center, hi = center;
  int k, s, d2;

  if (job->pri_strength > 0) {
    for (k = 0; k < 2; ++k) {
      for (s = -1; s <= 1; s += 2) {
        const int xx = px + dirx[job->dir] * (k + 1) * s;
        const int yy = py + diry[job->dir] * (k + 1) * s;
        int v;
        if (xx < 0 || xx >= job->plane_w || yy < 0 || yy >= job->plane_h) continue;
        v = (int)src[src_base + yy * job->src_stride + xx];
        /* the second tap is twice as far away: the spec halves its threshold */
        {
          const int strength = k ? (job->pri_strength >> 1) : job->pri_strength;
          sum += pri_taps[k] * av1g_clip(center - v, -strength, strength);
        }
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        ++count;
      }
    }
    if (job->sec_strength > 0) {
      for (d2 = 0; d2 < 2; ++d2) {
        const int nd = (job->dir + (d2 ? 6 : 2)) & 7;
        for (k = 0; k < 2; ++k) {
          for (s = -1; s <= 1; s += 2) {
            const int xx = px + dirx[nd] * (k + 1) * s;
            const int yy = py + diry[nd] * (k + 1) * s;
            int v;
            if (xx < 0 || xx >= job->plane_w || yy < 0 || yy >= job->plane_h) continue;
            v = (int)src[src_base + yy * job->src_stride + xx];
            {
              const int strength = k ? (job->sec_strength >> 1) : job->sec_strength;
              sum += sec_taps[k] * av1g_clip(center - v, -strength, strength);
            }
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            ++count;
          }
        }
      }
    }
  }

  {
    int out = center;
    if (count > 0) out = av1g_clip_px(av1g_clip(center + av1g_rshift_rnd((long long)sum, 4),
                                                lo, hi), maxv);
    dst[job->dst_off + y * job->dst_stride + x] = (av1g_px)out;
  }
}

#if !defined(AV1G_DEVICE)
AV1G_HOST_FN void av1g_cdef_block(const Av1gCdefJob *job, const av1g_px *src,
                             av1g_px *dst, int maxv) {
  int x, y;
  for (y = 0; y < 8; ++y)
    for (x = 0; x < 8; ++x)
      av1g_cdef_px(job, src, dst, x, y, maxv);
}

#endif
/* ------------------------------------------------------------------ */
/* film grain                                                         */
/* ------------------------------------------------------------------ */

AV1G_FN void av1g_grain_px(const Av1gGrainJob *job, AV1G_GLOBAL av1g_px *dst,
                           AV1G_GLOBAL const signed char *grain,
                           AV1G_GLOBAL const int *lut, int x, int y, int maxv) {
  const int shift = job->bitdepth - 8;
  const int idx = (job->y + y) % job->grain_h;
  const int gx = (job->x + x) % job->grain_stride;
  const int t = (int)grain[job->grain_off + idx * job->grain_stride + gx];
  AV1G_GLOBAL av1g_px *p = dst + job->dst_off + y * job->dst_stride + x;
  const int cur = (int)*p;
  const int scale = lut[job->lut_off + av1g_clip((cur >> shift) >> 2, 0, 63)];
  int g = av1g_rshift_rnd((long long)t * scale, 6);
  /* fade the grain towards block borders to avoid visible seams */
  if (job->overlap) {
    const int ex = (x < (job->w - 1 - x)) ? x : (job->w - 1 - x);
    const int ey = (job->overlap == 2)
                       ? ((y < (job->h - 1 - y)) ? y : (job->h - 1 - y))
                       : y;
    if (ex < 2 || ey < 2) g = (g * 2 + 1) >> 2;
  }
  *p = (av1g_px)av1g_clip_px(cur + g, maxv);
}

#if !defined(AV1G_DEVICE)
AV1G_HOST_FN void av1g_grain_block(const Av1gGrainJob *job, av1g_px *dst,
                              const signed char *grain, const int *lut, int maxv) {
  int x, y;
  for (y = 0; y < job->h; ++y)
    for (x = 0; x < job->w; ++x) av1g_grain_px(job, dst, grain, lut, x, y, maxv);
}

#endif
/* ------------------------------------------------------------------ */
/* YUV -> RGB                                                         */
/* ------------------------------------------------------------------ */

AV1G_FN void av1g_color_px(const Av1gColorJob *job, AV1G_GLOBAL const av1g_px *y_plane,
                           AV1G_GLOBAL const av1g_px *u_plane,
                           AV1G_GLOBAL const av1g_px *v_plane,
                           AV1G_GLOBAL unsigned char *rgb, int x, int y) {
  /* range/level offsets are folded into job->o[] by the host */
  const int yy = (int)y_plane[job->y_off + y * job->y_stride + x];
  const int uu = (int)u_plane[job->u_off + (y >> job->ss_y) * job->c_stride +
                              (x >> job->ss_x)];
  const int vv = (int)v_plane[job->v_off + (y >> job->ss_y) * job->c_stride +
                              (x >> job->ss_x)];
  const int r = (job->m[0] * yy + job->m[1] * vv + job->o[0]) >> 16;
  const int g = (job->m[2] * yy + job->m[3] * uu + job->m[4] * vv + job->o[1]) >> 16;
  const int b = (job->m[5] * yy + job->m[6] * uu + job->o[2]) >> 16;
  AV1G_GLOBAL unsigned char *out = rgb + (job->dst_off + y * job->dst_stride + x) * 3;
  out[0] = (unsigned char)av1g_clip(r, 0, 255);
  out[1] = (unsigned char)av1g_clip(g, 0, 255);
  out[2] = (unsigned char)av1g_clip(b, 0, 255);
}

#if !defined(AV1G_DEVICE)
AV1G_HOST_FN void av1g_color_block(const Av1gColorJob *job, const av1g_px *y_plane,
                              const av1g_px *u_plane, const av1g_px *v_plane,
                              unsigned char *rgb) {
  int x, y;
  for (y = 0; y < job->h; ++y)
    for (x = 0; x < job->w; ++x)
      av1g_color_px(job, y_plane, u_plane, v_plane, rgb, x, y);
}

#endif
#endif /* AV1GPU_KERNELS_H */
