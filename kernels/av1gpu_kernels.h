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

/* the specification's transform and prediction tables (generated) */
#include "av1gpu_tables.h"

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
  int tx_type;    /* AV1G_TX_*: which pair of 1D transforms                 */
  int eob;        /* coded coefficient count; 0 => the residual is zero      */
  int row_shift;  /* Transform_Row_Shift[txSz]; 0 when lossless             */
  int lossless;
  int bitdepth;
} Av1gItxJob;

/* One intra predicted transform block. Intra prediction is per transform
 * block, not per coding block: a 16x16 block coded with 4x4 transforms runs
 * four predictions, each seeing the samples the previous ones wrote. */
typedef struct {
  int dst_off;
  int dst_stride;
  int x, y;          /* position inside the plane, in samples               */
  int w, h;
  int mode;          /* AV1G_PRED_*                                        */
  int angle;         /* angle delta, in 3 degree steps                     */
  int bitdepth;
  int have_left;
  int have_above;
  int have_above_right; /* the row above continues past this block         */
  int have_below_left;  /* the column to the left continues below it       */
  int filter_type;   /* 1 when a neighbour used a smooth mode; -1 = no edge
                        filtering at all (used by tests and by callers that
                        already know the edge is not to be filtered) */
  int filter_intra;  /* 1: use the recursive filter intra predictor        */
  int filter_intra_mode;
  int cfl_alpha;     /* Q3 alpha for AV1G_PRED_CFL                         */
  int luma_off;      /* CFL: offset of the co-located luma samples         */
  int luma_stride;
  int luma_w, luma_h; /* CFL: the co-located luma block's right/bottom edge */
  int ss_x, ss_y;    /* chroma subsampling shifts (0/1)                    */
  int max_x, max_y;  /* last valid sample of this plane                    */
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

/* the eight directional modes are contiguous: that is what makes the mode to
 * angle table and the "is this directional" test simple */
#define AV1G_PRED_DC 0
#define AV1G_PRED_V 1
#define AV1G_PRED_H 2
#define AV1G_PRED_D45 3
#define AV1G_PRED_D135 4
#define AV1G_PRED_D113 5
#define AV1G_PRED_D157 6
#define AV1G_PRED_D203 7
#define AV1G_PRED_D67 8
#define AV1G_PRED_SMOOTH 9
#define AV1G_PRED_SMOOTH_V 10
#define AV1G_PRED_SMOOTH_H 11
#define AV1G_PRED_PAETH 12
#define AV1G_PRED_CFL 13

/* the sixteen transform types, numbered as the specification numbers them */
#define AV1G_TX_DCT_DCT 0
#define AV1G_TX_ADST_DCT 1
#define AV1G_TX_DCT_ADST 2
#define AV1G_TX_ADST_ADST 3
#define AV1G_TX_FLIPADST_DCT 4
#define AV1G_TX_DCT_FLIPADST 5
#define AV1G_TX_FLIPADST_FLIPADST 6
#define AV1G_TX_ADST_FLIPADST 7
#define AV1G_TX_FLIPADST_ADST 8
#define AV1G_TX_IDTX 9
#define AV1G_TX_V_DCT 10
#define AV1G_TX_H_DCT 11
#define AV1G_TX_V_ADST 12
#define AV1G_TX_H_ADST 13
#define AV1G_TX_V_FLIPADST 14
#define AV1G_TX_H_FLIPADST 15

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

/* ------------------------------------------------------------------ */
/* inverse transform                                                  */
/*                                                                    */
/* The specification's own integer transforms, transcribed step for   */
/* step: the butterfly network exists so that every intermediate      */
/* value fits in a known number of bits, and it is rounded at specific */
/* points. A plain matrix multiply would compute the same maths but   */
/* round differently, and the difference would grow through the two   */
/* passes into visible drift, so the butterflies are what a bit exact */
/* decoder has to implement.                                          */
/* ------------------------------------------------------------------ */

AV1G_FN int av1g_min(int a, int b) { return a < b ? a : b; }
AV1G_FN int av1g_max(int a, int b) { return a > b ? a : b; }

/* cos128(angle) = round(4096 * cos(angle * pi / 128)), extended to the whole
 * circle by symmetry. sin128(angle) is cos128(angle - 64). */
AV1G_FN int av1g_cos128(int angle) {
  const int a = angle & 255;
  if (a <= 64) return av1g_tbl_cos128[a];
  if (a <= 128) return -av1g_tbl_cos128[128 - a];
  if (a <= 192) return -av1g_tbl_cos128[a - 128];
  return av1g_tbl_cos128[256 - a];
}
AV1G_FN int av1g_sin128(int angle) { return av1g_cos128(angle - 64); }

/* Round2: round to nearest, halves away from zero (the specification's
 * Round2 and Round2Signed differ only in what they do to negatives, so both
 * are provided and the transform uses the plain one where the range allows). */
AV1G_FN int av1g_round2(int v, int n) {
  if (n <= 0) return v;
  return (v + (1 << (n - 1))) >> n;
}

AV1G_FN int av1g_round2_signed(int v, int n) {
  if (n <= 0) return v;
  if (v >= 0) return (v + (1 << (n - 1))) >> n;
  return -((-v + (1 << (n - 1))) >> n);
}

/* bit-reversal of `bits` bits of x */
AV1G_FN int av1g_brev(int bits, int x) {
  int t = 0;
  int i;
  for (i = 0; i < bits; ++i) t += ((x >> i) & 1) << (bits - 1 - i);
  return t;
}

/* B(a, b, angle, flip): the butterfly rotation of the transform network.
 * `r` is the intermediate precision the specification declares (it is not
 * needed to compute the rotation, only to state the conformance bound). */
AV1G_FN void av1g_bfly(int *t, int a, int b, int angle, int flip, int r) {
  int x, y;
  (void)r;
  if ((angle & 63) == 32) {
    /* cos and sin have the same magnitude here: two fewer multiplies */
    const int v = (angle & 64) ? t[a] + t[b] : t[a] - t[b];
    const int w = (angle & 64) ? -t[a] + t[b] : t[a] + t[b];
    const int c = av1g_cos128(angle);
    x = v * c;
    y = w * c;
  } else {
    const int c = av1g_cos128(angle);
    const int s = av1g_sin128(angle);
    x = t[a] * c - t[b] * s;
    y = t[a] * s + t[b] * c;
  }
  t[a] = av1g_round2(x, 12);
  t[b] = av1g_round2(y, 12);
  if (flip) {
    const int tmp = t[a];
    t[a] = t[b];
    t[b] = tmp;
  }
}

/* H(a, b, flip, r): the Hadamard rotation, clipped to r bits.
 *
 * With flip clear this is the plain sum and difference. With flip set the
 * specification defines H(a,b,1,r) as H(b,a,0,r), which *negates* what lands in
 * a - it is not a swap of the two results: reading x from b and y from a gives
 * a = x - y = T[b] - T[a] and b = x + y. Getting that backwards turns the DCT
 * network into a different (wrong) transform, and it only shows up on the odd
 * coefficients, so it is easy to miss. */
AV1G_FN void av1g_hadamard(int *t, int a, int b, int flip, int r) {
  const int lo = -(1 << (r - 1));
  const int hi = (1 << (r - 1)) - 1;
  const int x = t[a], y = t[b];
  const int sum = av1g_clip(x + y, lo, hi);
  const int dif = av1g_clip(flip ? y - x : x - y, lo, hi);
  if (flip) {
    t[a] = dif;
    t[b] = sum;
  } else {
    t[a] = sum;
    t[b] = dif;
  }
}

/* Inverse DCT of length 2^n, n = 2..6, in place. */
AV1G_FN void av1g_idct(int *t, int n, int r) {
  int i, j, tmp[64];
  const int len = 1 << n;
  /* the network expects the coefficients in bit-reversed order */
  for (i = 0; i < len; ++i) tmp[i] = t[av1g_brev(n, i)];
  for (i = 0; i < len; ++i) t[i] = tmp[i];

  if (n == 6)
    for (i = 0; i < 16; ++i) av1g_bfly(t, 32 + i, 63 - i, 63 - 4 * av1g_brev(4, i), 0, r);
  if (n >= 5)
    for (i = 0; i < 8; ++i)
      av1g_bfly(t, 16 + i, 31 - i, 6 + (av1g_brev(3, 7 - i) << 3), 0, r);
  if (n == 6)
    for (i = 0; i < 16; ++i) av1g_hadamard(t, 32 + i * 2, 33 + i * 2, i & 1, r);
  if (n >= 4)
    for (i = 0; i < 4; ++i)
      av1g_bfly(t, 8 + i, 15 - i, 12 + (av1g_brev(2, 3 - i) << 4), 0, r);
  if (n >= 5)
    for (i = 0; i < 8; ++i) av1g_hadamard(t, 16 + 2 * i, 17 + 2 * i, i & 1, r);
  if (n == 6)
    for (i = 0; i < 4; ++i)
      for (j = 0; j < 2; ++j)
        av1g_bfly(t, 62 - i * 4 - j, 33 + i * 4 + j,
                  60 - 16 * av1g_brev(2, i) + 64 * j, 1, r);
  if (n >= 3)
    for (i = 0; i < 2; ++i) av1g_bfly(t, 4 + i, 7 - i, 56 - 32 * i, 0, r);
  if (n >= 4)
    for (i = 0; i < 4; ++i) av1g_hadamard(t, 8 + 2 * i, 9 + 2 * i, i & 1, r);
  if (n >= 5)
    for (i = 0; i < 2; ++i)
      for (j = 0; j < 2; ++j)
        av1g_bfly(t, 30 - 4 * i - j, 17 + 4 * i + j, 24 + (j << 6) + ((1 - i) << 5), 1, r);
  if (n == 6)
    for (i = 0; i < 8; ++i)
      for (j = 0; j < 2; ++j) av1g_hadamard(t, 32 + i * 4 + j, 35 + i * 4 - j, i & 1, r);
  for (i = 0; i < 2; ++i) av1g_bfly(t, 2 * i, 2 * i + 1, 32 + 16 * i, 1 - i, r);
  if (n >= 3)
    for (i = 0; i < 2; ++i) av1g_hadamard(t, 4 + 2 * i, 5 + 2 * i, i, r);
  if (n >= 4)
    for (i = 0; i < 2; ++i) av1g_bfly(t, 14 - i, 9 + i, 48 + 64 * i, 1, r);
  if (n >= 5)
    for (i = 0; i < 4; ++i)
      for (j = 0; j < 2; ++j) av1g_hadamard(t, 16 + 4 * i + j, 19 + 4 * i - j, i & 1, r);
  if (n == 6)
    for (i = 0; i < 2; ++i)
      for (j = 0; j < 4; ++j)
        av1g_bfly(t, 61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, 1, r);
  for (i = 0; i < 2; ++i) av1g_hadamard(t, i, 3 - i, 0, r);
  if (n >= 3) av1g_bfly(t, 6, 5, 32, 1, r);
  if (n >= 4)
    for (i = 0; i < 2; ++i)
      for (j = 0; j < 2; ++j) av1g_hadamard(t, 8 + 4 * i + j, 11 + 4 * i - j, i, r);
  if (n >= 5)
    for (i = 0; i < 4; ++i) av1g_bfly(t, 29 - i, 18 + i, 48 + (i >> 1) * 64, 1, r);
  if (n == 6)
    for (i = 0; i < 4; ++i)
      for (j = 0; j < 4; ++j) av1g_hadamard(t, 32 + 8 * i + j, 39 + 8 * i - j, i & 1, r);
  if (n >= 3)
    for (i = 0; i < 4; ++i) av1g_hadamard(t, i, 7 - i, 0, r);
  if (n >= 4)
    for (i = 0; i < 2; ++i) av1g_bfly(t, 13 - i, 10 + i, 32, 1, r);
  if (n >= 5)
    for (i = 0; i < 2; ++i)
      for (j = 0; j < 4; ++j) av1g_hadamard(t, 16 + i * 8 + j, 23 + i * 8 - j, i, r);
  if (n == 6)
    for (i = 0; i < 8; ++i) av1g_bfly(t, 59 - i, 36 + i, i < 4 ? 48 : 112, 1, r);
  if (n >= 4)
    for (i = 0; i < 8; ++i) av1g_hadamard(t, i, 15 - i, 0, r);
  if (n >= 5)
    for (i = 0; i < 4; ++i) av1g_bfly(t, 27 - i, 20 + i, 32, 1, r);
  if (n == 6)
    for (i = 0; i < 8; ++i) {
      av1g_hadamard(t, 32 + i, 47 - i, 0, r);
      av1g_hadamard(t, 48 + i, 63 - i, 1, r);
    }
  if (n >= 5)
    for (i = 0; i < 16; ++i) av1g_hadamard(t, i, 31 - i, 0, r);
  if (n == 6)
    for (i = 0; i < 8; ++i) av1g_bfly(t, 55 - i, 40 + i, 32, 1, r);
  if (n == 6)
    for (i = 0; i < 32; ++i) av1g_hadamard(t, i, 63 - i, 0, r);
}

/* the two permutations the inverse ADST is bracketed with */
AV1G_FN void av1g_adst_in_perm(int *t, int n) {
  int copy[16];
  const int n0 = 1 << n;
  int i;
  for (i = 0; i < n0; ++i) copy[i] = t[i];
  for (i = 0; i < n0; ++i) {
    const int idx = (i & 1) ? (i - 1) : (n0 - i - 1);
    t[i] = copy[idx];
  }
}

AV1G_FN void av1g_adst_out_perm(int *t, int n) {
  int copy[16];
  const int n0 = 1 << n;
  int i;
  for (i = 0; i < n0; ++i) copy[i] = t[i];
  for (i = 0; i < n0; ++i) {
    const int a = (i >> 3) & 1;
    const int b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
    const int c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
    const int d = (i & 1) ^ ((i >> 1) & 1);
    const int idx = (((d << 3) | (c << 2) | (b << 1) | a)) >> (4 - n);
    t[i] = (i & 1) ? -copy[idx] : copy[idx];
  }
}

/* Inverse ADST of length 4, in place. */
AV1G_FN void av1g_adst4(int *t, int r) {
  int s[7], x[4];
  (void)r;
  s[0] = AV1G_SINPI_1_9 * t[0];
  s[1] = AV1G_SINPI_2_9 * t[0];
  s[2] = AV1G_SINPI_3_9 * t[1];
  s[3] = AV1G_SINPI_4_9 * t[2];
  s[4] = AV1G_SINPI_1_9 * t[2];
  s[5] = AV1G_SINPI_2_9 * t[3];
  s[6] = AV1G_SINPI_4_9 * t[3];
  {
    const int a7 = t[0] - t[2];
    const int b7 = a7 + t[3];
    s[0] = s[0] + s[3];
    s[1] = s[1] - s[4];
    s[3] = s[2];
    s[2] = AV1G_SINPI_3_9 * b7;
    s[0] = s[0] + s[5];
    s[1] = s[1] - s[6];
    x[0] = s[0] + s[3];
    x[1] = s[1] + s[3];
    x[2] = s[2];
    x[3] = s[0] + s[1];
    x[3] = x[3] - s[3];
  }
  t[0] = av1g_round2(x[0], 12);
  t[1] = av1g_round2(x[1], 12);
  t[2] = av1g_round2(x[2], 12);
  t[3] = av1g_round2(x[3], 12);
}

AV1G_FN void av1g_adst8(int *t, int r) {
  int i, j;
  av1g_adst_in_perm(t, 3);
  for (i = 0; i < 4; ++i) av1g_bfly(t, 2 * i, 2 * i + 1, 60 - 16 * i, 1, r);
  for (i = 0; i < 4; ++i) av1g_hadamard(t, i, 4 + i, 0, r);
  for (i = 0; i < 2; ++i) av1g_bfly(t, 4 + 3 * i, 5 + i, 48 - 32 * i, 1, r);
  for (i = 0; i < 2; ++i)
    for (j = 0; j < 2; ++j) av1g_hadamard(t, 4 * j + i, 2 + 4 * j + i, 0, r);
  for (i = 0; i < 2; ++i) av1g_bfly(t, 2 + 4 * i, 3 + 4 * i, 32, 1, r);
  av1g_adst_out_perm(t, 3);
}

AV1G_FN void av1g_adst16(int *t, int r) {
  int i, j;
  av1g_adst_in_perm(t, 4);
  for (i = 0; i < 8; ++i) av1g_bfly(t, 2 * i, 2 * i + 1, 62 - 8 * i, 1, r);
  for (i = 0; i < 8; ++i) av1g_hadamard(t, i, 8 + i, 0, r);
  for (i = 0; i < 2; ++i) {
    av1g_bfly(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, 1, r);
    av1g_bfly(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, 1, r);
  }
  for (i = 0; i < 4; ++i)
    for (j = 0; j < 2; ++j) av1g_hadamard(t, 8 * j + i, 4 + 8 * j + i, 0, r);
  for (i = 0; i < 2; ++i)
    for (j = 0; j < 2; ++j)
      av1g_bfly(t, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, 1, r);
  for (i = 0; i < 2; ++i)
    for (j = 0; j < 4; ++j) av1g_hadamard(t, 4 * j + i, 2 + 4 * j + i, 0, r);
  for (i = 0; i < 4; ++i) av1g_bfly(t, 2 + 4 * i, 3 + 4 * i, 32, 1, r);
  av1g_adst_out_perm(t, 4);
}

AV1G_FN void av1g_iadst(int *t, int n, int r) {
  if (n == 2) {
    av1g_adst4(t, r);
  } else if (n == 3) {
    av1g_adst8(t, r);
  } else {
    av1g_adst16(t, r);
  }
}

/* The identity transform: the coefficients are already samples, only the
 * scale has to be brought back to the Q12 convention the other transforms
 * produce. */
AV1G_FN void av1g_iidentity(int *t, int n) {
  int i;
  if (n == 2) {
    for (i = 0; i < 4; ++i) t[i] = av1g_round2(t[i] * 5793, 12);
  } else if (n == 3) {
    for (i = 0; i < 8; ++i) t[i] = t[i] * 2;
  } else if (n == 4) {
    for (i = 0; i < 16; ++i) t[i] = av1g_round2(t[i] * 11586, 12);
  } else {
    for (i = 0; i < 32; ++i) t[i] = t[i] * 4;
  }
}

/* Inverse Walsh-Hadamard transform, for lossless blocks. */
AV1G_FN void av1g_wht(int *t, int shift) {
  int a = t[0] >> shift;
  int c = t[1] >> shift;
  int d = t[2] >> shift;
  int b = t[3] >> shift;
  int e;
  a += c;
  d -= b;
  e = (a - d) >> 1;
  b = e - b;
  c = e - c;
  a -= b;
  d += c;
  t[0] = a;
  t[1] = b;
  t[2] = c;
  t[3] = d;
}

/* Which one dimensional transform a row or a column of a given 2D transform
 * type uses. The two names of an AV1 transform type are the pair of one
 * dimensional transforms, the first for the columns, the second for the rows;
 * the 1D types transform one direction and pass the other through. */
#define AV1G_TX1D_DCT 0
#define AV1G_TX1D_ADST 1
#define AV1G_TX1D_IDTX 2

AV1G_FN int av1g_row_kind(int tx_type) {
  switch (tx_type) {
    case AV1G_TX_DCT_DCT:
    case AV1G_TX_ADST_DCT:
    case AV1G_TX_FLIPADST_DCT:
    case AV1G_TX_H_DCT:
      return AV1G_TX1D_DCT;
    case AV1G_TX_DCT_ADST:
    case AV1G_TX_ADST_ADST:
    case AV1G_TX_DCT_FLIPADST:
    case AV1G_TX_FLIPADST_FLIPADST:
    case AV1G_TX_ADST_FLIPADST:
    case AV1G_TX_FLIPADST_ADST:
    case AV1G_TX_H_ADST:
    case AV1G_TX_H_FLIPADST:
      return AV1G_TX1D_ADST;
    default:
      return AV1G_TX1D_IDTX;
  }
}

AV1G_FN int av1g_col_kind(int tx_type) {
  switch (tx_type) {
    case AV1G_TX_DCT_DCT:
    case AV1G_TX_DCT_ADST:
    case AV1G_TX_DCT_FLIPADST:
    case AV1G_TX_V_DCT:
      return AV1G_TX1D_DCT;
    case AV1G_TX_ADST_DCT:
    case AV1G_TX_ADST_ADST:
    case AV1G_TX_FLIPADST_DCT:
    case AV1G_TX_FLIPADST_FLIPADST:
    case AV1G_TX_ADST_FLIPADST:
    case AV1G_TX_FLIPADST_ADST:
    case AV1G_TX_V_ADST:
    case AV1G_TX_V_FLIPADST:
      return AV1G_TX1D_ADST;
    default:
      return AV1G_TX1D_IDTX;
  }
}

/* flipUD / flipLR: the flip transforms are the same networks with their
 * output reversed, which the specification applies to the finished residual. */
AV1G_FN int av1g_flip_ud(int tx_type) {
  return tx_type == AV1G_TX_FLIPADST_DCT || tx_type == AV1G_TX_FLIPADST_ADST ||
         tx_type == AV1G_TX_FLIPADST_FLIPADST || tx_type == AV1G_TX_V_FLIPADST;
}
AV1G_FN int av1g_flip_lr(int tx_type) {
  return tx_type == AV1G_TX_DCT_FLIPADST || tx_type == AV1G_TX_ADST_FLIPADST ||
         tx_type == AV1G_TX_FLIPADST_FLIPADST || tx_type == AV1G_TX_H_FLIPADST;
}

AV1G_FN void av1g_tx1d(int *t, int n, int kind, int r) {
  if (kind == AV1G_TX1D_DCT)
    av1g_idct(t, n, r);
  else if (kind == AV1G_TX1D_ADST)
    av1g_iadst(t, n, r);
  else
    av1g_iidentity(t, n);
}

AV1G_FN int av1g_log2(int v) {
  int n = 0;
  while ((1 << n) < v) ++n;
  return n;
}

/* pass 1: transform coefficient row `y` along x into the scratch buffer */
AV1G_FN void av1g_itx_row(const Av1gItxJob *job, AV1G_GLOBAL const int *coef,
                          AV1G_GLOBAL int *tmp, int y) {
  int t[64];
  const int log2w = av1g_log2(job->w);
  const int log2h = av1g_log2(job->h);
  /* the first pass runs at row precision but its output is clamped to the
   * column precision, because that is what the second pass reads back */
  const int rowClampRange = job->bitdepth + 8;
  const int colClampRange = av1g_max(job->bitdepth + 6, 16);
  const int lo = -(1 << (colClampRange - 1));
  const int hi = (1 << (colClampRange - 1)) - 1;
  int j;

  for (j = 0; j < job->w; ++j)
    t[j] = (y < 32 && j < 32) ? coef[job->coef_off + y * job->w + j] : 0;

  /* a 2:1 rectangle pre-scales: the round trip through the two passes is not
   * exactly the identity for those sizes */
  if (av1g_abs(log2w - log2h) == 1) {
    for (j = 0; j < job->w; ++j) t[j] = av1g_round2(t[j] * 2896, 12);
  }

  if (job->lossless)
    av1g_wht(t, 2);
  else
    av1g_tx1d(t, log2w, av1g_row_kind(job->tx_type), rowClampRange);

  for (j = 0; j < job->w; ++j) {
    const int r = job->lossless ? t[j] : av1g_round2(t[j], job->row_shift);
    tmp[job->tmp_off + y * job->w + j] = av1g_clip(r, lo, hi);
  }
}

/* pass 2: transform column `x` along y and add the residual to the plane */
AV1G_FN void av1g_itx_col_add(const Av1gItxJob *job, AV1G_GLOBAL const int *tmp,
                              AV1G_GLOBAL av1g_px *dst, int x, int maxv) {
  int t[64];
  const int log2h = av1g_log2(job->h);
  const int colClampRange = av1g_max(job->bitdepth + 6, 16);
  const int fud = av1g_flip_ud(job->tx_type);
  const int flr = av1g_flip_lr(job->tx_type);
  int i;
  if (job->eob == 0) return;
  for (i = 0; i < job->h; ++i) t[i] = tmp[job->tmp_off + i * job->w + x];

  if (job->lossless)
    av1g_wht(t, 0);
  else
    av1g_tx1d(t, log2h, av1g_col_kind(job->tx_type), colClampRange);

  for (i = 0; i < job->h; ++i) {
    const int r = job->lossless ? t[i] : av1g_round2(t[i], 4);
    const int yy = fud ? job->h - 1 - i : i;
    const int xx = flr ? job->w - 1 - x : x;
    AV1G_GLOBAL av1g_px *p = dst + job->dst_off + yy * job->dst_stride + xx;
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
/*                                                                    */
/* Intra prediction reads the reconstructed samples above and to the  */
/* left of the block, smooths or extends them along the prediction    */
/* angle, and writes one sample per position. The edge handling is    */
/* most of the difficulty: the specification filters the edge        */
/* (three taps, strength chosen by the block size and the angle),     */
/* optionally upsamples it to half sample steps, and treats the       */
/* corner sample separately, and each of those changes the result.    */
/* ------------------------------------------------------------------ */

/* The edge arrays are indexed relative to the block: AboveRow[-1] is the
 * corner, AboveRow[0..] the samples above, LeftCol likewise. The largest
 * block is 64x64, so 128 samples of edge are enough before upsampling, which
 * doubles it; a few slots on either side keep the negative indices legal. */
#define AV1G_INTRA_EDGE 264
#define AV1G_INTRA_EDGE_OFF 4

/* Derives AboveRow and LeftCol exactly as the specification does, including
 * the values used when a neighbour is missing. */
AV1G_FN void av1g_intra_edges(const Av1gIntraJob *job, AV1G_GLOBAL const av1g_px *plane,
                              int *above, int *left) {
  const int w = job->w, h = job->h;
  const int stride = job->dst_stride;
  AV1G_GLOBAL const av1g_px *p = plane + job->dst_off;
  const int mid = 1 << (job->bitdepth - 1);
  int i;

  if (!job->have_above && job->have_left) {
    for (i = 0; i < w + h; ++i) above[i] = (int)p[-1];
  } else if (!job->have_above && !job->have_left) {
    for (i = 0; i < w + h; ++i) above[i] = mid - 1;
  } else {
    const int limit = av1g_min(job->max_x, job->x + (job->have_above_right ? 2 * w : w) - 1);
    for (i = 0; i < w + h; ++i) {
      int xx = job->x + i;
      if (xx > limit) xx = limit;
      above[i] = (int)p[-stride + (xx - job->x)];
    }
  }

  if (!job->have_left && job->have_above) {
    for (i = 0; i < w + h; ++i) left[i] = (int)p[-stride];
  } else if (!job->have_left && !job->have_above) {
    for (i = 0; i < w + h; ++i) left[i] = mid + 1;
  } else {
    const int limit = av1g_min(job->max_y, job->y + (job->have_below_left ? 2 * h : h) - 1);
    for (i = 0; i < w + h; ++i) {
      int yy = job->y + i;
      if (yy > limit) yy = limit;
      left[i] = (int)p[(yy - job->y) * stride - 1];
    }
  }

  if (job->have_above && job->have_left)
    above[-1] = (int)p[-stride - 1];
  else if (job->have_above)
    above[-1] = (int)p[-stride];
  else if (job->have_left)
    above[-1] = (int)p[-1];
  else
    above[-1] = mid;
  left[-1] = above[-1];
}

AV1G_FN int av1g_is_directional(int mode) { return mode >= 1 && mode <= 8; }

/* Intra_Edge_Kernel convolved along the edge; `buf` is in the block relative
 * indexing, so entry i-1 is the i'th sample of the edge. */
AV1G_FN void av1g_edge_filter(int *buf, int sz, int strength) {
  int edge[AV1G_INTRA_EDGE];
  int i, j, s, k;
  if (strength <= 0) return;
  if (sz > AV1G_INTRA_EDGE) sz = AV1G_INTRA_EDGE;
  for (i = 0; i < sz; ++i) edge[i] = buf[i - 1];
  for (i = 1; i < sz; ++i) {
    s = 0;
    for (j = 0; j < 5; ++j) {
      k = i - 2 + j;
      if (k < 0) k = 0;
      if (k > sz - 1) k = sz - 1;
      s += av1g_tbl_edge_kernel[strength - 1][j] * edge[k];
    }
    buf[i - 1] = (s + 8) >> 4;
  }
}

/* Doubles the resolution of an edge, so that the prediction can step half a
 * sample at a time. */
AV1G_FN void av1g_edge_upsample(int *buf, int num_px, int maxv) {
  int dup[AV1G_INTRA_EDGE];
  int i;
  if (num_px > AV1G_INTRA_EDGE - 4) num_px = AV1G_INTRA_EDGE - 4;
  dup[0] = buf[-1];
  for (i = -1; i < num_px; ++i) dup[i + 2] = buf[i];
  dup[num_px + 2] = buf[num_px - 1];
  buf[-2] = dup[0];
  for (i = 0; i < num_px; ++i) {
    int s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
    s = av1g_clip_px(av1g_round2(s, 4), maxv);
    buf[2 * i - 1] = s;
    buf[2 * i] = dup[i + 2];
  }
}

/* Intra edge filter strength, from the block size, the filter type and the
 * angle difference. */
AV1G_FN int av1g_edge_strength(int w, int h, int filter_type, int delta) {
  const int d = av1g_abs(delta);
  const int blk = w + h;
  int strength = 0;
  if (filter_type == 0) {
    if (blk <= 8) {
      if (d >= 56) strength = 1;
    } else if (blk <= 12) {
      if (d >= 40) strength = 1;
    } else if (blk <= 16) {
      if (d >= 40) strength = 1;
    } else if (blk <= 24) {
      if (d >= 8) strength = 1;
      if (d >= 16) strength = 2;
      if (d >= 32) strength = 3;
    } else if (blk <= 32) {
      strength = 1;
      if (d >= 4) strength = 2;
      if (d >= 32) strength = 3;
    } else {
      strength = 3;
    }
  } else {
    if (blk <= 8) {
      if (d >= 40) strength = 1;
      if (d >= 64) strength = 2;
    } else if (blk <= 16) {
      if (d >= 20) strength = 1;
      if (d >= 48) strength = 2;
    } else if (blk <= 24) {
      if (d >= 4) strength = 3;
    } else {
      strength = 3;
    }
  }
  return strength;
}

AV1G_FN int av1g_edge_upsample_sel(int w, int h, int filter_type, int delta) {
  const int d = av1g_abs(delta);
  const int blk = w + h;
  if (d <= 0 || d >= 40) return 0;
  if (filter_type == 0) return blk <= 16;
  return blk <= 8;
}

/* The directional predictor: walks the edge at the prediction angle. */
AV1G_FN void av1g_intra_directional(const Av1gIntraJob *job, int *above, int *left,
                                    int pangle, int upsample_above, int upsample_left,
                                    AV1G_GLOBAL av1g_px *dst, int maxv) {
  const int w = job->w, h = job->h, stride = job->dst_stride;
  int dx = 0, dy = 0, i, j;
  if (pangle < 90)
    dx = av1g_tbl_dr_deriv[pangle];
  else if (pangle > 90 && pangle < 180)
    dx = av1g_tbl_dr_deriv[180 - pangle];
  if (pangle > 90 && pangle < 180)
    dy = av1g_tbl_dr_deriv[pangle - 90];
  else if (pangle > 180)
    dy = av1g_tbl_dr_deriv[270 - pangle];

  if (pangle < 90) {
    const int max_base = (w + h - 1) << upsample_above;
    for (i = 0; i < h; ++i) {
      const int idx = (i + 1) * dx;
      for (j = 0; j < w; ++j) {
        const int base = (idx >> (6 - upsample_above)) + (j << upsample_above);
        const int shift = ((idx << upsample_above) >> 1) & 0x1F;
        int v;
        if (base < max_base)
          v = av1g_round2(above[base] * (32 - shift) + above[base + 1] * shift, 5);
        else
          v = above[max_base];
        dst[job->dst_off + i * stride + j] = (av1g_px)av1g_clip_px(v, maxv);
      }
    }
    return;
  }
  if (pangle > 90 && pangle < 180) {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
        int idx = (j << 6) - (i + 1) * dx;
        int base = idx >> (6 - upsample_above);
        int v;
        if (base >= -(1 << upsample_above)) {
          const int shift = ((idx << upsample_above) >> 1) & 0x1F;
          v = av1g_round2(above[base] * (32 - shift) + above[base + 1] * shift, 5);
        } else {
          idx = (i << 6) - (j + 1) * dy;
          base = idx >> (6 - upsample_left);
          {
            const int shift = ((idx << upsample_left) >> 1) & 0x1F;
            v = av1g_round2(left[base] * (32 - shift) + left[base + 1] * shift, 5);
          }
        }
        dst[job->dst_off + i * stride + j] = (av1g_px)av1g_clip_px(v, maxv);
      }
    }
    return;
  }
  if (pangle > 180) {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
        const int idx = (j + 1) * dy;
        const int base = (idx >> (6 - upsample_left)) + (i << upsample_left);
        const int shift = ((idx << upsample_left) >> 1) & 0x1F;
        const int v = av1g_round2(left[base] * (32 - shift) + left[base + 1] * shift, 5);
        dst[job->dst_off + i * stride + j] = (av1g_px)av1g_clip_px(v, maxv);
      }
    }
    return;
  }
  /* exactly horizontal or vertical */
  for (i = 0; i < h; ++i) {
    for (j = 0; j < w; ++j) {
      const int v = pangle == 90 ? above[j] : left[i];
      dst[job->dst_off + i * stride + j] = (av1g_px)av1g_clip_px(v, maxv);
    }
  }
}

/* The smooth predictors: a linear ramp between the two edges. */
AV1G_FN int av1g_sm_weight(int n, int i) {
  if (n == 4) return av1g_tbl_sm_w4[i];
  if (n == 8) return av1g_tbl_sm_w8[i];
  if (n == 16) return av1g_tbl_sm_w16[i];
  if (n == 32) return av1g_tbl_sm_w32[i];
  return av1g_tbl_sm_w64[i];
}

AV1G_FN void av1g_intra_smooth(const Av1gIntraJob *job, int *above, int *left, int mode,
                               AV1G_GLOBAL av1g_px *dst, int maxv) {
  const int w = job->w, h = job->h, stride = job->dst_stride;
  int i, j;
  for (i = 0; i < h; ++i) {
    for (j = 0; j < w; ++j) {
      int pred;
      if (mode == AV1G_PRED_SMOOTH) {
        const int wy = av1g_sm_weight(h, i), wx = av1g_sm_weight(w, j);
        pred = wy * above[j] + (256 - wy) * left[h - 1] + wx * left[i] +
               (256 - wx) * above[w - 1];
        pred = av1g_round2(pred, 9);
      } else if (mode == AV1G_PRED_SMOOTH_V) {
        const int wy = av1g_sm_weight(h, i);
        pred = av1g_round2(wy * above[j] + (256 - wy) * left[h - 1], 8);
      } else {
        const int wx = av1g_sm_weight(w, j);
        pred = av1g_round2(wx * left[i] + (256 - wx) * above[w - 1], 8);
      }
      dst[job->dst_off + i * stride + j] = (av1g_px)av1g_clip_px(pred, maxv);
    }
  }
}

/* Paeth: pick the neighbour the plane through the corner predicts best. */
AV1G_FN void av1g_intra_paeth(const Av1gIntraJob *job, int *above, int *left,
                              AV1G_GLOBAL av1g_px *dst, int maxv) {
  const int w = job->w, h = job->h, stride = job->dst_stride;
  int i, j;
  for (i = 0; i < h; ++i) {
    for (j = 0; j < w; ++j) {
      const int base = above[j] + left[i] - above[-1];
      const int pl = av1g_abs(base - left[i]);
      const int pt = av1g_abs(base - above[j]);
      const int ptl = av1g_abs(base - above[-1]);
      int v;
      if (pl <= pt && pl <= ptl)
        v = left[i];
      else if (pt <= ptl)
        v = above[j];
      else
        v = above[-1];
      dst[job->dst_off + i * stride + j] = (av1g_px)av1g_clip_px(v, maxv);
    }
  }
}

/* The recursive (filter intra) predictor: a 4x2 block at a time, seven
 * neighbouring samples through one of five 7-tap filters. */
AV1G_FN void av1g_intra_filter(const Av1gIntraJob *job, int *above, int *left,
                               AV1G_GLOBAL av1g_px *dst, int maxv) {
  const int w = job->w, h = job->h, stride = job->dst_stride;
  AV1G_GLOBAL av1g_px *out = dst + job->dst_off;
  const int w4 = w >> 2, h2 = h >> 1;
  int i2, j4, i, i1, j1;
  for (i2 = 0; i2 < h2; ++i2) {
    for (j4 = 0; j4 < w4; ++j4) {
      int p[7];
      int pr;
      for (i = 0; i < 5; ++i) {
        if (i2 == 0)
          p[i] = above[(j4 << 2) + i - 1];
        else if (j4 == 0 && i == 0)
          p[i] = left[(i2 << 1) - 1];
        else
          p[i] = (int)out[((i2 << 1) - 1) * stride + (j4 << 2) + i - 1];
      }
      for (i = 5; i < 7; ++i) {
        if (j4 == 0)
          p[i] = left[(i2 << 1) + i - 5];
        else
          p[i] = (int)out[((i2 << 1) + i - 5) * stride + (j4 << 2) - 1];
      }
      for (i1 = 0; i1 < 2; ++i1) {
        for (j1 = 0; j1 < 4; ++j1) {
          pr = 0;
          for (i = 0; i < 7; ++i)
            pr += av1g_tbl_filter_intra[job->filter_intra_mode][(i1 << 2) + j1][i] * p[i];
          out[((i2 << 1) + i1) * stride + (j4 << 2) + j1] =
              (av1g_px)av1g_clip_px(av1g_round2(pr, 4), maxv);
        }
      }
    }
  }
}

/* DC prediction: the average of whichever edges exist. */
AV1G_FN int av1g_intra_dc(const Av1gIntraJob *job, const int *above, const int *left) {
  const int w = job->w, h = job->h;
  int sum = 0, k;
  if (job->have_above && job->have_left) {
    for (k = 0; k < w; ++k) sum += above[k];
    for (k = 0; k < h; ++k) sum += left[k];
    return (sum + ((w + h) >> 1)) / (w + h);
  }
  if (job->have_left) {
    for (k = 0; k < h; ++k) sum += left[k];
    return (sum + (h >> 1)) >> av1g_log2(h);
  }
  if (job->have_above) {
    for (k = 0; k < w; ++k) sum += above[k];
    return (sum + (w >> 1)) >> av1g_log2(w);
  }
  return 1 << (job->bitdepth - 1);
}

/* The whole prediction of one transform block. Called once per block, not once
 * per sample: the edge filtering, the recursive predictor and the luma
 * projection of chroma from luma all look at more than one sample. */
AV1G_FN void av1g_intra_block(const Av1gIntraJob *job, AV1G_GLOBAL av1g_px *plane,
                              int maxv) {
  int above_buf[AV1G_INTRA_EDGE];
  int left_buf[AV1G_INTRA_EDGE];
  int *above = above_buf + AV1G_INTRA_EDGE_OFF;
  int *left = left_buf + AV1G_INTRA_EDGE_OFF;
  const int w = job->w, h = job->h, stride = job->dst_stride;
  AV1G_GLOBAL av1g_px *out = plane + job->dst_off;
  int i, j;

  if (job->mode == AV1G_PRED_CFL) {
    /* Chroma from luma: the block is DC predicted first (the specification's
     * CfL process reads the chroma samples back and adds the luma's high
     * frequencies to them), then the projected luma is added on top. */
    const int subx = job->ss_x, suby = job->ss_y;
    long long sum = 0;
    int dc;
    av1g_intra_edges(job, plane, above, left);
    dc = av1g_intra_dc(job, above, left);
    for (i = 0; i < h; ++i)
      for (j = 0; j < w; ++j) out[i * stride + j] = (av1g_px)av1g_clip_px(dc, maxv);
    for (i = 0; i < h; ++i) {
      int ly = (job->y + i) << suby;
      ly = av1g_min(ly, job->luma_h - (1 << suby));
      for (j = 0; j < w; ++j) {
        int lx = (job->x + j) << subx;
        int t = 0, dy, dx, v;
        lx = av1g_min(lx, job->luma_w - (1 << subx));
        for (dy = 0; dy <= suby; ++dy)
          for (dx = 0; dx <= subx; ++dx)
            t += (int)plane[job->luma_off + (ly + dy) * job->luma_stride + lx + dx];
        v = t << (3 - subx - suby);
        sum += v;
        /* the luma sample is stored here so that the second pass is free of
         * the addressing; it is overwritten below */
        out[i * stride + j] = (av1g_px)av1g_clip_px(v, 65535);
      }
    }
    {
      const int shift = av1g_log2(w) + av1g_log2(h);
      const long long avg = (sum + (1LL << (shift - 1))) >> shift;
      for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
          const int l = (int)out[i * stride + j];
          const int scaled = av1g_round2_signed(job->cfl_alpha * (l - (int)avg), 6);
          out[i * stride + j] = (av1g_px)av1g_clip_px(dc + scaled, maxv);
        }
      }
    }
    return;
  }

  av1g_intra_edges(job, plane, above, left);

  if (job->filter_intra && job->w >= 4 && job->h >= 2) {
    av1g_intra_filter(job, above, left, plane, maxv);
    return;
  }

  if (av1g_is_directional(job->mode)) {
    const int pangle = av1g_tbl_mode_to_angle[job->mode] + job->angle * 3;
    int upsample_above = 0, upsample_left = 0;
    if (job->filter_type >= 0) {
      if (pangle != 90 && pangle != 180) {
        if (pangle > 90 && pangle < 180 && w + h >= 24) {
          const int corner = av1g_round2(left[0] * 5 + above[-1] * 6 + above[0] * 5, 4);
          above[-1] = corner;
          left[-1] = corner;
        }
        if (job->have_above) {
          const int strength = av1g_edge_strength(w, h, job->filter_type, pangle - 90);
          const int num = av1g_min(w, job->max_x - job->x + 1) + (pangle < 90 ? h : 0) + 1;
          av1g_edge_filter(above, num, strength);
        }
        if (job->have_left) {
          const int strength = av1g_edge_strength(w, h, job->filter_type, pangle - 180);
          const int num = av1g_min(h, job->max_y - job->y + 1) + (pangle > 180 ? w : 0) + 1;
          av1g_edge_filter(left, num, strength);
        }
        upsample_above = av1g_edge_upsample_sel(w, h, job->filter_type, pangle - 90);
        if (upsample_above) av1g_edge_upsample(above, w + (pangle < 90 ? h : 0), maxv);
        upsample_left = av1g_edge_upsample_sel(w, h, job->filter_type, pangle - 180);
        if (upsample_left) av1g_edge_upsample(left, h + (pangle > 180 ? w : 0), maxv);
      }
    }
    av1g_intra_directional(job, above, left, pangle, upsample_above, upsample_left, plane,
                           maxv);
    return;
  }

  if (job->mode == AV1G_PRED_SMOOTH || job->mode == AV1G_PRED_SMOOTH_V ||
      job->mode == AV1G_PRED_SMOOTH_H) {
    av1g_intra_smooth(job, above, left, job->mode, plane, maxv);
    return;
  }
  if (job->mode == AV1G_PRED_PAETH) {
    av1g_intra_paeth(job, above, left, plane, maxv);
    return;
  }
  /* DC: the same flat predictor, computed from whichever edges exist */
  {
    const int dc = av1g_intra_dc(job, above, left);
    for (i = 0; i < h; ++i)
      for (j = 0; j < w; ++j) out[i * stride + j] = (av1g_px)av1g_clip_px(dc, maxv);
  }
}

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
 * Filters one group of four samples along an edge.  p1 is at
 * job->p_off + g * job->step, p0 one stride step later, q0 two, q1 three, so
 * `g` is a *sample* offset along the edge (0, 4, 8, ...) - the kernels get a
 * group index and have to multiply it by four.  Edge lengths are multiples of
 * four because block boundaries are.
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
