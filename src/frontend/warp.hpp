/*
 * AV1onGPU - local warped motion
 *
 * The second of the two ways a block can be predicted by an affine model
 * instead of by its motion vector alone. Global motion (predict_inter steps
 * 5-7) reads the model from the frame header: every block that uses the same
 * reference frame is displaced the same way, which is what a camera pan or a
 * zoom looks like. Local warped motion (7.11.3.8) is the opposite: the model
 * is fitted to the block's *neighbours*, so it can describe a rotation or a
 * zoom that only part of the picture goes through - a spinning object in front
 * of a static background, or a scene where the camera pans while something
 * moves the other way.
 *
 * What the bitstream codes for such a block is one symbol (motion_mode), not
 * the model: the model is derived, and it is derived from the motion vectors
 * the block's neighbours already carry. That is why this must be evaluated
 * while the frame is being decoded - the fit reads the motion vector grid as
 * it stands at that block, and the grid of a block's own row is finished by
 * the time the block is parsed.
 *
 * Two steps, in specification order:
 *
 *   find_warp_samples (7.11.3.8)   collect up to eight neighbours that use
 *                                  this block's reference frame, as the
 *                                  positions and displaced positions of their
 *                                  block centres.
 *   warp estimation (7.11.3.8)     least squares fit of the affine model
 *                                  (a, b, c, d, e, f) through those points.
 *
 * The fit is a plain 2x2 normal equations solve, but nothing about it fits in
 * 32 bits: the sums run over positions in 1/8 luma samples (so an 8K frame is
 * 6e4 of them) squared, times eight samples. The determinant is a product of
 * two of those sums, which is why the divisor here is resolved at 64 bit
 * rather than reusing the 32 bit one the global motion shear path uses.
 *
 * The result is fed to the same place a global model is: the block's warp
 * matrix at WARPEDMODEL_PREC_BITS, validated by the setup shear process, after
 * which the same kernel predicts the block. Nothing here is GPU specific.
 */
#ifndef AV1GPU_FRONTEND_WARP_HPP
#define AV1GPU_FRONTEND_WARP_HPP

#include <cstdint>

#include "frontend/headers.hpp"

namespace av1gpu {
namespace frontend {

/* LEAST_SQUARES_SAMPLES_MAX: the most neighbours the fit ever sees. The scan
 * stops as soon as this many have been looked at, valid or not. */
constexpr int kWarpSamplesMax = 8;

/* LS_MV_MAX: a sample whose vector disagrees with the block's by this much or
 * more is dropped from the fit instead of being clipped. */
constexpr int kLsMvMax = 256;

/* WARPEDMODEL_TRANS_CLAMP and WARPEDMODEL_NONDIAGAFFINE_CLAMP: the ranges the
 * fitted parameters are clamped to. The diagonal entries are clamped around
 * the identity's 1 << WARPEDMODEL_PREC_BITS instead of around zero. */
constexpr int kWarpedmodelTransClamp = 1 << 23;
constexpr int kWarpedmodelNondiagClamp = 1 << 13;

/* Div_Lut (7.11.3.7): the reciprocal table the resolve divisor process looks
 * its factor up in, Q14, indexed by the divisor's mantissa. */
constexpr int kDivLutBits = 8;
constexpr int kDivLutPrecBits = 14;
constexpr int kDivLutNum = 257;

constexpr uint16_t kDivLut[kDivLutNum] = {
    16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768, 15709, 15650,
    15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142, 15087, 15033, 14980, 14926,
    14873, 14821, 14769, 14717, 14665, 14614, 14564, 14513, 14463, 14413, 14364, 14315, 14266,
    14218, 14170, 14122, 14075, 14028, 13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662,
    13618, 13574, 13530, 13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107,
    13066, 13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633, 12596,
    12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228, 12193, 12157, 12122,
    12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848, 11815, 11782, 11749, 11716, 11683,
    11651, 11619, 11586, 11555, 11523, 11491, 11460, 11429, 11398, 11367, 11336, 11305, 11275,
    11245, 11215, 11185, 11155, 11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894,
    10866, 10838, 10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
    10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255, 10230, 10205,
    10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986,  9963,  9939,  9916,  9892,
    9869,  9846,  9823,  9800,  9777,  9754,  9732,  9709,  9687,  9664,  9642,  9620,  9598,
    9576,  9554,  9533,  9511,  9489,  9468,  9447,  9425,  9404,  9383,  9362,  9341,  9321,
    9300,  9279,  9259,  9239,  9218,  9198,  9178,  9158,  9138,  9118,  9098,  9079,  9059,
    9039,  9020,  9001,  8981,  8962,  8943,  8924,  8905,  8886,  8867,  8849,  8830,  8812,
    8793,  8775,  8756,  8738,  8720,  8702,  8684,  8666,  8648,  8630,  8613,  8595,  8577,
    8560,  8542,  8525,  8508,  8490,  8473,  8456,  8439,  8422,  8405,  8389,  8372,  8355,
    8339,  8322,  8306,  8289,  8273,  8257,  8240,  8224,  8208,  8192,
};

/* floor_log2 of a 64 bit value: the position of the top set bit. */
inline int floor_log2_64(uint64_t v) {
  int n = 0;
  while (v >> (n + 1)) ++n;
  return n;
}

/* resolve divisor (7.11.3.7), 64 bit: the reciprocal of `d` as a Q14 factor and
 * a shift, so that `v / d` is `Round2Signed(v * factor, shift)`. The mantissa
 * keeps eight bits, which is what makes one table of 257 entries enough. */
inline void resolve_divisor_64(int64_t d, int* shift, int64_t* factor) {
  const uint64_t abs_d = d < 0 ? static_cast<uint64_t>(-d) : static_cast<uint64_t>(d);
  const int n = floor_log2_64(abs_d);
  const int64_t e = static_cast<int64_t>(abs_d - (static_cast<uint64_t>(1) << n));
  const int64_t f = n > kDivLutBits
                        ? (e + (static_cast<int64_t>(1) << (n - kDivLutBits - 1))) >>
                              (n - kDivLutBits)
                        : e << (kDivLutBits - n);
  *shift = n + kDivLutPrecBits;
  *factor = d < 0 ? -static_cast<int64_t>(kDivLut[f]) : static_cast<int64_t>(kDivLut[f]);
}

/* Round2Signed at 64 bit: rounds towards the larger value, so a negative input
 * rounds towards zero. `n` is zero when the divisor's shift is zero, where the
 * rounding term is zero as well. */
inline int64_t round2s_64(int64_t v, int n) {
  const int64_t round = n > 0 ? (static_cast<int64_t>(1) << (n - 1)) : 0;
  return v >= 0 ? (v + round) >> n : -((-v + round) >> n);
}

inline int clip3_local(int lo, int v, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* The warp estimation process (7.11.3.8).
 *
 * `cand` holds NumSamples entries of {source y, source x, destination y,
 * destination x} in 1/8 luma samples, as find_warp_samples produced them.
 * `mid_x`/`mid_y` are the block's centre in luma samples and `mv_x`/`mv_y` its
 * vector, in the component order the rest of this decoder uses (x first; the
 * specification writes the row component first).
 *
 * Writes the six LocalWarpParams and returns LocalValid: false when the normal
 * equations are singular, which is the only way the fit itself can fail. The
 * setup shear process is a separate check and belongs to the caller, because it
 * is the same code the global motion path runs.
 */
inline bool warp_estimation(const int cand[][4], int num_samples, int mid_x, int mid_y, int mv_x,
                            int mv_y, int params[6]) {
  int a00 = 0, a01 = 0, a11 = 0;
  int bx0 = 0, bx1 = 0;
  int by0 = 0, by1 = 0;

  const int suy = mid_y * 8;
  const int sux = mid_x * 8;
  const int duy = suy + mv_y;
  const int dux = sux + mv_x;

  /* ls_product: the least squares fit works on positions scaled by 2^-2, twice
   * shifted so the normal equations stay linear in the unknown. */
  const auto ls_product = [](int a, int b) { return ((a * b) >> 2) + (a + b); };

  for (int i = 0; i < num_samples; ++i) {
    const int sy = cand[i][0] - suy;
    const int sx = cand[i][1] - sux;
    const int dy = cand[i][2] - duy;
    const int dx = cand[i][3] - dux;
    const int diff_x = sx - dx;
    const int diff_y = sy - dy;
    if ((diff_x < 0 ? -diff_x : diff_x) < kLsMvMax && (diff_y < 0 ? -diff_y : diff_y) < kLsMvMax) {
      a00 += ls_product(sx, sx) + 8;
      a01 += ls_product(sx, sy) + 4;
      a11 += ls_product(sy, sy) + 8;
      bx0 += ls_product(sx, dx) + 8;
      bx1 += ls_product(sy, dx) + 4;
      by0 += ls_product(sx, dy) + 4;
      by1 += ls_product(sy, dy) + 8;
    }
  }

  const int64_t det = static_cast<int64_t>(a00) * a11 - static_cast<int64_t>(a01) * a01;
  if (det == 0) return false;

  int div_shift = 0;
  int64_t div_factor = 0;
  resolve_divisor_64(det, &div_shift, &div_factor);
  div_shift -= kWarpmodelPrecBits;
  if (div_shift < 0) {
    div_factor <<= -div_shift;
    div_shift = 0;
  }

  /* diag: a diagonal entry is a scale around one, so it is clamped around
   * 1 << WARPEDMODEL_PREC_BITS; the two off-diagonal entries are shears and are
   * clamped around zero. */
  const auto scaled = [&](int64_t v) { return round2s_64(v * div_factor, div_shift); };
  const auto diag = [&](int64_t v) {
    return clip3_local(kGmIdentityParam - kWarpedmodelNondiagClamp + 1, static_cast<int>(scaled(v)),
                       kGmIdentityParam + kWarpedmodelNondiagClamp - 1);
  };
  const auto nondiag = [&](int64_t v) {
    return clip3_local(-kWarpedmodelNondiagClamp + 1, static_cast<int>(scaled(v)),
                       kWarpedmodelNondiagClamp - 1);
  };

  params[2] = diag(static_cast<int64_t>(a11) * bx0 - static_cast<int64_t>(a01) * bx1);
  params[3] = nondiag(static_cast<int64_t>(a00) * bx1 - static_cast<int64_t>(a01) * bx0);
  params[4] = nondiag(static_cast<int64_t>(a11) * by0 - static_cast<int64_t>(a01) * by1);
  params[5] = diag(static_cast<int64_t>(a00) * by1 - static_cast<int64_t>(a01) * by0);

  /* The translation is whatever puts the block's own centre on its own vector
   * once the linear part is fixed. */
  const int64_t vx =
      static_cast<int64_t>(mv_x) * (1 << (kWarpmodelPrecBits - 3)) -
      (static_cast<int64_t>(mid_x) * (params[2] - kGmIdentityParam) +
       static_cast<int64_t>(mid_y) * params[3]);
  const int64_t vy = static_cast<int64_t>(mv_y) * (1 << (kWarpmodelPrecBits - 3)) -
                     (static_cast<int64_t>(mid_x) * params[4] +
                      static_cast<int64_t>(mid_y) * (params[5] - kGmIdentityParam));
  params[0] = clip3_local(-kWarpedmodelTransClamp, static_cast<int>(vx),
                          kWarpedmodelTransClamp - 1);
  params[1] = clip3_local(-kWarpedmodelTransClamp, static_cast<int>(vy),
                          kWarpedmodelTransClamp - 1);
  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_WARP_HPP */
