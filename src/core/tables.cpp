/*
 * AV1onGPU - shared tables
 *
 * What lives here is the host side data the backends share: the subpel filter
 * weights, the film grain generator, and the colour transforms. The transform
 * and prediction tables are not here: they are normative, so they are generated
 * from the specification (tools/gen_kernel_tables.py) and compiled into the
 * kernel header, which every backend includes. Keeping one copy means the CPU
 * reference path and the GPU kernels cannot drift apart.
 */
#include "core/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace av1gpu {
namespace internal {
namespace tables {
namespace {

/* ---- subpel filters ---------------------------------------------------- */

double sinc(double x) {
  if (std::fabs(x) < 1e-9) return 1.0;
  const double px = M_PI * x;
  return std::sin(px) / px;
}

double lanczos(double x, double a) {
  if (x <= -a || x >= a) return 0.0;
  return sinc(x) * sinc(x / a);
}

struct FilterSpec {
  int taps;      /* 2, 4 or 8                                              */
  double window; /* Lanczos a; smaller is sharper                          */
  int first_tap; /* index in the padded 8 tap row                          */
};

FilterSpec filter_spec(int filt) {
  switch (filt) {
    case AV1G_FILT_SMOOTH:
      return {4, 2.0, 2};
    case AV1G_FILT_SHARP:
      return {8, 2.0, 0};
    case AV1G_FILT_BILINEAR:
      return {2, 1.0, 3};
    case AV1G_FILT_REGULAR:
    default:
      return {8, 4.0, 0};
  }
}

struct FilterTable {
  std::vector<int> data;
  FilterTable() {
    data.assign(AV1G_FILT_COUNT * 8 * 8, 0);
    for (int filt = 0; filt < AV1G_FILT_COUNT; ++filt) {
      const FilterSpec spec = filter_spec(filt);
      for (int phase = 0; phase < 8; ++phase) {
        const double t = phase / 8.0;
        int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int sum = 0;
        for (int i = 0; i < spec.taps; ++i) {
          /* tap i sits at sample offset (first_tap + i) - 3 = -3..+4 */
          const int centre = i + spec.first_tap - 3;
          double v = lanczos(centre - t, spec.window);
          if (spec.taps == 2) v = (i == 0) ? 1.0 - t : t; /* exactly bilinear */
          w[spec.first_tap + i] = static_cast<int>(std::floor(v * 128.0 + 0.5));
          sum += w[spec.first_tap + i];
        }
        /* force an exact unity DC gain */
        int best = 0;
        for (int i = 1; i < 8; ++i)
          if (w[i] > w[best]) best = i;
        w[best] += 128 - sum;
        for (int i = 0; i < 8; ++i)
          data[((filt * 8) + phase) * 8 + i] = w[i];
      }
    }
  }
};

} /* namespace */

/* Transform_Row_Shift, keyed by the transform size.
 *
 * The specification's table is
 *
 *   Transform_Row_Shift[ TX_SIZES_ALL ] = {
 *     0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2
 *   }
 *
 * indexed by the specification's Tx_Size numbers, in this order: 4x4, 8x8,
 * 16x16, 32x32, 64x64, 4x8, 8x4, 8x16, 16x8, 16x32, 32x16, 32x64, 64x32,
 * 4x16, 16x4, 8x32, 32x8, 16x64, 64x16. Two things in it surprise: the shift
 * stops growing at 32x32 (larger transforms are coded with the same scale),
 * and the tall thin sizes 4x16/16x4 shift by 1 while 8x32/32x8 shift by 2 -
 * it is not a function of the width alone, so it has to be the table, not a
 * formula. */
int tx_row_shift(int w, int h) {
  static const struct {
    int w, h, shift;
  } kTable[] = {
      {4, 4, 0},   {8, 8, 1},   {16, 16, 2}, {32, 32, 2}, {64, 64, 2},
      {4, 8, 0},   {8, 4, 0},   {8, 16, 1},  {16, 8, 1},  {16, 32, 1},
      {32, 16, 1}, {32, 64, 1}, {64, 32, 1}, {4, 16, 1},  {16, 4, 1},
      {8, 32, 2},  {32, 8, 2},  {16, 64, 2}, {64, 16, 2},
  };
  for (const auto& e : kTable)
    if (e.w == w && e.h == h) return e.shift;
  return 0;
}

const std::vector<int>& filters() {
  static const FilterTable table;
  return table.data;
}

/* ------------------------------------------------------------------------- */
/* film grain template                                                       */
/* ------------------------------------------------------------------------- */

namespace {

/* xorshift, matching the distribution the AV1 film grain model expects (zero
 * mean, roughly gaussian tails). */
uint32_t rng_state = 1u;

uint32_t rng_next() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

} /* namespace */

void grain_template(int seed, int stride, int rows, std::vector<signed char>* out) {
  out->assign(static_cast<size_t>(stride) * rows, 0);
  rng_state = static_cast<uint32_t>(seed) * 2654435761u + 12345u;
  if (rng_state == 0) rng_state = 1;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < stride; ++x) {
      /* sum of two uniforms gives a triangular, zero mean distribution */
      const int a = static_cast<int>(rng_next() % 129) - 64;
      const int b = static_cast<int>(rng_next() % 129) - 64;
      int v = (a + b) / 2;
      if (v < -64) v = -64;
      if (v > 63) v = 63;
      (*out)[static_cast<size_t>(y) * stride + x] = static_cast<signed char>(v);
    }
  }
}

/* ------------------------------------------------------------------------- */
/* colour                                                                    */
/* ------------------------------------------------------------------------- */

void color_matrix(ColorMatrix matrix, ColorRange range, int bit_depth, int m[8],
                  int o[3]) {
  /* Y'CbCr -> R'G'B', Q16, samples already normalised to 8 bit range by the
   * kernel.  The offsets fold in the 16 / 128 / 0 nominal levels. */
  const int shift = bit_depth - 8;
  const int y_off = (range == ColorRange::Limited) ? (16 << shift) : 0;
  const int c_off = (range == ColorRange::Limited) ? (128 << shift)
                                                  : (1 << (bit_depth - 1));
  const int half = 1 << 15;
  switch (matrix) {
    case ColorMatrix::Bt2020: {
      m[0] = 76293;   /* 1.1644 * Y            */
      m[1] = 109969;  /* 1.6781 * (V - 128)    */
      m[2] = 76293;
      m[3] = -12255;  /* -0.1870 * (U - 128)   */
      m[4] = -42598;  /* -0.6500 * (V - 128)   */
      m[5] = 76293;
      m[6] = 121634;  /* 1.8556 * (U - 128)    */
      o[0] = half - m[0] * y_off - m[1] * c_off;
      o[1] = half - m[2] * y_off - m[3] * c_off - m[4] * c_off;
      o[2] = half - m[5] * y_off - m[6] * c_off;
      break;
    }
    case ColorMatrix::Identity: {
      /* GBR: the three planes are already R, G and B */
      m[0] = 65536;
      m[1] = 0;
      m[2] = 65536;
      m[3] = 0;
      m[4] = 0;
      m[5] = 65536;
      m[6] = 0;
      o[0] = 0;
      o[1] = 0;
      o[2] = 0;
      break;
    }
    case ColorMatrix::Bt709:
    default: {
      m[0] = 76293;   /* 1.1644 * Y            */
      m[1] = 117504;  /* 1.7930 * (V - 128)    */
      m[2] = 76293;
      m[3] = -13926;  /* -0.2125 * (U - 128)   */
      m[4] = -34937;  /* -0.5331 * (V - 128)   */
      m[5] = 76293;
      m[6] = 138412;  /* 2.1124 * (U - 128)    */
      o[0] = half - m[0] * y_off - m[1] * c_off;
      o[1] = half - m[2] * y_off - m[3] * c_off - m[4] * c_off;
      o[2] = half - m[5] * y_off - m[6] * c_off;
      break;
    }
  }
}

void frame_to_rgb(const Frame& frame, ColorMatrix matrix, ColorRange range,
                  std::vector<uint8_t>* rgb) {
  rgb->resize(static_cast<size_t>(frame.width) * frame.height * 3);
  int m[8];
  int o[3];
  /* the nominal-range offsets are folded into o[] by color_matrix(), exactly
   * like the kernel gets them */
  color_matrix(matrix, range, frame.bit_depth, m, o);

  for (int y = 0; y < frame.height; ++y) {
    const uint16_t* yr = frame.plane(0) + static_cast<size_t>(y) * frame.stride[0];
    const uint16_t* ur = frame.plane(1) + static_cast<size_t>(y >> frame.ss_y) * frame.stride[1];
    const uint16_t* vr = frame.plane(2) + static_cast<size_t>(y >> frame.ss_y) * frame.stride[2];
    uint8_t* row = rgb->data() + static_cast<size_t>(y) * frame.width * 3;
    for (int x = 0; x < frame.width; ++x) {
      const int yy = static_cast<int>(yr[x]);
      const int uu = static_cast<int>(ur[x >> frame.ss_x]);
      const int vv = static_cast<int>(vr[x >> frame.ss_x]);
      int r = (m[0] * yy + m[1] * vv + o[0]) >> 16;
      int g = (m[2] * yy + m[3] * uu + m[4] * vv + o[1]) >> 16;
      int b = (m[5] * yy + m[6] * uu + o[2]) >> 16;
      row[3 * x + 0] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
      row[3 * x + 1] = static_cast<uint8_t>(g < 0 ? 0 : (g > 255 ? 255 : g));
      row[3 * x + 2] = static_cast<uint8_t>(b < 0 ? 0 : (b > 255 ? 255 : b));
    }
  }
}

} /* namespace tables */
} /* namespace internal */
} /* namespace av1gpu */
