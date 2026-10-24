/*
 * AV1onGPU - shared tables
 *
 * The transform bases are generated here, on the host, from the exact
 * definitions, and then uploaded once per device. Doing it this way (instead of
 * a hard coded table) keeps the numbers identical on every backend regardless
 * of the maths library the device compiler ships, and makes the orientation
 * obvious: B is the *synthesis* matrix.
 *
 *       inverse:  out[i] = ( sum_k in[k] * B[i][k] + 2048 ) >> 12
 *       forward:  out[k] = ( sum_i in[i] * B[i][k] + 2048 ) >> 12
 *
 * B is orthonormal for the DCT-II (with the sqrt(1/n) / sqrt(2/n) scaling) and
 * for the DST-IV, so the forward transform is exactly the transpose.
 */
#include "core/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace av1gpu {
namespace internal {
namespace tables {
namespace {

const int kDctSizes[] = {4, 8, 16, 32, 64};
const int kDstSizes[] = {4, 8, 16};

struct BasisTable {
  std::vector<int> data;
  int dct[65];
  int adst[65];
  int idtx[65];

  BasisTable() {
    std::memset(dct, 0, sizeof(dct));
    std::memset(adst, 0, sizeof(adst));
    std::memset(idtx, 0, sizeof(idtx));

    for (int n : kDctSizes) {
      dct[n] = static_cast<int>(data.size());
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
          const double scale = (k == 0) ? std::sqrt(1.0 / n) : std::sqrt(2.0 / n);
          const double v = scale * std::cos(M_PI * (2.0 * i + 1.0) * k / (2.0 * n));
          data.push_back(quantise(v));
        }
      }
    }
    for (int n : kDstSizes) {
      adst[n] = static_cast<int>(data.size());
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
          const double v = std::sqrt(2.0 / n) *
                           std::sin(M_PI * (2.0 * i + 1.0) * (2.0 * k + 1.0) / (4.0 * n));
          data.push_back(quantise(v));
        }
      }
    }
    for (int n : kDctSizes) {
      idtx[n] = static_cast<int>(data.size());
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) data.push_back(i == k ? 4096 : 0);
    }
  }

  static int quantise(double v) {
    const double x = v * 4096.0;
    return static_cast<int>(std::floor(x + 0.5));
  }
};

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

const std::vector<int>& basis() {
  static const BasisTable table;
  return table.data;
}

int basis_dct(int n) {
  static const BasisTable table;
  return n < 65 ? table.dct[n] : -1;
}

int basis_adst(int n) {
  static const BasisTable table;
  return n < 65 ? table.adst[n] : -1;
}

int basis_identity(int n) {
  static const BasisTable table;
  return n < 65 ? table.idtx[n] : -1;
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
