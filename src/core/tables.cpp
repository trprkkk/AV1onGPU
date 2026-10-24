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
#include <cstring>

namespace av1gpu {
namespace internal {
namespace tables {
namespace {

/* ---- subpel filters ---------------------------------------------------- */

/* Subpel_Filters[6][16][8] from the specification (section 7.11.3.5).
 *
 * Rows 0..3 are the four filter classes a block can name - regular, smooth,
 * sharp, bilinear - and are indexed by AV1G_FILT_*. Rows 4 and 5 are the four
 * tap versions of the regular and smooth filters that a prediction block only
 * four samples wide or high uses. All six rows are stored with eight taps so
 * the filter loops stay uniform: the four tap rows carry zeros at both ends,
 * which puts their taps at the same sample offsets the specification uses.
 *
 * The weights are Q7 (each row sums to 128) and there are sixteen phases per
 * row, one per 1/16 sample. The fractional part of a motion vector is
 * therefore *not* the index: a vector stored in 1/8 samples has even phases in
 * luma and every phase in chroma (the chroma vector is halved). Phase 0 is a
 * straight sample copy.
 *
 * These are the specification's numbers, not an approximation of them: an
 * interpolated prediction is only bit exact against a reference decoder if the
 * weights, the rounding and the phase resolution all match. */
constexpr int kFilterRows = 6;
constexpr int kFilterPhases = 16;
constexpr int kFilterTaps = 8;

const int kSubpelFilters[kFilterRows][kFilterPhases][kFilterTaps] = {
    { /* regular (EIGHTTAP) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 2, -6, 126, 8, -2, 0, 0},
        {0, 2, -10, 122, 18, -4, 0, 0},
        {0, 2, -12, 116, 28, -8, 2, 0},
        {0, 2, -14, 110, 38, -10, 2, 0},
        {0, 2, -14, 102, 48, -12, 2, 0},
        {0, 2, -16, 94, 58, -12, 2, 0},
        {0, 2, -14, 84, 66, -12, 2, 0},
        {0, 2, -14, 76, 76, -14, 2, 0},
        {0, 2, -12, 66, 84, -14, 2, 0},
        {0, 2, -12, 58, 94, -16, 2, 0},
        {0, 2, -12, 48, 102, -14, 2, 0},
        {0, 2, -10, 38, 110, -14, 2, 0},
        {0, 2, -8, 28, 116, -12, 2, 0},
        {0, 0, -4, 18, 122, -10, 2, 0},
        {0, 0, -2, 8, 126, -6, 2, 0},
    },
    { /* smooth (EIGHTTAP_SMOOTH) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 2, 28, 62, 34, 2, 0, 0},
        {0, 0, 26, 62, 36, 4, 0, 0},
        {0, 0, 22, 62, 40, 4, 0, 0},
        {0, 0, 20, 60, 42, 6, 0, 0},
        {0, 0, 18, 58, 44, 8, 0, 0},
        {0, 0, 16, 56, 46, 10, 0, 0},
        {0, -2, 16, 54, 48, 12, 0, 0},
        {0, -2, 14, 52, 52, 14, -2, 0},
        {0, 0, 12, 48, 54, 16, -2, 0},
        {0, 0, 10, 46, 56, 16, 0, 0},
        {0, 0, 8, 44, 58, 18, 0, 0},
        {0, 0, 6, 42, 60, 20, 0, 0},
        {0, 0, 4, 40, 62, 22, 0, 0},
        {0, 0, 4, 36, 62, 26, 0, 0},
        {0, 0, 2, 34, 62, 28, 2, 0},
    },
    { /* sharp (EIGHTTAP_SHARP) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {-2, 2, -6, 126, 8, -2, 2, 0},
        {-2, 6, -12, 124, 16, -6, 4, -2},
        {-2, 8, -18, 120, 26, -10, 6, -2},
        {-4, 10, -22, 116, 38, -14, 6, -2},
        {-4, 10, -22, 108, 48, -18, 8, -2},
        {-4, 10, -24, 100, 60, -20, 8, -2},
        {-4, 10, -24, 90, 70, -22, 10, -2},
        {-4, 12, -24, 80, 80, -24, 12, -4},
        {-2, 10, -22, 70, 90, -24, 10, -4},
        {-2, 8, -20, 60, 100, -24, 10, -4},
        {-2, 8, -18, 48, 108, -22, 10, -4},
        {-2, 6, -14, 38, 116, -22, 10, -4},
        {-2, 6, -10, 26, 120, -18, 8, -2},
        {-2, 4, -6, 16, 124, -12, 6, -2},
        {0, 2, -2, 8, 126, -6, 2, -2},
    },
    { /* bilinear (BILINEAR) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, 0, 120, 8, 0, 0, 0},
        {0, 0, 0, 112, 16, 0, 0, 0},
        {0, 0, 0, 104, 24, 0, 0, 0},
        {0, 0, 0, 96, 32, 0, 0, 0},
        {0, 0, 0, 88, 40, 0, 0, 0},
        {0, 0, 0, 80, 48, 0, 0, 0},
        {0, 0, 0, 72, 56, 0, 0, 0},
        {0, 0, 0, 64, 64, 0, 0, 0},
        {0, 0, 0, 56, 72, 0, 0, 0},
        {0, 0, 0, 48, 80, 0, 0, 0},
        {0, 0, 0, 40, 88, 0, 0, 0},
        {0, 0, 0, 32, 96, 0, 0, 0},
        {0, 0, 0, 24, 104, 0, 0, 0},
        {0, 0, 0, 16, 112, 0, 0, 0},
        {0, 0, 0, 8, 120, 0, 0, 0},
    },
    { /* four tap regular, for blocks 4 wide or high */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, -4, 126, 8, -2, 0, 0},
        {0, 0, -8, 122, 18, -4, 0, 0},
        {0, 0, -10, 116, 28, -6, 0, 0},
        {0, 0, -12, 110, 38, -8, 0, 0},
        {0, 0, -12, 102, 48, -10, 0, 0},
        {0, 0, -14, 94, 58, -10, 0, 0},
        {0, 0, -12, 84, 66, -10, 0, 0},
        {0, 0, -12, 76, 76, -12, 0, 0},
        {0, 0, -10, 66, 84, -12, 0, 0},
        {0, 0, -10, 58, 94, -14, 0, 0},
        {0, 0, -10, 48, 102, -12, 0, 0},
        {0, 0, -8, 38, 110, -12, 0, 0},
        {0, 0, -6, 28, 116, -10, 0, 0},
        {0, 0, -4, 18, 122, -8, 0, 0},
        {0, 0, -2, 8, 126, -4, 0, 0},
    },
    { /* four tap smooth, for blocks 4 wide or high */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, 30, 62, 34, 2, 0, 0},
        {0, 0, 26, 62, 36, 4, 0, 0},
        {0, 0, 22, 62, 40, 4, 0, 0},
        {0, 0, 20, 60, 42, 6, 0, 0},
        {0, 0, 18, 58, 44, 8, 0, 0},
        {0, 0, 16, 56, 46, 10, 0, 0},
        {0, 0, 14, 54, 48, 12, 0, 0},
        {0, 0, 12, 52, 52, 12, 0, 0},
        {0, 0, 12, 48, 54, 14, 0, 0},
        {0, 0, 10, 46, 56, 16, 0, 0},
        {0, 0, 8, 44, 58, 18, 0, 0},
        {0, 0, 6, 42, 60, 20, 0, 0},
        {0, 0, 4, 40, 62, 22, 0, 0},
        {0, 0, 4, 36, 62, 26, 0, 0},
        {0, 0, 2, 34, 62, 30, 0, 0},
    },
};


struct FilterTable {
  std::vector<int> data;
  FilterTable() {
    data.resize(static_cast<size_t>(kFilterRows) * kFilterPhases * kFilterTaps);
    for (int row = 0; row < kFilterRows; ++row)
      for (int phase = 0; phase < kFilterPhases; ++phase)
        for (int tap = 0; tap < kFilterTaps; ++tap)
          data[(static_cast<size_t>(row) * kFilterPhases + phase) * kFilterTaps + tap] =
              kSubpelFilters[row][phase][tap];
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

/* ---- masked compound: the wedge codebook ------------------------------- */

namespace {

/* Wedge_Codebook[3][16][3] from the specification (section 7.11.3.11): the
 * master mask direction and the offset, in eighths of the block size, that
 * places the wedge for one block shape. The three rows are the tall shapes, the
 * wide ones and the square ones - the wedge table is indexed by shape rather
 * than by size, so an 8x16 and a 16x32 block share a row. */
struct WedgeCode {
  int dir;
  int xoff;
  int yoff;
};

const WedgeCode kWedgeCodebook[3][16] = {
    {{AV1G_WEDGE_OBLIQUE27, 4, 4}, {AV1G_WEDGE_OBLIQUE63, 4, 4},
     {AV1G_WEDGE_OBLIQUE117, 4, 4}, {AV1G_WEDGE_OBLIQUE153, 4, 4},
     {AV1G_WEDGE_HORIZONTAL, 4, 2}, {AV1G_WEDGE_HORIZONTAL, 4, 4},
     {AV1G_WEDGE_HORIZONTAL, 4, 6}, {AV1G_WEDGE_VERTICAL, 4, 4},
     {AV1G_WEDGE_OBLIQUE27, 4, 2}, {AV1G_WEDGE_OBLIQUE27, 4, 6},
     {AV1G_WEDGE_OBLIQUE153, 4, 2}, {AV1G_WEDGE_OBLIQUE153, 4, 6},
     {AV1G_WEDGE_OBLIQUE63, 2, 4}, {AV1G_WEDGE_OBLIQUE63, 6, 4},
     {AV1G_WEDGE_OBLIQUE117, 2, 4}, {AV1G_WEDGE_OBLIQUE117, 6, 4}},
    {{AV1G_WEDGE_OBLIQUE27, 4, 4}, {AV1G_WEDGE_OBLIQUE63, 4, 4},
     {AV1G_WEDGE_OBLIQUE117, 4, 4}, {AV1G_WEDGE_OBLIQUE153, 4, 4},
     {AV1G_WEDGE_VERTICAL, 2, 4}, {AV1G_WEDGE_VERTICAL, 4, 4},
     {AV1G_WEDGE_VERTICAL, 6, 4}, {AV1G_WEDGE_HORIZONTAL, 4, 4},
     {AV1G_WEDGE_OBLIQUE27, 4, 2}, {AV1G_WEDGE_OBLIQUE27, 4, 6},
     {AV1G_WEDGE_OBLIQUE153, 4, 2}, {AV1G_WEDGE_OBLIQUE153, 4, 6},
     {AV1G_WEDGE_OBLIQUE63, 2, 4}, {AV1G_WEDGE_OBLIQUE63, 6, 4},
     {AV1G_WEDGE_OBLIQUE117, 2, 4}, {AV1G_WEDGE_OBLIQUE117, 6, 4}},
    {{AV1G_WEDGE_OBLIQUE27, 4, 4}, {AV1G_WEDGE_OBLIQUE63, 4, 4},
     {AV1G_WEDGE_OBLIQUE117, 4, 4}, {AV1G_WEDGE_OBLIQUE153, 4, 4},
     {AV1G_WEDGE_HORIZONTAL, 4, 2}, {AV1G_WEDGE_HORIZONTAL, 4, 6},
     {AV1G_WEDGE_VERTICAL, 2, 4}, {AV1G_WEDGE_VERTICAL, 6, 4},
     {AV1G_WEDGE_OBLIQUE27, 4, 2}, {AV1G_WEDGE_OBLIQUE27, 4, 6},
     {AV1G_WEDGE_OBLIQUE153, 4, 2}, {AV1G_WEDGE_OBLIQUE153, 4, 6},
     {AV1G_WEDGE_OBLIQUE63, 2, 4}, {AV1G_WEDGE_OBLIQUE63, 6, 4},
     {AV1G_WEDGE_OBLIQUE117, 2, 4}, {AV1G_WEDGE_OBLIQUE117, 6, 4}},
};

/* Wedge_Bits[bsize] > 0: the shapes that carry a wedge at all. Everything
 * larger than 32x32 uses the difference weighted mask instead, which is why the
 * master mask only ever has to cover 32 samples. */
bool wedge_shape(int w, int h, int* shape) {
  if (w < 8 || h < 8 || w > 32 || h > 32) return false;
  *shape = h > w ? 0 : (h < w ? 1 : 2);
  return true;
}

} /* namespace */

bool wedge_entry(int block_w, int block_h, int wedge_index, int* dir, int* xoff, int* yoff,
                 int* flip) {
  int shape = 0;
  if (wedge_index < 0 || wedge_index >= 16 || !wedge_shape(block_w, block_h, &shape))
    return false;
  const WedgeCode& c = kWedgeCodebook[shape][wedge_index];
  *dir = c.dir;
  *xoff = 32 - ((c.xoff * block_w) >> 3);
  *yoff = 32 - ((c.yoff * block_h) >> 3);
  /* flipSign: the specification stores both signs of every wedge and picks one
   * by the sign of the mask's mean. Only the first row and the first column are
   * summed, which is enough: the wedge is monotone along both. */
  int sum = 0;
  for (int i = 0; i < block_w; ++i) sum += av1g_wedge_master(c.dir, *yoff, *xoff + i);
  for (int i = 1; i < block_h; ++i) sum += av1g_wedge_master(c.dir, *yoff + i, *xoff);
  const int n = block_w + block_h - 1;
  const int avg = (sum + n / 2) / n;
  *flip = avg < 32 ? 1 : 0;
  return true;
}

} /* namespace tables */
} /* namespace internal */
} /* namespace av1gpu */
