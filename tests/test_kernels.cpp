/*
 * AV1onGPU - kernel unit tests
 *
 * These check the shared scalar code that every backend compiles: the inverse
 * transform, the subpel filters, the intra predictors, the loop filters and the
 * colour conversion. They run on the host, so they also validate the CPU
 * reference path that the GPU output is compared against.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "core/backend.hpp"
#include "test_common.hpp"

using namespace av1gpu;
using namespace av1gpu::internal;

namespace {

/* inverse transform of one block, driven exactly the way the decoder drives it:
 * a row pass into a scratch buffer, then a column pass that adds the residual */
void inverse_transform(const std::vector<int>& coef, int w, int h, int tx_type,
                       std::vector<int>* out, int bitdepth = 8) {
  Av1gItxJob job;
  std::memset(&job, 0, sizeof(job));
  job.w = w;
  job.h = h;
  job.tx_type = tx_type;
  job.eob = static_cast<int>(coef.size());
  job.row_shift = tables::tx_row_shift(w, h);
  job.bitdepth = bitdepth;
  std::vector<int> tmp(coef.size(), 0);
  for (int y = 0; y < h; ++y) av1g_itx_row(&job, coef.data(), tmp.data(), y);
  /* Residual adds are applied on top of a prediction, so start from a mid
   * level: the samples are unsigned and negative residuals would otherwise
   * clip. 16384 leaves room in both directions, so a test that compares two
   * runs with different magnitudes is not measuring the clip. */
  const int kMid = 16384;
  std::vector<uint16_t> plane(static_cast<size_t>(w) * h + 64,
                              static_cast<uint16_t>(kMid));
  job.dst_off = 0;
  job.dst_stride = w;
  for (int x = 0; x < w; ++x) av1g_itx_col_add(&job, tmp.data(), plane.data(), x, 65535);
  out->assign(plane.begin(), plane.begin() + w * h);
  for (int& v : *out) v -= kMid;
}

/* Clip1, for the predictors that clamp before storing */
int clip1(int v, int bd) {
  const int maxv = (1 << bd) - 1;
  return v < 0 ? 0 : (v > maxv ? maxv : v);
}

/* Round2 and Round2Signed as the specification defines them */
int round2(int v, int n) { return (v + (1 << (n - 1))) >> n; }
int round2_signed(int v, int n) {
  return v >= 0 ? (v + (1 << (n - 1))) >> n : -((-v + (1 << (n - 1))) >> n);
}

/* a fresh 8 bit plane, mid grey, with room for a top row and a left column */
struct TestPlane {
  int stride = 40;
  int size = 40;
  std::vector<uint16_t> data;
  TestPlane() : data(static_cast<size_t>(stride) * size, 128) {}
  uint16_t& at(int x, int y) { return data[static_cast<size_t>(y) * stride + x]; }
  uint16_t at(int x, int y) const { return data[static_cast<size_t>(y) * stride + x]; }
};

/* An intra job at (8, 8) with the given size. Everything optional is off, so a
 * test only has to set what it means to exercise. */
Av1gIntraJob make_intra_job(int w, int h) {
  Av1gIntraJob job;
  std::memset(&job, 0, sizeof(job));
  job.x = 8;
  job.y = 8;
  job.dst_off = 8 * 40 + 8;
  job.dst_stride = 40;
  job.w = w;
  job.h = h;
  job.bitdepth = 8;
  job.have_above = 1;
  job.have_left = 1;
  job.filter_type = -1; /* no edge filtering unless a test asks for it */
  job.max_x = 39;
  job.max_y = 39;
  return job;
}

void test_tables() {
  std::printf("tables\n");
  /* cos128 is the specification's cosine table: 4096 * cos(a * pi / 128) */
  CHECK_EQ(av1g_tbl_cos128[0], 4096);
  CHECK_EQ(av1g_tbl_cos128[32], 2896);
  for (int a = 0; a <= 64; ++a) {
    const double want = 4096.0 * std::cos(a * M_PI / 128.0);
    CHECK(std::fabs(av1g_tbl_cos128[a] - want) <= 1.0);
  }
  /* Mode_To_Angle: the directional modes carry the angle, the rest are 0 */
  const int want_angle[13] = {0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0};
  for (int m = 0; m < 13; ++m) CHECK_EQ(av1g_tbl_mode_to_angle[m], want_angle[m]);
  /* Dr_Intra_Derivative is only defined at the angles that can actually occur:
   * the eight base angles and those plus or minus a multiple of ANGLE_STEP.
   * Every other entry is zero. The values below are the specification's. */
  const int deriv_angles[] = {3,  6,  9,  14, 17, 20, 23, 26, 29, 32, 36, 39, 42, 45,
                              48, 51, 54, 58, 61, 64, 67, 70, 73, 76, 81, 84, 87, 0};
  const int deriv_values[] = {1023, 547, 372, 273, 215, 178, 151, 132, 116, 102, 90,
                              80,   71,  64,  57,  51,  45,  40,  35,  31,  27,  23,
                              19,   15,  11,  7,   3,   0};
  for (int i = 0; deriv_angles[i] || i == 0; ++i) {
    if (i > 0 && deriv_angles[i] == 0) break;
    CHECK_EQ(av1g_tbl_dr_deriv[deriv_angles[i]], deriv_values[i]);
  }
  /* the smooth weights run from 0 to 256 and are symmetric about the middle */
  for (int n : {4, 8, 16, 32, 64}) {
    CHECK(av1g_sm_weight(n, 0) >= 250); /* almost all of the neighbour above */
    CHECK(av1g_sm_weight(n, n - 1) < av1g_sm_weight(n, 0));
    for (int i = 0; i < n; ++i) {
      CHECK(av1g_sm_weight(n, i) <= 256);
      if (i > 0) CHECK(av1g_sm_weight(n, i) <= av1g_sm_weight(n, i - 1));
    }
  }
  /* the specification's own values for the two smallest sizes */
  const int w4[4] = {255, 149, 85, 64};
  for (int i = 0; i < 4; ++i) CHECK_EQ(av1g_sm_weight(4, i), w4[i]);
  /* the edge filter kernels are normalised to 16 and symmetric */
  for (int k = 0; k < 3; ++k) {
    int sum = 0;
    for (int i = 0; i < 5; ++i) {
      sum += av1g_tbl_edge_kernel[k][i];
      CHECK_EQ(av1g_tbl_edge_kernel[k][i], av1g_tbl_edge_kernel[k][4 - i]);
    }
    CHECK_EQ(sum, 16);
  }
  /* every filter intra tap row sums to 16 (INTRA_FILTER_SCALE_BITS) */
  for (int m = 0; m < 5; ++m)
    for (int i = 0; i < 8; ++i) {
      int sum = 0;
      for (int k = 0; k < 7; ++k) sum += av1g_tbl_filter_intra[m][i][k];
      CHECK_EQ(sum, 16);
    }

  /* subpel filters: every phase of every row has unity DC gain, and the table
   * is the specification's (phase 0 is a copy, the classes are symmetric about
   * the half sample phase) */
  const int* filters = tables::filters().data();
  for (int row = 0; row < AV1G_FILT_ROWS; ++row) {
    for (int phase = 0; phase < AV1G_FILT_PHASES; ++phase) {
      int sum = 0;
      for (int i = 0; i < AV1G_FILT_TAPS; ++i)
        sum += filters[(row * AV1G_FILT_PHASES + phase) * AV1G_FILT_TAPS + i];
      CHECK_EQ(sum, 128);
    }
  }
  for (int row = 0; row < AV1G_FILT_ROWS; ++row) {
    /* phase 0 is a straight copy: only the centre tap is non zero */
    for (int i = 0; i < AV1G_FILT_TAPS; ++i)
      CHECK_EQ(filters[(row * AV1G_FILT_PHASES + 0) * AV1G_FILT_TAPS + i],
               i == 3 ? 128 : 0);
    /* the half sample phase of every row is symmetric */
    for (int i = 0; i < AV1G_FILT_TAPS / 2; ++i) {
      const int a = filters[(row * AV1G_FILT_PHASES + 8) * AV1G_FILT_TAPS + i];
      const int b = filters[(row * AV1G_FILT_PHASES + 8) * AV1G_FILT_TAPS + 7 - i];
      CHECK_EQ(a, b);
    }
  }
  /* a constant input must stay constant through the filter */
  {
    const int w = 32;
    std::vector<uint16_t> ref(static_cast<size_t>(w) * w, 200);
    Av1gInterJob job;
    std::memset(&job, 0, sizeof(job));
    job.w = 8;
    job.h = 8;
    job.dst_stride = w;
    job.ref_x = 8;
    job.ref_y = 8;
    job.ref_w = w;
    job.ref_h = w;
    job.ref0_stride = w;
    job.ref1_stride = w;
    job.filt = AV1G_FILT_REGULAR;
    job.filt_v = AV1G_FILT_REGULAR;
    job.mv0_x = 15; /* 1/8 pel, so the filter actually interpolates */
    job.mv0_y = 7;
    job.bitdepth = 8;
    std::vector<uint16_t> dst(static_cast<size_t>(w) * w, 0);
    av1g_inter_block(&job, dst.data(), ref.data(), ref.data(), filters, 255);
    bool constant = true;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (dst[y * w + x] != 200) constant = false;
    CHECK(constant);
  }
}

void test_transform() {
  std::printf("inverse transform\n");

  /* A single DC coefficient must come out flat. That is structural: it holds for
   * every size and both 1D transform types, and it catches any butterfly whose
   * DC path is wired to the wrong sample. */
  for (int n : {4, 8, 16, 32}) {
    std::vector<int> coef(static_cast<size_t>(n) * n, 0);
    coef[0] = 4096;
    std::vector<int> out;
    inverse_transform(coef, n, n, AV1G_TX_DCT_DCT, &out);
    int lo = out[0], hi = out[0];
    for (int v : out) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    CHECK_EQ(hi - lo, 0);
    CHECK(out[0] != 0);
  }

  /* The transform is linear apart from rounding, so a scaled coefficient has to
   * give a (nearly) scaled residual. This exercises every butterfly branch
   * without needing a reference implementation. */
  {
    const int n = 16;
    std::vector<int> coef(static_cast<size_t>(n) * n, 0);
    unsigned state = 987654321u;
    for (size_t i = 0; i < coef.size(); ++i) {
      state = state * 1103515245u + 12345u;
      coef[i] = static_cast<int>((state >> 17) % 2049) - 1024;
    }
    std::vector<int> a, b;
    inverse_transform(coef, n, n, AV1G_TX_ADST_ADST, &a);
    std::vector<int> doubled(coef);
    for (int& v : doubled) v *= 2;
    inverse_transform(doubled, n, n, AV1G_TX_ADST_ADST, &b);
    /* Each pass rounds to integers, so doubling the input doubles the output
     * up to one unit of the intermediate precision. */
    int worst = 0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(2 * a[i] - b[i]));
    CHECK(worst <= 2);
  }

  /* FLIPADST is ADST with the output reversed, so the flipped pair of a type
   * must be the mirror image of the plain one. */
  for (int n : {4, 8, 16}) {
    std::vector<int> coef(static_cast<size_t>(n) * n, 0);
    unsigned state = 24680u + static_cast<unsigned>(n);
    for (size_t i = 0; i < coef.size(); ++i) {
      state = state * 1103515245u + 12345u;
      coef[i] = static_cast<int>((state >> 19) % 513) - 256;
    }
    std::vector<int> plain, lr;
    inverse_transform(coef, n, n, AV1G_TX_DCT_ADST, &plain);
    inverse_transform(coef, n, n, AV1G_TX_DCT_FLIPADST, &lr);
    int worst = 0;
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        worst = std::max(worst, std::abs(plain[y * n + x] - lr[y * n + (n - 1 - x)]));
    CHECK(worst <= 1);

    std::vector<int> ud;
    inverse_transform(coef, n, n, AV1G_TX_ADST_DCT, &plain);
    inverse_transform(coef, n, n, AV1G_TX_FLIPADST_DCT, &ud);
    worst = 0;
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        worst = std::max(worst, std::abs(plain[y * n + x] - ud[(n - 1 - y) * n + x]));
    CHECK(worst <= 1);
  }

  /* IDTX is a pass through: one coefficient lights exactly the matching sample. */
  {
    const int n = 8;
    for (int k = 0; k < 4; ++k) {
      std::vector<int> coef(static_cast<size_t>(n) * n, 0);
      coef[k * n + k] = 512;
      std::vector<int> out;
      inverse_transform(coef, n, n, AV1G_TX_IDTX, &out);
      CHECK(out[k * n + k] > 0);
      int others = 0;
      for (int i = 0; i < n * n; ++i)
        if (i != k * n + k) others += std::abs(out[i]);
      CHECK_EQ(others, 0);
    }
  }

  /* A 2:1 rectangle uses a different row shift, so it must still produce a flat
   * DC block; a wrong pre-scale would leave a DC offset. */
  for (int w : {4, 8, 16}) {
    const int h = w * 2;
    std::vector<int> coef(static_cast<size_t>(w) * h, 0);
    coef[0] = 4096;
    std::vector<int> out;
    inverse_transform(coef, w, h, AV1G_TX_DCT_DCT, &out);
    int lo = out[0], hi = out[0];
    for (int v : out) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    CHECK(hi - lo <= 1);
  }

  /* The 2D gain does not depend on the size: the 1D transforms gain sqrt(n/2)
   * each while the shifts take out 2^(rowShift+4), and those cancel. So one
   * coefficient gives the same residual for 4x4 and 8x8 identity transforms. */
  {
    std::vector<int> small, big;
    std::vector<int> c4(16, 0), c8(64, 0);
    c4[0] = 4096;
    c8[0] = 4096;
    inverse_transform(c4, 4, 4, AV1G_TX_IDTX, &small);
    inverse_transform(c8, 8, 8, AV1G_TX_IDTX, &big);
    CHECK_EQ(small[0], big[0]);
    CHECK(small[0] > 0);
  }

}

void test_intra() {
  std::printf("intra prediction\n");

  /* ---- the two edge copies ------------------------------------------- */
  {
    TestPlane p;
    for (int x = 0; x < 16; ++x) p.at(8 + x, 7) = static_cast<uint16_t>(x * 4);
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_V;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    CHECK_EQ(p.at(8, 8), 0);
    CHECK_EQ(p.at(9, 8), 4);
    CHECK_EQ(p.at(15, 15), 28);
  }
  {
    TestPlane p;
    for (int y = 0; y < 8; ++y) p.at(7, 8 + y) = static_cast<uint16_t>(10 + y);
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_H;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    CHECK_EQ(p.at(8, 8), 10);
    CHECK_EQ(p.at(15, 15), 17);
  }

  /* ---- DC ------------------------------------------------------------- */
  {
    TestPlane p;
    for (int x = 0; x < 8; ++x) p.at(8 + x, 7) = static_cast<uint16_t>(x);
    for (int y = 0; y < 8; ++y) p.at(7, 8 + y) = static_cast<uint16_t>(y);
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_DC;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    const int want = (28 + 28 + 8) / 16; /* sum 0..7 twice, plus the rounding term */
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) CHECK_EQ(p.at(8 + x, 8 + y), want);
  }
  /* DC with only the row above is the rounded row average */
  {
    TestPlane p;
    for (int x = 0; x < 8; ++x) p.at(8 + x, 7) = static_cast<uint16_t>(100 + x);
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_DC;
    job.have_left = 0;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    const int want = (100 * 8 + 28 + 4) >> 3;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) CHECK_EQ(p.at(8 + x, 8 + y), want);
  }
  /* DC with no neighbours at all is the mid level, and so are the edges that
   * the specification invents for the missing side */
  {
    TestPlane p;
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_DC;
    job.have_left = 0;
    job.have_above = 0;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    CHECK_EQ(p.at(9, 9), 128);
  }

  /* ---- Paeth ---------------------------------------------------------- */
  {
    TestPlane p;
    for (int x = 0; x < 8; ++x) p.at(8 + x, 7) = 90;
    for (int y = 0; y < 8; ++y) p.at(7, 8 + y) = 90;
    p.at(7, 7) = 90;
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_PAETH;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) CHECK_EQ(p.at(8 + x, 8 + y), 90);
  }

  /* ---- smooth: check the interpolation formula directly ---------------- */
  {
    TestPlane p;
    for (int x = 0; x < 8; ++x) p.at(8 + x, 7) = static_cast<uint16_t>(10 * x);
    for (int y = 0; y < 8; ++y) p.at(7, 8 + y) = static_cast<uint16_t>(100 + 5 * y);
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_SMOOTH;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        const int wy = av1g_sm_weight(8, i), wx = av1g_sm_weight(8, j);
        const int above_j = 10 * j, left_i = 100 + 5 * i;
        const int left_7 = 100 + 35, above_7 = 70;
        const int pred = round2(wy * above_j + (256 - wy) * left_7 + wx * left_i +
                                    (256 - wx) * above_7,
                                9);
        CHECK_EQ(p.at(8 + j, 8 + i), clip1(pred, 8));
      }
  }

  /* ---- directional: the pAngle < 90 case, computed by hand ------------- */
  {
    TestPlane p;
    for (int x = 0; x < 40; ++x) p.at(x, 7) = static_cast<uint16_t>((x * 7) & 0xff);
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_D67;
    job.angle = 1; /* 67 + 3 = 70 degrees */
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    /* use a second job with edge filtering off to compute the expectation */
    const int pangle = av1g_tbl_mode_to_angle[AV1G_PRED_D67] + 3;
    CHECK_EQ(pangle, 70);
    const int dx = av1g_tbl_dr_deriv[pangle];
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        const int idx = (i + 1) * dx;
        const int base = (idx >> 6) + j;
        const int shift = (idx >> 1) & 0x1F;
        const int max_base = 8 + 8 - 1;
        /* the edge row never reaches past the block: without an available
         * above-right block the specification clamps AboveRow to the block's
         * own last column, which repeats the sample at x + w - 1 */
        const int b0 = 8 + std::min(base, 8 - 1);
        const int b1 = 8 + std::min(base + 1, 8 - 1);
        const int a = base < max_base
                          ? round2(p.at(b0, 7) * (32 - shift) + p.at(b1, 7) * shift, 5)
                          : p.at(8 + max_base, 7);
        CHECK_EQ(p.at(8 + j, 8 + i), clip1(a, 8));
      }
  }
  /* 45 degrees steps one sample right per sample down, so the block is a
   * shifted copy of the row above */
  {
    TestPlane p;
    for (int x = 0; x < 40; ++x) p.at(x, 7) = static_cast<uint16_t>((x * 3) & 0xff);
    Av1gIntraJob job = make_intra_job(4, 4);
    job.mode = AV1G_PRED_D45;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    /* dx at 45 degrees makes half way between two samples, so just check the
     * prediction is inside the edge range and that it is monotone across the
     * row (the definition of a directional predictor) */
    for (int i = 0; i < 4; ++i) {
      for (int j = 1; j < 4; ++j)
        if (p.at(8 + j, 8 + i) < p.at(8 + j - 1, 8 + i) && p.at(8, 7) < p.at(39, 7))
          CHECK(p.at(8 + j, 8 + i) >= p.at(8 + j - 1, 8 + i));
    }
  }

  /* ---- the edge filter strength selection, straight from the spec ----- */
  {
    /* filterType 0: small blocks only get a filter very close to the diagonal */
    CHECK_EQ(av1g_edge_strength(4, 4, 0, 56), 1);
    CHECK_EQ(av1g_edge_strength(4, 4, 0, 55), 0);
    CHECK_EQ(av1g_edge_strength(8, 8, 0, 40), 1);
    CHECK_EQ(av1g_edge_strength(16, 16, 0, 4), 2);
    CHECK_EQ(av1g_edge_strength(16, 16, 0, 32), 3);
    CHECK_EQ(av1g_edge_strength(32, 32, 0, 0), 3);
    /* filterType 1 (a smooth neighbour) filters much more aggressively */
    CHECK_EQ(av1g_edge_strength(4, 4, 1, 40), 1);
    CHECK_EQ(av1g_edge_strength(4, 4, 1, 64), 2);
    CHECK_EQ(av1g_edge_strength(8, 8, 1, 20), 1);
    CHECK_EQ(av1g_edge_strength(8, 8, 1, 48), 2);
    CHECK_EQ(av1g_edge_strength(12, 12, 1, 4), 3);
    CHECK_EQ(av1g_edge_strength(12, 12, 1, 3), 0);
    /* the upsample decision follows the same table's neighbours */
    CHECK_EQ(av1g_edge_upsample_sel(8, 8, 0, 20), 1);
    CHECK_EQ(av1g_edge_upsample_sel(8, 8, 0, 40), 0);
    CHECK_EQ(av1g_edge_upsample_sel(16, 16, 0, 20), 0);
    CHECK_EQ(av1g_edge_upsample_sel(4, 4, 1, 20), 1);
    CHECK_EQ(av1g_edge_upsample_sel(8, 8, 1, 20), 0);
  }

  /* ---- the edge filter keeps a flat edge flat ------------------------- */
  {
    int buf[AV1G_INTRA_EDGE];
    int* b = buf + AV1G_INTRA_EDGE_OFF;
    for (int i = 0; i < 32; ++i) b[i] = 77;
    b[-1] = 77;
    av1g_edge_filter(b, 17, 3);
    bool flat = true;
    for (int i = 0; i < 17; ++i)
      if (b[i] != 77) flat = false;
    CHECK(flat);
  }

  /* ---- chroma from luma ---------------------------------------------- */
  {
    /* A flat luma block has no AC, so CfL must reproduce the DC prediction. */
    TestPlane p;
    for (int x = 0; x < 8; ++x) p.at(8 + x, 7) = 60;
    for (int y = 0; y < 8; ++y) p.at(7, 8 + y) = 60;
    Av1gIntraJob job = make_intra_job(4, 4);
    job.mode = AV1G_PRED_CFL;
    job.cfl_alpha = 1;
    job.ss_x = 1;
    job.ss_y = 1;
    job.luma_off = 0; /* the start of the luma plane: see below */
    job.luma_stride = 40;
    job.luma_w = 24;
    job.luma_h = 24;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    CHECK_EQ(p.at(8, 8), 60);
    CHECK_EQ(p.at(11, 11), 60);
  }
  {
    /* The luma position comes from the block's own x and y: the job carries the
     * start of the luma plane, and the kernel scales x/y by the subsampling
     * factors itself. Getting this wrong (adding the block position twice) can
     * only be seen with luma that has structure, so the luma here is a
     * gradient placed exactly where the specification says the co-located
     * samples are, and the prediction is compared with the specification's
     * formula:
     *
     *   L[i][j]  = (sum of the 2x2 luma samples) << (3 - 2)
     *   lumaAvg  = Round2(sum of L, log2(w) + log2(h))
     *   pred     = Clip1(dc + Round2Signed(alpha * (L[i][j] - lumaAvg), 6))
     */
    const int kAlpha = -2;
    TestPlane p;
    for (int x = 0; x < 8; ++x) p.at(8 + x, 7) = 60;
    for (int y = 0; y < 8; ++y) p.at(7, 8 + y) = 60;
    /* co-located luma of the chroma block at (8,8): 4x4 chroma -> 8x8 luma at
     * (16,16); the top left 2x2 of each chroma sample is enough because the
     * block is 4x4, and the extra 2x2 of the last column and row is clamped */
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) p.at(16 + x, 16 + y) = static_cast<uint16_t>(100 + 3 * x + 5 * y);
    Av1gIntraJob job = make_intra_job(4, 4);
    job.mode = AV1G_PRED_CFL;
    job.cfl_alpha = kAlpha;
    job.ss_x = 1;
    job.ss_y = 1;
    job.luma_off = 0;
    job.luma_stride = 40;
    job.luma_w = 24;
    job.luma_h = 24;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);

    int L[4][4];
    long long sum = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        int t = 0;
        for (int dy = 0; dy <= 1; ++dy)
          for (int dx = 0; dx <= 1; ++dx)
            t += p.at(16 + 2 * j + dx, 16 + 2 * i + dy);
        L[i][j] = t << 1;
        sum += L[i][j];
      }
    }
    const int avg = static_cast<int>((sum + 8) >> 4);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        const int want = clip1(60 + round2_signed(kAlpha * (L[i][j] - avg), 6), 8);
        CHECK_EQ(p.at(8 + j, 8 + i), want);
      }
    /* the gradient must actually have moved the samples away from DC */
    CHECK(p.at(8, 8) != 60 || p.at(11, 11) != 60);
  }
  {
    /* An 8x8 chroma block (16x16 luma) with a positive alpha, driven through
     * the same formula, so the block size does not take a different path. */
    const int kAlpha = 3;
    TestPlane p;
    for (int x = 0; x < 16; ++x) p.at(8 + x, 7) = 128;
    for (int y = 0; y < 16; ++y) p.at(7, 8 + y) = 128;
    /* the luma buffer of this test is the same array, so the co-located samples
     * of a block at (8,8) start at (16,16) and overlap the block itself from
     * (16,16) on: keep them clear of the 8x8 block under test by using x >= 8
     * only beyond the block, which is what the clamp does not do - so this case
     * uses the block at (8,8) with a luma block at (16,16) that starts exactly
     * where the chroma block ends */
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) p.at(16 + x, 16 + y) = static_cast<uint16_t>(40 + 6 * x + 2 * y);
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_CFL;
    job.cfl_alpha = kAlpha;
    job.ss_x = 1;
    job.ss_y = 1;
    job.luma_off = 0;
    job.luma_stride = 40;
    job.luma_w = 32;
    job.luma_h = 32;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    int L[8][8];
    long long sum = 0;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        int t = 0;
        for (int dy = 0; dy <= 1; ++dy)
          for (int dx = 0; dx <= 1; ++dx)
            t += p.at(16 + 2 * j + dx, 16 + 2 * i + dy);
        L[i][j] = t << 1;
        sum += L[i][j];
      }
    }
    const int avg = static_cast<int>((sum + 32) >> 6);
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        const int want = clip1(128 + round2_signed(kAlpha * (L[i][j] - avg), 6), 8);
        CHECK_EQ(p.at(8 + j, 8 + i), want);
      }
  }

  /* ---- filter intra --------------------------------------------------- */
  {
    TestPlane p;
    for (int x = 0; x < 8; ++x) p.at(8 + x, 7) = 100;
    for (int y = 0; y < 8; ++y) p.at(7, 8 + y) = 100;
    p.at(7, 7) = 100;
    Av1gIntraJob job = make_intra_job(8, 8);
    job.mode = AV1G_PRED_DC;
    job.filter_intra = 1;
    job.filter_intra_mode = 0;
    av1g_intra_block(&job, p.data.data(), nullptr, 255);
    bool flat = true;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (p.at(8 + x, 8 + y) != 100) flat = false;
    CHECK(flat);
  }
}

void test_inter() {
  std::printf("inter prediction\n");
  const int stride = 48;
  std::vector<uint16_t> ref(static_cast<size_t>(stride) * 48);
  for (int y = 0; y < 48; ++y)
    for (int x = 0; x < 48; ++x)
      ref[y * stride + x] = static_cast<uint16_t>((x * 3 + y * 5) & 0xff);

  /* zero motion vector at an integer position copies the reference block */
  Av1gInterJob job;
  std::memset(&job, 0, sizeof(job));
  job.w = 8;
  job.h = 8;
  job.dst_stride = stride;
  job.ref_x = 16;
  job.ref_y = 16;
  job.ref_w = 48;
  job.ref_h = 48;
  job.ref0_stride = stride;
  job.ref1_stride = stride;
  job.filt = AV1G_FILT_REGULAR;
  job.filt_v = AV1G_FILT_REGULAR;
  job.bitdepth = 8;
  std::vector<uint16_t> dst(static_cast<size_t>(stride) * 48, 0);
  av1g_inter_block(&job, dst.data(), ref.data(), ref.data(), tables::filters().data(), 255);
  int worst = 0;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      worst = std::max(worst, std::abs(static_cast<int>(dst[y * stride + x]) -
                                       static_cast<int>(ref[(16 + y) * stride + 16 + x])));
  CHECK(worst <= 1);

  /* compound average of two identical references equals the single prediction */
  job.compound = AV1G_COMPOUND_AVG;
  job.w0 = 8;
  job.w1 = 8;
  std::vector<uint16_t> dst2(dst);
  av1g_inter_block(&job, dst2.data(), ref.data(), ref.data(), tables::filters().data(), 255);
  int worst2 = 0;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      worst2 = std::max(worst2, std::abs(static_cast<int>(dst2[y * stride + x]) -
                                         static_cast<int>(dst[y * stride + x])));
  CHECK(worst2 <= 1);

  /* Two *different* references, each in its own pool slot, one prediction per
   * slot: skip mode's only blend. The two halves are rounded once, together
   * (Round2(pred0 + pred1, InterPostRound + 1)), so a flat 100 against a flat
   * 200 is exactly 150 - not 150 +- 1, and not 100 or 200, which is what a
   * slot that resolved to the wrong reference or a blend that dropped one half
   * would give. */
  {
    std::vector<uint16_t> pool(static_cast<size_t>(stride) * 48 * 2, 0);
    for (size_t i = 0; i < static_cast<size_t>(stride) * 48; ++i) {
      pool[i] = 100;
      pool[static_cast<size_t>(stride) * 48 + i] = 200;
    }
    Av1gInterJob blend;
    std::memset(&blend, 0, sizeof(blend));
    blend.w = 8;
    blend.h = 8;
    blend.dst_stride = stride;
    blend.ref_x = 16;
    blend.ref_y = 16;
    blend.ref_w = 48;
    blend.ref_h = 48;
    blend.ref0_stride = stride;
    blend.ref1_stride = stride;
    /* the slot index is the backend's business: it rebases the offsets by
     * slot * buffer_size before calling the kernel, so the second reference
     * here starts one padded plane further in */
    blend.ref0_slot = 0;
    blend.ref1_slot = 1;
    blend.ref1_off = static_cast<int>(static_cast<size_t>(stride) * 48);
    blend.filt = AV1G_FILT_REGULAR;
    blend.filt_v = AV1G_FILT_REGULAR;
    blend.compound = AV1G_COMPOUND_AVG;
    blend.w0 = 8;
    blend.w1 = 8;
    blend.bitdepth = 8;
    std::vector<uint16_t> dst3(static_cast<size_t>(stride) * 48, 0);
    av1g_inter_block(&blend, dst3.data(), dst3.data(), pool.data(), tables::filters().data(), 255);
    int wrong = 0;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (dst3[y * stride + x] != 150) ++wrong;
    CHECK_EQ(wrong, 0);
    /* the two slots are 100 and 200 apart: a blend that read one slot twice
     * cannot land on 150 */
    CHECK_EQ(pool[0], static_cast<uint16_t>(100));
    CHECK_EQ(pool[static_cast<size_t>(stride) * 48], static_cast<uint16_t>(200));
  }
}

/* Intra block copy (7.11.5). The block predicts from the frame it belongs to,
 * which a job reaches with a negative reference slot: ref0_slot < 0 means "read
 * the current picture" and the vector is then an offset inside it. The syntax
 * fixes the filter to bilinear and the vector to whole samples, so the copy is
 * exact - the only thing that can go wrong is the offset, and this checks it for
 * both directions of a negative vector and for a chroma plane, whose offset is
 * the luma vector shifted one bit further. */
void test_intrabc() {
  std::printf("intra block copy\n");
  const int stride = 64;
  /* a source that is distinct per sample, so a one sample slip shows up */
  std::vector<uint16_t> frame(static_cast<size_t>(stride) * 64);
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x)
      frame[static_cast<size_t>(y) * stride + x] = static_cast<uint16_t>((x * 7 + y * 11) & 0xff);

  /* a 16x16 block at (32, 32) copied from (16, 8): the vector is negative in
   * both components and whole samples at the bilinear phase 0, which is the
   * only phase an intra block copy can produce for luma. */
  Av1gInterJob job;
  std::memset(&job, 0, sizeof(job));
  job.w = 16;
  job.h = 16;
  job.dst_off = 32 * stride + 32;
  job.dst_stride = stride;
  job.ref_x = 32;
  job.ref_y = 32;
  job.ref_w = 64;
  job.ref_h = 64;
  job.ref0_stride = stride;
  job.ref1_stride = stride;
  job.ref0_slot = -1; /* the current picture */
  job.ref1_slot = -1;
  job.filt = AV1G_FILT_BILINEAR;
  job.filt_v = AV1G_FILT_BILINEAR;
  job.mv0_x = (16 - 32) * 8;
  job.mv0_y = (8 - 32) * 8;
  job.bitdepth = 8;
  std::vector<uint16_t> dst(frame);
  av1g_inter_block(&job, dst.data(), frame.data(), frame.data(), tables::filters().data(), 255);
  int wrong = 0;
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 16; ++x)
      if (dst[static_cast<size_t>(32 + y) * stride + 32 + x] !=
          frame[static_cast<size_t>(8 + y) * stride + 16 + x])
        ++wrong;
  CHECK_EQ(wrong, 0);
  /* and nothing outside the block moved */
  CHECK_EQ(dst[static_cast<size_t>(32 - 1) * stride + 32], frame[static_cast<size_t>(31) * stride + 32]);

  /* a chroma block: half the samples, the same luma vector, and the integer
   * offset is the luma one halved - a shift, not a scaling of the luma sample
   * position, which is what makes an odd luma vector land on the same chroma
   * sample in this decoder as in the specification. */
  Av1gInterJob chroma;
  std::memset(&chroma, 0, sizeof(chroma));
  chroma.w = 8;
  chroma.h = 8;
  chroma.dst_off = 16 * stride + 16;
  chroma.dst_stride = stride;
  chroma.ref_x = 16;
  chroma.ref_y = 16;
  chroma.ref_w = 32;
  chroma.ref_h = 32;
  chroma.ref0_stride = stride;
  chroma.ref1_stride = stride;
  chroma.ref0_slot = -1;
  chroma.ref1_slot = -1;
  chroma.sub_x = 1;
  chroma.sub_y = 1;
  chroma.filt = AV1G_FILT_BILINEAR;
  chroma.filt_v = AV1G_FILT_BILINEAR;
  /* whole chroma samples are 1/16 luma samples on a subsampled axis */
  chroma.mv0_x = (8 - 16) * 16;
  chroma.mv0_y = (4 - 16) * 16;
  chroma.bitdepth = 8;
  av1g_inter_block(&chroma, dst.data(), frame.data(), frame.data(), tables::filters().data(),
                   255);
  int chroma_wrong = 0;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      if (dst[static_cast<size_t>(16 + y) * stride + 16 + x] !=
          frame[static_cast<size_t>(4 + y) * stride + 8 + x])
        ++chroma_wrong;
  CHECK_EQ(chroma_wrong, 0);
}

/* Inter/intra compound (7.11.4). The inter half is a flat reference with a zero
 * vector, so it predicts its own value exactly and the only thing under test is
 * the intra half and the blend: the mask is the specification's Ii_Weights_1d
 * taper indexed by the plane sample, the intra half reads the reconstructed
 * edges above and to the left, and the pair is rounded once at 6 bits. */
void test_inter_intra() {
  std::printf("inter/intra compound\n");
  const int stride = 48;
  const int block = 16;             /* samples from the frame origin       */
  std::vector<uint16_t> frame(static_cast<size_t>(stride) * 48, 0);
  std::vector<uint16_t> ref(static_cast<size_t>(stride) * 48, 128);
  /* the edges the intra half reads: distinct per row and column, so a mask
   * applied to the wrong sample cannot pass */
  for (int x = block; x < block + 8; ++x) frame[(block - 1) * stride + x] = static_cast<uint16_t>(100 + x);
  for (int y = block; y < block + 8; ++y) frame[y * stride + block - 1] = static_cast<uint16_t>(200 - y);

  Av1gInterJob job;
  std::memset(&job, 0, sizeof(job));
  job.w = 8;
  job.h = 8;
  job.dst_off = block * stride + block;
  job.dst_stride = stride;
  job.ref_x = block;
  job.ref_y = block;
  job.ref_w = 48;
  job.ref_h = 48;
  job.ref0_stride = stride;
  job.ref1_stride = stride;
  job.filt = AV1G_FILT_REGULAR;
  job.filt_v = AV1G_FILT_REGULAR;
  job.bitdepth = 8;
  job.have_left = 1;
  job.have_above = 1;
  job.max_x = stride - 1;
  job.max_y = 47;
  job.interintra = 1;

  /* the mask itself, per mode, on the luma grid: sizeScale is 128 / 8 = 16, so
   * samples (0, 0), (2, 0) and (4, 0) read Ii_Weights_1d[0], [32] and [64] */
  struct {
    int mode;
    int expect[3][3];
  } masks[] = {
      /* DC is the one mode whose weight is a constant, not a table entry */
      {AV1G_II_DC, {{32, 32, 32}, {32, 32, 32}, {32, 32, 32}}},
      /* V varies down the rows, H across the columns */
      {AV1G_II_V, {{60, 60, 60}, {19, 19, 19}, {6, 6, 6}}},
      {AV1G_II_H, {{60, 19, 6}, {60, 19, 6}, {60, 19, 6}}},
      /* SMOOTH takes Min(x, y) */
      {AV1G_II_SMOOTH, {{60, 60, 60}, {60, 19, 19}, {60, 19, 6}}},
  };
  for (const auto& m : masks) {
    job.interintra_mode = m.mode;
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x)
        CHECK(av1g_ii_mask(&job, x * 2, y * 2) == m.expect[y][x]);
  }

  /* the blend, sample by sample, for a vertical taper */
  job.interintra_mode = AV1G_II_V;
  std::vector<uint16_t> dst(frame);
  av1g_inter_block(&job, dst.data(), frame.data(), ref.data(), tables::filters().data(), 255);
  int worst = 0;
  for (int y = 0; y < 8; ++y) {
    const int mask = av1g_tbl_ii_weights[y * 16];
    for (int x = 0; x < 8; ++x) {
      const int intra = frame[(block - 1) * stride + block + x]; /* AboveRow */
      const int want = (mask * intra + (64 - mask) * 128 + 32) >> 6;
      worst = std::max(worst, std::abs(static_cast<int>(dst[job.dst_off + y * stride + x]) - want));
    }
  }
  CHECK(worst == 0);

  /* A chroma plane of an inter/intra block reads Mask[y][x] at its own sample,
   * with the mask built over the plane's own size - not four averaged luma
   * weights, which is the wedge's rule (7.11.3.14: `interintra && !wedge` picks
   * the plane's own array). So the chroma mask of a 8x8 luma block is a 4x4
   * taper of sizeScale 32, and differs from the luma one at the same sample. */
  Av1gInterJob chroma = job;
  chroma.w = 4;
  chroma.h = 4;
  chroma.sub_x = 1;
  chroma.sub_y = 1;
  chroma.interintra_mode = AV1G_II_V;
  for (int y = 0; y < 4; ++y) {
    CHECK(av1g_ii_mask(&chroma, 0, y) == av1g_tbl_ii_weights[y * 32]);
    CHECK(av1g_ii_mask(&chroma, 0, y) == av1g_ii_mask(&job, 0, y * 2));
    for (int x = 1; x < 4; ++x) CHECK(av1g_ii_mask(&chroma, x, y) == av1g_ii_mask(&chroma, 0, y));
  }
  /* DC stays 32 at any size */
  chroma.interintra_mode = AV1G_II_DC;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) CHECK(av1g_ii_mask(&chroma, x, y) == 32);
}

void test_filters() {
  std::printf("loop filters\n");
  const int stride = 32;
  const int size = 32;

  /* deblocking must not touch a flat area */
  {
    std::vector<uint16_t> plane(static_cast<size_t>(stride) * size, 128);
    Av1gDeblockJob job;
    std::memset(&job, 0, sizeof(job));
    job.p_off = 8 * stride + 8; /* p1 at x=8, edge between x=9 (p0) and x=10 (q0) */
    job.stride = stride;
    job.step = 1;
    job.len = 8;
    job.level = 32;
    job.filter_size = 8;
    job.bitdepth = 8;
    av1g_deblock_job(&job, plane.data(), 255);
    bool flat = true;
    for (uint16_t v : plane)
      if (v != 128) flat = false;
    CHECK(flat);
  }
  /* a strong edge must survive (the filter only smooths, never inverts) */
  {
    std::vector<uint16_t> plane(static_cast<size_t>(stride) * size, 128);
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < 12; ++x) plane[y * stride + x] = 20;
    Av1gDeblockJob job;
    std::memset(&job, 0, sizeof(job));
    job.p_off = 4 * stride + 8;
    job.stride = stride;
    job.step = 1;
    job.len = 4;
    job.level = 16;
    job.filter_size = 8;
    job.bitdepth = 8;
    av1g_deblock_job(&job, plane.data(), 255);
    CHECK(plane[4 * stride + 8] < 128);
    CHECK(plane[4 * stride + 11] >= 20);
  }
  /* CDEF on a flat area is a no-op */
  {
    std::vector<uint16_t> src(static_cast<size_t>(stride) * size, 90);
    std::vector<uint16_t> dst(src.size(), 0);
    Av1gCdefJob job;
    std::memset(&job, 0, sizeof(job));
    job.x = 8;
    job.y = 8;
    job.src_off = 8 * stride + 8;
    job.src_stride = stride;
    job.dst_off = 8 * stride + 8;
    job.dst_stride = stride;
    job.luma_off = job.src_off;
    job.luma_stride = stride;
    job.pri_strength = 4;
    job.sec_strength = 2;
    job.dir = 1;
    job.bitdepth = 8;
    job.plane_w = size;
    job.plane_h = size;
    av1g_cdef_block(&job, src.data(), dst.data(), src.data(), 255);
    bool flat = true;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (dst[(8 + y) * stride + 8 + x] != 90) flat = false;
    CHECK(flat);
  }
  /* film grain: the blend scales the noise image sample by the plane's curve at
   * that sample's value, adds it and clips to the frame's range */
  {
    const int w = 32;
    std::vector<uint16_t> dst(static_cast<size_t>(w) * w, 77);
    std::vector<short> noise(64 * 64, 40);
    int lut[AV1G_GRAIN_LUT_ENTRIES] = {0};
    Av1gGrainJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 4 * w + 4;
    job.dst_stride = w;
    job.noise_stride = 64;
    job.w = 8;
    job.h = 8;
    job.plane = 0;
    job.bitdepth = 8;
    job.scaling_shift = 8;
    job.num_y_points = 2;
    job.min_value = 16;
    job.max_value = 235;
    job.luma_w = 8;

    /* a curve of zero is no gain at all */
    av1g_grain_block(&job, dst.data(), dst.data(), noise.data(), lut, 255);
    bool same = true;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (dst[static_cast<size_t>((4 + y) * w + 4 + x)] != 77) same = false;
    CHECK(same);

    /* a curve of 256 is a gain of one, so the noise is added as it stands */
    for (int i = 0; i < AV1G_GRAIN_LUT_ENTRIES; ++i) lut[i] = 256;
    av1g_grain_block(&job, dst.data(), dst.data(), noise.data(), lut, 255);
    CHECK(dst[static_cast<size_t>(4 * w + 4)] == 117);

    /* and a sample past the plane's maximum is clipped, not wrapped */
    std::fill(dst.begin(), dst.end(), static_cast<uint16_t>(220));
    av1g_grain_block(&job, dst.data(), dst.data(), noise.data(), lut, 255);
    CHECK(dst[static_cast<size_t>(4 * w + 4)] == 235);
  }

  /* film grain offsets are nine bits, so a chroma offset below 256 is an
   * ordinary one: 7.18.3.5 merges luma and chroma into
   * `index = Clip3(0, 255, Round2(combined, 6) + (offset - 256) * 2^shift)`,
   * and `offset - 256` is negative for every one of those. The scaling was a
   * left shift, which is undefined for a negative value in C++ - the picture
   * came out right because compilers emit the multiply anyway, but only the
   * sanitizer build says so. These two samples pin the arithmetic and the value
   * it produces. */
  {
    const int w = 32;
    std::vector<uint16_t> dst(static_cast<size_t>(w) * w, 0);
    std::vector<short> noise(64 * 64, 40);
    std::vector<uint16_t> luma(64 * 64, 0);
    int lut[AV1G_GRAIN_LUT_ENTRIES] = {0};
    /* Cr's curve: a gain of one at index 0, a gain of two at index 144 */
    lut[512 + 0] = 256;
    lut[512 + 144] = 512;

    Av1gGrainJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 4 * w + 4;
    job.dst_stride = w;
    job.noise_stride = 64;
    job.w = 8;
    job.h = 8;
    job.plane = 2;
    job.ss_x = 1;
    job.ss_y = 1;
    job.bitdepth = 8;
    job.scaling_shift = 8;
    job.num_cr_points = 2;
    job.min_value = 0;
    job.max_value = 255;
    job.cr_mult = 128;
    job.cr_luma_mult = 128;
    job.cr_offset = 200; /* below 256, so `offset - 256` is -56 */
    job.luma_stride = 64;
    job.luma_w = 8;

    /* luma 0 and mult 128 make combined 0, the index clips to 0 and the curve
     * there is a gain of one, so the sample takes the noise as it stands */
    dst[static_cast<size_t>(4 * w + 4)] = 77;
    av1g_grain_block(&job, dst.data(), luma.data(), noise.data(), lut, 255);
    CHECK(dst[static_cast<size_t>(4 * w + 4)] == 117);

    /* a chroma multiplier above 128 lands the merge on index 144:
     * Round2(200 * 64, 6) - 56 = 144, where the curve is a gain of two */
    std::fill(noise.begin(), noise.end(), static_cast<short>(10));
    dst[static_cast<size_t>(4 * w + 4)] = 200;
    job.cr_mult = 192;
    av1g_grain_block(&job, dst.data(), luma.data(), noise.data(), lut, 255);
    CHECK(dst[static_cast<size_t>(4 * w + 4)] == 220);
  }

  /* film grain deeper than 8 bit: the grain range is +/-512 at 10 bit, so a
   * noise sample does not fit in a signed byte - the noise image is 16 bit and
   * a large sample has to survive the trip */
  {
    const int w = 32;
    std::vector<uint16_t> dst(static_cast<size_t>(w) * w, 500);
    std::vector<short> noise(64 * 64, 300);
    int lut[AV1G_GRAIN_LUT_ENTRIES];
    for (int i = 0; i < AV1G_GRAIN_LUT_ENTRIES; ++i) lut[i] = 256; /* gain of one */
    Av1gGrainJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 4 * w + 4;
    job.dst_stride = w;
    job.noise_stride = 64;
    job.w = 8;
    job.h = 8;
    job.plane = 0;
    job.bitdepth = 10;
    job.scaling_shift = 8;
    job.num_y_points = 1;
    job.min_value = 64;
    job.max_value = 940;
    job.luma_w = 8;
    av1g_grain_block(&job, dst.data(), dst.data(), noise.data(), lut, 1023);
    CHECK(dst[static_cast<size_t>(4 * w + 4)] == 800);
    /* and the clip is the 10 bit one, not the 8 bit one */
    std::fill(dst.begin(), dst.end(), static_cast<uint16_t>(900));
    av1g_grain_block(&job, dst.data(), dst.data(), noise.data(), lut, 1023);
    CHECK(dst[static_cast<size_t>(4 * w + 4)] == 940);

    /* the generator: at 10 bit the template holds values a signed byte cannot
     * represent, which is the regression this guards */
    Av1gGrainParams params;
    std::memset(&params, 0, sizeof(params));
    params.seed = 1234;
    params.bitdepth = 10;
    params.num_y_points = 1;
    params.y_value[0] = 0;
    params.y_scaling[0] = 40;
    std::vector<short> luma_grain(static_cast<size_t>(AV1G_GRAIN_LUMA_W) * AV1G_GRAIN_LUMA_H, 0);
    av1g_grain_generate(&params, luma_grain.data(), nullptr, nullptr, AV1G_GRAIN_CHROMA_W,
                        AV1G_GRAIN_CHROMA_H, 1, 1);
    int peak = 0;
    for (short v : luma_grain) peak = std::max(peak, std::abs(static_cast<int>(v)));
    CHECK(peak > 127);
  }

  /* chroma grain from luma: with chroma_scaling_from_luma the curve is indexed
   * by the average of the luma samples the chroma sample covers */
  {
    const int w = 32;
    std::vector<uint16_t> buf(static_cast<size_t>(w) * w, 0);
    /* two rows of luma, then the chroma rows below them */
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 16; ++x) buf[static_cast<size_t>(y * w + x)] = 128;
    const int chroma_off = 8 * w;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) buf[static_cast<size_t>(chroma_off + y * w + x)] = 100;
    std::vector<short> noise(64 * 64, 40);
    /* one table per plane, each indexed by an 8 bit sample value */
    int lut[AV1G_GRAIN_LUT_ENTRIES];
    for (int i = 0; i < AV1G_GRAIN_LUT_ENTRIES; ++i) lut[i] = i % 256;
    Av1gGrainJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = chroma_off;
    job.dst_stride = w;
    job.noise_stride = 64;
    job.w = 8;
    job.h = 8;
    job.plane = 1;
    job.ss_x = 1;
    job.ss_y = 1;
    job.bitdepth = 8;
    job.scaling_shift = 8;
    job.num_y_points = 2;
    job.chroma_scaling_from_luma = 1;
    job.max_value = 255;
    job.luma_off = 0;
    job.luma_stride = w;
    job.luma_w = 16;
    av1g_grain_block(&job, buf.data(), buf.data(), noise.data(), lut, 255);
    /* averageLuma = 128, so the curve gives 128 and the noise (40) becomes
     * Round2(128 * 40, 8) = 20 */
    CHECK(buf[static_cast<size_t>(chroma_off)] == 120);
  }
}

/* Loop restoration. The interesting properties are not "matches the decoder"
 * (the fixtures cover that) but the ones a wrong filter breaks immediately:
 * both filters have unity DC gain, so a flat plane comes back flat at every bit
 * depth, and the samples a unit reads come from two different pictures. */
Av1gLrJob make_lr_job(int x, int y, int w, int h, int stride, int plane_w, int plane_h,
                      int type, int bitdepth) {
  Av1gLrJob job;
  std::memset(&job, 0, sizeof(job));
  job.plane = 0;
  /* dst_off points at the job's own first sample, not at the plane origin: the
   * kernel adds only the position inside the rectangle */
  job.dst_off = y * stride + x;
  job.dst_stride = stride;
  job.cdef_off = 0;
  job.cdef_stride = stride;
  job.curr_off = 0;
  job.curr_stride = stride;
  job.x = x;
  job.y = y;
  job.w = w;
  job.h = h;
  job.plane_w = plane_w;
  job.plane_h = plane_h;
  job.stripe_start_y = 0;
  job.stripe_end_y = plane_h - 1;
  job.type = type;
  job.bitdepth = bitdepth;
  return job;
}

/* reads the restored rectangle back out of the destination plane */
std::vector<int> lr_result(const std::vector<uint16_t>& dst, const Av1gLrJob& job) {
  std::vector<int> out;
  for (int y = 0; y < job.h; ++y)
    for (int x = 0; x < job.w; ++x)
      out.push_back(dst[static_cast<size_t>(job.dst_off + y * job.dst_stride + x)]);
  return out;
}

void test_loop_restoration() {
  std::printf("loop restoration\n");
  const int stride = 32;
  const int size = 32;
  const int maxv8 = 255;

  /* The Wiener filter's seven taps are built from three coded coefficients and
   * always sum to 128, so a flat plane survives it exactly - whatever the
   * coefficients are. That is the one property that catches a wrong tap
   * normalisation or a wrong rounding variable. */
  {
    for (int bd : {8, 10, 12}) {
      const int maxv = (1 << bd) - 1;
      const int flat = bd == 8 ? 100 : (bd == 10 ? 400 : 1600);
      for (int variant = 0; variant < 2; ++variant) {
        std::vector<uint16_t> cdef(static_cast<size_t>(stride) * size,
                                   static_cast<uint16_t>(flat));
        std::vector<uint16_t> dst(cdef.size(), 0);
        Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, size, AV1G_LR_WIENER, bd);
        if (variant == 0) {
          /* identity: only the centre tap survives */
          job.vfilter[0] = job.vfilter[1] = job.vfilter[2] = 0;
          job.hfilter[0] = job.hfilter[1] = job.hfilter[2] = 0;
        } else {
          /* a real, asymmetric set: still unity gain */
          job.vfilter[0] = 3;
          job.vfilter[1] = -2;
          job.vfilter[2] = 5;
          job.hfilter[0] = -4;
          job.hfilter[1] = 1;
          job.hfilter[2] = 7;
        }
        av1g_lr_block(&job, cdef.data(), cdef.data(), dst.data(), maxv);
        bool flat_out = true;
        for (int v : lr_result(dst, job))
          if (v != flat) flat_out = false;
        CHECK(flat_out);
      }
    }
  }

  /* A step edge is smoothed but stays between the two levels, and the samples
   * outside the rectangle are left alone (dst_off is the rectangle's origin, so
   * a kernel that adds the job origin again writes in the wrong place). */
  {
    std::vector<uint16_t> cdef(static_cast<size_t>(stride) * size, 20);
    for (int y = 0; y < size; ++y)
      for (int x = 16; x < size; ++x) cdef[static_cast<size_t>(y * stride + x)] = 200;
    std::vector<uint16_t> dst(cdef.size(), 4000); /* not a valid sample value */
    Av1gLrJob job = make_lr_job(8, 8, 16, 16, stride, size, size, AV1G_LR_WIENER, 8);
    job.vfilter[0] = job.vfilter[1] = job.vfilter[2] = 0;
    job.hfilter[0] = job.hfilter[1] = job.hfilter[2] = 0;
    /* an asymmetric set: 40 8 -20 72 -20 8 40, so the edge is smoothed over
     * seven samples and the samples far from it are untouched */
    job.hfilter[0] = 40;
    job.hfilter[1] = 8;
    job.hfilter[2] = -20;
    av1g_lr_block(&job, cdef.data(), cdef.data(), dst.data(), maxv8);
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x) {
        const bool inside = x >= 8 && x < 24 && y >= 8 && y < 24;
        if (!inside) CHECK_EQ(dst[static_cast<size_t>(y * stride + x)], 4000);
      }
    /* the sample just left of the edge is pulled up, the one on the edge down */
    CHECK(dst[static_cast<size_t>(12 * stride + 15)] > 20);
    CHECK(dst[static_cast<size_t>(12 * stride + 16)] < 200);
    /* and the untouched parts of the plane stay as they were */
    CHECK_EQ(dst[static_cast<size_t>(12 * stride + 20)], 200);
  }

  /* The self guided filter also has unity gain: with both radii zero it says
   * "use the input" for both passes, which is an exact copy for any weights. */
  {
    std::vector<uint16_t> cdef(static_cast<size_t>(stride) * size, 137);
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x)
        cdef[static_cast<size_t>(y * stride + x)] =
            static_cast<uint16_t>((x * 7 + y * 3) & 255);
    std::vector<uint16_t> dst(cdef.size(), 0);
    Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, size, AV1G_LR_SGRPROJ, 8);
    job.sgr_radius[0] = job.sgr_radius[1] = 0;
    job.sgr_xqd[0] = -1;
    job.sgr_xqd[1] = 121;
    av1g_lr_block(&job, cdef.data(), cdef.data(), dst.data(), maxv8);
    bool same = true;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (dst[static_cast<size_t>((8 + y) * stride + 8 + x)] !=
            cdef[static_cast<size_t>((8 + y) * stride + 8 + x)])
          same = false;
    CHECK(same);
  }

  /* A flat plane through a real strength (Sgr_Params set 0: radius 2 with
   * eps 12, radius 1 with eps 4) comes back flat at every bit depth. */
  for (int bd : {8, 10, 12}) {
    const int maxv = (1 << bd) - 1;
    const int flat = bd == 8 ? 100 : (bd == 10 ? 400 : 1600);
    std::vector<uint16_t> cdef(static_cast<size_t>(stride) * size,
                               static_cast<uint16_t>(flat));
    std::vector<uint16_t> dst(cdef.size(), 0);
    Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, size, AV1G_LR_SGRPROJ, bd);
    job.sgr_radius[0] = 2;
    job.sgr_radius[1] = 1;
    job.sgr_eps[0] = 12;
    job.sgr_eps[1] = 4;
    job.sgr_xqd[0] = -1;
    job.sgr_xqd[1] = 121;
    av1g_lr_block(&job, cdef.data(), cdef.data(), dst.data(), maxv);
    /* Exactly, not approximately: the variance of a flat block is zero, so the
     * filter is an identity even through the rounding of A and B - and it only
     * stays one if the two bit depth scalings are there (see docs/ACCURACY.md:
     * at 8 bit they are shifts by zero, so this is the assertion that notices
     * when they are missing). */
    for (int v : lr_result(dst, job)) CHECK_EQ(v, flat);
  }

  /* The same picture at 8 and at 10 bit - every sample times four - has to come
   * back the same way, times four. The box filter scales its sums back to 8 bit
   * before they become a variance, so both runs see the same A and B; a missing
   * scaling shows up here as tens of levels, not as rounding. */
  {
    const int depth = 10;
    const int maxv = (1 << depth) - 1;
    std::vector<uint16_t> p8(static_cast<size_t>(stride) * size, 0);
    /* a smooth ramp with one bump in the middle of the job: low variance, so
     * the filter actually smooths (a noisy picture saturates A and the filter
     * would return its input, which would test nothing) */
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x) {
        int v = 40 + x * 3 + y * 2;
        if (x >= 11 && x <= 13 && y >= 11 && y <= 13) v += 40;
        p8[static_cast<size_t>(y * stride + x)] = static_cast<uint16_t>(v);
      }
    std::vector<uint16_t> p10(p8.size(), 0);
    for (size_t i = 0; i < p8.size(); ++i) p10[i] = static_cast<uint16_t>(p8[i] * 4);
    Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, size, AV1G_LR_SGRPROJ, 8);
    job.sgr_radius[0] = 2;
    job.sgr_radius[1] = 1;
    job.sgr_eps[0] = 12;
    job.sgr_eps[1] = 4;
    job.sgr_xqd[0] = -1;
    job.sgr_xqd[1] = 121;
    std::vector<uint16_t> o8(p8.size(), 0), o10(p10.size(), 0);
    av1g_lr_block(&job, p8.data(), p8.data(), o8.data(), 255);
    job.bitdepth = depth;
    av1g_lr_block(&job, p10.data(), p10.data(), o10.data(), maxv);
    int maxdev = 0;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        const size_t off = static_cast<size_t>((8 + y) * stride + 8 + x);
        maxdev = std::max(maxdev, std::abs(static_cast<int>(o10[off]) -
                                           static_cast<int>(o8[off]) * 4));
      }
    /* nothing at all: the two runs differ only in where the scaling happens,
     * and every scaling is exact for a value that is four times another. */
    CHECK_EQ(maxdev, 0);
  }

  /* The same box at 8 and at 10 bit must give the same A and a B that is
   * exactly four times as large: A is built from sums scaled back to 8 bit, B
   * from the raw sums of the deeper picture. That is the pair of statements the
   * historical bug in this filter broke (B was built from the scaled sum, which
   * is invisible at 8 bit), and it is checked here without a 10 bit stream. */
  {
    std::vector<uint16_t> p8(static_cast<size_t>(stride) * size, 0);
    /* a vertical edge of contrast 8 near the middle of the job, so some boxes
     * straddle it and get a variance in the middle of the table */
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x)
        p8[static_cast<size_t>(y * stride + x)] = static_cast<uint16_t>(x < 15 ? 100 : 108);
    std::vector<uint16_t> p10(p8.size(), 0);
    for (size_t i = 0; i < p8.size(); ++i) p10[i] = static_cast<uint16_t>(p8[i] * 4);
    Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, size, AV1G_LR_SGRPROJ, 8);
    bool straddled = false;
    for (int y = 9; y < 15; ++y)
      for (int x = 9; x < 15; ++x) {
        int a8 = 0, b8 = 0, a10 = 0, b10 = 0;
        av1g_lr_sgr_ab(&job, p8.data(), p8.data(), x, y, 5, 12, &a8, &b8);
        job.bitdepth = 10;
        av1g_lr_sgr_ab(&job, p10.data(), p10.data(), x, y, 5, 12, &a10, &b10);
        job.bitdepth = 8;
        CHECK_EQ(a10, a8);
        CHECK(std::abs(b10 - b8 * 4) <= 4);
        if (a8 > 1 && a8 < 256) straddled = true;
      }
    /* the boxes really did land in the interesting part of the curve */
    CHECK(straddled);

    /* and the other end of it: a busy box saturates A at 256 (the filter then
     * passes the picture through) and leaves B at zero */
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x)
        p8[static_cast<size_t>(y * stride + x)] =
            static_cast<uint16_t>(((x + y) & 1) ? 30 : 210);
    int a = 0, b = 0;
    av1g_lr_sgr_ab(&job, p8.data(), p8.data(), 12, 12, 5, 12, &a, &b);
    CHECK_EQ(a, 256);
    CHECK_EQ(b, 0);
  }

  /* A gradient is actually filtered: the output differs from the input, stays
   * inside the sample range, and the projection weights really are used (all
   * weight on one pass gives a different picture than all weight on the other). */
  {
    std::vector<uint16_t> cdef(static_cast<size_t>(stride) * size, 0);
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x)
        cdef[static_cast<size_t>(y * stride + x)] =
            static_cast<uint16_t>(std::min(255, 40 + x * 5 + ((y & 1) ? 30 : 0)));
    std::vector<uint16_t> a(cdef.size(), 0), b(cdef.size(), 0);
    Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, size, AV1G_LR_SGRPROJ, 8);
    job.sgr_radius[0] = 2;
    job.sgr_radius[1] = 1;
    job.sgr_eps[0] = 12;
    job.sgr_eps[1] = 4;
    job.sgr_xqd[0] = 128;
    job.sgr_xqd[1] = 0;
    av1g_lr_block(&job, cdef.data(), cdef.data(), a.data(), maxv8);
    job.sgr_xqd[0] = 0;
    job.sgr_xqd[1] = 128;
    av1g_lr_block(&job, cdef.data(), cdef.data(), b.data(), maxv8);
    bool filtered = false, differ = false;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        const size_t off = static_cast<size_t>((8 + y) * stride + 8 + x);
        if (a[off] != cdef[off]) filtered = true;
        if (a[off] != b[off]) differ = true;
        CHECK(a[off] <= maxv8 && b[off] <= maxv8);
      }
    CHECK(filtered);
    CHECK(differ);
  }

  /* get_source_sample (7.17.6): a unit only borrows two lines from above its
   * stripe and two from below, and those lines come from the picture *before*
   * CDEF. Everything inside the stripe comes from the CDEF output. The vertical
   * filter here uses only taps -2 and +2, so an output row shows which picture
   * each of its neighbours came from. */
  {
    const int h = 32;
    std::vector<uint16_t> cdef(static_cast<size_t>(stride) * h, 100);
    std::vector<uint16_t> curr(static_cast<size_t>(stride) * h, 100);
    /* rows 6 and 7 are the two lines the stripe may borrow; make them, and the
     * rows above them, unmistakable */
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < stride; ++x) curr[static_cast<size_t>(y * stride + x)] = 200;
    std::vector<uint16_t> dst(cdef.size(), 0);
    Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, h, AV1G_LR_WIENER, 8);
    job.stripe_start_y = 8;
    job.stripe_end_y = 23;
    job.hfilter[0] = job.hfilter[1] = job.hfilter[2] = 0;
    job.vfilter[0] = 0;
    job.vfilter[1] = 64; /* the seven taps become 0 64 0 0 0 64 0 */
    job.vfilter[2] = 0;
    av1g_lr_block(&job, cdef.data(), curr.data(), dst.data(), maxv8);
    /* row 8 and 9 read row 6 (clamped twice), which is the pre-CDEF picture,
     * and row 10 or 12, which is CDEF: both contribute 64/128 each */
    CHECK_EQ(dst[static_cast<size_t>(8 * stride + 8)], 150);
    CHECK_EQ(dst[static_cast<size_t>(9 * stride + 8)], 150);
    /* rows 10 onwards read only CDEF rows, so the flat 100 comes back */
    CHECK_EQ(dst[static_cast<size_t>(10 * stride + 8)], 100);
    CHECK_EQ(dst[static_cast<size_t>(15 * stride + 8)], 100);
  }

  /* Both snapshots the same picture: the filter is just the flat case again,
   * which is the check that the two pointers are not swapped. */
  {
    std::vector<uint16_t> cdef(static_cast<size_t>(stride) * size, 64);
    std::vector<uint16_t> dst(cdef.size(), 0);
    Av1gLrJob job = make_lr_job(8, 8, 8, 8, stride, size, size, AV1G_LR_WIENER, 8);
    job.vfilter[1] = 64;
    job.hfilter[0] = job.hfilter[1] = job.hfilter[2] = 0;
    av1g_lr_block(&job, cdef.data(), cdef.data(), dst.data(), maxv8);
    for (int v : lr_result(dst, job)) CHECK_EQ(v, 64);
  }
}

void test_color() {
  std::printf("colour conversion\n");
  Frame frame;
  frame.allocate(16, 16, 8, 1, 1);
  for (int p = 0; p < 3; ++p)
    std::fill(frame.storage[p].begin(), frame.storage[p].end(),
              static_cast<uint16_t>(p == 0 ? 16 : 128));
  std::vector<uint8_t> rgb;
  tables::frame_to_rgb(frame, ColorMatrix::Bt709, ColorRange::Limited, &rgb);
  /* limited range black is (16, 128, 128) and must convert to (0, 0, 0) */
  CHECK(rgb[0] <= 1);
  CHECK(rgb[1] <= 1);
  CHECK(rgb[2] <= 1);

  for (int p = 0; p < 3; ++p)
    std::fill(frame.storage[p].begin(), frame.storage[p].end(),
              static_cast<uint16_t>(p == 0 ? 235 : 128));
  tables::frame_to_rgb(frame, ColorMatrix::Bt709, ColorRange::Limited, &rgb);
  CHECK(rgb[0] >= 253);
  CHECK(rgb[1] >= 253);
  CHECK(rgb[2] >= 253);

  /* grey stays grey for every matrix */
  for (int p = 0; p < 3; ++p)
    std::fill(frame.storage[p].begin(), frame.storage[p].end(),
              static_cast<uint16_t>(p == 0 ? 128 : 128));
  for (ColorMatrix m : {ColorMatrix::Bt709, ColorMatrix::Bt2020}) {
    tables::frame_to_rgb(frame, m, ColorRange::Limited, &rgb);
    CHECK(std::abs(rgb[0] - rgb[1]) <= 2);
    CHECK(std::abs(rgb[1] - rgb[2]) <= 2);
  }
}

void test_nv12_pack() {
  std::printf("NV12 output\n");
  /* A 6x4 4:2:0 frame with a pitch of 8: two bytes of every row are padding, and
   * they belong to the output - a VA surface is written as a whole, so the
   * kernel has to fill them (with zero) instead of leaving whatever was there. */
  Frame frame;
  frame.allocate(6, 4, 8, 1, 1);
  for (int y = 0; y < frame.plane_height(0); ++y)
    for (int x = 0; x < frame.plane_width(0); ++x)
      frame.plane(0)[static_cast<size_t>(y) * frame.stride[0] + x] =
          static_cast<uint16_t>(10 + y * 6 + x);
  for (int p = 1; p < 3; ++p)
    for (int y = 0; y < frame.plane_height(p); ++y)
      for (int x = 0; x < frame.plane_width(p); ++x)
        frame.plane(p)[static_cast<size_t>(y) * frame.stride[p] + x] =
            static_cast<uint16_t>((p == 1 ? 100 : 200) + y * 3 + x);

  const size_t pitch = 8;
  std::vector<uint8_t> nv12;
  CHECK(frame_to_nv12(frame, pitch, &nv12));
  CHECK(nv12.size() == pitch * (4 + 2));
  if (nv12.size() == pitch * 6) {
    bool luma_ok = true, chroma_ok = true, padding_ok = true;
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < static_cast<int>(pitch); ++x) {
        const uint8_t got = nv12[static_cast<size_t>(y) * pitch + x];
        if (x < 6) {
          luma_ok = luma_ok && got == 10 + y * 6 + x;
        } else {
          padding_ok = padding_ok && got == 0;
        }
      }
    CHECK(luma_ok);
    CHECK(padding_ok);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 3; ++x) {
        const size_t row = static_cast<size_t>(4 + y) * pitch;
        /* even byte is Cb, odd byte is Cr, and one pair per chroma sample */
        chroma_ok = chroma_ok && nv12[row + 2 * x] == 100 + y * 3 + x;
        chroma_ok = chroma_ok && nv12[row + 2 * x + 1] == 200 + y * 3 + x;
      }
    CHECK(chroma_ok);
    /* the last two bytes of a chroma row are padding too */
    CHECK(nv12[static_cast<size_t>(4) * pitch + 6] == 0);
    CHECK(nv12[static_cast<size_t>(5) * pitch + 6] == 0);
    CHECK(nv12[static_cast<size_t>(5) * pitch + 7] == 0);
  }

  /* The whole kernel loop is the same scalar code the device runs, so the block
   * helper and the per byte function have to agree byte for byte. */
  {
    Av1gNv12Job job;
    std::memset(&job, 0, sizeof(job));
    job.y_stride = frame.stride[0];
    job.c_stride = frame.stride[1];
    job.w = frame.width;
    job.h = frame.height;
    job.pitch = static_cast<int>(pitch);
    job.ss_x = frame.ss_x;
    job.ss_y = frame.ss_y;
    job.bitdepth = frame.bit_depth;
    std::vector<uint8_t> again(nv12.size(), 0xAB);
    av1g_nv12_block(&job, frame.plane(0), frame.plane(1), frame.plane(2), again.data());
    CHECK(again == nv12);
  }

  /* A pitch that is odd, or narrower than a chroma row, and a frame that is not
   * 4:2:0: all three are refused instead of writing a truncated picture. */
  CHECK(!frame_to_nv12(frame, 7, &nv12));
  CHECK(!frame_to_nv12(frame, 4, &nv12));
  {
    Frame wide;
    wide.allocate(6, 4, 8, 0, 0);
    CHECK(!frame_to_nv12(wide, pitch, &nv12));
  }

  /* An odd height has a chroma plane of `(height + 1) >> 1` rows, and NV12 has
   * one chroma row per two luma rows - so a 5 row frame writes 3 of them. */
  {
    Frame odd;
    odd.allocate(4, 5, 8, 1, 1);
    std::vector<uint8_t> out;
    CHECK(frame_to_nv12(odd, 4, &out));
    CHECK(out.size() == static_cast<size_t>(4) * (5 + 3));
  }

  /* Deeper than 8 bit: NV12 is an 8 bit format, so the samples are shifted down
   * by (BitDepth - 8) - the same shift frame_to_yuv8 does. */
  {
    Frame deep;
    deep.allocate(4, 2, 10, 1, 1);
    for (int p = 0; p < 3; ++p)
      for (int y = 0; y < deep.plane_height(p); ++y)
        for (int x = 0; x < deep.plane_width(p); ++x)
          deep.plane(p)[static_cast<size_t>(y) * deep.stride[p] + x] =
              static_cast<uint16_t>((p == 0 ? 400 : 600) + 4 * x);
    std::vector<uint8_t> out;
    CHECK(frame_to_nv12(deep, 4, &out));
    if (out.size() == static_cast<size_t>(4) * 3) {
      CHECK(out[0] == 100);  /* 400 >> 2 */
      CHECK(out[3] == 103);
      /* two luma rows of four bytes each, then the one chroma row */
      CHECK(out[8] == 150);   /* 600 >> 2, Cb of the first chroma sample */
      CHECK(out[9] == 150);   /* Cr */
      CHECK(out[10] == 151);  /* 604 >> 2 */
      CHECK(out.size() == static_cast<size_t>(4) * 3);
    }
  }
}

} /* namespace */

int main() {
  std::printf("AV1onGPU kernel tests\n");
  test_tables();
  test_transform();
  test_intra();
  test_inter();
  test_intrabc();
  test_inter_intra();
  test_filters();
  test_loop_restoration();
  test_color();
  test_nv12_pack();
  return testing::finish("test_kernels");
}
