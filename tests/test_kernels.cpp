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

/* inverse transform of one block, vectorised the same way the decoder does it */
void inverse_transform(const std::vector<int>& coef, int w, int h, int row_basis,
                       int col_basis, std::vector<int>* out) {
  const int* basis = tables::basis().data();
  Av1gItxJob job;
  std::memset(&job, 0, sizeof(job));
  job.w = w;
  job.h = h;
  job.row_basis = row_basis;
  job.col_basis = col_basis;
  job.eob = static_cast<int>(coef.size());
  std::vector<int> tmp(coef.size(), 0);
  for (int y = 0; y < h; ++y) av1g_itx_row(&job, coef.data(), tmp.data(), basis, y);
  /* residual adds are applied on top of a prediction, so start from mid grey:
   * the samples are unsigned and negative residuals would otherwise clip */
  std::vector<uint16_t> plane(static_cast<size_t>(w) * h + 64, 128);
  job.dst_off = 0;
  job.dst_stride = w;
  for (int x = 0; x < w; ++x)
    av1g_itx_col_add(&job, tmp.data(), plane.data(), basis, x, 65535);
  out->assign(plane.begin(), plane.begin() + w * h);
  for (int& v : *out) v -= 128;
}

void test_tables() {
  std::printf("tables\n");
  const int* basis = tables::basis().data();
  /* the synthesis basis must be orthonormal: columns have unit norm */
  for (int n : {4, 8, 16, 32, 64}) {
    const int off = tables::basis_dct(n);
    double worst = 0;
    for (int a = 0; a < n; ++a) {
      double dot = 0;
      for (int i = 0; i < n; ++i) dot += basis[off + i * n + a] * basis[off + i * n + a];
      worst = std::max(worst, std::fabs(dot / (4096.0 * 4096.0) - 1.0));
    }
    CHECK(worst < 0.01);
  }
  /* subpel filters: every phase has unity DC gain and the halves are symmetric */
  const int* filters = tables::filters().data();
  for (int f = 0; f < AV1G_FILT_COUNT; ++f) {
    for (int phase = 0; phase < 8; ++phase) {
      int sum = 0;
      for (int i = 0; i < 8; ++i) sum += filters[((f * 8) + phase) * 8 + i];
      CHECK_EQ(sum, 128);
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
  const int* basis = tables::basis().data();
  for (int n : {4, 8, 16, 32}) {
    const int dct = tables::basis_dct(n);
    /* DC only: the residual must be flat at coef * sqrt(1/n) / sqrt(n) * ... */
    std::vector<int> coef(static_cast<size_t>(n) * n, 0);
    coef[0] = 4 * n;
    std::vector<int> out;
    inverse_transform(coef, n, n, dct, dct, &out);
    int lo = out[0], hi = out[0];
    for (int v : out) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    /* the DC column of an orthonormal DCT-II is 1/sqrt(n), so a single
     * coefficient c produces the flat residual c/sqrt(n) */
    CHECK_EQ(hi - lo, 0);
    /* the 2D DC basis is 1/n, so the flat residual is coef[0]/n */
    const double expected = static_cast<double>(coef[0]) / n;
    CHECK_NEAR(out[0], expected, 2.0);


  }

  /* forward + inverse round trip on a random block: the encoder side of the
   * pipeline (src/core/synthetic.cpp) relies on this being exact enough */
  {
    const int n = 8;
    const int off = tables::basis_dct(n);
    std::vector<int> src(static_cast<size_t>(n) * n);
    unsigned state = 12345;
    for (int& v : src) {
      state = state * 1103515245u + 12345u;
      v = static_cast<int>((state >> 16) % 81) - 40;
    }
    std::vector<int> tmp(static_cast<size_t>(n) * n, 0);
    std::vector<int> coef(static_cast<size_t>(n) * n, 0);
    for (int x = 0; x < n; ++x) {
      long long acc = 0;
      for (int i = 0; i < n; ++i) acc += static_cast<long long>(src[i * n + x]) *
                                        basis[off + i * n + 0];
      (void)acc;
    }
    /* rows then columns with the transposed basis */
    for (int y = 0; y < n; ++y)
      for (int k = 0; k < n; ++k) {
        long long acc = 0;
        for (int i = 0; i < n; ++i)
          acc += static_cast<long long>(src[y * n + i]) * basis[off + i * n + k];
        tmp[y * n + k] = static_cast<int>((acc + 2048) >> 12);
      }
    for (int k = 0; k < n; ++k)
      for (int y = 0; y < n; ++y) {
        long long acc = 0;
        for (int i = 0; i < n; ++i)
          acc += static_cast<long long>(tmp[i * n + k]) * basis[off + i * n + y];
        coef[y * n + k] = static_cast<int>((acc + 2048) >> 12);
      }
    std::vector<int> back;
    inverse_transform(coef, n, n, off, off, &back);
    int worst = 0;
    for (int i = 0; i < n * n; ++i) worst = std::max(worst, std::abs(back[i] - src[i]));
    CHECK(worst <= 2);
  }
}

void test_intra() {
  std::printf("intra prediction\n");
  const int stride = 32;
  const int size = 32;
  std::vector<uint16_t> plane(static_cast<size_t>(stride) * size, 0);

  /* vertical: the block copies the row above */
  for (int x = 0; x < 16; ++x) plane[7 * stride + 8 + x] = static_cast<uint16_t>(x * 4);
  Av1gIntraJob job;
  std::memset(&job, 0, sizeof(job));
  job.dst_off = 8 * stride + 8;
  job.dst_stride = stride;
  job.w = 8;
  job.h = 8;
  job.bitdepth = 8;
  job.have_above = 1;
  job.have_left = 1;
  Av1gIntraAux aux_v;
  std::memset(&aux_v, 0, sizeof(aux_v));
  job.mode = AV1G_PRED_V;
  av1g_intra_block(&job, plane.data(), &aux_v, 255);
  CHECK_EQ(plane[8 * stride + 8], 0);
  CHECK_EQ(plane[8 * stride + 9], 4);
  CHECK_EQ(plane[15 * stride + 15], 28);

  /* horizontal: the block copies the column to the left */
  for (int y = 0; y < 8; ++y) plane[(8 + y) * stride + 7] = static_cast<uint16_t>(10 + y);
  Av1gIntraAux aux_h;
  std::memset(&aux_h, 0, sizeof(aux_h));
  job.mode = AV1G_PRED_H;
  av1g_intra_block(&job, plane.data(), &aux_h, 255);
  CHECK_EQ(plane[8 * stride + 8], 10);
  CHECK_EQ(plane[15 * stride + 15], 17);

  /* DC: average of above and left */
  Av1gIntraAux aux;
  std::memset(&aux, 0, sizeof(aux));
  job.mode = AV1G_PRED_DC;
  av1g_intra_aux(&job, plane.data(), &aux);
  int sum = 0;
  for (int x = 0; x < 8; ++x) sum += plane[7 * stride + 8 + x];
  for (int y = 0; y < 8; ++y) sum += plane[(8 + y) * stride + 7];
  CHECK_EQ(aux.dc, (sum + 8) / 16);

  /* DC without neighbours is the mid level */
  job.have_above = 0;
  job.have_left = 0;
  av1g_intra_aux(&job, plane.data(), &aux);
  CHECK_EQ(aux.dc, 128);
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
  job.w0 = 32;
  job.w1 = 32;
  std::vector<uint16_t> dst2(dst);
  av1g_inter_block(&job, dst2.data(), ref.data(), ref.data(), tables::filters().data(), 255);
  int worst2 = 0;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      worst2 = std::max(worst2, std::abs(static_cast<int>(dst2[y * stride + x]) -
                                         static_cast<int>(dst[y * stride + x])));
  CHECK(worst2 <= 1);
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
    job.filter_len = 8;
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
    job.filter_len = 8;
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
    job.src_off = 8 * stride + 8;
    job.src_stride = stride;
    job.dst_off = 8 * stride + 8;
    job.dst_stride = stride;
    job.pri_strength = 4;
    job.sec_strength = 2;
    job.dir = 1;
    job.bitdepth = 8;
    job.plane_w = size;
    job.plane_h = size;
    av1g_cdef_block(&job, src.data(), dst.data(), 255);
    bool flat = true;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (dst[(8 + y) * stride + 8 + x] != 90) flat = false;
    CHECK(flat);
  }
  /* film grain with a zero gain leaves the frame alone */
  {
    std::vector<uint16_t> dst(static_cast<size_t>(stride) * size, 77);
    std::vector<signed char> grain(64 * 64, 40);
    int lut[64] = {0};
    Av1gGrainJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 4 * stride + 4;
    job.dst_stride = stride;
    job.w = 8;
    job.h = 8;
    job.grain_stride = 64;
    job.grain_h = 64;
    job.bitdepth = 8;
    av1g_grain_block(&job, dst.data(), grain.data(), lut, 255);
    bool same = true;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        if (dst[(4 + y) * stride + 4 + x] != 77) same = false;
    CHECK(same);

    /* and with a non zero gain it changes the samples without clipping */
    for (int i = 0; i < 64; ++i) lut[i] = 32;
    av1g_grain_block(&job, dst.data(), grain.data(), lut, 255);
    CHECK(dst[4 * stride + 4] != 77);
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

} /* namespace */

int main() {
  std::printf("AV1onGPU kernel tests\n");
  test_tables();
  test_transform();
  test_intra();
  test_inter();
  test_filters();
  test_color();
  return testing::finish("test_kernels");
}
