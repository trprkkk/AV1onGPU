/*
 * AV1onGPU - synthetic content generator and closed loop encoder
 *
 * See include/av1gpu/synthetic.hpp for what this is and what it is not.
 *
 * The important property is that the *reconstruction* path is the same code the
 * decoder runs (kernels/av1gpu_kernels.h), so the encoder's reference frame and
 * the decoder's reference frame stay identical and PSNR measurements are
 * meaningful even after many frames.
 */
#include "av1gpu/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "core/backend.hpp"

namespace av1gpu {
namespace {

constexpr int kMaxValue(int bd) { return (1 << bd) - 1; }

inline int clamp_px(int v, int bd) {
  const int maxv = (1 << bd) - 1;
  return v < 0 ? 0 : (v > maxv ? maxv : v);
}

/* ---- source picture ----------------------------------------------------- */

void render_plane(uint16_t* data, int stride, int w, int h, int bd, int frame_index,
                  int plane) {
  const int maxv = (1 << bd) - 1;
  const double t = frame_index * 0.35;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      /* background: two gradients plus a slow rotation, enough texture to make
       * the deblocking and CDEF stages do visible work */
      const double fx = static_cast<double>(x) / w;
      const double fy = static_cast<double>(y) / h;
      double v = 0.45 + 0.22 * std::sin(6.0 * fx + t) + 0.18 * std::cos(5.0 * fy - t * 0.7);
      v += 0.10 * std::sin(11.0 * (fx + fy) + 2.0 * t);

      /* three moving squares, one per plane so chroma has real edges too */
      for (int s = 0; s < 3; ++s) {
        const double cx = 0.5 + 0.32 * std::cos(t * (0.6 + 0.2 * s) + s * 2.1) +
                          (plane == 0 ? 0.0 : 0.04 * s);
        const double cy = 0.5 + 0.30 * std::sin(t * (0.5 + 0.15 * s) + s * 1.3);
        const double half = 0.10 + 0.02 * s;
        if (std::fabs(fx - cx) < half && std::fabs(fy - cy) < half) {
          const double tint = (s == plane) ? 0.30 : -0.12;
          v += tint;
        }
      }
      const int value = static_cast<int>(v * maxv + 0.5);
      data[static_cast<size_t>(y) * stride + x] = static_cast<uint16_t>(clamp_px(value, bd));
    }
  }
}

/* ---- forward transform -------------------------------------------------- */
/* The synthetic encoder needs a forward transform to make coefficients, but the
 * decoder's inverse transform is the specification's butterfly network, not a
 * matrix product. Rather than reusing a generic orthonormal basis (which would
 * make the encoder and the decoder disagree by a scale factor), the basis here
 * is the same orthonormal DCT-II *scaled the way AV1 scales it*: the 1D inverse
 * transform is s(n) * B with s(n) = sqrt(n/2), so the forward pass multiplies by
 * 1/s(n) = sqrt(2/n), which is exactly 4096/2896 for n = 4 and so on. The row
 * and column shifts of the inverse are undone by scaling the residual up by
 * 2^(rowShift + colShift) before the analysis.
 *
 * Everything below is derived from that one identity:
 *
 *     residual = ( A_w (x) A_h ) / 2^(rowShift + colShift)  applied to coef
 *     coef     = ( A_w (x) A_h )^-1 * 2^(rowShift + colShift) applied to residual
 */

/* round-half-away-from-zero division by 2^n, matching Round2 */
long long round_shift(long long v, int n) {
  if (n <= 0) return v;
  const long long half = 1LL << (n - 1);
  return (v + half) >> n;
}

const std::vector<int>& dct_basis(int n) {
  static std::map<int, std::vector<int>> cache;
  auto it = cache.find(n);
  if (it != cache.end()) return it->second;
  std::vector<int> b(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      const double scale = (k == 0) ? std::sqrt(1.0 / n) : std::sqrt(2.0 / n);
      const double v = scale * std::cos(M_PI * (2.0 * i + 1.0) * k / (2.0 * n));
      b[static_cast<size_t>(i) * n + k] = static_cast<int>(std::floor(v * 4096.0 + 0.5));
    }
  }
  return cache.emplace(n, std::move(b)).first->second;
}

/* 4096 / sqrt(n / 2), the reciprocal of the factor AV1's 1D transform carries */
int forward_scale(int n) {
  switch (n) {
    case 4: return 2896;
    case 8: return 2048;
    case 16: return 1448;
    case 32: return 1024;
    default: return 724; /* 64 */
  }
}

/* Forward transform of a residual block, producing coefficients scaled the way
 * the inverse transform expects them. */
void forward_transform(const int* src, int stride, int w, int h, int* out) {
  const std::vector<int>& bw = dct_basis(w);
  const std::vector<int>& bh = dct_basis(h);
  const int sb = forward_scale(w), sa = forward_scale(h);
  const int up = 4 + internal::tables::tx_row_shift(w, h);
  std::vector<long long> tmp(static_cast<size_t>(w) * h, 0);
  /* vertical analysis */
  for (int x = 0; x < w; ++x)
    for (int k = 0; k < h; ++k) {
      long long acc = 0;
      for (int i = 0; i < h; ++i) {
        const long long s = static_cast<long long>(src[static_cast<size_t>(i) * stride + x]) << up;
        acc += s * bh[static_cast<size_t>(i) * h + k];
      }
      tmp[static_cast<size_t>(k) * w + x] = round_shift(acc * sa, 24);
    }
  /* horizontal analysis */
  for (int k = 0; k < h; ++k)
    for (int j = 0; j < w; ++j) {
      long long acc = 0;
      for (int i = 0; i < w; ++i)
        acc += tmp[static_cast<size_t>(k) * w + i] * bw[static_cast<size_t>(i) * w + j];
      out[k * w + j] = static_cast<int>(round_shift(acc * sb, 24));
    }
}

int sad_region(const uint16_t* a, int a_stride, const uint16_t* b, int b_stride, int w,
               int h) {
  int sum = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      sum += std::abs(static_cast<int>(a[static_cast<size_t>(y) * a_stride + x]) -
                      static_cast<int>(b[static_cast<size_t>(y) * b_stride + x]));
  return sum;
}

} /* namespace */

void synthetic_source(const SyntheticConfig& cfg, int frame_index, Frame* out) {
  if (!out->matches(cfg.width, cfg.height, cfg.bit_depth, cfg.chroma_420 ? 1 : 0,
                    cfg.chroma_420 ? 1 : 0))
    out->allocate(cfg.width, cfg.height, cfg.bit_depth, cfg.chroma_420 ? 1 : 0,
                  cfg.chroma_420 ? 1 : 0);
  for (int p = 0; p < 3; ++p)
    render_plane(out->plane(p), out->stride[p], out->plane_width(p), out->plane_height(p),
                 cfg.bit_depth, frame_index, p);
}

FrameSyntax synthetic_encode(const SyntheticConfig& cfg, int frame_index,
                             const Frame* reference, Frame* source, Frame* recon_out) {
  Frame src;
  synthetic_source(cfg, frame_index, &src);
  if (source) *source = src;

  const int ss = cfg.chroma_420 ? 1 : 0;
  Frame recon;
  recon.allocate(cfg.width, cfg.height, cfg.bit_depth, ss, ss);
  /* the key frame starts from mid grey so the DC predictor has sane neighbours */
  for (int p = 0; p < 3; ++p)
    std::fill(recon.storage[p].begin(), recon.storage[p].end(),
              static_cast<uint16_t>(1 << (cfg.bit_depth - 1)));

  FrameSyntax syntax;
  syntax.width = cfg.width;
  syntax.height = cfg.height;
  syntax.bit_depth = cfg.bit_depth;
  syntax.ss_x = ss;
  syntax.ss_y = ss;
  syntax.sharpness = 0;
  syntax.base_q_idx = cfg.qstep * 4;

  const int bs = std::max(4, cfg.block_size);
  const int ts = std::max(4, cfg.transform_size);
  const int maxv = kMaxValue(cfg.bit_depth);
  const int* filters = internal::tables::filters().data();

  const int blocks_x = (cfg.width + bs - 1) / bs;
  const int blocks_y = (cfg.height + bs - 1) / bs;
  const bool inter = reference != nullptr;
  if (inter) syntax.reference_slots = {0};

  for (int by = 0; by < blocks_y; ++by) {
    for (int bx = 0; bx < blocks_x; ++bx) {
      for (int plane = 0; plane < (cfg.chroma_420 ? 3 : 1); ++plane) {
        const int shift = plane ? ss : 0;
        const int x = bx * (bs >> shift);
        const int y = by * (bs >> shift);
        const int w = std::min(bs >> shift, src.plane_width(plane) - x);
        const int h = std::min(bs >> shift, src.plane_height(plane) - y);
        if (w <= 0 || h <= 0) continue;

        BlockSyntax block;
        block.x = x;
        block.y = y;
        block.w = w;
        block.h = h;
        block.plane = plane;
        block.bit_depth = cfg.bit_depth;
        block.have_left = x > 0;
        block.have_above = y > 0;

        /* ---- prediction ------------------------------------------------- */
        std::vector<uint16_t> pred(static_cast<size_t>(w) * h, 0);
        const int stride = recon.stride[plane];
        const int off = static_cast<int>(y) * stride + x;

        if (!inter) {
          block.is_intra = true;
          block.mode = PredictionMode::Dc;
          /* run the real predictor on the reconstruction buffer, then copy the
           * block out so the residual is measured against it */
          Av1gIntraJob job;
          std::memset(&job, 0, sizeof(job));
          job.dst_off = off;
          job.dst_stride = stride;
          job.w = w;
          job.h = h;
          job.x = x;
          job.y = y;
          job.mode = AV1G_PRED_DC;
          job.bitdepth = cfg.bit_depth;
          job.have_left = block.have_left;
          job.have_above = block.have_above;
          job.have_above_right = block.have_left && block.have_above;
          job.filter_type = -1; /* the encoder writes the edge as it is */
          job.max_x = recon.plane_width(plane) - 1;
          job.max_y = recon.plane_height(plane) - 1;
          av1g_intra_block(&job, recon.plane(plane), maxv);
          for (int yy = 0; yy < h; ++yy)
            std::memcpy(pred.data() + static_cast<size_t>(yy) * w,
                        recon.plane(plane) + off + static_cast<size_t>(yy) * stride,
                        static_cast<size_t>(w) * sizeof(uint16_t));
        } else {
          /* motion search against the *reconstructed* previous frame */
          int best_mv_x = 0, best_mv_y = 0;
          int best_sad = 1 << 30;
          const int radius = cfg.search_radius;
          const int step = radius >= 8 ? 2 : 1;
          for (int dy = -radius; dy <= radius; dy += step) {
            for (int dx = -radius; dx <= radius; dx += step) {
              const int rx = x + dx;
              const int ry = y + dy;
              if (rx < 0 || ry < 0 || rx + w > reference->plane_width(plane) ||
                  ry + h > reference->plane_height(plane))
                continue;
              const int sad = sad_region(src.plane(plane) + off, src.stride[plane],
                                         reference->plane(plane) +
                                             static_cast<size_t>(ry) *
                                                 reference->stride[plane] +
                                             rx,
                                         reference->stride[plane], w, h);
              if (sad < best_sad) {
                best_sad = sad;
                best_mv_x = dx;
                best_mv_y = dy;
              }
            }
          }
          block.is_intra = false;
          block.ref[0] = 0;
          block.mv[0][0] = best_mv_x << 3; /* 1/8 pel, integer MV  */
          block.mv[0][1] = best_mv_y << 3;
          block.filter = FilterType::Regular;
          block.compound = CompoundType::None;

          Av1gInterJob job;
          std::memset(&job, 0, sizeof(job));
          job.dst_off = 0;
          job.dst_stride = w;
          job.w = w;
          job.h = h;
          job.ref_x = x;
          job.ref_y = y;
          job.ref_w = reference->plane_width(plane);
          job.ref_h = reference->plane_height(plane);
          job.mv0_x = block.mv[0][0];
          job.mv0_y = block.mv[0][1];
          job.ref0_stride = reference->stride[plane];
          job.ref1_stride = reference->stride[plane];
          job.filt = AV1G_FILT_REGULAR;
          job.compound = AV1G_COMPOUND_NONE;
          job.bitdepth = cfg.bit_depth;
          std::vector<uint16_t> tmp(static_cast<size_t>(w) * h, 0);
          const uint16_t* refp = reference->plane(plane);
          av1g_inter_block(&job, tmp.data(), refp, refp, filters, maxv);
          std::memcpy(pred.data(), tmp.data(), tmp.size() * sizeof(uint16_t));
        }

        /* the decoder writes the prediction into the frame before adding the
         * residual; mirror that exactly */
        for (int yy = 0; yy < h; ++yy)
          std::memcpy(recon.plane(plane) + off + static_cast<size_t>(yy) * stride,
                      pred.data() + static_cast<size_t>(yy) * w,
                      static_cast<size_t>(w) * sizeof(uint16_t));

        /* ---- residual, transform, quantisation -------------------------- */
        std::vector<int> residual(static_cast<size_t>(w) * h, 0);
        for (int yy = 0; yy < h; ++yy) {
          for (int xx = 0; xx < w; ++xx) {
            const int s = src.plane(plane)[off + static_cast<size_t>(yy) * src.stride[plane] + xx];
            residual[static_cast<size_t>(yy) * w + xx] =
                s - static_cast<int>(pred[static_cast<size_t>(yy) * w + xx]);
          }
        }

        const int tw = std::min(ts, w);
        const int th = std::min(ts, h);
        const int tiles_x = w / tw;
        const int tiles_y = h / th;
        for (int ty = 0; ty < tiles_y; ++ty) {
          for (int tx = 0; tx < tiles_x; ++tx) {
            std::vector<int> block_res(static_cast<size_t>(tw) * th, 0);
            for (int yy = 0; yy < th; ++yy)
              for (int xx = 0; xx < tw; ++xx)
                block_res[static_cast<size_t>(yy) * tw + xx] =
                    residual[static_cast<size_t>(ty * th + yy) * w + tx * tw + xx];
            std::vector<int> coef(static_cast<size_t>(tw) * th, 0);
            forward_transform(block_res.data(), tw, tw, th, coef.data());

            TransformBlock tblock;
            tblock.x = tx * tw;
            tblock.y = ty * th;
            tblock.w = tw;
            tblock.h = th;
            tblock.coef_offset = static_cast<int>(syntax.coefficients.size());
            int last = -1;
            const int q = std::max(1, cfg.qstep);
            for (size_t i = 0; i < coef.size(); ++i) {
              const int qc = coef[i] >= 0 ? (coef[i] + q / 2) / q : -((-coef[i] + q / 2) / q);
              const int deq = qc * q;
              syntax.coefficients.push_back(deq);
              if (deq != 0) last = static_cast<int>(i);
            }
            tblock.eob = last + 1;
            block.transforms.push_back(tblock);
          }
        }
        syntax.blocks.push_back(block);

        /* ---- mirror the decoder reconstruction --------------------------- */
        {
          const int plane_off = off;
          for (const TransformBlock& tblock : block.transforms) {
            Av1gItxJob job;
            std::memset(&job, 0, sizeof(job));
            job.coef_off = tblock.coef_offset;
            job.tmp_off = 0;
            job.dst_off = plane_off + tblock.y * stride + tblock.x;
            job.dst_stride = stride;
            job.w = tblock.w;
            job.h = tblock.h;
            job.tx_type = static_cast<int>(tblock.tx_type);
            job.eob = tblock.eob;
            job.row_shift = internal::tables::tx_row_shift(tblock.w, tblock.h);
            job.bitdepth = cfg.bit_depth;
            std::vector<int> tmp(static_cast<size_t>(tblock.w) * tblock.h, 0);
            for (int yy = 0; yy < tblock.h; ++yy)
              av1g_itx_row(&job, syntax.coefficients.data(), tmp.data(), yy);
            for (int xx = 0; xx < tblock.w; ++xx)
              av1g_itx_col_add(&job, tmp.data(), recon.plane(plane), xx, maxv);
          }
        }
      }
    }
  }

  /* ---- deblocking edges on every block boundary ------------------------- */
  if (cfg.deblock_level > 0) {
    for (int p = 0; p < (cfg.chroma_420 ? 3 : 1); ++p) {
      const int shift = p ? ss : 0;
      const int pw = recon.plane_width(p);
      const int ph = recon.plane_height(p);
      const int step = bs >> shift;
      for (int y = 0; y < ph; y += step) {
        for (int x = step; x < pw; x += step) {
          DeblockEdge edge;
          edge.plane = p;
          edge.x = x;
          edge.y = y;
          edge.vertical = true;
          edge.length = std::min(step, ph - y);
          if (edge.length < 4) continue;
          edge.level = cfg.deblock_level;
          edge.filter_len = (p == 0) ? 8 : 6;
          syntax.deblock.push_back(edge);
        }
      }
      for (int y = step; y < ph; y += step) {
        for (int x = 0; x < pw; x += step) {
          DeblockEdge edge;
          edge.plane = p;
          edge.x = x;
          edge.y = y;
          edge.vertical = false;
          edge.length = std::min(step, pw - x);
          if (edge.length < 4) continue;
          edge.level = cfg.deblock_level;
          edge.filter_len = (p == 0) ? 8 : 6;
          syntax.deblock.push_back(edge);
        }
      }
    }
  }

  /* ---- CDEF ------------------------------------------------------------- */
  if (cfg.cdef) {
    const int strength = std::max(1, cfg.qstep / 2);
    for (int p = 0; p < (cfg.chroma_420 ? 3 : 1); ++p) {
      const int shift = p ? ss : 0;
      const int pw = recon.plane_width(p);
      const int ph = recon.plane_height(p);
      for (int y = 0; y + 8 <= ph; y += 8) {
        for (int x = 0; x + 8 <= pw; x += 8) {
          CdefBlock c;
          c.plane = p;
          c.x = x;
          c.y = y;
          c.pri_strength = std::max(0, strength - shift * 2);
          c.sec_strength = std::max(0, strength / 2 - shift);
          c.damping = 3 + shift;
          c.dir = ((x * 3 + y * 5) >> 6) & 7;
          syntax.cdef.push_back(c);
        }
      }
    }
  }

  /* ---- film grain -------------------------------------------------------- */
  if (cfg.grain) {
    syntax.apply_grain = true;
    syntax.grain.seed = cfg.seed + frame_index;
    syntax.grain.overlap = 0;
    syntax.grain.template_size = 128;
    for (int i = 0; i < 32; ++i) {
      const double v = 0.12 * (1.0 - i / 31.0); /* less grain in the highlights */
      syntax.grain.lut[i] = static_cast<int>(std::floor(v * 64.0 + 0.5));
    }
  }

  /* the reference the *decoder* will use has the loop filters applied, so the
   * encoder must apply them too or the next inter frame drifts away */
  internal::apply_post_filters_host(&recon, syntax);

  if (recon_out) *recon_out = recon;
  return syntax;
}

} /* namespace av1gpu */
