/*
 * AV1onGPU - frame planner and public Decoder
 *
 * Turning a FrameSyntax into GPU work happens here. The interesting part is the
 * scheduler: intra prediction reads reconstructed neighbours, so the blocks of a
 * frame cannot all run at the same time. The engine records exactly which block
 * owns every sample of every plane, works out precisely which samples each
 * block's prediction reads, and gives every block the earliest "wave" in which
 * all of them are already reconstructed:
 *
 *     wave(block) = 1 + max(wave(owner(sample)) for every sample it reads)
 *
 * where the samples are the row above (extended by another block width when
 * above-right is available), the column to the left (extended by another block
 * height for below-left) and the corner, plus the co-located luma samples for
 * chroma-from-luma. Inter blocks read reference frames only, so they always
 * land in wave 0. Waves are executed in order and inside a wave the jobs are
 * completely independent, which is exactly what a GPU wants; the number of
 * waves is the length of the longest such chain, not a tunable.
 *
 * Frames live in one flat buffer (three planes back to back) so every job can
 * carry a plain sample offset and the backends need a single allocation.
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/backend.hpp"

namespace av1gpu {
namespace internal {
namespace {

} /* namespace */

class EngineDecoder final : public av1gpu::Decoder {
 public:
  EngineDecoder(std::unique_ptr<Backend> backend, const BackendOptions& options)
      : backend_(std::move(backend)), options_(options) {}

  ~EngineDecoder() override {
    for (int slot : held_slots_) backend_->release_reference(slot);
  }

  const DeviceInfo& device() const override { return backend_->info(); }
  const BackendOptions& options() const override { return options_; }
  const DecodeStats& stats() const override { return stats_; }

  void reset() override {
    for (int slot : held_slots_) backend_->release_reference(slot);
    held_slots_.clear();
    backend_->reset();
    stats_ = DecodeStats();
  }

  void release_reference(int slot) override {
    auto it = std::find(held_slots_.begin(), held_slots_.end(), slot);
    if (it != held_slots_.end()) held_slots_.erase(it);
    backend_->release_reference(slot);
  }

  bool decode(const FrameSyntax& syntax, Frame* out, std::string* error) override {
    /* A frame nobody keeps needs no pool slot at all: asking for a temporary
     * one would starve a pipelined caller whose queued frames still hold the
     * slots the pool just dropped. */
    FramePlan plan;
    if (!plan_for(syntax, &plan, error)) return false;
    int slot = -1;
    return backend_->decode(plan, false, out, &slot, &stats_, error);
  }

  int reserve_reference() override { return backend_->reserve_reference(); }

  int decode_reference(const FrameSyntax& syntax, Frame* out, std::string* error) override {
    FramePlan plan;
    if (!plan_for(syntax, &plan, error)) return -1;

    int slot = -1;
    if (!backend_->decode(plan, true, out, &slot, &stats_, error)) return -1;
    if (slot >= 0) held_slots_.push_back(slot);
    return slot;
  }

  int decode_reference_reserved(const FrameSyntax& syntax, int slot, Frame* out,
                                std::string* error) override {
    if (slot < 0) {
      /* no reserved slot: decode and drop the device copy */
      decode(syntax, out, error);
      return -1;
    }
    FramePlan plan;
    if (!plan_for(syntax, &plan, error)) return -1;

    int got = -1;
    if (!backend_->decode_reserved(plan, slot, out, &got, &stats_, error)) return -1;
    if (got >= 0) held_slots_.push_back(got);
    return got;
  }

 private:
  bool plan_for(const FrameSyntax& syntax, FramePlan* plan, std::string* error) {
    std::string local;
    if (!error) error = &local;
    error->clear();

    FrameGeometry geom;
    geom.width = syntax.width;
    geom.height = syntax.height;
    geom.bit_depth = syntax.bit_depth;
    geom.ss_x = syntax.ss_x;
    geom.ss_y = syntax.ss_y;
    if (geom.width <= 0 || geom.height <= 0) {
      *error = "frame has no size";
      return false;
    }

    const int coded_width = syntax.coded_width > 0 ? syntax.coded_width : syntax.width;
    return build_frame_plan(syntax, backend_->prepare(geom, coded_width), options_,
                            backend_->reference_capacity(), plan, error);
  }

 public:

  bool to_rgb(int slot, ColorMatrix matrix, ColorRange range, std::vector<uint8_t>* rgb,
              std::string* error) override {
    return backend_->to_rgb(slot, matrix, range, rgb, error);
  }

  bool to_nv12(int slot, size_t pitch, std::vector<uint8_t>* nv12,
               std::string* error) override {
    return backend_->to_nv12(slot, pitch, nv12, error);
  }

 private:
  std::unique_ptr<Backend> backend_;
  BackendOptions options_;
  DecodeStats stats_;
  std::vector<int> held_slots_;
};

/* ------------------------------------------------------------------------- */
/* plan building                                                              */
/*                                                                             */
/* This is a free function (not a member) so that tools and tests can build a  */
/* plan for a backend they drive themselves - tests/opencl_pipeline.py uses it */
/* to replay exactly the jobs the real backends submit.                        */
/* ------------------------------------------------------------------------- */

/* Mode_To_Angle (the table the kernels use): the prediction angle of each intra
 * mode in degrees, so the scheduler can tell which edge a block reads. */
static int intra_mode_angle(PredictionMode mode) {
  static const int angles[] = {0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0};
  const int m = static_cast<int>(mode);
  if (m < 0 || m >= static_cast<int>(sizeof(angles) / sizeof(angles[0]))) return 0;
  return angles[m];
}

static int resolve_slot(const FrameSyntax& syntax, int ref_id) {
  if (ref_id < 0) return -1;
  if (ref_id >= static_cast<int>(syntax.reference_slots.size())) return -1;
  return syntax.reference_slots[ref_id];
}

bool build_frame_plan(const FrameSyntax& syntax, const PlanLayout& layout,
                      const BackendOptions& options, int reference_capacity,
                      FramePlan* plan, std::string* error) {
    FrameGeometry geom;
    geom.width = syntax.width;
    geom.height = syntax.height;
    geom.bit_depth = syntax.bit_depth;
    geom.ss_x = syntax.ss_x;
    geom.ss_y = syntax.ss_y;
    if (geom.width <= 0 || geom.height <= 0) {
      *error = "frame has no size";
      return false;
    }
    plan->layout = layout;
    const PlanLayout& g = plan->layout;
    plan->maxv = (1 << geom.bit_depth) - 1;
    /* The tile data codes the *coded* size; the display size the layout was
     * built from is what the upscaling process produces. Everything below that
     * clamps to the frame edge clamps to the coded size, because that is the
     * picture a block is predicted in. */
    const int frame_w = g.plane_w[0];
    const int frame_h = g.plane_h[0];

    auto at = [&](int plane, int x, int y) {
      return g.plane_off[plane] + y * g.stride[plane] + x;
    };

    /* ---- coefficient and residual pools ---------------------------------- */
    size_t coef_size = 0;
    size_t residual_size = 0;
    for (const BlockSyntax& b : syntax.blocks) {
      for (const TransformBlock& tx : b.transforms) {
        const size_t end = tx.coef_offset +
                           static_cast<size_t>(tx.w) * tx.h;
        if (b.residual_provided)
          residual_size = std::max(residual_size, end);
        else
          coef_size = std::max(coef_size, end);
      }
    }
    plan->coefficients.assign(coef_size + 64, 0);
    if (!syntax.coefficients.empty())
      std::memcpy(plan->coefficients.data(), syntax.coefficients.data(),
                  std::min(plan->coefficients.size(), syntax.coefficients.size()) *
                      sizeof(int32_t));
    plan->residual.assign(residual_size + 64, 0);
    if (!syntax.residual.empty())
      std::memcpy(plan->residual.data(), syntax.residual.data(),
                  std::min(plan->residual.size(), syntax.residual.size()) *
                      sizeof(int32_t));
    /* The palette maps are one byte per sample and indexed by the block that
     * codes them, so the pool is copied as it is. The trailing slack keeps a
     * backend's buffer allocation non-zero for a frame without a palette. */
    plan->palette_map.assign(syntax.palette_map.begin(), syntax.palette_map.end());
    plan->palette_map.resize(plan->palette_map.size() + 8, 0);

    /* ---- reference slots -------------------------------------------------- */
    for (int i = 0; i < 2; ++i) {
      plan->ref_slot[i] = -1;
      if (i < static_cast<int>(syntax.reference_slots.size())) {
        const int s = syntax.reference_slots[i];
        if (s >= 0 && s < reference_capacity) plan->ref_slot[i] = s;
      }
    }

    /* ---- wavefront schedule ---------------------------------------------- */
    /* Every block has to run after the blocks that own the samples it reads.
     * That is a property of the *reconstruction*, not of the syntax: a block
     * may read samples of a block that comes later in coding order (the
     * above-right and below-left neighbours of intra prediction), and it must
     * read them once their residual has been added, not just their
     * prediction. So instead of a coarse "who owns this area" grid, work out
     * exactly which block owns every sample and let each block wait for the
     * owners of the samples it touches.
     *
     * Within a wave the backends run every prediction first and every residual
     * add afterwards, so a same-wave neighbour would be read before its
     * residual exists. The schedule below therefore requires a *strictly*
     * earlier wave for every sample a block reads. */
    /* The grid this covers is the *coded* one, MiCols * MI_SIZE - the frame
     * size rounded up to a multiple of eight - and not the visible size. A
     * frame 99 samples tall has blocks down to row 103, and the last visible
     * row band is predicted from the left column below it: those samples are
     * outside the picture but inside the grid, so a dependency tracker that
     * stopped at the visible height would let a block read a neighbour that has
     * not been reconstructed yet. Every multiple of eight size makes the two
     * grids the same, which is why this only shows up as a handful of samples
     * on the last rows of an odd sized frame. */
    std::vector<int> grid_w(3), grid_h(3);
    for (int p = 0; p < 3; ++p) {
      const int sub_x = p > 0 && geom.ss_x ? 1 : 0;
      const int sub_y = p > 0 && geom.ss_y ? 1 : 0;
      grid_w[p] = ((frame_w + 7) & ~7) >> sub_x;
      grid_h[p] = ((frame_h + 7) & ~7) >> sub_y;
    }
    std::vector<std::vector<int>> owner(3);
    for (int p = 0; p < 3; ++p)
      owner[p].assign(static_cast<size_t>(grid_w[p]) * grid_h[p], -1);
    for (size_t bi = 0; bi < syntax.blocks.size(); ++bi) {
      const BlockSyntax& b = syntax.blocks[bi];
      const int p = b.plane;
      const int x1 = std::min(b.x + b.w, grid_w[p]);
      const int y1 = std::min(b.y + b.h, grid_h[p]);
      for (int y = std::max(b.y, 0); y < y1; ++y)
        for (int x = std::max(b.x, 0); x < x1; ++x)
          owner[p][static_cast<size_t>(y) * grid_w[p] + x] = static_cast<int>(bi);
    }
    auto sample_owner = [&](int p, int x, int y) {
      if (x < 0 || y < 0 || x >= grid_w[p] || y >= grid_h[p]) return -1;
      return owner[p][static_cast<size_t>(y) * grid_w[p] + x];
    };
    /* True when every in-frame sample of the rectangle already belongs to a
     * block earlier in coding order - the question BlockDecoded answers for
     * the above-right and below-left neighbours. */
    auto already_decoded = [&](int p, int x0, int x1, int y0, int y1, size_t bi) {
      for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
          const int o = sample_owner(p, x, y);
          if (o >= 0 && static_cast<size_t>(o) >= bi) return false;
        }
      }
      return true;
    };

    std::vector<int> block_wave(syntax.blocks.size(), 0);
    std::vector<char> above_right(syntax.blocks.size(), 0);
    std::vector<char> below_left(syntax.blocks.size(), 0);
    int max_wave = 0;
    /* Dependencies point backwards in coding order for everything the syntax
     * guarantees, so one pass is enough; the repeat loop is a safety net for a
     * plan that violates that (nothing should, but a wrong wave is a silent
     * pixel error, so it is not worth trusting a single pass). */
    bool changed = true;
    for (int pass = 0; changed && pass < 4; ++pass) {
      changed = false;
      for (size_t bi = 0; bi < syntax.blocks.size(); ++bi) {
        const BlockSyntax& b = syntax.blocks[bi];
        const int p = b.plane;
        int dep = -1;
        /* An inter/intra block belongs here too: its intra half reads the same
         * edges an intra block of that size does, so it cannot be scheduled with
         * the other inter work, which always lands in wave 0. The extensions the
         * specification allows (above-right, below-left, corner) change the
         * result of the directional and recursive intra modes only, and an
         * inter/intra block cannot use those, so it waits for the plain edge
         * alone. */
        if (b.is_intra || b.interintra) {
          if (b.is_intra) {
            above_right[bi] = static_cast<char>(
                b.have_above_right >= 0
                    ? (b.have_above_right != 0)
                    : (b.y > 0 && already_decoded(p, b.x + b.w, b.x + 2 * b.w - 1, b.y - 1,
                                                  b.y - 1, bi)));
            below_left[bi] = static_cast<char>(
                b.have_below_left >= 0
                    ? (b.have_below_left != 0)
                    : (b.x > 0 && already_decoded(p, b.x - 1, b.x - 1, b.y + b.h,
                                                  b.y + 2 * b.h - 1, bi)));
          }
          /* The predictor does not read every edge for every mode, and waiting
           * for a sample nothing looks at costs a wave: a directional block
           * reads two blocks of above samples only when its angle is below 90,
           * two blocks of left samples only when it is above 180, and every
           * other mode (DC, the smooths, Paeth, filter intra, chroma from luma)
           * reads the plain edges. A palette block reads no edge at all - the
           * colour map is coded, not predicted - so it waits for nothing here.
           * The three planes are one dependency grid, so an intra frame costs
           * the length of its diagonal, not one diagonal per plane. */
          int left_y1 = b.y + b.h - 1;
          int above_x1 = b.x + b.w - 1;
          if (b.is_intra && !b.use_palette) {
            const int mode = static_cast<int>(b.mode);
            /* the directional modes are Vertical .. D67, the eight entries
             * after Dc; anything else is not directional */
            if (mode >= static_cast<int>(PredictionMode::Vertical) &&
                mode <= static_cast<int>(PredictionMode::D67)) {
              const int pangle = intra_mode_angle(b.mode) + 3 * b.angle_delta;
              if (pangle < 90 && above_right[bi]) above_x1 = b.x + 2 * b.w - 1;
              if (pangle > 180 && below_left[bi]) left_y1 = b.y + 2 * b.h - 1;
            }
          } else if (b.is_intra) {
            /* a palette block: no edge samples are read */
            above_x1 = b.x - 1;
            left_y1 = b.y - 1;
          }
          const int left_x = b.x - 1;
          for (int y = b.y; y <= left_y1; ++y) {
            const int o = sample_owner(p, left_x, y);
            if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
          }
          const int above_y = b.y - 1;
          for (int x = b.x; x <= above_x1; ++x) {
            const int o = sample_owner(p, x, above_y);
            if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
          }
          /* the sample diagonally above-left, used as the corner (and by the
           * recursive filter, which reads its 2x2 neighbourhood) */
          if (above_x1 >= b.x || left_y1 >= b.y) {
            const int o = sample_owner(p, left_x, above_y);
            if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
          }
        }
        /* An intra block copy block reads a rectangle of *this* frame, so it
         * has to run after every block owning a sample of that rectangle - and
         * after their residual, which is what a strictly earlier wave gives. */
        if (b.intrabc) {
          /* the vector is coded in 1/8 luma samples, so a chroma plane shifts
           * it one bit further; both shifts are arithmetic, which is what puts
           * a negative vector below the sample rather than above it */
          const int mv_shift_x = 3 + (p > 0 && geom.ss_x ? 1 : 0);
          const int mv_shift_y = 3 + (p > 0 && geom.ss_y ? 1 : 0);
          const int sx = b.x + (b.mv[0][0] >> mv_shift_x);
          const int sy = b.y + (b.mv[0][1] >> mv_shift_y);
          const int x1 = std::min(sx + b.w, grid_w[p]);
          const int y1 = std::min(sy + b.h, grid_h[p]);
          for (int y = std::max(sy, 0); y < y1; ++y) {
            for (int x = std::max(sx, 0); x < x1; ++x) {
              const int o = sample_owner(p, x, y);
              if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
            }
          }
        }
        if (b.mode == PredictionMode::ChromaFromLuma) {
          /* the co-located luma samples, one per chroma sample */
          for (int y = 0; y < b.h; ++y) {
            for (int x = 0; x < b.w; ++x) {
              const int o = sample_owner(0, (b.x + x) << geom.ss_x, (b.y + y) << geom.ss_y);
              if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
            }
          }
        }
        const int wave = dep + 1;
        if (wave > block_wave[bi]) {
          block_wave[bi] = wave;
          changed = true;
        }
        max_wave = std::max(max_wave, block_wave[bi]);
      }
    }
    plan->waves.assign(static_cast<size_t>(max_wave) + 1, WaveJobs());

    /* ---- reference scaling ------------------------------------------------ */
    /* A superres frame is coded narrower than the pictures it predicts from, so
     * a motion vector addresses a *larger* reference frame: 7.11.3.3 scales the
     * sampling position by RefUpscaledWidth / FrameWidth. The references all
     * have to be the same display size as this frame for one pair of scales to
     * serve the whole frame, which is the geometry this decoder supports (a
     * stream that changes its picture size mid-stream already needs the pool
     * to hold mixed sizes, which it does not). */
    int scale_x = AV1G_SCALE_ONE, scale_y = AV1G_SCALE_ONE;
    for (int i = 0; i < 7; ++i) {
      const int rw = syntax.reference_width[i] > 0 ? syntax.reference_width[i] : syntax.width;
      const int rh = syntax.reference_height[i] > 0 ? syntax.reference_height[i] : syntax.height;
      if (rw == syntax.width && rh == syntax.height) continue;
      *error = "engine: reference scaling between pictures of different sizes is not supported";
      return false;
    }
    if (g.upscaled()) {
      scale_x = ((syntax.width << AV1G_REF_SCALE_SHIFT) + (frame_w / 2)) / frame_w;
      scale_y = ((syntax.height << AV1G_REF_SCALE_SHIFT) + (frame_h / 2)) / frame_h;
    }

    /* ---- jobs ------------------------------------------------------------- */
    /* The transform scratch is handed out in job order, one w * h block per
     * job. Two jobs never share a region, so the passes of a wave stay
     * independent even when the coefficient areas are smaller. */
    size_t tmp_cursor = 0;
    for (size_t bi = 0; bi < syntax.blocks.size(); ++bi) {
      const BlockSyntax& b = syntax.blocks[bi];
      WaveJobs& wj = plan->waves[block_wave[bi]];
      const int plane = b.plane;
      const int stride = g.stride[plane];
      const int dst_off = at(plane, b.x, b.y);

      if (b.is_intra) {
        Av1gIntraJob job;
        std::memset(&job, 0, sizeof(job));
        job.dst_off = dst_off;
        job.dst_stride = stride;
        job.w = b.w;
        job.h = b.h;
        job.mode = static_cast<int>(b.mode);
        job.bitdepth = b.bit_depth;
        job.x = b.x;
        job.y = b.y;
        job.have_left = b.have_left && b.x > 0;
        job.have_above = b.have_above && b.y > 0;
        job.have_above_right =
            b.have_above_right >= 0 ? b.have_above_right != 0 : above_right[bi] != 0;
        job.have_below_left =
            b.have_below_left >= 0 ? b.have_below_left != 0 : below_left[bi] != 0;
        /* The edge filter strength table depends on filterType, which comes
         * from the neighbours, and on whether the sequence enables the filter
         * at all. -1 means "do not touch the edge". */
        job.filter_type = syntax.enable_intra_edge_filter ? b.filter_type : -1;
        job.filter_intra = b.use_filter_intra && plane == 0;
        job.filter_intra_mode = static_cast<int>(b.filter_intra_mode);
        job.angle = b.angle_delta;
        job.cfl_alpha = b.cfl_alpha;
        /* The kernel derives the co-located luma position from the block's own
         * x/y (`(job.x + j) << subX`), so this is the start of the luma plane,
         * not the block's first luma sample. */
        job.luma_off = g.plane_off[0];
        job.luma_stride = g.stride[0];
        job.luma_w = (b.x + b.w) << geom.ss_x;
        job.luma_h = (b.y + b.h) << geom.ss_y;
        job.ss_x = geom.ss_x;
        job.ss_y = geom.ss_y;
        /* maxX/maxY are derived from the frame size rounded up to a multiple
         * of 8 (MiCols * MI_SIZE), not from the visible size. */
        job.max_x = (((frame_w + 7) & ~7) >> (plane ? geom.ss_x : 0)) - 1;
        job.max_y = (((frame_h + 7) & ~7) >> (plane ? geom.ss_y : 0)) - 1;
        if (b.use_palette) {
          job.palette = 1;
          job.palette_size = b.palette_size;
          for (int i = 0; i < 8; ++i) job.palette_color[i] = b.palette[i];
          job.palette_off = b.palette_off;
          job.palette_stride = b.palette_stride;
        }
        wj.intra.push_back(job);
      } else {
        Av1gInterJob job;
        std::memset(&job, 0, sizeof(job));
        job.dst_off = dst_off;
        job.dst_stride = stride;
        job.plane = plane;
        job.w = b.w;
        job.h = b.h;
        job.ref_x = b.x; /* patch origin, in plane samples */
        job.ref_y = b.y;
        /* The reference pool holds the *upscaled* picture, so a prediction
         * clamps to the display size even when this frame is coded narrower. */
        job.ref_w = g.out_w[plane];
        job.ref_h = g.out_h[plane];
        job.scale_x = scale_x;
        job.scale_y = scale_y;
        if (b.intrabc) {
          /* 7.11.4 step 11: a block copy's "reference frame" is the picture
           * being decoded, sized MiCols * MI_SIZE - the 4x4 grid rounded up,
           * which is wider than the visible picture when the width is not a
           * multiple of four. The plane padding covers the difference. */
          const int sub_x = plane > 0 && geom.ss_x ? 1 : 0;
          const int sub_y = plane > 0 && geom.ss_y ? 1 : 0;
          job.ref_w = ((frame_w + 3) & ~3) >> sub_x;
          job.ref_h = ((frame_h + 3) & ~3) >> sub_y;
          /* a block copy reads this frame, which is coded, not upscaled */
          job.scale_x = AV1G_SCALE_ONE;
          job.scale_y = AV1G_SCALE_ONE;
        }
        job.border = 0;
        job.mv0_x = b.mv[0][0];
        job.mv0_y = b.mv[0][1];
        job.mv1_x = b.mv[1][0];
        job.mv1_y = b.mv[1][1];
        job.filt = static_cast<int>(b.filter);
        job.filt_v = static_cast<int>(b.filter_v);
        job.sub_x = geom.ss_x && plane > 0 ? 1 : 0;
        job.sub_y = geom.ss_y && plane > 0 ? 1 : 0;
        job.compound = static_cast<int>(b.compound);
        job.w0 = b.blend[0];
        job.w1 = b.blend[1];
        job.bitdepth = b.bit_depth;
        /* Masked compound. The mask lives on the luma grid, so the wedge's
         * codebook entry is looked up for the luma size of this block (a chroma
         * block's own size would select the wrong entry) and a chroma job also
         * carries the luma plane: the difference weighted mask is a function of
         * the two *luma* predictions, which a chroma job has no other way to
         * reach. */
        job.mask_kind = AV1G_MASK_NONE;
        if (b.compound == CompoundType::Wedge || b.compound == CompoundType::DiffWtd) {
          const int luma_w = b.w << (plane > 0 ? geom.ss_x : 0);
          const int luma_h = b.h << (plane > 0 ? geom.ss_y : 0);
          job.mask_kind = b.compound == CompoundType::Wedge ? AV1G_MASK_WEDGE : AV1G_MASK_DIFFWTD;
          job.mask_sign = b.mask_sign;
          if (job.mask_kind == AV1G_MASK_WEDGE) {
            int dir = AV1G_WEDGE_HORIZONTAL, xoff = 0, yoff = 0, flip = 0;
            if (!tables::wedge_entry(luma_w, luma_h, b.wedge_index, &dir, &xoff, &yoff, &flip)) {
              *error = "engine: a wedge on a block size that has none";
              return false;
            }
            job.wedge_dir = dir;
            job.wedge_xoff = xoff;
            job.wedge_yoff = yoff;
            job.wedge_flip = flip;
          }
          /* A difference weighted mask is a function of the two luma
           * predictions on every chroma plane, subsampled or not: at 4:4:4 a
           * chroma sample covers exactly one luma sample, but the two are still
           * different pictures, so the chroma job carries the luma plane. */
          if (job.mask_kind == AV1G_MASK_DIFFWTD && plane > 0) {
            job.luma_ref0_off = g.plane_off[0];
            job.luma_ref1_off = g.plane_off[0];
            job.luma_stride = g.stride[0];
            job.luma_x = b.x << geom.ss_x;
            job.luma_y = b.y << geom.ss_y;
            job.luma_ref_w = g.out_w[0];
            job.luma_ref_h = g.out_h[0];
          }
        }
        /* Inter/intra compound (7.11.4). The intra half is predicted from the
         * frame being decoded, so the job carries the neighbours it may read and
         * the same maxX/maxY clamp an intra job carries; a wedge blend comes in
         * through the codebook entry a masked compound uses, with a sign of zero
         * because the syntax codes none for an inter/intra block. */
        job.interintra = b.interintra ? 1 : 0;
        job.interintra_mode = b.interintra_mode;
        job.have_left = b.have_left && b.x > 0;
        job.have_above = b.have_above && b.y > 0;
        job.max_x = (((frame_w + 7) & ~7) >> (plane ? geom.ss_x : 0)) - 1;
        job.max_y = (((frame_h + 7) & ~7) >> (plane ? geom.ss_y : 0)) - 1;
        if (b.interintra && b.interintra_wedge) {
          const int luma_w = b.w << (plane > 0 ? geom.ss_x : 0);
          const int luma_h = b.h << (plane > 0 ? geom.ss_y : 0);
          int dir = AV1G_WEDGE_HORIZONTAL, xoff = 0, yoff = 0, flip = 0;
          if (!tables::wedge_entry(luma_w, luma_h, b.wedge_index, &dir, &xoff, &yoff,
                                   &flip)) {
            *error = "engine: an inter/intra wedge on a block size that has none";
            return false;
          }
          job.interintra_wedge = 1;
          job.wedge_dir = dir;
          job.wedge_xoff = xoff;
          job.wedge_yoff = yoff;
          job.wedge_flip = flip;
        }
        job.ref0_slot = resolve_slot(syntax, b.ref[0]);
        job.ref1_slot = resolve_slot(syntax, b.ref[1]);
        job.ref0_off = g.plane_off[plane];
        job.ref1_off = g.plane_off[plane];
        job.ref0_stride = stride;
        job.ref1_stride = stride;
        /* OBMC strips: the neighbour's vector, filters and reference frame. The
         * offset starts at the plane and the backend adds the slot, exactly as
         * it does for ref0/ref1. */
        job.obmc_count = 0;
        for (const ObmcCandidate& c : b.obmc) {
          if (job.obmc_count >= static_cast<int>(sizeof(job.obmc) / sizeof(job.obmc[0])))
            break;
          Av1gObmcCandidate& o = job.obmc[job.obmc_count];
          std::memset(&o, 0, sizeof(o));
          o.pass = c.pass;
          o.rx = c.x;
          o.ry = c.y;
          o.w = c.w;
          o.h = c.h;
          o.mv_x = c.mv[0];
          o.mv_y = c.mv[1];
          o.filt = static_cast<int>(c.filter);
          o.filt_v = static_cast<int>(c.filter_v);
          o.ref_slot = resolve_slot(syntax, c.ref);
          o.ref_off = g.plane_off[plane];
          ++job.obmc_count;
        }
        /* Global warped motion: the shear is computed here, once per job,
         * because the model is a frame property and the two passes of the
         * warp need it. A model whose shear is too steep to apply is not
         * warped at all - the block falls back to its vector, which is the
         * specification's warpValid. */
        const bool warp_disabled = std::getenv("AV1GPU_NO_WARP") != nullptr;
        for (int i = 0; i < 2; ++i) {
          job.warp[i] = 0;
          job.mask_warp[i] = 0;
          if ((!b.warp[i] && !b.mask_warp[i]) || warp_disabled) continue;
          /* a chroma region that is too small to warp may still need the luma
           * region's model to build a difference weighted mask */
          if (b.warp[i]) job.warp[i] = 1;
          if (b.mask_warp[i]) job.mask_warp[i] = 1;
          int alpha = 0, beta = 0, gamma = 0, delta = 0;
          if (!tables::gm_setup_shear(b.warp_params[i], &alpha, &beta, &gamma, &delta))
            continue;
          for (int j = 0; j < 6; ++j) job.warp_params[i][j] = b.warp_params[i][j];
          job.warp_abcd[i][0] = alpha;
          job.warp_abcd[i][1] = beta;
          job.warp_abcd[i][2] = gamma;
          job.warp_abcd[i][3] = delta;
          if (std::getenv("AV1GPU_WARPDEBUG"))
            std::fprintf(stderr, "WARP ref=%d plane=%d x=%d y=%d w=%d h=%d params=%d,%d,%d,%d,%d,%d abcd=%d,%d,%d,%d\n",
                         i, plane, b.x, b.y, b.w, b.h, b.warp_params[i][0], b.warp_params[i][1],
                         b.warp_params[i][2], b.warp_params[i][3], b.warp_params[i][4],
                         b.warp_params[i][5], alpha, beta, gamma, delta);
        }
        wj.inter.push_back(job);
      }

      for (const TransformBlock& tx : b.transforms) {
        Av1gItxJob itx;
        std::memset(&itx, 0, sizeof(itx));
        itx.coef_off = tx.coef_offset;
        itx.tmp_off = static_cast<int>(tmp_cursor);
        tmp_cursor += static_cast<size_t>(tx.w) * static_cast<size_t>(tx.h);
        plan->tmp_size = tmp_cursor;
        itx.dst_off = dst_off + tx.y * stride + tx.x;
        itx.dst_stride = stride;
        itx.w = tx.w;
        itx.h = tx.h;
        itx.tx_type = static_cast<int>(tx.tx_type);
        itx.eob = tx.eob;
        itx.lossless = b.lossless ? 1 : 0;
        itx.row_shift = b.lossless ? 0 : tables::tx_row_shift(tx.w, tx.h);
        itx.bitdepth = b.bit_depth;
        if (itx.eob <= 0) continue; /* nothing to add */
        if (b.residual_provided)
          wj.residual.push_back(itx);
        else
          wj.itx.push_back(itx);
      }
    }

    /* ---- deblock ---------------------------------------------------------- */
    for (const DeblockEdge& e : syntax.deblock) {
      Av1gDeblockJob job;
      std::memset(&job, 0, sizeof(job));
      const int plane = e.plane;
      /* e.x / e.y is the first sample on the far side of the edge, so p1 (two
       * samples in) sits two steps before it. `stride` steps across the edge,
       * `step` runs along it. */
      job.p_off = e.vertical ? at(plane, e.x - 2, e.y) : at(plane, e.x, e.y - 2);
      job.stride = e.vertical ? 1 : g.stride[plane];
      job.step = e.vertical ? g.stride[plane] : 1;
      job.len = e.length;
      job.level = e.level;
      job.sharpness = syntax.sharpness;
      job.filter_size = e.filter_size;
      job.is_chroma = plane != 0;
      job.bitdepth = geom.bit_depth;
      (e.vertical ? plan->deblock_v : plan->deblock_h).push_back(job);
    }

    /* ---- CDEF -------------------------------------------------------------- */
    for (const CdefBlock& c : syntax.cdef) {
      Av1gCdefJob job;
      std::memset(&job, 0, sizeof(job));
      const int plane = c.plane;
      const bool chroma = plane > 0;
      job.x = c.x;
      job.y = c.y;
      job.src_off = at(plane, c.x, c.y);
      job.src_stride = g.stride[plane];
      job.dst_off = job.src_off;
      job.dst_stride = g.stride[plane];
      /* the direction always comes from the luma, so a chroma block carries the
       * luma block it is co-located with */
      job.luma_off = at(0, chroma ? c.x << geom.ss_x : c.x,
                        chroma ? c.y << geom.ss_y : c.y);
      job.luma_stride = g.stride[0];
      job.pri_strength = c.pri_strength;
      job.sec_strength = c.sec_strength;
      job.damping = c.damping;
      job.dir = c.dir;
      job.uv = chroma ? 1 : 0;
      job.ss_x = geom.ss_x;
      job.ss_y = geom.ss_y;
      job.bitdepth = geom.bit_depth;
      /* The block grid and the halo are the *coded* extent (MiCols * MI_SIZE,
       * rounded to eight), not the visible one: 7.15 copies and writes the
       * whole 8x8 block and decides availability in mi units, so the last
       * partial block of a frame whose size is not a multiple of eight filters
       * against samples that exist in the border but not in the picture. */
      job.plane_w = grid_w[plane];
      job.plane_h = grid_h[plane];
      plan->cdef.push_back(job);
    }

    /* ---- superres ---------------------------------------------------------- */
    /* The upscaling process (7.15) runs on the CDEF output and before the loop
     * restoration and the film grain, so both the output and the reference
     * store are in the display domain. One job per plane: the backend splits it
     * into rows (CPU) or samples (the GPU ones). */
    if (g.upscaled()) {
      for (int plane = 0; plane < 3; ++plane) {
        Av1gUpscaleJob job;
        std::memset(&job, 0, sizeof(job));
        const int sub_x = plane > 0 ? geom.ss_x : 0;
        job.src_off = g.plane_off[plane];
        job.dst_off = g.plane_off[plane];
        job.src_stride = g.stride[plane];
        job.dst_stride = g.stride[plane];
        job.y0 = 0;
        job.rows = g.out_h[plane];
        job.upscaled_w = g.out_w[plane];
        /* The last coded sample of the plane. The specification clamps to
         * miW * MI_SIZE - 1 (the block grid, rounded up to eight luma
         * samples), which for a width that is a multiple of eight - every
         * width an encoder produces - is the visible width. A decoder with a
         * padded plane clamps to the visible width, and that is what the
         * reference decoders do, so the two only differ for a width no encoder
         * writes. */
        job.max_x = g.plane_w[plane] - 1;
        (void)sub_x;
        av1g_superres_params(g.plane_w[plane], g.out_w[plane], &job.step_x,
                             &job.initial_subpel_x);
        job.bitdepth = geom.bit_depth;
        plan->upscale.push_back(job);
      }
    }

    /* ---- loop restoration (7.17) ------------------------------------------- */
    /* The picture is restored in stripes, and one job is one stripe's part of
     * one restoration unit: inside the stripe the samples come from the CDEF
     * output, the two lines either side of it from the deblocked picture, so a
     * job may only see one of the two.
     *
     * Which unit filters a row is not simply its position divided by the unit
     * size: 7.17.1 offsets the row by eight luma samples before the division
     * (the same eight samples the stripes are shifted by), so a unit owns the
     * samples up to eight rows past its nominal end. The offset is a multiple
     * of the stripe grid, so one stripe piece always belongs to one unit, and
     * the pieces are the right granularity. Nothing here depends on the order
     * the jobs run in, which is what lets the GPU backends run them together. */
    for (int plane = 0; plane < 3; ++plane) {
      if (syntax.restoration_type[plane] == kRestoreNone) continue;
      const int sub_x = plane > 0 ? geom.ss_x : 0;
      const int sub_y = plane > 0 ? geom.ss_y : 0;
      const int unit_size = 1 << syntax.restoration_unit_log2[plane];
      const int unit_cols = std::max(syntax.restoration_unit_cols[plane], 1);
      const int plane_w = g.out_w[plane];
      const int plane_h = g.out_h[plane];
      /* count_units_in_frame( unitSize, Round2( frameSize, subY ) ) */
      const int frame_h = (geom.height + (sub_y ? 1 << (sub_y - 1) : 0)) >> sub_y;
      const int unit_rows = std::max((frame_h + (unit_size >> 1)) / unit_size, 1);
      for (int y = 0; y < plane_h;) {
        const int stripe = ((y << sub_y) + 8) / 64;
        const int stripe_start = (-8 + stripe * 64) >> sub_y;
        const int stripe_end = stripe_start + (64 >> sub_y) - 1;
        const int y1 = std::min(plane_h, stripe_end + 1);
        const int unit_row =
            std::min(unit_rows - 1, (((y << sub_y) + 8) >> sub_y) / unit_size);
        for (int unit_col = 0; unit_col < unit_cols; ++unit_col) {
          const size_t index = static_cast<size_t>(unit_row) * unit_cols + unit_col;
          if (index >= syntax.restoration[plane].size()) continue;
          const RestorationUnit& unit = syntax.restoration[plane][index];
          if (unit.type == kRestoreNone) continue;
          const int ux0 = unit_col * unit_size;
          /* 7.17.1 clamps the unit index, so the last unit of a row or of the
           * frame extends over the samples past its nominal end. */
          const int ux1 = unit_col + 1 == unit_cols ? plane_w
                                                    : std::min(ux0 + unit_size, plane_w);
          if (ux1 <= ux0) continue;
          Av1gLrJob job;
          std::memset(&job, 0, sizeof(job));
          job.plane = plane;
          job.dst_off = at(plane, ux0, y);
          job.dst_stride = g.stride[plane];
          job.cdef_off = at(plane, 0, 0);
          job.cdef_stride = g.stride[plane];
          job.curr_off = at(plane, 0, 0);
          job.curr_stride = g.stride[plane];
          job.x = ux0;
          job.y = y;
          job.w = ux1 - ux0;
          job.h = y1 - y;
          job.plane_w = plane_w;
          job.plane_h = plane_h;
          job.stripe_start_y = stripe_start;
          job.stripe_end_y = stripe_end;
          job.type = unit.type == kRestoreWiener ? AV1G_LR_WIENER : AV1G_LR_SGRPROJ;
          if (unit.type == kRestoreWiener) {
            for (int i = 0; i < 3; ++i) {
              /* pass 0 is the vertical filter, which the bitstream codes
               * first */
              job.vfilter[i] = unit.wiener[0][i];
              job.hfilter[i] = unit.wiener[1][i];
            }
          } else {
            for (int i = 0; i < 2; ++i) {
              job.sgr_radius[i] = kSgrParams[unit.sgr_set & 15][i * 2];
              job.sgr_eps[i] = kSgrParams[unit.sgr_set & 15][i * 2 + 1];
              job.sgr_xqd[i] = unit.sgr_xqd[i];
            }
          }
          job.bitdepth = geom.bit_depth;
          plan->restoration.push_back(job);
        }
        y = y1;
      }
      (void)sub_x;
    }

    /* ---- film grain -------------------------------------------------------- */
    /* film grain is added to the *output* picture, i.e. after the upscaling, so
     * the noise is cut for the display size. The sequential half of the
     * synthesis happens here, on the host (see src/core/grain.cpp); the plan
     * carries the finished noise image. */
    if (syntax.grain.apply) {
      const bool identity = syntax.color.matrix == ColorMatrix::Identity;
      build_grain(syntax.grain, geom.bit_depth, geom.ss_x, geom.ss_y, identity, g.out_w[0],
                  g.out_h[0], &plan->grain_noise);
      for (int plane = 0; plane < 3; ++plane) {
        append_grain_jobs(plan->grain_noise, plane, g.out_w[plane], g.out_h[plane],
                          g.plane_off[plane], g.stride[plane], /*luma_off=*/g.plane_off[0],
                          g.stride[0], g.out_w[0], geom.ss_x, geom.ss_y, &plan->grain_chroma,
                          &plan->grain_luma);
      }
    }

    plan->blocks = static_cast<int>(syntax.blocks.size());
    return true;
}



} /* namespace internal */

/* ------------------------------------------------------------------------- */

bool Decoder::decode(const FrameSyntax& syntax, Frame* out, std::string* error) {
  const int slot = decode_reference(syntax, out, error);
  if (slot < 0) return false;
  release_reference(slot);
  return true;
}

std::unique_ptr<Decoder> Decoder::create(const BackendOptions& options, std::string* error) {
  std::string local;
  if (!error) error = &local;
  error->clear();

  std::vector<BackendKind> order;
  switch (options.kind) {
    case BackendKind::Cpu:
      order = {BackendKind::Cpu};
      break;
    case BackendKind::OpenCL:
      order = {BackendKind::OpenCL};
      if (options.allow_cpu_fallback) order.push_back(BackendKind::Cpu);
      break;
    case BackendKind::CUDA:
      order = {BackendKind::CUDA};
      if (options.allow_cpu_fallback) order.push_back(BackendKind::Cpu);
      break;
    case BackendKind::Auto:
    default:
      order = {BackendKind::CUDA, BackendKind::OpenCL, BackendKind::Cpu};
      break;
  }

  std::string last_error;
  for (BackendKind kind : order) {
    std::string err;
    std::unique_ptr<internal::Backend> backend;
    switch (kind) {
      case BackendKind::Cpu:
        backend = internal::create_cpu_backend(options, &err);
        break;
      case BackendKind::OpenCL:
        backend = internal::create_opencl_backend(options, &err);
        break;
      case BackendKind::CUDA:
        backend = internal::create_cuda_backend(options, &err);
        break;
      default:
        break;
    }
    if (backend)
      return std::unique_ptr<Decoder>(
          new internal::EngineDecoder(std::move(backend), options));
    if (!err.empty()) last_error = err;
    if (options.kind != BackendKind::Auto && !options.allow_cpu_fallback) break;
  }
  *error = last_error.empty() ? "no usable backend" : last_error;
  return nullptr;
}

std::unique_ptr<Decoder> create_best_decoder(const BackendOptions& options, std::string* error) {
  return Decoder::create(options, error);
}

} /* namespace av1gpu */
