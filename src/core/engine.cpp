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
    if (it == held_slots_.end()) return;
    held_slots_.erase(it);
    backend_->release_reference(slot);
  }

  bool decode(const FrameSyntax& syntax, Frame* out, std::string* error) override {
    const int slot = decode_reference(syntax, out, error);
    if (slot < 0) return false;
    release_reference(slot);
    return true;
  }

  int decode_reference(const FrameSyntax& syntax, Frame* out, std::string* error) override {
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
      return -1;
    }

    FramePlan plan;
    if (!build_frame_plan(syntax, backend_->prepare(geom), options_,
                          backend_->reference_capacity(), &plan, error))
      return -1;

    int slot = -1;
    if (!backend_->decode(plan, true, out, &slot, &stats_, error)) return -1;
    if (slot >= 0) held_slots_.push_back(slot);
    return slot;
  }

  bool to_rgb(int slot, ColorMatrix matrix, ColorRange range, std::vector<uint8_t>* rgb,
              std::string* error) override {
    return backend_->to_rgb(slot, matrix, range, rgb, error);
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
    std::vector<std::vector<int>> owner(3);
    for (int p = 0; p < 3; ++p)
      owner[p].assign(static_cast<size_t>(g.plane_w[p]) * g.plane_h[p], -1);
    for (size_t bi = 0; bi < syntax.blocks.size(); ++bi) {
      const BlockSyntax& b = syntax.blocks[bi];
      const int p = b.plane;
      const int x1 = std::min(b.x + b.w, g.plane_w[p]);
      const int y1 = std::min(b.y + b.h, g.plane_h[p]);
      for (int y = b.y; y < y1; ++y)
        for (int x = b.x; x < x1; ++x)
          owner[p][static_cast<size_t>(y) * g.plane_w[p] + x] = static_cast<int>(bi);
    }
    auto sample_owner = [&](int p, int x, int y) {
      if (x < 0 || y < 0 || x >= g.plane_w[p] || y >= g.plane_h[p]) return -1;
      return owner[p][static_cast<size_t>(y) * g.plane_w[p] + x];
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
        if (b.is_intra) {
          /* The edge samples, plus the extensions the specification allows:
           * the row above by another block width and the column to the left by
           * another block height. */
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
          const int left_x = b.x - 1;
          const int left_y1 = b.y + (below_left[bi] ? 2 * b.h : b.h) - 1;
          for (int y = b.y; y <= left_y1; ++y) {
            const int o = sample_owner(p, left_x, y);
            if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
          }
          const int above_y = b.y - 1;
          const int above_x1 = b.x + (above_right[bi] ? 2 * b.w : b.w) - 1;
          for (int x = b.x; x <= above_x1; ++x) {
            const int o = sample_owner(p, x, above_y);
            if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
          }
          /* the sample diagonally above-left, used as the corner */
          const int o = sample_owner(p, left_x, above_y);
          if (o >= 0) dep = std::max(dep, block_wave[static_cast<size_t>(o)]);
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

    /* ---- jobs ------------------------------------------------------------- */
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
        job.max_x = (((geom.width + 7) & ~7) >> (plane ? geom.ss_x : 0)) - 1;
        job.max_y = (((geom.height + 7) & ~7) >> (plane ? geom.ss_y : 0)) - 1;
        wj.intra.push_back(job);
      } else {
        Av1gInterJob job;
        std::memset(&job, 0, sizeof(job));
        job.dst_off = dst_off;
        job.dst_stride = stride;
        job.w = b.w;
        job.h = b.h;
        job.ref_x = b.x; /* patch origin, in plane samples */
        job.ref_y = b.y;
        job.ref_w = g.plane_w[plane];
        job.ref_h = g.plane_h[plane];
        job.border = 0;
        job.mv0_x = b.mv[0][0];
        job.mv0_y = b.mv[0][1];
        job.mv1_x = b.mv[1][0];
        job.mv1_y = b.mv[1][1];
        job.filt = static_cast<int>(b.filter);
        job.compound = static_cast<int>(b.compound);
        job.w0 = b.blend[0];
        job.w1 = b.blend[1];
        job.bitdepth = b.bit_depth;
        job.ref0_slot = resolve_slot(syntax, b.ref[0]);
        job.ref1_slot = resolve_slot(syntax, b.ref[1]);
        job.ref0_off = g.plane_off[plane];
        job.ref1_off = g.plane_off[plane];
        job.ref0_stride = stride;
        job.ref1_stride = stride;
        wj.inter.push_back(job);
      }

      for (const TransformBlock& tx : b.transforms) {
        Av1gItxJob itx;
        std::memset(&itx, 0, sizeof(itx));
        itx.coef_off = tx.coef_offset;
        itx.tmp_off = static_cast<int>(tx.coef_offset);
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
      job.src_off = at(plane, c.x, c.y);
      job.src_stride = g.stride[plane];
      job.dst_off = job.src_off;
      job.dst_stride = g.stride[plane];
      job.pri_strength = c.pri_strength;
      job.sec_strength = c.sec_strength;
      job.damping = c.damping;
      job.dir = c.dir;
      job.bitdepth = geom.bit_depth;
      job.plane_w = g.plane_w[plane];
      job.plane_h = g.plane_h[plane];
      plan->cdef.push_back(job);
    }

    /* ---- film grain -------------------------------------------------------- */
    if (syntax.apply_grain && syntax.grain.template_size > 0) {
      plan->grain_stride = std::max(64, syntax.grain.template_size);
      plan->grain_h = std::max(64, syntax.grain.template_size);
      tables::grain_template(syntax.grain.seed, plan->grain_stride, plan->grain_h,
                             &plan->grain_template);
      for (int i = 0; i < 32; ++i) {
        plan->grain_lut[i] = syntax.grain.lut[i];
        plan->grain_lut[32 + i] = syntax.grain.lut[i];
      }
      const int tile = 128;
      for (int plane = 0; plane < 3; ++plane) {
        const int pw = g.plane_w[plane];
        const int ph = g.plane_h[plane];
        for (int y = 0; y < ph; y += tile) {
          for (int x = 0; x < pw; x += tile) {
            Av1gGrainJob job;
            std::memset(&job, 0, sizeof(job));
            job.dst_off = at(plane, x, y);
            job.dst_stride = g.stride[plane];
            job.x = x;
            job.y = y;
            job.w = std::min(tile, pw - x);
            job.h = std::min(tile, ph - y);
            job.bitdepth = geom.bit_depth;
            job.grain_stride = plan->grain_stride;
            job.grain_h = plan->grain_h;
            job.overlap = syntax.grain.overlap;
            plan->grain.push_back(job);
          }
        }
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
