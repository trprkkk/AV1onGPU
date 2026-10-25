/*
 * AV1onGPU - CPU backend
 *
 * This is not just a fallback: it is the reference. It runs exactly the same
 * scalar functions from kernels/av1gpu_kernels.h that the OpenCL and CUDA
 * backends compile for the device, so CPU and GPU output must be bit identical
 * (tests/test_pipeline.cpp checks that).
 *
 * The unit of work is a job, in the same order the GPU sees it, which makes this
 * backend a good debugging tool: if the CPU result matches the reference decoder
 * and the GPU does not, the bug is in the device path.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/backend.hpp"

#if defined(AV1GPU_HAVE_OPENMP)
#include <omp.h>
#endif

namespace av1gpu {
namespace internal {
namespace {

double now_ms() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point start = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

/* AV1 has eight reference slots, and a stream may keep a different picture in
 * every one of them. The ninth buffer is the picture being decoded: it cannot
 * be written into a buffer whose samples the prediction still has to read, and
 * a frame is allowed to refresh a slot it also predicts from, so the output
 * always needs a buffer of its own. */
constexpr int kReferenceSlots = 9;

/* The slots live in one flat buffer, slot after slot, so a job can name any
 * reference (a compound prediction, or the neighbour of an OBMC strip) with an
 * absolute offset instead of a second pointer - which is exactly what the GPU
 * backends do, where every reference is an offset into one pool buffer. */
struct RefFrame {
  bool used = false;
  int width = 0, height = 0, bit_depth = 8, ss_x = 1, ss_y = 1;
};

class CpuBackend final : public Backend {
 public:
  explicit CpuBackend(const BackendOptions& options) : options_(options) {
    info_.kind = BackendKind::Cpu;
    info_.name = "CPU reference";
    info_.vendor = "AV1onGPU";
#if defined(__aarch64__)
    info_.version = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    info_.version = "x86-64";
#else
    info_.version = "generic";
#endif
#if defined(_OPENMP)
    info_.detail = "OpenMP enabled";
#else
    info_.detail = "single threaded";
#endif
    info_.available = true;
    info_.compute_units = 1;
  }

  const DeviceInfo& info() const override { return info_; }
  const BackendOptions& options() const override { return options_; }

  PlanLayout prepare(const FrameGeometry& geometry) override {
    const PlanLayout g = make_plan_layout(geometry, 32);
    const size_t size = g.buffer_size;
    if (frame_.size() < size) frame_.resize(size, 0);
    if (scratch_.size() < size) scratch_.resize(size, 0);
    if (ref_pool_.size() < size * kReferenceSlots)
      ref_pool_.assign(size * kReferenceSlots, 0);
    layout_ = g;
    return g;
  }

  int upload_reference(const Frame& frame, std::string* error) override {
    const int slot = allocate_slot();
    if (slot < 0) {
      if (error) *error = "CPU reference pool exhausted";
      return -1;
    }
    /* pack into the same flat layout the jobs use */
    const PlanLayout& g = layout_;
    RefFrame& ref = refs_[slot];
    ref.used = true;
    ref.width = frame.width;
    ref.height = frame.height;
    ref.bit_depth = frame.bit_depth;
    ref.ss_x = frame.ss_x;
    ref.ss_y = frame.ss_y;
    uint16_t* base = ref_pool_.data() + static_cast<size_t>(slot) * g.buffer_size;
    std::fill(base, base + g.buffer_size, 0);
    for (int p = 0; p < 3; ++p) {
      const int w = g.plane_w[p];
      const int h = g.plane_h[p];
      for (int y = 0; y < h; ++y)
        std::memcpy(base + g.plane_off[p] + static_cast<size_t>(y) * g.stride[p],
                    frame.plane(p) + static_cast<size_t>(y) * frame.stride[p],
                    static_cast<size_t>(w) * sizeof(uint16_t));
    }
    return slot;
  }

  void release_reference(int slot) override {
    if (slot >= 0 && slot < kReferenceSlots) refs_[slot].used = false;
  }

  int reference_capacity() const override { return kReferenceSlots; }

  bool decode(const FramePlan& plan, bool keep_reference, Frame* out, int* out_slot,
              DecodeStats* stats, std::string* error) override {
    std::string local;
    if (!error) error = &local;
    error->clear();

    const double t_start = now_ms();
    const PlanLayout& g = plan.layout;
    const int maxv = plan.maxv;
    const int* filters = tables::filters().data();
    uint16_t* frame = frame_.data();

    /* AV1GPU_INTERRECT=x0,y0,x1,y1 prints every inter prediction job whose
     * origin falls inside that rectangle of a chroma plane, with the frame it
     * belongs to. Comparing a wrong sample with the job that should have
     * produced it is the quickest way to see a vector or a reference slot go
     * astray, which is why the dump is on the job rather than on the block. */
    static int g_decode_count = 0;
    ++g_decode_count;
    /* AV1GPU_DUMPCODED=<file> appends every decoded picture, shown or not, in
     * coding order. Comparing against a reference decoder's invisible frame
     * output (dav1d's output_invisible_frames) is the only way to tell which
     * frame a drift starts on: a display-order dump cannot, because a stream
     * with show_existing_frame reorders them. */
    const char* dump_coded = std::getenv("AV1GPU_DUMPCODED");

    if (std::getenv("AV1GPU_PLAN_STATS")) {
      size_t ni = 0, nt = 0, nr = 0, nx = 0;
      for (const WaveJobs& w : plan.waves) {
        ni += w.intra.size();
        nt += w.itx.size();
        nr += w.residual.size();
        nx += w.inter.size();
      }
      std::fprintf(stderr,
                   "PLAN waves=%zu intra=%zu itx=%zu residual=%zu inter=%zu deblock=%zu/%zu "
                   "cdef=%zu grain=%zu\n",
                   plan.waves.size(), ni, nt, nr, nx, plan.deblock_v.size(),
                   plan.deblock_h.size(), plan.cdef.size(), plan.grain.size());
    }

    /* ---- prediction, transform, residual --------------------------------- */
    std::vector<int> tmp(plan.tmp_size + 64, 0);
    /* the palette colour maps of the frame, read by the blocks that code one */
    const unsigned char* palette = plan.palette_map.data();
    for (const WaveJobs& wave : plan.waves) {
      /* Inside a wave no job reads another job's output, so every stage of a
       * wave is embarrassingly parallel - the same property the GPU backends
       * rely on. The order within a wave (intra, then transforms) is required,
       * because a block's residual is added on top of its prediction. */
      const int ninter = static_cast<int>(wave.inter.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
      for (int i = 0; i < ninter; ++i) {
        /* every reference - the two a compound block names, and the ones its
         * OBMC strips name - is an absolute offset into the pool */
        Av1gInterJob job = wave.inter[static_cast<size_t>(i)];
        if (job.ref0_slot >= 0)
          job.ref0_off += job.ref0_slot * static_cast<int>(g.buffer_size);
        if (job.ref1_slot >= 0)
          job.ref1_off += job.ref1_slot * static_cast<int>(g.buffer_size);
        /* a difference weighted chroma block builds its mask on the luma plane, so
         * those offsets need the same slot rebase the two named planes get */
        if (job.mask_kind == AV1G_MASK_DIFFWTD) {
          if (job.ref0_slot >= 0)
            job.luma_ref0_off += job.ref0_slot * static_cast<int>(g.buffer_size);
          if (job.ref1_slot >= 0)
            job.luma_ref1_off += job.ref1_slot * static_cast<int>(g.buffer_size);
        }
        for (int k = 0; k < job.obmc_count; ++k)
          if (job.obmc[k].ref_slot >= 0)
            job.obmc[k].ref_off += job.obmc[k].ref_slot * static_cast<int>(g.buffer_size);
        if (job.obmc_count > 0 && std::getenv("AV1GPU_OBMCDUMP")) {
          static int obmc_dumped = 0;
          if (obmc_dumped < 4) {
            ++obmc_dumped;
            std::fprintf(stderr, "OUROBMC w=%d h=%d refxy=%d,%d mv=%d,%d filt=%d/%d comp=%d "
                                 "slots=%d,%d strips=%d warp=%d,%d\n",
                         job.w, job.h, job.ref_x, job.ref_y, job.mv0_x, job.mv0_y, job.filt,
                         job.filt_v, job.compound, job.ref0_slot, job.ref1_slot,
                         job.obmc_count, job.warp[0], job.warp[1]);
            for (int k = 0; k < job.obmc_count; ++k)
              std::fprintf(stderr, "  strip %d pass=%d rect=%d,%d %dx%d mv=%d,%d filt=%d/%d "
                                   "slot=%d off=%d\n",
                           k, job.obmc[k].pass, job.obmc[k].rx, job.obmc[k].ry, job.obmc[k].w,
                           job.obmc[k].h, job.obmc[k].mv_x, job.obmc[k].mv_y, job.obmc[k].filt,
                           job.obmc[k].filt_v, job.obmc[k].ref_slot, job.obmc[k].ref_off);
          }
        }
        if (std::getenv("AV1GPU_MCDUMP") && job.dst_off == g.plane_off[0]) {
          static int dumped = 0;
          if (dumped < 3) {
            ++dumped;
            std::fprintf(stderr,
                         "OURMC w=%d h=%d refxy=%d,%d mv=%d,%d sub=%d,%d filt=%d/%d comp=%d "
                         "slots=%d,%d obmc=%d dst=%d\n",
                         job.w, job.h, job.ref_x, job.ref_y, job.mv0_x, job.mv0_y, job.sub_x,
                         job.sub_y, job.filt, job.filt_v, job.compound, job.ref0_slot,
                         job.ref1_slot, job.obmc_count, job.dst_off);
          }
        }
        if (const char* rect = std::getenv("AV1GPU_INTERRECT")) {
          int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
          if (std::sscanf(rect, "%d,%d,%d,%d", &x0, &y0, &x1, &y1) == 4) {
            for (int p = 0; p < 3; ++p) {
              if (job.dst_off < g.plane_off[p]) continue;
              if (p < 2 && job.dst_off >= g.plane_off[p + 1]) continue;
              const int rel = job.dst_off - g.plane_off[p];
              const int jx = rel % g.stride[p], jy = rel / g.stride[p];
              if (jx < x0 || jx > x1 || jy < y0 || jy > y1) continue;
              std::fprintf(stderr,
                           "IJOB frame=%d plane=%d x=%d y=%d w=%d h=%d mv=%d,%d/%d,%d "
                           "filter=%d/%d compound=%d slots=%d,%d obmc=%d warp=%d,%d\n",
                           g_decode_count - 1, p, jx, jy, job.w, job.h, job.mv0_x,
                           job.mv0_y, job.mv1_x, job.mv1_y, job.filt, job.filt_v,
                           job.compound, job.ref0_slot, job.ref1_slot, job.obmc_count,
                           job.warp[0], job.warp[1]);
            }
          }
        }
        av1g_inter_block(&job, frame, frame, ref_pool_.data(), filters, maxv);
        if (job.obmc_count > 0 && std::getenv("AV1GPU_OBMCDUMP")) {
          static int obmc_dumped = 0;
          if (obmc_dumped < 4) {
            ++obmc_dumped;
            std::fprintf(stderr, "OUROBMC w=%d h=%d refxy=%d,%d mv=%d,%d filt=%d/%d comp=%d "
                                 "slots=%d,%d strips=%d\n",
                         job.w, job.h, job.ref_x, job.ref_y, job.mv0_x, job.mv0_y, job.filt,
                         job.filt_v, job.compound, job.ref0_slot, job.ref1_slot, job.obmc_count);
            for (int k = 0; k < job.obmc_count; ++k)
              std::fprintf(stderr, "  strip %d pass=%d rect=%d,%d %dx%d mv=%d,%d filt=%d/%d "
                                   "slot=%d off=%d\n",
                           k, job.obmc[k].pass, job.obmc[k].rx, job.obmc[k].ry, job.obmc[k].w,
                           job.obmc[k].h, job.obmc[k].mv_x, job.obmc[k].mv_y, job.obmc[k].filt,
                           job.obmc[k].filt_v, job.obmc[k].ref_slot, job.obmc[k].ref_off);
          }
        }
        if (std::getenv("AV1GPU_MCDUMP") && job.dst_off == g.plane_off[0]) {
          static int dumped2 = 0;
          if (dumped2 < 3) {
            ++dumped2;
            std::fprintf(stderr, "OURDST dst=%d:", job.dst_off);
            for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %d", static_cast<int>(frame[i]));
            std::fprintf(stderr, "\n");
          }
        }
      }
      const int nintra = static_cast<int>(wave.intra.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
      for (int i = 0; i < nintra; ++i)
        av1g_intra_block(&wave.intra[static_cast<size_t>(i)], frame, palette, maxv);
      const int nitx = static_cast<int>(wave.itx.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
      for (int i = 0; i < nitx; ++i) {
        const Av1gItxJob& job = wave.itx[static_cast<size_t>(i)];
        for (int y = 0; y < job.h; ++y)
          av1g_itx_row(&job, plan.coefficients.data(), tmp.data(), y);
        for (int x = 0; x < job.w; ++x)
          av1g_itx_col_add(&job, tmp.data(), frame, x, maxv);
      }
      const int nres = static_cast<int>(wave.residual.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
      for (int i = 0; i < nres; ++i) {
        const Av1gItxJob& job = wave.residual[static_cast<size_t>(i)];
        for (int y = 0; y < job.h; ++y)
          for (int x = 0; x < job.w; ++x)
            av1g_add_residual_px(&job, plan.residual.data(), frame, x, y, maxv);
      }
    }

    const double t_recon = now_ms();

    /* ---- deblock, CDEF, film grain --------------------------------------- */
    /* Vertical and horizontal edges cross each other, so the two groups are
     * sequential passes (the order the specification defines). Inside one group
     * the edges are disjoint, which makes the pass fully parallel. */
    /* AV1GPU_NO_DEBLOCK is a debug bitmask of planes to leave unfiltered. */
    const char* no_lf_env = std::getenv("AV1GPU_NO_DEBLOCK");
    const int no_lf = no_lf_env ? std::atoi(no_lf_env) : 0;
    for (const std::vector<Av1gDeblockJob>* pass : {&plan.deblock_v, &plan.deblock_h}) {
      const int n = static_cast<int>(pass->size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 8)
#endif
      for (int i = 0; i < n; ++i) {
        const Av1gDeblockJob& job = (*pass)[static_cast<size_t>(i)];
        if (no_lf & (job.is_chroma ? 2 : 1)) continue;
        av1g_deblock_job(&job, frame, maxv);
      }
    }

    /* AV1GPU_NO_CDEF is a debug bitmask of planes to leave unfiltered. */
    const char* no_cdef_env = std::getenv("AV1GPU_NO_CDEF");
    const int no_cdef = no_cdef_env ? std::atoi(no_cdef_env) : 0;
    if (!plan.cdef.empty()) {
      /* CDEF reads a copy of the frame so that neighbouring blocks cannot
       * observe each other's filtering, exactly like the GPU scratch buffer. */
      std::memcpy(scratch_.data(), frame, g.buffer_size * sizeof(uint16_t));
      const int n = static_cast<int>(plan.cdef.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 8)
#endif
      for (int i = 0; i < n; ++i) {
        const Av1gCdefJob& job = plan.cdef[static_cast<size_t>(i)];
        /* chroma is one bit: the CDEF jobs carry both planes together */
        if (no_cdef & (job.uv ? 2 : 1)) continue;
        av1g_cdef_block(&job, scratch_.data(), frame, scratch_.data(), maxv);
      }
    }
    {
      const int n = static_cast<int>(plan.grain.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 8)
#endif
      for (int i = 0; i < n; ++i)
        av1g_grain_block(&plan.grain[static_cast<size_t>(i)], frame,
                         plan.grain_template.data(), plan.grain_lut, maxv);
    }

    if (std::getenv("AV1GPU_MCDUMP")) {
      static int dumped3 = 0;
      if (dumped3 < 3) {
        ++dumped3;
        std::fprintf(stderr, "OURFINAL:");
        for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %d", static_cast<int>(frame[i]));
        std::fprintf(stderr, "\n");
      }
    }

    if (dump_coded) {
      if (FILE* df = std::fopen(dump_coded, "ab")) {
        for (int p = 0; p < 3; ++p) {
          const int w = g.plane_w[p];
          const int h = g.plane_h[p];
          for (int y = 0; y < h; ++y) {
            const uint16_t* row = frame + g.plane_off[p] + static_cast<size_t>(y) * g.stride[p];
            if (g.frame.bit_depth > 8) {
              std::fwrite(row, sizeof(uint16_t), static_cast<size_t>(w), df);
            } else {
              std::vector<unsigned char> bytes(static_cast<size_t>(w));
              for (int i = 0; i < w; ++i) bytes[static_cast<size_t>(i)] = static_cast<unsigned char>(row[i]);
              std::fwrite(bytes.data(), 1, static_cast<size_t>(w), df);
            }
          }
        }
        std::fclose(df);
      }
    }

    const double t_filter = now_ms();

    /* ---- read back -------------------------------------------------------- */
    if (out) {
      if (!out->matches(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                        g.frame.ss_y))
        out->allocate(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                      g.frame.ss_y);
      for (int p = 0; p < 3; ++p) {
        const int w = g.plane_w[p];
        const int h = g.plane_h[p];
        for (int y = 0; y < h; ++y)
          std::memcpy(out->plane(p) + static_cast<size_t>(y) * out->stride[p],
                      frame + g.plane_off[p] + static_cast<size_t>(y) * g.stride[p],
                      static_cast<size_t>(w) * sizeof(uint16_t));
      }
    }

    int slot = -1;
    if (keep_reference) {
      slot = allocate_slot();
      if (slot < 0) {
        *error = "CPU reference pool exhausted";
        return false;
      }
      RefFrame& ref = refs_[slot];
      ref.used = true;
      ref.width = g.frame.width;
      ref.height = g.frame.height;
      ref.bit_depth = g.frame.bit_depth;
      ref.ss_x = g.frame.ss_x;
      ref.ss_y = g.frame.ss_y;
      std::memcpy(ref_pool_.data() + static_cast<size_t>(slot) * g.buffer_size, frame,
                  g.buffer_size * sizeof(uint16_t));
    }
    if (out_slot) *out_slot = slot;
    last_slot_ = slot;

    if (stats) {
      stats->blocks = plan.blocks;
      stats->waves = static_cast<int>(plan.waves.size());
      stats->kernel_launches = 0;
      stats->reconstruct_ms = t_recon - t_start;
      stats->filter_ms = t_filter - t_recon;
      stats->readback_ms = now_ms() - t_filter;
      stats->total_ms = now_ms() - t_start;
    }
    return true;
  }

  bool to_rgb(int slot, ColorMatrix matrix, ColorRange range, std::vector<uint8_t>* rgb,
              std::string* error) override {
    const uint16_t* samples = nullptr;
    if (slot >= 0 && slot < kReferenceSlots && refs_[slot].used)
      samples = ref_pool_.data() + static_cast<size_t>(slot) * layout_.buffer_size;
    else if (last_slot_ >= 0 && refs_[last_slot_].used)
      samples = ref_pool_.data() + static_cast<size_t>(last_slot_) * layout_.buffer_size;
    else
      samples = frame_.data();
    if (!samples) {
      if (error) *error = "no frame to convert";
      return false;
    }

    Frame frame;
    frame.allocate(layout_.frame.width, layout_.frame.height, layout_.frame.bit_depth,
                   layout_.frame.ss_x, layout_.frame.ss_y);
    for (int p = 0; p < 3; ++p) {
      for (int y = 0; y < layout_.plane_h[p]; ++y)
        std::memcpy(frame.plane(p) + static_cast<size_t>(y) * frame.stride[p],
                    samples + layout_.plane_off[p] +
                        static_cast<size_t>(y) * layout_.stride[p],
                    static_cast<size_t>(layout_.plane_w[p]) * sizeof(uint16_t));
    }
    tables::frame_to_rgb(frame, matrix, range, rgb);
    if (error) error->clear();
    return true;
  }

  void reset() override {
    for (int i = 0; i < kReferenceSlots; ++i) refs_[i].used = false;
    std::fill(frame_.begin(), frame_.end(), 0);
    std::fill(ref_pool_.begin(), ref_pool_.end(), 0);
  }

 private:
  int allocate_slot() {
    for (int i = 0; i < kReferenceSlots; ++i)
      if (!refs_[i].used) return i;
    return -1;
  }

  BackendOptions options_;
  DeviceInfo info_;
  PlanLayout layout_;
  std::vector<uint16_t> frame_;
  std::vector<uint16_t> scratch_;
  /* kReferenceSlots pictures of buffer_size samples, one after another */
  std::vector<uint16_t> ref_pool_;
  RefFrame refs_[kReferenceSlots];
  int last_slot_ = -1;
};

} /* namespace */

std::unique_ptr<Backend> create_cpu_backend(const BackendOptions& options, std::string* error) {
  if (error) error->clear();
  return std::unique_ptr<Backend>(new CpuBackend(options));
}

} /* namespace internal */
} /* namespace av1gpu */
