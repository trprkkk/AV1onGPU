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

constexpr int kReferenceSlots = 6;

struct RefFrame {
  bool used = false;
  int width = 0, height = 0, bit_depth = 8, ss_x = 1, ss_y = 1;
  std::vector<uint16_t> samples;
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
    ref.samples.assign(g.buffer_size, 0);
    for (int p = 0; p < 3; ++p) {
      const int w = g.plane_w[p];
      const int h = g.plane_h[p];
      for (int y = 0; y < h; ++y)
        std::memcpy(ref.samples.data() + g.plane_off[p] + static_cast<size_t>(y) * g.stride[p],
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

    /* ---- prediction, transform, residual --------------------------------- */
    std::vector<int> tmp(plan.coefficients.size() + 64, 0);
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
        const Av1gInterJob& job = wave.inter[static_cast<size_t>(i)];
        const uint16_t* r0 = job.ref0_slot >= 0 ? refs_[job.ref0_slot].samples.data() : frame;
        const uint16_t* r1 = job.ref1_slot >= 0 ? refs_[job.ref1_slot].samples.data() : r0;
        av1g_inter_block(&job, frame, r0, r1, filters, maxv);
      }
      const int nintra = static_cast<int>(wave.intra.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
      for (int i = 0; i < nintra; ++i)
        av1g_intra_block(&wave.intra[static_cast<size_t>(i)], frame, maxv);
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
    for (const std::vector<Av1gDeblockJob>* pass : {&plan.deblock_v, &plan.deblock_h}) {
      const int n = static_cast<int>(pass->size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 8)
#endif
      for (int i = 0; i < n; ++i)
        av1g_deblock_job(&(*pass)[static_cast<size_t>(i)], frame, maxv);
    }

    if (!plan.cdef.empty()) {
      /* CDEF reads a copy of the frame so that neighbouring blocks cannot
       * observe each other's filtering, exactly like the GPU scratch buffer. */
      std::memcpy(scratch_.data(), frame, g.buffer_size * sizeof(uint16_t));
      const int n = static_cast<int>(plan.cdef.size());
#if defined(AV1GPU_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 8)
#endif
      for (int i = 0; i < n; ++i)
        av1g_cdef_block(&plan.cdef[static_cast<size_t>(i)], scratch_.data(), frame,
                        scratch_.data(), maxv);
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
      ref.samples.assign(frame, frame + g.buffer_size);
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
      samples = refs_[slot].samples.data();
    else if (last_slot_ >= 0 && refs_[last_slot_].used)
      samples = refs_[last_slot_].samples.data();
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
