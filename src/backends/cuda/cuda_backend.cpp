/*
 * AV1onGPU - CUDA backend
 *
 * Same job arrays as the OpenCL backend, compiled with NVRTC at run time and
 * executed through the CUDA driver API. This is the path that lets GPUs without
 * any AV1 fixed function hardware decode in software at shader speed.
 *
 * The whole frame (three planes back to back) lives in one device buffer, the
 * reference frames live in a second pool, so a decoded frame becomes a reference
 * with a device to device copy and never touches host memory unless the caller
 * actually wants the pixels.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "backends/cuda/cuda_min.hpp"
#include "core/backend.hpp"
#include "core/kernel_source.hpp"

namespace av1gpu {
namespace internal {

using namespace ::av1gpu::cuda;

namespace {

double now_ms() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point start = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

inline size_t round_up(size_t v, size_t m) { return ((v + m - 1) / m) * m; }

using cuda::CUdeviceptr;

/* argument pack for cuLaunchKernel: it wants an array of pointers to the
 * argument values, which is easy to get wrong by hand. */
class Launcher {
 public:
  void add_int(int value) {
    ints_[count_] = value;
    slots_[count_] = &ints_[count_];
    ++count_;
  }
  void add_ptr(CUdeviceptr value) {
    ptrs_[count_] = value;
    slots_[count_] = &ptrs_[count_];
    ++count_;
  }
  void** values() { return slots_; }
  int count() const { return count_; }

 private:
  int count_ = 0;
  void* slots_[16] = {nullptr};
  int ints_[16] = {0};
  CUdeviceptr ptrs_[16] = {0};
};

class CudaBackend final : public Backend {
 public:
  explicit CudaBackend(const BackendOptions& options) : options_(options) {}
  ~CudaBackend() override { shutdown(); }

  const DeviceInfo& info() const override { return info_; }
  const BackendOptions& options() const override { return options_; }

  bool init(std::string* error) {
    if (!cuda::load_driver(&driver_, &driver_lib_, error)) return false;
    if (!cuda::load_nvrtc(&nvrtc_, &nvrtc_lib_, error)) return false;

    if (driver_.Init(0) != cuda::CUDA_SUCCESS) {
      *error = "cuInit failed (no CUDA driver or no device)";
      return false;
    }
    int count = 0;
    driver_.DeviceGetCount(&count);
    if (count <= 0) {
      *error = "no CUDA device";
      return false;
    }
    if (options_.device_index >= count) {
      *error = "CUDA device index out of range";
      return false;
    }
    if (driver_.DeviceGet(&device_, options_.device_index) != cuda::CUDA_SUCCESS) {
      *error = "cuDeviceGet failed";
      return false;
    }
    char name[256] = {0};
    driver_.DeviceGetName(name, sizeof(name), device_);
    size_t mem = 0;
    driver_.DeviceTotalMem(&mem, device_);
    int sms = 0, major = 0, minor = 0;
    driver_.DeviceGetAttribute(&sms, cuda::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_);
    driver_.DeviceGetAttribute(&major, cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                               device_);
    driver_.DeviceGetAttribute(&minor, cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                               device_);
    compute_major_ = major;
    compute_minor_ = minor;

    info_.kind = BackendKind::CUDA;
    info_.name = name;
    info_.vendor = "NVIDIA";
    info_.version = "compute capability " + std::to_string(major) + "." +
                    std::to_string(minor);
    info_.detail = std::to_string(sms) + " multiprocessors, kernels built with NVRTC";
    info_.global_mem = mem;
    info_.compute_units = sms;
    info_.available = true;

    CUresult rc = driver_.CtxCreate(&context_, 0, device_);
    if (rc != cuda::CUDA_SUCCESS) {
      *error = "cuCtxCreate failed";
      return false;
    }
    return build_module(error);
  }

  PlanLayout prepare(const FrameGeometry& geometry) override {
    const PlanLayout g = make_plan_layout(geometry, 64);
    const size_t size = g.buffer_size;
    layout_ = g;
    ensure(&frame_, &frame_cap_, size * sizeof(uint16_t));
    ensure(&scratch_, &scratch_cap_, size * sizeof(uint16_t));
    ensure(&ref_pool_, &ref_cap_, size * kReferenceSlots * sizeof(uint16_t));
    if (!tables_uploaded_) {
      upload_tables();
      tables_uploaded_ = true;
    }
    return g;
  }

  int upload_reference(const Frame& frame, std::string* error) override {
    if (layout_.buffer_size == 0) {
      if (error) *error = "upload_reference before prepare";
      return -1;
    }
    const int slot = allocate_slot();
    if (slot < 0) {
      if (error) *error = "CUDA reference pool exhausted";
      return -1;
    }
    const size_t size = layout_.buffer_size;
    std::vector<uint16_t> packed(size, 0);
    for (int p = 0; p < 3; ++p) {
      const int w = layout_.plane_w[p];
      const int h = layout_.plane_h[p];
      for (int y = 0; y < h; ++y)
        std::memcpy(packed.data() + layout_.plane_off[p] +
                        static_cast<size_t>(y) * layout_.stride[p],
                    frame.plane(p) + static_cast<size_t>(y) * frame.stride[p],
                    static_cast<size_t>(w) * sizeof(uint16_t));
    }
    const CUdeviceptr target =
        ref_pool_ + static_cast<size_t>(slot) * size * sizeof(uint16_t);
    if (driver_.MemcpyHtoD(target, packed.data(), size * sizeof(uint16_t)) !=
        cuda::CUDA_SUCCESS) {
      if (error) *error = "cuMemcpyHtoD failed";
      return -1;
    }
    slots_[slot] = true;
    return slot;
  }

  void release_reference(int slot) override {
    if (slot >= 0 && slot < kReferenceSlots) slots_[slot] = false;
  }

  int reference_capacity() const override { return kReferenceSlots; }

  bool decode(const FramePlan& plan, bool keep_reference, Frame* out, int* out_slot,
              DecodeStats* stats, std::string* error) override {
    error->clear();
    const double t0 = now_ms();
    const PlanLayout& g = plan.layout;
    const int maxv = plan.maxv;
    int launches = 0;

    ensure(&coef_, &coef_cap_, (plan.coefficients.size() + 1) * sizeof(int));
    if (!plan.coefficients.empty())
      driver_.MemcpyHtoD(coef_, plan.coefficients.data(),
                         plan.coefficients.size() * sizeof(int));
    ensure(&tmp_, &tmp_cap_, (plan.tmp_size + 1) * sizeof(int));
    if (!plan.residual.empty()) {
      ensure(&residual_, &residual_cap_, (plan.residual.size() + 1) * sizeof(int));
      driver_.MemcpyHtoD(residual_, plan.residual.data(),
                         plan.residual.size() * sizeof(int));
    }

    for (const WaveJobs& wave : plan.waves) {
      if (!wave.inter.empty()) {
        std::vector<Av1gInterJob> jobs = wave.inter;
        for (Av1gInterJob& job : jobs) {
          if (job.ref0_slot >= 0)
            job.ref0_off += job.ref0_slot * static_cast<int>(g.buffer_size);
          if (job.ref1_slot >= 0)
            job.ref1_off += job.ref1_slot * static_cast<int>(g.buffer_size);
          /* a difference weighted chroma block builds its mask on the luma
           * plane, so those offsets need the same slot rebase */
          if (job.mask_kind == AV1G_MASK_DIFFWTD) {
            if (job.ref0_slot >= 0)
              job.luma_ref0_off += job.ref0_slot * static_cast<int>(g.buffer_size);
            if (job.ref1_slot >= 0)
              job.luma_ref1_off += job.ref1_slot * static_cast<int>(g.buffer_size);
          }
          for (int k = 0; k < job.obmc_count; ++k)
            if (job.obmc[k].ref_slot >= 0)
              job.obmc[k].ref_off += job.obmc[k].ref_slot * static_cast<int>(g.buffer_size);
        }
        int total = 0;
        CUdeviceptr job_mem = 0, pref_mem = 0;
        if (!upload_jobs(jobs, &job_mem, &pref_mem, &total,
                         [](const Av1gInterJob& j) { return j.w * j.h; }))
          return fail(error, "inter job upload failed");
        Launcher a;
        add_mem(a, job_mem);
        add_mem(a, pref_mem);
        add_int(a, static_cast<int>(jobs.size()));
        add_int(a, total);
        add_mem(a, frame_);
        add_mem(a, ref_pool_);
        add_mem(a, frame_);
        add_mem(a, filters_);
        add_int(a, maxv);
        if (!launch("av1g_k_inter", total, a)) return fail(error, "av1g_k_inter failed");
        ++launches;
      }
      if (!wave.intra.empty()) {
        const int n = static_cast<int>(wave.intra.size());
        if (!ensure(&palette_, &palette_cap_, plan.palette_map.size()))
          return fail(error, "palette map upload failed");
        if (!plan.palette_map.empty() &&
            driver_.MemcpyHtoD(palette_, plan.palette_map.data(),
                               plan.palette_map.size()) != cuda::CUDA_SUCCESS)
          return fail(error, "palette map write failed");
        int total = 0;
        CUdeviceptr job_mem = 0, unused = 0;
        if (!upload_jobs(wave.intra, &job_mem, &unused, &total,
                         [](const Av1gIntraJob&) { return 1; }))
          return fail(error, "intra job upload failed");
        (void)total;
        {
          Launcher a;
          add_mem(a, job_mem);
          add_int(a, n);
          add_mem(a, frame_);
          add_mem(a, palette_);
          add_int(a, maxv);
          if (!launch("av1g_k_intra", n, a)) return fail(error, "av1g_k_intra failed");
          ++launches;
        }
      }
      if (!wave.itx.empty()) {
        int rows = 0, cols = 0;
        CUdeviceptr job_mem = 0, row_pref = 0, col_pref = 0;
        if (!upload_jobs(wave.itx, &job_mem, &row_pref, &rows,
                         [](const Av1gItxJob& j) { return j.h; }))
          return fail(error, "itx job upload failed");
        if (!upload_prefix(wave.itx, &col_pref, &cols,
                           [](const Av1gItxJob& j) { return j.w; }))
          return fail(error, "itx prefix upload failed");
        {
          Launcher a;
          add_mem(a, job_mem);
          add_mem(a, row_pref);
          add_int(a, static_cast<int>(wave.itx.size()));
          add_int(a, rows);
          add_mem(a, coef_);
          add_mem(a, tmp_);
          if (!launch("av1g_k_itx_rows", rows, a))
            return fail(error, "av1g_k_itx_rows failed");
          ++launches;
        }
        {
          Launcher a;
          add_mem(a, job_mem);
          add_mem(a, col_pref);
          add_int(a, static_cast<int>(wave.itx.size()));
          add_int(a, cols);
          add_mem(a, tmp_);
          add_mem(a, frame_);
          add_int(a, maxv);
          if (!launch("av1g_k_itx_cols", cols, a))
            return fail(error, "av1g_k_itx_cols failed");
          ++launches;
        }
      }
      if (!wave.residual.empty()) {
        int total = 0;
        CUdeviceptr job_mem = 0, pref_mem = 0;
        if (!upload_jobs(wave.residual, &job_mem, &pref_mem, &total,
                         [](const Av1gItxJob& j) { return j.w * j.h; }))
          return fail(error, "residual job upload failed");
        Launcher a;
        add_mem(a, job_mem);
        add_mem(a, pref_mem);
        add_int(a, static_cast<int>(wave.residual.size()));
        add_int(a, total);
        add_mem(a, residual_);
        add_mem(a, frame_);
        add_int(a, maxv);
        if (!launch("av1g_k_add_residual", total, a))
          return fail(error, "av1g_k_add_residual failed");
        ++launches;
      }
    }

    const double t1 = now_ms();

    /* vertical edges first, then horizontal ones: the two groups cross */
    for (const std::vector<Av1gDeblockJob>* pass : {&plan.deblock_v, &plan.deblock_h}) {
      if (pass->empty()) continue;
      int total = 0;
      CUdeviceptr job_mem = 0, pref_mem = 0;
      if (!upload_jobs(*pass, &job_mem, &pref_mem, &total,
                       [](const Av1gDeblockJob& j) { return (j.len + 3) / 4; }))
        return fail(error, "deblock job upload failed");
      Launcher a;
      add_mem(a, job_mem);
      add_mem(a, pref_mem);
      add_int(a, static_cast<int>(pass->size()));
      add_int(a, total);
      add_mem(a, frame_);
      add_int(a, maxv);
      if (!launch("av1g_k_deblock", total, a)) return fail(error, "av1g_k_deblock failed");
      ++launches;
    }

    if (!plan.cdef.empty()) {
      ensure(&scratch_, &scratch_cap_, g.buffer_size * sizeof(uint16_t));
      driver_.MemcpyDtoD(scratch_, frame_, g.buffer_size * sizeof(uint16_t));
      int total = 0;
      CUdeviceptr job_mem = 0, pref_mem = 0;
      if (!upload_jobs(plan.cdef, &job_mem, &pref_mem, &total,
                       [](const Av1gCdefJob&) { return 1; }))
        return fail(error, "CDEF job upload failed");
      Launcher a;
      add_mem(a, job_mem);
      add_mem(a, pref_mem);
      add_int(a, static_cast<int>(plan.cdef.size()));
      add_int(a, total);
      add_mem(a, scratch_);
      add_mem(a, frame_);
      add_int(a, maxv);
      if (!launch("av1g_k_cdef", total, a)) return fail(error, "av1g_k_cdef failed");
      ++launches;
    }

    if (!plan.grain.empty()) {
      ensure(&grain_, &grain_cap_, plan.grain_template.size());
      driver_.MemcpyHtoD(grain_, plan.grain_template.data(), plan.grain_template.size());
      ensure(&lut_, &lut_cap_, sizeof(plan.grain_lut));
      driver_.MemcpyHtoD(lut_, plan.grain_lut, sizeof(plan.grain_lut));
      int total = 0;
      CUdeviceptr job_mem = 0, pref_mem = 0;
      if (!upload_jobs(plan.grain, &job_mem, &pref_mem, &total,
                       [](const Av1gGrainJob& j) { return j.w * j.h; }))
        return fail(error, "grain job upload failed");
      Launcher a;
      add_mem(a, job_mem);
      add_mem(a, pref_mem);
      add_int(a, static_cast<int>(plan.grain.size()));
      add_int(a, total);
      add_mem(a, frame_);
      add_mem(a, grain_);
      add_mem(a, lut_);
      add_int(a, maxv);
      if (!launch("av1g_k_grain", total, a)) return fail(error, "av1g_k_grain failed");
      ++launches;
    }

    driver_.CtxSynchronize();
    const double t2 = now_ms();

    if (out) {
      if (!out->matches(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                        g.frame.ss_y))
        out->allocate(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                      g.frame.ss_y);
      if (staging_.size() < g.buffer_size) staging_.resize(g.buffer_size);
      driver_.MemcpyDtoH(staging_.data(), frame_, g.buffer_size * sizeof(uint16_t));
      for (int p = 0; p < 3; ++p) {
        const int w = g.plane_w[p];
        const int h = g.plane_h[p];
        for (int y = 0; y < h; ++y)
          std::memcpy(out->plane(p) + static_cast<size_t>(y) * out->stride[p],
                      staging_.data() + g.plane_off[p] +
                          static_cast<size_t>(y) * g.stride[p],
                      static_cast<size_t>(w) * sizeof(uint16_t));
      }
    }
    const double t3 = now_ms();

    int slot = -1;
    if (keep_reference) {
      slot = allocate_slot();
      if (slot < 0) return fail(error, "CUDA reference pool exhausted");
      cuda::CUresult rc = driver_.MemcpyDtoD(
          ref_pool_ + static_cast<size_t>(slot) * g.buffer_size * sizeof(uint16_t), frame_,
          g.buffer_size * sizeof(uint16_t));
      if (rc != cuda::CUDA_SUCCESS) return fail(error, "reference copy failed");
      driver_.CtxSynchronize();
      slots_[slot] = true;
    }
    if (out_slot) *out_slot = slot;
    last_slot_ = slot;

    if (stats) {
      stats->blocks = plan.blocks;
      stats->waves = static_cast<int>(plan.waves.size());
      stats->kernel_launches = launches;
      stats->reconstruct_ms = t1 - t0;
      stats->filter_ms = t2 - t1;
      stats->readback_ms = t3 - t2;
      stats->total_ms = now_ms() - t0;
    }
    return true;
  }

  bool to_rgb(int slot, ColorMatrix matrix, ColorRange range, std::vector<uint8_t>* rgb,
              std::string* error) override {
    error->clear();
    if (layout_.buffer_size == 0) {
      *error = "prepare() has not been called";
      return false;
    }
    const PlanLayout& g = layout_;
    CUdeviceptr source = frame_;
    int base = 0;
    if (slot >= 0) {
      if (slot >= kReferenceSlots || !slots_[slot]) {
        *error = "invalid reference slot";
        return false;
      }
      source = ref_pool_;
      base = slot * static_cast<int>(g.buffer_size);
    } else if (last_slot_ >= 0 && slots_[last_slot_]) {
      source = ref_pool_;
      base = last_slot_ * static_cast<int>(g.buffer_size);
    }

    Av1gColorJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 0;
    job.dst_stride = g.frame.width;
    job.w = g.frame.width;
    job.h = g.frame.height;
    job.y_off = base + g.plane_off[0];
    job.u_off = base + g.plane_off[1];
    job.v_off = base + g.plane_off[2];
    job.y_stride = g.stride[0];
    job.c_stride = g.stride[1];
    job.ss_x = g.frame.ss_x;
    job.ss_y = g.frame.ss_y;
    job.bitdepth = g.frame.bit_depth;
    tables::color_matrix(matrix, range, g.frame.bit_depth, job.m, job.o);

    const size_t rgb_bytes = static_cast<size_t>(job.w) * job.h * 3;
    if (!ensure(&rgb_, &rgb_cap_, rgb_bytes)) {
      *error = "out of device memory for RGB output";
      return false;
    }
    std::vector<Av1gColorJob> jobs(1, job);
    int total = 0;
    CUdeviceptr job_mem = 0, pref_mem = 0;
    if (!upload_jobs(jobs, &job_mem, &pref_mem, &total,
                     [](const Av1gColorJob& j) { return j.w * j.h; })) {
      *error = "RGB job upload failed";
      return false;
    }
    Launcher a;
    add_mem(a, job_mem);
    add_mem(a, pref_mem);
    add_int(a, 1);
    add_int(a, total);
    add_mem(a, source);
    add_mem(a, source);
    add_mem(a, source);
    add_mem(a, rgb_);
    if (!launch("av1g_k_color", total, a)) {
      *error = "av1g_k_color failed";
      return false;
    }
    driver_.CtxSynchronize();
    rgb->resize(rgb_bytes);
    driver_.MemcpyDtoH(rgb->data(), rgb_, rgb_bytes);
    return true;
  }

  void reset() override {
    for (int i = 0; i < kReferenceSlots; ++i) slots_[i] = false;
    last_slot_ = -1;
  }

 private:
  /* one pool slot per AV1 reference slot plus the picture being decoded; see
   * the CPU backend */
  static constexpr int kReferenceSlots = 9;

  void shutdown() {
    if (module_ && driver_.ModuleLoadDataEx) {
      /* no cuModuleUnload in the minimal set: the context teardown frees it */
    }
    CUdeviceptr buffers[] = {frame_,    scratch_, coef_,     tmp_,   residual_,
                             filters_,  ref_pool_, grain_,
                             lut_,      rgb_,     job_,      pref_};
    for (CUdeviceptr b : buffers)
      if (b && driver_.MemFree) driver_.MemFree(b);
    frame_ = scratch_ = coef_ = tmp_ = residual_ = 0;
    filters_ = ref_pool_ = grain_ = 0;
    lut_ = rgb_ = job_ = pref_ = pref_b_ = 0;
    if (context_ && driver_.CtxDestroy) driver_.CtxDestroy(context_);
    context_ = nullptr;
    if (driver_lib_) {
      cuda::close_library(driver_lib_);
      driver_lib_ = nullptr;
    }
    if (nvrtc_lib_) {
      cuda::close_library(nvrtc_lib_);
      nvrtc_lib_ = nullptr;
    }
  }

  bool build_module(std::string* error) {
    const std::string& source = kernel_source::cuda();
    void* program = nullptr;
    if (nvrtc_.CreateProgram(&program, source.c_str(), "av1gpu.cu", 0, nullptr, nullptr) !=
        0) {
      *error = "nvrtcCreateProgram failed";
      return false;
    }
    /* NVRTC spells the minor version as two digits for 10.x and later, so
     * build the string with enough room for "compute_1200" */
    char arch[48];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", compute_major_,
                  compute_minor_ >= 5 ? compute_minor_ : 0);
    const char* options[] = {arch, "-default-device"};
    int rc = nvrtc_.CompileProgram(program, 2, options);
    size_t log_size = 0;
    nvrtc_.GetProgramLogSize(program, &log_size);
    std::string log(log_size + 1, '\0');
    if (log_size) nvrtc_.GetProgramLog(program, &log[0]);
    if (rc != 0) {
      *error = "NVRTC build failed:\n" + log;
      nvrtc_.DestroyProgram(&program);
      return false;
    }
    if (options_.verbose) error->append(log);
    size_t ptx_size = 0;
    nvrtc_.GetPTXSize(program, &ptx_size);
    std::vector<char> ptx(ptx_size + 1, 0);
    nvrtc_.GetPTX(program, ptx.data());
    nvrtc_.DestroyProgram(&program);

    if (driver_.ModuleLoadDataEx(&module_, ptx.data(), 0, nullptr, nullptr) !=
        cuda::CUDA_SUCCESS) {
      *error = "cuModuleLoadDataEx failed";
      return false;
    }
    return true;
  }

  bool ensure(CUdeviceptr* buffer, size_t* capacity, size_t bytes) {
    if (*buffer && *capacity >= bytes) return true;
    if (*buffer) driver_.MemFree(*buffer);
    *buffer = 0;
    if (driver_.MemAlloc(buffer, bytes) != cuda::CUDA_SUCCESS) {
      *capacity = 0;
      return false;
    }
    *capacity = bytes;
    return true;
  }

  void upload_tables() {
    const std::vector<int>& filters = tables::filters();
    if (ensure(&filters_, &filters_cap_, filters.size() * sizeof(int)))
      driver_.MemcpyHtoD(filters_, filters.data(), filters.size() * sizeof(int));
  }

  /* Uploads only the prefix-sum table. The transform passes need two of them:
   * the row pass has one thread per row (job.h), the column pass one per column
   * (job.w), and both read the same job array. */
  template <typename JobT, typename Fn>
  bool upload_prefix(const std::vector<JobT>& jobs, CUdeviceptr* pref_mem,
                     int* total, Fn element_count) {
    if (!ensure(&pref_b_, &pref_b_cap_, (jobs.size() + 2) * sizeof(int))) return false;
    *pref_mem = pref_b_;
    std::vector<int> prefix(jobs.size() + 2, 0);
    int sum = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
      prefix[i] = sum;
      sum += element_count(jobs[i]);
    }
    prefix[jobs.size()] = sum;
    prefix[jobs.size() + 1] = sum;
    *total = sum;
    if (sum == 0) return true;
    return driver_.MemcpyHtoD(pref_b_, prefix.data(), prefix.size() * sizeof(int)) ==
           cuda::CUDA_SUCCESS;
  }

  template <typename JobT, typename Fn>
  bool upload_jobs(const std::vector<JobT>& jobs, CUdeviceptr* job_mem,
                   CUdeviceptr* pref_mem, int* total, Fn element_count) {
    if (!ensure(&job_, &job_cap_, jobs.size() * sizeof(JobT))) return false;
    if (!ensure(&pref_, &pref_cap_, (jobs.size() + 2) * sizeof(int))) return false;
    *job_mem = job_;
    *pref_mem = pref_;
    std::vector<int> prefix(jobs.size() + 2, 0);
    int sum = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
      prefix[i] = sum;
      sum += element_count(jobs[i]);
    }
    prefix[jobs.size()] = sum;
    prefix[jobs.size() + 1] = sum;
    *total = sum;
    if (sum == 0) return true;
    if (driver_.MemcpyHtoD(job_, jobs.data(), jobs.size() * sizeof(JobT)) !=
        cuda::CUDA_SUCCESS)
      return false;
    return driver_.MemcpyHtoD(pref_, prefix.data(), prefix.size() * sizeof(int)) ==
           cuda::CUDA_SUCCESS;
  }

  void add_mem(Launcher& a, CUdeviceptr value) { a.add_ptr(value); }
  void add_int(Launcher& a, int value) { a.add_int(value); }

  bool launch(const char* name, int elements, Launcher& launcher) {
    if (elements <= 0) return true;
    if (launcher.count() > 16) return false;
    auto it = functions_.find(name);
    CUfunction function = nullptr;
    if (it != functions_.end()) {
      function = it->second;
    } else {
      if (driver_.ModuleGetFunction(&function, module_, name) != cuda::CUDA_SUCCESS)
        return false;
      functions_[name] = function;
    }
    const unsigned block = 64;
    const unsigned grid = static_cast<unsigned>((elements + block - 1) / block);
    return driver_.LaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr,
                                launcher.values(), nullptr) == cuda::CUDA_SUCCESS;
  }

  static bool fail(std::string* error, const char* message) {
    *error = message;
    return false;
  }

  int allocate_slot() {
    for (int i = 0; i < kReferenceSlots; ++i)
      if (!slots_[i]) return i;
    return -1;
  }

  BackendOptions options_;
  DeviceInfo info_;
  cuda::Driver driver_{};
  cuda::Nvrtc nvrtc_{};
  void* driver_lib_ = nullptr;
  void* nvrtc_lib_ = nullptr;
  cuda::CUdevice device_ = 0;
  cuda::CUcontext context_ = nullptr;
  cuda::CUmodule module_ = nullptr;
  int compute_major_ = 0, compute_minor_ = 0;
  std::map<std::string, cuda::CUfunction> functions_;

  PlanLayout layout_;
  bool tables_uploaded_ = false;
  CUdeviceptr frame_ = 0, scratch_ = 0, coef_ = 0, tmp_ = 0, residual_ = 0;
  CUdeviceptr filters_ = 0, ref_pool_ = 0, grain_ = 0, palette_ = 0;
  CUdeviceptr lut_ = 0, rgb_ = 0, job_ = 0, pref_ = 0, pref_b_ = 0;
  size_t frame_cap_ = 0, scratch_cap_ = 0, coef_cap_ = 0, tmp_cap_ = 0;
  size_t residual_cap_ = 0, filters_cap_ = 0, ref_cap_ = 0;
  size_t grain_cap_ = 0, lut_cap_ = 0, rgb_cap_ = 0, palette_cap_ = 0;
  size_t job_cap_ = 0, pref_cap_ = 0, pref_b_cap_ = 0;
  std::vector<uint16_t> staging_;
  bool slots_[kReferenceSlots] = {};
  int last_slot_ = -1;
};

} /* namespace */

} /* namespace internal */

/* ------------------------------------------------------------------------- */

namespace cuda {

namespace {

void* open_first(const char* const* names, int count) {
  for (int i = 0; i < count; ++i) {
#if defined(_WIN32)
    void* handle = LoadLibraryA(names[i]);
#else
    void* handle = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
#endif
    if (handle) return handle;
  }
  return nullptr;
}

} /* namespace */

bool load_driver(Driver* driver, void** library, std::string* error) {
  const char* names[] = {
#if defined(_WIN32)
      "nvcuda.dll",
#else
      "libcuda.so.1", "libcuda.so",
#endif
  };
  *library = open_first(names, 2);
  if (!*library) {
    *error = "no CUDA driver library (libcuda.so.1)";
    return false;
  }
  std::string missing;
#define AV1GPU_CU(field, name)                                                 \
  do {                                                                         \
    *(void**)(&driver->field) = dlsym(*library, name);                         \
    if (!driver->field) missing += std::string(missing.empty() ? "" : ", ") + name; \
  } while (0)
  AV1GPU_CU(Init, "cuInit");
  AV1GPU_CU(DeviceGetCount, "cuDeviceGetCount");
  AV1GPU_CU(DeviceGet, "cuDeviceGet");
  AV1GPU_CU(DeviceGetName, "cuDeviceGetName");
  AV1GPU_CU(DeviceGetAttribute, "cuDeviceGetAttribute");
  AV1GPU_CU(DeviceTotalMem, "cuDeviceTotalMem_v2");
  AV1GPU_CU(CtxCreate, "cuCtxCreate_v2");
  AV1GPU_CU(CtxDestroy, "cuCtxDestroy_v2");
  AV1GPU_CU(CtxSynchronize, "cuCtxSynchronize");
  AV1GPU_CU(MemAlloc, "cuMemAlloc_v2");
  AV1GPU_CU(MemFree, "cuMemFree_v2");
  AV1GPU_CU(MemcpyHtoD, "cuMemcpyHtoD_v2");
  AV1GPU_CU(MemcpyDtoH, "cuMemcpyDtoH_v2");
  AV1GPU_CU(MemcpyDtoD, "cuMemcpyDtoD_v2");
  AV1GPU_CU(ModuleLoadDataEx, "cuModuleLoadDataEx");
  AV1GPU_CU(ModuleGetFunction, "cuModuleGetFunction");
  AV1GPU_CU(LaunchKernel, "cuLaunchKernel");
#undef AV1GPU_CU
  if (!missing.empty()) {
    *error = "CUDA driver is missing entry points: " + missing;
    return false;
  }
  return true;
}

bool load_nvrtc(Nvrtc* nvrtc, void** library, std::string* error) {
  const char* names[] = {
#if defined(_WIN32)
      "nvrtc64_120_0.dll", "nvrtc64_110_0.dll", "nvrtc64_102_0.dll",
#else
      "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so.11.1", "libnvrtc.so.11.0",
      "libnvrtc.so.10.2", "libnvrtc.so",
#endif
  };
  *library = open_first(names, 6);
  if (!*library) {
    *error = "no NVRTC library (libnvrtc.so); install the CUDA toolkit or the "
             "cuda-nvrtc package";
    return false;
  }
  std::string missing;
#define AV1GPU_NV(field, name)                                                 \
  do {                                                                         \
    *(void**)(&nvrtc->field) = dlsym(*library, name);                          \
    if (!nvrtc->field) missing += std::string(missing.empty() ? "" : ", ") + name; \
  } while (0)
  AV1GPU_NV(CreateProgram, "nvrtcCreateProgram");
  AV1GPU_NV(CompileProgram, "nvrtcCompileProgram");
  AV1GPU_NV(GetProgramLogSize, "nvrtcGetProgramLogSize");
  AV1GPU_NV(GetProgramLog, "nvrtcGetProgramLog");
  AV1GPU_NV(GetPTXSize, "nvrtcGetPTXSize");
  AV1GPU_NV(GetPTX, "nvrtcGetPTX");
  AV1GPU_NV(DestroyProgram, "nvrtcDestroyProgram");
#undef AV1GPU_NV
  if (!missing.empty()) {
    *error = "NVRTC is missing entry points: " + missing;
    return false;
  }
  return true;
}

void close_library(void* library) {
  if (!library) return;
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(library));
#else
  dlclose(library);
#endif
}

} /* namespace cuda */

namespace internal {

std::unique_ptr<Backend> create_cuda_backend(const BackendOptions& options,
                                             std::string* error) {
  std::string local;
  if (!error) error = &local;
  std::unique_ptr<CudaBackend> backend(new CudaBackend(options));
  if (!backend->init(error)) return nullptr;
  return std::unique_ptr<Backend>(backend.release());
}

std::vector<DeviceInfo> list_cuda_devices() {
  std::vector<DeviceInfo> out;
  cuda::Driver driver{};
  void* library = nullptr;
  std::string error;
  if (!cuda::load_driver(&driver, &library, &error)) return out;
  if (driver.Init(0) != cuda::CUDA_SUCCESS) {
    cuda::close_library(library);
    return out;
  }
  int count = 0;
  driver.DeviceGetCount(&count);
  for (int i = 0; i < count; ++i) {
    cuda::CUdevice device;
    if (driver.DeviceGet(&device, i) != cuda::CUDA_SUCCESS) continue;
    DeviceInfo info;
    info.kind = BackendKind::CUDA;
    char name[256] = {0};
    driver.DeviceGetName(name, sizeof(name), device);
    size_t mem = 0;
    driver.DeviceTotalMem(&mem, device);
    int sms = 0, major = 0, minor = 0;
    driver.DeviceGetAttribute(&sms, cuda::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    driver.DeviceGetAttribute(&major, cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                              device);
    driver.DeviceGetAttribute(&minor, cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                              device);
    info.name = name;
    info.vendor = "NVIDIA";
    info.version = "compute capability " + std::to_string(major) + "." +
                   std::to_string(minor);
    info.global_mem = mem;
    info.compute_units = sms;
    info.available = true;
    out.push_back(info);
  }
  cuda::close_library(library);
  return out;
}

} /* namespace internal */
} /* namespace av1gpu */
