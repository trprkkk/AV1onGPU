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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "backends/cuda/cuda_min.hpp"
#include "core/backend.hpp"
#include "core/kernel_source.hpp"
#include "frontend/coef_tables.hpp"
#include "frontend/thread_pool.hpp"

namespace av1gpu {
namespace internal {

/* Upload volume categories, for the per-50-frame log line: where the frame's
 * host-to-device bytes actually went. */
enum H2dTag {
  kH2dTagCoef = 0,    /* the dequantised raster pool */
  kH2dTagDequant = 1, /* the levels pool and the QM matrices (device dequant) */
  kH2dTagWave = 2,    /* the per-wave job arenas */
  kH2dTagBatch = 3,   /* the batched job arrays */
  kH2dTagMisc = 4,    /* palette, grain, tables */
  kH2dTagInter = 5,   /* the batched inter jobs and their prefix */
  kH2dTagItx = 6,     /* the batched itx jobs and their row prefix */
  kH2dTagRes = 7,     /* the batched residual jobs and their prefix */
  kH2dTagCount = 8
};

using namespace ::av1gpu::cuda;

namespace {

double now_ms() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point start = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

inline size_t round_up(size_t v, size_t m) { return ((v + m - 1) / m) * m; }

/* Where compiled kernels are cached. It is data a user can delete at any time,
 * so a missing or unwritable directory only costs the next build its compile. */
std::string kernel_cache_dir() {
#if defined(_WIN32)
  char path[MAX_PATH] = {0};
  const DWORD n = GetEnvironmentVariableA("LOCALAPPDATA", path, MAX_PATH);
  if (n == 0 || n >= MAX_PATH) return std::string();
  const std::string dir = std::string(path) + "\\av1gpu";
  CreateDirectoryA(dir.c_str(), nullptr);
  return dir;
#else
  const char* home = std::getenv("XDG_CACHE_HOME");
  std::string dir = home && *home ? std::string(home) : std::string();
  if (dir.empty()) {
    const char* h = std::getenv("HOME");
    if (!h || !*h) return std::string();
    dir = std::string(h) + "/.cache";
  }
  dir += "/av1gpu";
  mkdir(dir.c_str(), 0700);
  return dir;
#endif
}

bool read_binary(const std::string& path, std::vector<char>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size <= 0) {
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(size));
  const size_t got = std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  if (got != out->size()) {
    out->clear();
    return false;
  }
  return true;
}

bool write_binary(const std::string& path, const std::vector<char>& bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  return wrote == bytes.size();
}

using cuda::CUdeviceptr;

/* argument pack for cuLaunchKernel: it wants an array of pointers to the
 * argument values, which is easy to get wrong by hand. The bound is checked
 * here rather than at the launch site, because writing past the array would
 * corrupt the caller's stack and the launch would still look fine. */
class Launcher {
 public:
  static constexpr int kMaxArgs = 16;

  void add_int(int value) {
    if (count_ >= kMaxArgs) {
      overflow_ = true;
      return;
    }
    ints_[count_] = value;
    slots_[count_] = &ints_[count_];
    ++count_;
  }
  void add_ptr(CUdeviceptr value) {
    if (count_ >= kMaxArgs) {
      overflow_ = true;
      return;
    }
    ptrs_[count_] = value;
    slots_[count_] = &ptrs_[count_];
    ++count_;
  }
  void** values() { return slots_; }
  int count() const { return count_; }
  bool ok() const { return !overflow_; }

 private:
  int count_ = 0;
  bool overflow_ = false;
  void* slots_[kMaxArgs] = {nullptr};
  int ints_[kMaxArgs] = {0};
  CUdeviceptr ptrs_[kMaxArgs] = {0};

 public:
  /* expose per-argument storage so graph node args can be copied into
   * persistent memory (the stack Launcher cannot outlive launch_graph). */
  int value_kind(int idx) const { return idx < count_ ? (slots_[idx] == &ints_[idx] ? 0 : 1) : -1; }
  int int_val(int idx) const { return idx < count_ ? ints_[idx] : 0; }
  CUdeviceptr ptr_val(int idx) const { return idx < count_ ? ptrs_[idx] : 0; }
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

  /* A pipelined caller decodes from its own thread; make the context current
   * there (a no-op when it already is). Returns false when that fails. */
  bool ensure_context(std::string* error) {
    cuda::CUcontext current = nullptr;
    if (driver_.CtxGetCurrent &&
        (driver_.CtxGetCurrent(&current) != cuda::CUDA_SUCCESS || current != context_)) {
      if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS) {
        if (error) *error = "cuCtxSetCurrent failed";
        return false;
      }
    }
    return true;
  }

  PlanLayout prepare(const FrameGeometry& geometry, int coded_width) override {
    /* A pipelined caller plans on its own thread while the worker is inside
     * the driver. Touching the context from here - CtxSetCurrent, let alone
     * an allocation - serialises the two threads on the driver, which is what
     * made the pipeline cost the sum of its halves instead of the larger one.
     * The layout is pure arithmetic and the buffers are sized for the first
     * frame, so the steady state takes no driver call at all. */
    const PlanLayout cached =
        make_plan_layout(geometry, coded_width, 64);
    if (ready_ && cached.buffer_size == layout_.buffer_size &&
        cached.stride[0] == layout_.stride[0] && cached.stride[1] == layout_.stride[1] &&
        cached.stride[2] == layout_.stride[2])
      return layout_;
    ensure_context(nullptr); /* prepare cannot report an error; decode will */
    const PlanLayout g = cached;
    const size_t size = g.buffer_size;
    layout_ = g;
    /* prepare() has no way to report an error, so record the failure and let
     * decode() refuse the frame: silently decoding into a null pointer would
     * either fault the context or, worse, be caught by the sync check much
     * later as an unexplained device error. */
    alloc_failed_ = !ensure(&frame_, &frame_cap_, size * sizeof(uint16_t)) ||
                    !ensure(&scratch_, &scratch_cap_, size * sizeof(uint16_t)) ||
                    !ensure(&ref_pool_, &ref_cap_, size * kReferenceSlots * sizeof(uint16_t)) ||
                    /* Loop restoration reads two pictures the frame no longer
                     * holds by then - the deblocked one and a snapshot of the
                     * CDEF output - and, for a superres frame, the upscaled
                     * deblocked one. */
                    !ensure(&lr_curr_, &lr_curr_cap_, size * sizeof(uint16_t)) ||
                    !ensure(&lr_cdef_, &lr_cdef_cap_, size * sizeof(uint16_t)) ||
                    !ensure(&lr_prev_up_, &lr_prev_up_cap_, size * sizeof(uint16_t));
    if (!tables_uploaded_ && !alloc_failed_) {
      upload_tables();
      tables_uploaded_ = true;
    }
    ready_ = !alloc_failed_;
    return g;
  }

  int upload_reference(const Frame& frame, std::string* error) override {
    if (layout_.buffer_size == 0) {
      if (error) *error = "upload_reference before prepare";
      return -1;
    }
    int slot;
    {
      std::lock_guard<std::mutex> lk(slot_mutex_);
      slot = allocate_slot();
      if (slot >= 0) slots_[slot] = true;
    }
    if (slot < 0) {
      if (error) *error = "CUDA reference pool exhausted";
      return -1;
    }
    const size_t size = layout_.buffer_size;
    std::vector<uint16_t> packed(size, 0);
    for (int p = 0; p < 3; ++p) {
      const int w = layout_.out_w[p];
      const int h = layout_.out_h[p];
      for (int y = 0; y < h; ++y)
        std::memcpy(packed.data() + layout_.plane_off[p] +
                        static_cast<size_t>(y) * layout_.stride[p],
                    frame.plane(p) + static_cast<size_t>(y) * frame.stride[p],
                    static_cast<size_t>(w) * sizeof(uint16_t));
    }
    const CUdeviceptr target =
        ref_pool_ + static_cast<size_t>(slot) * size * sizeof(uint16_t);
    if (!upload_h2d(target, packed.data(), size * sizeof(uint16_t))) {
      std::lock_guard<std::mutex> lk(slot_mutex_);
      slots_[slot] = false;
      if (error) *error = "cuMemcpyHtoD failed";
      return -1;
    }
    return slot;
  }

  void release_reference(int slot) override {
    std::lock_guard<std::mutex> lk(slot_mutex_);
    if (slot >= 0 && slot < kReferenceSlots) slots_[slot] = false;
  }

  int reference_capacity() const override { return kReferenceSlots; }

  /* The levels are expanded into the coefficient pool on the device by
   * av1g_k_dequant (see the batching section of decode_impl), so the host
   * leaves the raster pool empty. Verified byte-exact against the host path
   * on the recorded hashes. AV1GPU_GPU_DEQUANT=0 turns it off for the A/B. */
  /* The levels are expanded into the coefficient pool on the device by
   * av1g_k_dequant (see the batching section of decode_impl), so the host
   * leaves the raster pool empty: at 4K that is ~29 MB of work a frame and the
   * pool's upload, both of which move onto the device. The expansion is a port
   * of the host's fill loop and was verified byte-exact against it (the
   * lw_t2, 4K and 1080p recorded hashes, with the pool upload switched off to
   * prove the kernel is what fills it).
   *
   * The engine only leaves the pool empty for a syntax that describes it as
   * levels; anything else keeps the host path and this flag is not consulted.
   * AV1GPU_GPU_DEQUANT=0 turns the path off for the A/B. */
  bool dequantises_on_device() const override {
    static const bool on = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_GPU_DEQUANT", f, sizeof(f)) == 0 ||
             f[0] != '0';
    }();
    return on;
  }

  /* A reference slot is refreshed with the display picture after film grain
   * (see the display copy in decode_impl), and to_nv12 packs it on the device,
   * so a pipeline can leave the picture there. */
  bool packs_nv12_on_device() const override { return true; }

  int reserve_reference() override {
    std::lock_guard<std::mutex> lk(slot_mutex_);
    const int slot = allocate_slot();
    if (slot >= 0) slots_[slot] = true; /* a later decode_reserved fills it */
    return slot;
  }

  bool decode(const FramePlan& plan, bool keep_reference, Frame* out, int* out_slot,
              DecodeStats* stats, std::string* error) override {
    return decode_impl(plan, keep_reference, -1, out, out_slot, stats, error);
  }

  bool decode_reserved(const FramePlan& plan, int slot, Frame* out, int* out_slot,
                       DecodeStats* stats, std::string* error) override {
    return decode_impl(plan, true, slot, out, out_slot, stats, error);
  }

  bool decode_impl(const FramePlan& plan, bool keep_reference, int forced_slot,
                   Frame* out, int* out_slot, DecodeStats* stats, std::string* error) {
    /* One thread at a time on the device side: the staging buffers the uploads
     * go through (job_, pref_, the arenas) are shared, so a caller that packs
     * an earlier frame while this decode runs - a pipeline's output stage
     * calling to_nv12 on its own thread - has to wait for it, and vice versa.
     * The planner's half never touches the driver, which is what keeps the
     * pipeline's overlap intact. */
    std::lock_guard<std::mutex> driver_guard(driver_mutex_);
    error->clear();
    if (!ensure_context(error)) return false;
    if (alloc_failed_)
      return fail(error, "out of device memory (prepare failed to allocate the frame buffers)");
    const double t0 = now_ms();
    const PlanLayout& g = plan.layout;
    const int maxv = plan.maxv;
    int launches = 0;
    /* temporary profiling: CPU-side upload vs launch vs wait time */
    auto qpc = [] { LARGE_INTEGER c; QueryPerformanceCounter(&c); return c.QuadPart; };
    LARGE_INTEGER qf; QueryPerformanceFrequency(&qf);
    const auto ms_between = [&](LARGE_INTEGER a, LARGE_INTEGER b) {
      return double(b.QuadPart - a.QuadPart) / double(qf.QuadPart) * 1000.0;
    };
    LARGE_INTEGER tp0{}, tp1{};
    (void)tp0; (void)tp1;

    /* Every allocation and every copy is checked: a failed cuMemAlloc leaves a
     * null device pointer, and a kernel launched against it faults the context
     * asynchronously, which would surface as wrong pixels in a later frame
     * rather than as an error in this one. */
    h2d_bytes = 0; /* this frame's real upload volume, reported below */
    h2d_tag_ = kH2dTagCoef;
    /* The device buffers are sized from the plan's pool sizes: with a device
     * dequantisation the host leaves the raster vectors empty and only the
     * sizes travel, so `coefficients.size()` would be one int and the kernel
     * would write past the buffer. */
    const size_t coef_pool_ints = plan.coef_pool_size ? plan.coef_pool_size
                                                      : plan.coefficients.size();
    const size_t residual_pool_ints = plan.residual_pool_size ? plan.residual_pool_size
                                                             : plan.residual.size();
    if (!ensure(&coef_, &coef_cap_, (coef_pool_ints + 1) * sizeof(int)) ||
        !ensure(&tmp_, &tmp_cap_, (plan.tmp_size + 1) * sizeof(int)))
      return fail(error, "out of device memory for the coefficient buffers");
    /* The device expansion (av1g_k_dequant, launched in the batching section
     * below) turns the frame's levels into the raster pool, so the host's pool
     * is left unbuilt - at 4K that is ~24 MB less per frame over the bus and
     * the ~10 ms the expansion costs, both of which move onto the device.
     *
     * The decision is the *plan's* (FramePlan::raster_from_levels), set by the
     * engine when it chose to leave the raster empty: the two sides must not
     * disagree, or one of them reads a pool the other never filled. A plan that
     * carries its own raster (the synthetic encoder's syntax has no levels to
     * expand) says so and is uploaded as usual. AV1GPU_GPU_DEQUANT=0 disables
     * the whole path for the A/B against the host one. */
    const bool dequant_on_device = plan.raster_from_levels;
    if (!plan.coefficients.empty() && !dequant_on_device &&
        !upload_h2d(coef_, plan.coefficients.data(),
                    plan.coefficients.size() * sizeof(int)))
      return fail(error, "coefficient upload failed");
    if (!plan.residual.empty()) {
      if (!ensure(&residual_, &residual_cap_,
                  (residual_pool_ints + 1) * sizeof(int)))
        return fail(error, "out of device memory for the residual buffer");
      if (!upload_h2d(residual_, plan.residual.data(),
                      plan.residual.size() * sizeof(int)))
        return fail(error, "residual upload failed");
    }

    /* The device expansion (below) reads the frame's coded levels and its
     * quantiser matrices: a few megabytes where the raster pool is ~25 MB at
     * 4K. The pool it writes is zeroed by the kernel itself, so nothing has
     * to be cleared on the host. */
    h2d_tag_ = kH2dTagDequant;
    if (dequant_on_device) {
      if (!ensure(&dq_levels_, &dq_levels_cap_, (plan.levels.size() + 1) * sizeof(short)))
        return fail(error, "out of device memory for the level pool");
      if (!upload_h2d(dq_levels_, plan.levels.data(), plan.levels.size() * sizeof(short)))
        return fail(error, "level pool upload failed");
      dq_level_n_ = static_cast<int>(plan.levels.size());
      if (!ensure(&dq_qm_, &dq_qm_cap_, (plan.qm_pool.size() + 15) * sizeof(unsigned char)))
        return fail(error, "out of device memory for the QM pool");
      if (!upload_h2d(dq_qm_, plan.qm_pool.data(), plan.qm_pool.size()))
        return fail(error, "QM pool upload failed");
      dq_qm_n_ = static_cast<int>(plan.qm_pool.size());
    }

    /* ---- legacy path (AV1GPU_LEGACY=1): upload and launch each wave in
     * turn, the way this backend worked before the parallel prep existed.
     * Kept for A/B profiling of the arena layout. */
    static const bool legacy_wave = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_LEGACY", f, sizeof(f)) > 0 &&
             f[0] == '1';
    }();
    double gpu_ms = -1.0;
    void *ev0 = nullptr, *ev1 = nullptr;
    if (legacy_wave && driver_.EventCreate) {
      driver_.EventCreate(&ev0, 0);
      driver_.EventCreate(&ev1, 0);
      if (ev0) driver_.EventRecord(ev0, nullptr);
    }
    double t1 = 0.0; /* end of the wave reconstruction section */
    if (legacy_wave) {
      for (const WaveJobs& wave : plan.waves) {
        if (!wave.inter.empty()) {
          std::vector<Av1gInterJob> jobs = wave.inter;
          for (Av1gInterJob& job : jobs) {
            if (job.ref0_slot >= 0)
              job.ref0_off += job.ref0_slot * static_cast<int>(g.buffer_size);
            if (job.ref1_slot >= 0)
              job.ref1_off += job.ref1_slot * static_cast<int>(g.buffer_size);
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
              !upload_h2d(palette_, plan.palette_map.data(),
                          plan.palette_map.size()))
            return fail(error, "palette map write failed");
          CUdeviceptr job_mem = 0, pref_mem = 0;
          int total = 0;
          if (!upload_jobs(wave.intra, &job_mem, &pref_mem, &total,
                           [](const Av1gIntraJob& j) { return j.h; }))
            return fail(error, "intra job upload failed");
          Launcher a;
          add_mem(a, job_mem);
          add_mem(a, pref_mem);
          add_int(a, n);
          add_int(a, total);
          add_mem(a, frame_);
          add_mem(a, palette_);
          add_int(a, maxv);
          if (!launch_grid("av1g_k_intra", static_cast<unsigned>(n), 64, a))
            return fail(error, "av1g_k_intra failed");
          ++launches;
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
      if (ev1) {
        driver_.EventRecord(ev1, nullptr);
        driver_.EventSynchronize(ev1);
        float ms = 0;
        driver_.EventElapsedTime(&ms, ev0, ev1);
        gpu_ms = ms;
        driver_.EventDestroy(ev1);
        driver_.EventDestroy(ev0);
      }
      const double t1p = now_ms();
      {
        static double acc_rec = 0, acc_gpu = 0, acc_waves = 0;
        static long n = 0;
        acc_rec += t1p - t0;
        acc_gpu += gpu_ms > 0 ? gpu_ms : 0;
        acc_waves += plan.waves.size();
        ++n;
        if (n % 50 == 0) {
          char msg[128];
          std::snprintf(msg, sizeof(msg),
                        "perf legacy: reconstruct %.1f, gpu kernel %.1f, "
                        "waves %.0f over %ld",
                        acc_rec / n, acc_gpu / n, acc_waves / n, n);
          char path[MAX_PATH];
          if (GetTempPathA(MAX_PATH, path)) {
            strncat_s(path, "av1gpu_ds.log", _TRUNCATE);
            if (FILE* f = std::fopen(path, "a")) {
              SYSTEMTIME st2;
              GetLocalTime(&st2);
              std::fprintf(f, "%02u:%02u:%02u.%03u pid %lu thr %lu  %s\n", st2.wHour,
                           st2.wMinute, st2.wSecond, st2.wMilliseconds,
                           GetCurrentProcessId(), GetCurrentThreadId(), msg);
              std::fclose(f);
            }
          }
        }
      }
      t1 = t1p;
    } else {
    /* ---- parallel wave prep -------------------------------------------------
     * A 1080p frame carries ~100k jobs in ~130 dependency waves; preparing a
     * wave (copy, slot rebase, prefix sums, host to device upload) is pure CPU
     * work that does not read any other wave, and it dwarfs the 2-3 ms the
     * kernels actually run. Prepare the waves on worker threads - each adopts
     * the CUDA context - then launch the kernels from this thread in wave
     * order. Every wave's arrays land in one per-frame arena at fixed offsets,
     * so the launch phase only needs the offsets. */
    struct WaveOffsets {
      /* The arena (host_job_/host_pref_, job_arena_): the kinds a per-wave
       * launch reads - intra and residual. Inter and itx are not staged here
       * any more: the prep writes them straight into the frame-level batch
       * arrays, so the 7.6 MB of inter jobs a 4K frame carries is copied and
       * uploaded once instead of twice. */
      size_t intra_job = 0, intra_pref = 0, res_job = 0, res_pref = 0;
      /* Element bases into the frame-level batch host arrays: where this
       * wave's inter and itx jobs, and their prefixes, belong. */
      size_t inter_base = 0, inter_pref_base = 0;
      size_t itx_base = 0, itx_pref_base = 0;
      /* The absolute prefix offset of this wave's entries: the batched launch
       * reads one prefix over every wave's jobs concatenated, so each wave's
       * run starts where the previous wave's element count left off. */
      int inter_abs_base = 0, itx_row_abs_base = 0;
    };
    std::vector<WaveLaunch> prepped(plan.waves.size());
    auto align64 = [](size_t x) { return (x + 63) & ~size_t(63); };
    std::vector<WaveOffsets> offs(plan.waves.size());
    /* The arena holds the per-wave kinds; the inter and itx jobs go into the
     * batch arrays directly, so the arena is only as big as intra and residual
     * are. Two more layouts matter: the frame-level batch concatenation (one
     * absolute prefix per kind, for the single batched launches) and the
     * per-wave prefixes rebased to zero, because a per-wave launch binary
     * searches its own slice of the batch array and needs joff[0] == 0. */
    size_t arena_job = 0, arena_pref = 0;
    size_t bi_n = 0, itx_n = 0;
    /* The prefix arrays hold n+2 entries per wave (the run plus the two
     * trailing duplicates the kernels read at joff[njobs]), so the totals count
     * those per wave - the absolute and the rebased arrays have the same
     * shape and the same count. */
    size_t inter_pref_n = 0, itx_pref_n = 0;
    int bi_total = 0, itx_rows_total = 0;
    /* Device buffers for the frame-level batch arrays, sized per geometry
     * maximum and reused across frames. */
    auto ensure_batch = [&](CUdeviceptr& ptr, size_t& cap, size_t bytes) -> bool {
      if (bytes == 0) return true; /* nothing to size: the pointer is kept */
      if (bytes > cap) {
        CUdeviceptr tmp = 0;
        if (driver_.MemAlloc(&tmp, bytes) != cuda::CUDA_SUCCESS) return false;
        if (ptr && driver_.MemFree(ptr) != cuda::CUDA_SUCCESS) {
        }
        ptr = tmp;
        cap = bytes;
      }
      return true;
    };
    for (size_t i = 0; i < plan.waves.size(); ++i) {
      const WaveJobs& w = plan.waves[i];
      WaveOffsets& o = offs[i];
      size_t job = align64(arena_job), pref = align64(arena_pref);
      if (!w.inter.empty()) {
        o.inter_base = bi_n;
        o.inter_pref_base = inter_pref_n;
        o.inter_abs_base = bi_total;
        bi_n += w.inter.size();
        inter_pref_n += w.inter.size() + 2;
        for (const Av1gInterJob& j : w.inter) bi_total += j.w * j.h;
      }
      if (!w.intra.empty()) {
        o.intra_job = align64(job);
        o.intra_pref = align64(pref);
        job = o.intra_job + w.intra.size() * sizeof(Av1gIntraJob);
        pref = o.intra_pref + (w.intra.size() + 2) * sizeof(int);
      }
      if (!w.itx.empty()) {
        o.itx_base = itx_n;
        o.itx_pref_base = itx_pref_n;
        o.itx_row_abs_base = itx_rows_total;
        itx_n += w.itx.size();
        itx_pref_n += w.itx.size() + 2;
        for (const Av1gItxJob& j : w.itx) itx_rows_total += j.h;
      }
      if (!w.residual.empty()) {
        o.res_job = align64(job);
        job = o.res_job + w.residual.size() * sizeof(Av1gItxJob);
        o.res_pref = align64(pref);
        pref = o.res_pref + (w.residual.size() + 2) * sizeof(int);
      }
      arena_job = job;
      arena_pref = pref;
    }
    /* The frame-level batch arrays, laid out back to back in one device buffer
     * and one host buffer, so the whole set travels in a single upload. The
     * offsets are byte offsets into that buffer; the prep's writes use the
     * per-wave *element* bases added to them.
     *
     * Two prefix shapes per kind: the compact one (one entry per job plus the
     * array's trailing pair) that the single batched launch binary-searches, and
     * the per-wave one rebased to zero that a launch covering a single wave
     * reads. Both hold n+2 entries per wave, so their counts are the same. */
    struct BatchLayout {
      size_t inter_job = 0, inter_pref = 0, pw_inter = 0;
      size_t itx_job = 0, itx_row = 0, pw_itx_col = 0, pw_itx_row = 0;
      size_t total = 0;
    };
    BatchLayout lay;
    auto align16 = [](size_t x) { return (x + 15) & ~size_t(15); };
    lay.inter_job = 0;
    lay.inter_pref = align16(bi_n * sizeof(Av1gInterJob));
    lay.pw_inter = align16(lay.inter_pref + (bi_n + 2) * sizeof(int));
    lay.itx_job = align16(lay.pw_inter + inter_pref_n * sizeof(int));
    lay.itx_row = align16(lay.itx_job + itx_n * sizeof(Av1gItxJob));
    lay.pw_itx_col = align16(lay.itx_row + (itx_n + 2) * sizeof(int));
    lay.pw_itx_row = align16(lay.pw_itx_col + itx_pref_n * sizeof(int));
    lay.total = align16(lay.pw_itx_row + itx_pref_n * sizeof(int));
    if (batch_host_.size() < lay.total) batch_host_.resize(lay.total);
    if (!ensure(&job_arena_, &job_arena_cap_, arena_job) ||
        !ensure(&pref_arena_, &pref_arena_cap_, arena_pref * sizeof(int)))
      return fail(error, "out of device memory for the wave arena");
    if (!ensure(&batch_dev_, &batch_dev_cap_, lay.total))
      return fail(error, "out of device memory for the batched jobs");

    /* the palette map is one shared upload for the whole frame; the buffer is
     * allocated even when the frame uses no palette, because the intra kernel
     * takes the pointer unconditionally and a null one would fault the
     * context asynchronously */
    if (!ensure(&palette_, &palette_cap_, plan.palette_map.size()))
      return fail(error, "palette map upload failed");
    if (!plan.palette_map.empty() &&
        !upload_h2d(palette_, plan.palette_map.data(), plan.palette_map.size()))
      return fail(error, "palette map write failed");

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    (void)hw; /* the shared pool picks the worker count itself */

    /* ---- build the arena on the host, then upload it once ----------------
     * The job arrays and their prefix sums go into one host buffer per arena
     * and travel to the device in a single copy each. The previous shape did a
     * pinned allocation and a cuMemcpyHtoD per wave *and* per kind - hundreds
     * of small transfers per frame - and needed the CUDA context on every
     * worker thread. Now the workers only touch plain memory (no CUDA call at
     * all), and the uploads happen on this thread afterwards. */
    if (host_job_.size() < arena_job) host_job_.resize(arena_job);
    if (host_pref_.size() < arena_pref) host_pref_.resize(arena_pref);
    char* const hjobs = host_job_.data();
    char* const hpref = host_pref_.data();
    char* const bh = batch_host_.data();
    Av1gInterJob* const binter =
        reinterpret_cast<Av1gInterJob*>(bh + lay.inter_job);
    int* const binter_pref = reinterpret_cast<int*>(bh + lay.inter_pref);
    int* const pw_inter_pref = reinterpret_cast<int*>(bh + lay.pw_inter);
    Av1gItxJob* const bitx = reinterpret_cast<Av1gItxJob*>(bh + lay.itx_job);
    int* const bitx_row_pref = reinterpret_cast<int*>(bh + lay.itx_row);
    int* const pw_itx_col_pref = reinterpret_cast<int*>(bh + lay.pw_itx_col);
    int* const pw_itx_row_pref = reinterpret_cast<int*>(bh + lay.pw_itx_row);

    auto prep_wave = [&](size_t i) {
      {
        const WaveJobs& w = plan.waves[i];
        WaveLaunch& L = prepped[i];
        const WaveOffsets& o = offs[i];
        if (!w.inter.empty()) {
          /* Straight into the frame-level batch array, with the absolute prefix
           * the single batched launch reads and the per-wave one rebased to
           * zero for a launch that covers this wave only. */
          Av1gInterJob* dst = binter + o.inter_base;
          std::memcpy(dst, w.inter.data(), w.inter.size() * sizeof(Av1gInterJob));
          int* prefix = binter_pref + o.inter_base;
          int* pw_prefix = pw_inter_pref + o.inter_pref_base;
          const int base = o.inter_abs_base;
          int sum = 0;
          for (size_t k = 0; k < w.inter.size(); ++k) {
            Av1gInterJob& job = dst[k];
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
            for (int b = 0; b < job.obmc_count; ++b)
              if (job.obmc[b].ref_slot >= 0)
                job.obmc[b].ref_off += job.obmc[b].ref_slot * static_cast<int>(g.buffer_size);
            prefix[k] = base + sum;
            pw_prefix[k] = sum;
            sum += job.w * job.h;
          }
          /* The compact array's two trailing entries are the whole array's
           * total, so only the last wave to write one puts them there - the
           * other waves would overwrite their neighbours. */
          if (o.inter_base + w.inter.size() == bi_n) {
            prefix[w.inter.size()] = base + sum;
            prefix[w.inter.size() + 1] = base + sum;
          }
          pw_prefix[w.inter.size()] = sum;
          pw_prefix[w.inter.size() + 1] = sum;
          L.inter = true;
          L.inter_n = static_cast<int>(w.inter.size());
          L.inter_total = sum;
          L.host_inter_job = o.inter_base;
          L.host_inter_pref = o.inter_pref_base;
        }
        if (!w.intra.empty()) {
          std::memcpy(hjobs + o.intra_job, w.intra.data(),
                      w.intra.size() * sizeof(Av1gIntraJob));
          int* prefix = reinterpret_cast<int*>(hpref + o.intra_pref);
          int sum = 0;
          for (size_t k = 0; k < w.intra.size(); ++k) {
            prefix[k] = sum;
            sum += w.intra[k].h;
          }
          prefix[w.intra.size()] = sum;
          prefix[w.intra.size() + 1] = sum;
          L.intra = true;
          L.intra_n = static_cast<int>(w.intra.size());
          L.intra_total = sum;
          L.host_intra_job = o.intra_job;
          L.host_intra_pref = o.intra_pref;
        }
        if (!w.itx.empty()) {
          /* Like inter: the jobs go into the frame-level batch array (the
           * batched itx_rows launch reads them there), with the row prefix
           * absolute over every wave and the column prefix rebased for the
           * per-wave itx_cols launch, which needs joff[0] == 0. */
          std::memcpy(bitx + o.itx_base, w.itx.data(),
                      w.itx.size() * sizeof(Av1gItxJob));
          int* row_prefix = bitx_row_pref + o.itx_base;
          int* pw_row = pw_itx_row_pref + o.itx_pref_base;
          int* col_prefix = pw_itx_col_pref + o.itx_pref_base;
          const int row_base = o.itx_row_abs_base;
          int rows_sum = 0, cols_sum = 0;
          for (size_t k = 0; k < w.itx.size(); ++k) {
            row_prefix[k] = row_base + rows_sum;
            pw_row[k] = rows_sum;
            rows_sum += w.itx[k].h;
            col_prefix[k] = cols_sum;
            cols_sum += w.itx[k].w;
          }
          /* the compact array's trailing pair is the whole array's total, so
           * only the last wave writes it: the others would overwrite the next
           * wave's entries */
          if (o.itx_base + w.itx.size() == itx_n) {
            row_prefix[w.itx.size()] = row_base + rows_sum;
            row_prefix[w.itx.size() + 1] = row_base + rows_sum;
          }
          col_prefix[w.itx.size()] = cols_sum;
          col_prefix[w.itx.size() + 1] = cols_sum;
          pw_row[w.itx.size()] = rows_sum;
          pw_row[w.itx.size() + 1] = rows_sum;
          L.itx = true;
          L.itx_n = static_cast<int>(w.itx.size());
          L.itx_rows = rows_sum;
          L.itx_cols = cols_sum;
          L.host_itx_job = o.itx_base;
          L.host_itx_row = o.itx_base;
          L.host_itx_col = o.itx_pref_base;
        }
        if (!w.residual.empty()) {
          std::memcpy(hjobs + o.res_job, w.residual.data(),
                      w.residual.size() * sizeof(Av1gItxJob));
          int* prefix = reinterpret_cast<int*>(hpref + o.res_pref);
          int sum = 0;
          for (size_t k = 0; k < w.residual.size(); ++k) {
            prefix[k] = sum;
            sum += w.residual[k].w * w.residual[k].h;
          }
          prefix[w.residual.size()] = sum;
          prefix[w.residual.size() + 1] = sum;
          L.residual = true;
          L.res_n = static_cast<int>(w.residual.size());
          L.res_total = sum;
          L.host_res_job = o.res_job;
          L.host_res_pref = o.res_pref;
        }
      }
    };

    frontend::parallel_chunks(plan.waves.size(), 6, [&](size_t, size_t b0, size_t e0) {
      for (size_t i = b0; i < e0; ++i) prep_wave(i);
    }, /*urgent=*/true);

    /* the device addresses, once the host copy is complete: the inter and itx
     * jobs live in the one frame-level batch buffer (a per-wave launch reads
     * its own slice of it), and only the intra and residual jobs are in the
     * arena. */
    for (size_t i = 0; i < plan.waves.size(); ++i) {
      WaveLaunch& L = prepped[i];
      const WaveOffsets& o = offs[i];
      if (L.inter) {
        L.inter_job = batch_dev_ + lay.inter_job +
                      o.inter_base * sizeof(Av1gInterJob);
        L.inter_pref = batch_dev_ + lay.pw_inter + o.inter_pref_base * sizeof(int);
      }
      if (L.intra) {
        L.intra_job = job_arena_ + o.intra_job;
        L.intra_pref = pref_arena_ + o.intra_pref;
      }
      if (L.itx) {
        L.itx_job = batch_dev_ + lay.itx_job + o.itx_base * sizeof(Av1gItxJob);
        L.itx_row_pref = batch_dev_ + lay.pw_itx_row + o.itx_pref_base * sizeof(int);
        L.itx_col_pref = batch_dev_ + lay.pw_itx_col + o.itx_pref_base * sizeof(int);
      }
      if (L.residual) {
        L.res_job = job_arena_ + o.res_job;
        L.res_pref = pref_arena_ + o.res_pref;
      }
    }
    h2d_tag_ = kH2dTagWave;
    if (arena_job && !upload_h2d(job_arena_, hjobs, arena_job))
      return fail(error, "wave job upload failed");
    if (arena_pref && !upload_h2d(pref_arena_, hpref, arena_pref))
      return fail(error, "wave prefix upload failed");
    const double t_arena_end = now_ms(); /* DEBUG: arena vs batch split */
    /* The whole frame-level batch set travels in ONE copy: the inter and itx
     * jobs, the compact prefixes the single batched launches read and the
     * per-wave ones rebased to zero, back to back in buffer_dev_. Seven
     * separate uploads measured ~1.8 ms of driver calls a frame at 4K, and one
     * memcpy plus one cuMemcpyHtoD replaces them. */
    h2d_tag_ = kH2dTagBatch;
    if (lay.total && !upload_h2d(batch_dev_, batch_host_.data(), lay.total))
      return fail(error, "batched jobs upload failed");

    /* ---- kernel batching: concatenate all waves' jobs per kind ------------
     * Instead of one kernel launch per wave per kind (thousands per 4K
     * keyframe on WDDM), build one frame-level array per kind and launch a
     * single kernel per kind per frame. Kernels use AV1G_JOB_LOOKUP (jobs,
     * joff, njobs, total) to binary-search element to job, so concatenation is
     * straightforward.
     *
     * Batchable kinds (order-independent across waves): inter, itx_rows,
     * itx_cols, residual. Intra keeps per-wave launches: its edge-pixel reads
     * depend on the block above, which belongs to an earlier wave.
     *
     *       * The prep writes these arrays itself, at the per-wave bases the offsets
      * above computed, so the launch phase has nothing to concatenate: the
      * inter and itx jobs of every wave are already in wave order, with an
      * absolute prefix per kind for the single batched launch and a per-wave
      * prefix rebased to zero for the launches that cover one wave. */
    /* The counts the offsets above produced; the launch phase only reads them. */
    /* The frame-level arrays are already on the device (the prep filled the
     * host side and the uploads above sent them), so this phase is only the
     * boundary marker for the arena/batch split below. */
    const double t_batch_end = now_ms(); /* DEBUG: arena vs batch split */

    /* The batched residual array is deliberately not built: no launch reads it.
     * The per-wave sequential launches take each wave's residual jobs straight
     * from the arena (L.res_job), which is where the prep already put them, so
     * a second frame-level copy of the same jobs would be pure PCIe traffic.
     * The same is true of the itx *column* prefix: only the row prefix is
     * launched (av1g_k_itx_rows), the columns run per wave off pref_arena_. */

    /* Device dequantisation: the frame's coded levels are expanded into the
     * raster pool here instead of the host building (and uploading) the whole
     * pool. The kernel reads the same jobs the itx kernels will, so it runs
     * off the very same array - and only the itx jobs: a residual-provided
     * transform takes its data from plan.residual, which is uploaded as
     * usual. The three scan tables are constant, so they go up once. */
    if (dequant_on_device) {
      if (!dq_tables_ready_) {
        const size_t two_d = sizeof(frontend::coef_tables::kDefaultScan);
        const size_t row = sizeof(frontend::coef_tables::kMrowScan);
        const size_t col = sizeof(frontend::coef_tables::kMcolScan);
        if (!ensure(&dq_scan_2d_, &dq_scan_2d_cap_, two_d) ||
            !ensure(&dq_scan_row_, &dq_scan_row_cap_, row) ||
            !ensure(&dq_scan_col_, &dq_scan_col_cap_, col))
          return fail(error, "out of device memory for the dequant scan tables");
        if (!upload_h2d(dq_scan_2d_, frontend::coef_tables::kDefaultScan, two_d) ||
            !upload_h2d(dq_scan_row_, frontend::coef_tables::kMrowScan, row) ||
            !upload_h2d(dq_scan_col_, frontend::coef_tables::kMcolScan, col))
          return fail(error, "dequant scan table upload failed");
        dq_tables_ready_ = true;
      }
      if (itx_n > 0) {
        Launcher a;
        add_mem(a, batch_dev_ + lay.itx_job);
        add_int(a, static_cast<int>(itx_n));
        add_mem(a, dq_levels_);
        add_mem(a, dq_qm_);
        add_mem(a, dq_scan_2d_);
        add_mem(a, dq_scan_row_);
        add_mem(a, dq_scan_col_);
        add_mem(a, coef_);
        if (!launch("av1g_k_dequant", static_cast<int>(itx_n), a))
          return fail(error, "av1g_k_dequant failed");
      }
    }

    /* Ensure all batch uploads complete before any kernel launches */
    const double t_prep = now_ms(); /* DEBUG */

    /* launch phase: kernels must start in wave order */
    static const bool skip_launch = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_NO_LAUNCH", f, sizeof(f)) > 0 &&
             f[0] == '1';
    }();
    /* CUDA graphs crash on WDDM driver 13.1 even with PTX-loaded kernels
     * when using real decoder kernels (access violation in first chunk).
     * Disabled by default; AV1GPU_USE_GRAPH=1 to force-enable. */
    static const bool use_graph = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_USE_GRAPH", f, sizeof(f)) > 0 &&
             f[0] == '1';
    }();
    static const unsigned kind_mask = [] {
      char f[16] = {0};
      if (GetEnvironmentVariableA("AV1GPU_ONLY_KIND", f, sizeof(f)) == 0)
        return 0xFu;
      unsigned m = 0;
      if (std::strchr(f, 'I')) m |= 1u; /* inter */
      if (std::strchr(f, 'n')) m |= 2u; /* intra */
      if (std::strchr(f, 't')) m |= 4u; /* itx */
      if (std::strchr(f, 'r')) m |= 8u; /* residual */
      return m ? m : 0xFu;
    }();
    void *ev0 = nullptr, *ev1 = nullptr; /* GPU side timing (optional via env) */
    static const bool want_gpu_timing = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_GPU_TIMING", f, sizeof(f)) > 0 && f[0] == '1';
    }();
    if (want_gpu_timing && driver_.EventCreate && !skip_launch) {
      driver_.EventCreate(&ev0, 0);
      driver_.EventCreate(&ev1, 0);
      if (ev0) driver_.EventRecord(ev0, nullptr);
    }
    /* ---- batched kernel launch phase -------------------------------------
     * Launch one kernel per kind per frame for batchable kinds (inter,
     * itx_rows, itx_cols, residual), plus intra per-wave (cross-wave edge
     * dependencies).  Kernels take (jobs, joff, njobs, total) and use
     * AV1G_JOB_LOOKUP, so concatenated frame-level arrays work directly.
     * This collapses thousands of per-wave launches into ~4-5 launches per
     * frame, which is the dominant 4K win on WDDM where each cuLaunchKernel
     * costs ~0.5-1 ms. */
    static const bool use_batch = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_USE_BATCH", f, sizeof(f)) == 0 ||
             f[0] != '0';
    }();

    /* Persistent single-launch wave kernel: one kernel processes all waves
     * with software barriers, eliminating per-wave launch overhead on WDDM.
     * Enabled by default; AV1GPU_USE_WAVE=0 disables. */
    static const bool use_wave = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_USE_WAVE", f, sizeof(f)) > 0 &&
             f[0] == '1';
    }();

    if (!skip_launch) {
      bool all = false;
      /* Try CUDA graph first (one launch per frame - eliminates WDDM per-wave
       * launch overhead). Fall back to batching, then per-wave. */
      if (use_graph && driver_.GraphCreate && driver_.GraphAddKernelNode &&
          driver_.GraphInstantiate && driver_.GraphLaunch &&
          driver_.GraphAddDependencies) {
        all = launch_graph(prepped, kind_mask, maxv);
      }
      if (!all && use_wave) {
        /* Wave-independent kinds first: batch inter + itx_rows */
        if (bi_n > 0 && (kind_mask & 1u)) {
          Launcher a;
          add_mem(a, batch_dev_ + lay.inter_job);
          add_mem(a, batch_dev_ + lay.inter_pref);
          add_int(a, static_cast<int>(bi_n));
          add_int(a, bi_total);
          add_mem(a, frame_);
          add_mem(a, ref_pool_);
          add_mem(a, frame_);
          add_mem(a, filters_);
          add_int(a, maxv);
          if (!launch("av1g_k_inter", bi_total, a)) all = false;
          else ++launches;
        }
        if (itx_n > 0 && (kind_mask & 4u)) {
          Launcher a;
          add_mem(a, batch_dev_ + lay.itx_job);
          add_mem(a, batch_dev_ + lay.itx_row);
          add_int(a, static_cast<int>(itx_n));
          add_int(a, itx_rows_total);
          add_mem(a, coef_);
          add_mem(a, tmp_);
          if (!launch("av1g_k_itx_rows", itx_rows_total, a)) all = false;
          else ++launches;
        }
        if (all) {
          all = launch_wave_kernel(prepped, kind_mask, maxv, lay.itx_job,
                                   lay.pw_itx_col);
          if (all) ++launches;
        }
      }
      if (!all && use_batch) {
        /* The batched launch sequence, as a function so the stream capture
         * path can record it and the plain path can run it. */
        auto batch_launch = [&]() -> bool {
          bool ok = true;
        /* Batch launch order matters: intra of wave N+1 reads frame_
         * reconstructed by wave N's itx_cols+residual. So we must:
         *   1) batch inter (all waves) - independent
         *   2) batch itx_rows (all waves) - independent
         *   3) per-wave sequential: intra -> itx_cols -> residual */
        /* inter: single launch for all waves */
        if (bi_n > 0 && (kind_mask & 1u)) {
          Launcher a;
          add_mem(a, batch_dev_ + lay.inter_job);
          add_mem(a, batch_dev_ + lay.inter_pref);
          add_int(a, static_cast<int>(bi_n));
          add_int(a, bi_total);
          add_mem(a, frame_);
          add_mem(a, ref_pool_);
          add_mem(a, frame_);
          add_mem(a, filters_);
          add_int(a, maxv);
          if (!launch("av1g_k_inter", bi_total, a)) ok = false;
          else ++launches;
        }
        /* itx_rows: single launch for all waves */
        if (itx_n > 0 && (kind_mask & 4u)) {
          Launcher a;
          add_mem(a, batch_dev_ + lay.itx_job);
          add_mem(a, batch_dev_ + lay.itx_row);
          add_int(a, static_cast<int>(itx_n));
          add_int(a, itx_rows_total);
          add_mem(a, coef_);
          add_mem(a, tmp_);
          if (!launch("av1g_k_itx_rows", itx_rows_total, a)) ok = false;
          else ++launches;
        }
        /* intra, itx_cols, residual: per-wave sequential (cross-wave frame_
         * dependencies - intra of wave N+1 reads residual of wave N).
         * NOTE: 2-stream event pipelining was tried and is ~8x SLOWER on WDDM
         * (events cost ~300 us each there), so we stay on the default stream. */
        for (size_t i = 0; i < plan.waves.size() && ok; ++i) {
          const WaveLaunch& L = prepped[i];
          if (L.intra && (kind_mask & 2u)) {
            Launcher a;
            add_mem(a, L.intra_job);
            add_mem(a, L.intra_pref);
            add_int(a, L.intra_n);
            add_int(a, L.intra_total);
            add_mem(a, frame_);
            add_mem(a, palette_);
            add_int(a, maxv);
            if (!launch_grid("av1g_k_intra", static_cast<unsigned>(L.intra_n), 64, a))
              ok = false;
            else ++launches;
          }
          if (L.itx && (kind_mask & 4u)) {
            Launcher a;
            add_mem(a, L.itx_job);
            add_mem(a, L.itx_col_pref);
            add_int(a, L.itx_n);
            add_int(a, L.itx_cols);
            add_mem(a, tmp_);
            add_mem(a, frame_);
            add_int(a, maxv);
            if (!launch("av1g_k_itx_cols", L.itx_cols, a)) ok = false;
            else ++launches;
          }
          if (L.residual && (kind_mask & 8u)) {
            Launcher a;
            add_mem(a, L.res_job);
            add_mem(a, L.res_pref);
            add_int(a, L.res_n);
            add_int(a, L.res_total);
            add_mem(a, residual_);
            add_mem(a, frame_);
            add_int(a, maxv);
            if (!launch("av1g_k_add_residual", L.res_total, a)) ok = false;
            else ++launches;
          }
        }
        return ok;
        };
        /* Stream capture: record the launches into a CUDA graph and submit
         * them with a single cuGraphLaunch.  A frame is hundreds of waves and
         * one launch costs tens of microseconds of driver time on WDDM, so
         * the launch count is what the frame time is made of.  Unlike the
         * manual cuGraphAddKernelNode path this does not depend on the
         * CUDA_KERNEL_NODE_PARAMS layout, which differs between drivers.
         * On by default; AV1GPU_USE_CAPTURE=0 turns it off. */
        static const bool use_capture = [] {
          char f[8] = {0};
          if (GetEnvironmentVariableA("AV1GPU_USE_CAPTURE", f, sizeof(f)) == 0)
            return true;
          return f[0] != '0';
        }();
        if (use_capture && capture_begin()) {
          batch_launch();
          if (capture_end_and_launch()) {
            all = true;
          } else {
            /* Nothing ran: the capture failed before the graph was submitted,
             * so the plain launch path can still do the work. */
            all = false;
          }
        }
        if (!all) all = batch_launch();
      }
      if (!all) {
        /* Graph and batch both failed/unavailable: plain per-wave launches. */
        all = true;
        for (size_t i = 0; i < plan.waves.size() && all; ++i) {
            const WaveLaunch& L = prepped[i];
            if (L.inter && (kind_mask & 1u)) {
              Launcher a;
              add_mem(a, L.inter_job);
              add_mem(a, L.inter_pref);
              add_int(a, L.inter_n);
              add_int(a, L.inter_total);
              add_mem(a, frame_);
              add_mem(a, ref_pool_);
              add_mem(a, frame_);
              add_mem(a, filters_);
              add_int(a, maxv);
              if (!launch("av1g_k_inter", L.inter_total, a)) all = false;
              else ++launches;
            }
            if (L.intra && (kind_mask & 2u)) {
              Launcher a;
              add_mem(a, L.intra_job);
              add_mem(a, L.intra_pref);
              add_int(a, L.intra_n);
              add_int(a, L.intra_total);
              add_mem(a, frame_);
              add_mem(a, palette_);
              add_int(a, maxv);
              if (!launch_grid("av1g_k_intra", static_cast<unsigned>(L.intra_n), 64, a))
                all = false;
              else ++launches;
            }
            if (L.itx && (kind_mask & 4u)) {
              {
                Launcher a;
                add_mem(a, L.itx_job);
                add_mem(a, L.itx_row_pref);
                add_int(a, L.itx_n);
                add_int(a, L.itx_rows);
                add_mem(a, coef_);
                add_mem(a, tmp_);
                if (!launch("av1g_k_itx_rows", L.itx_rows, a)) all = false;
                else ++launches;
              }
              {
                Launcher a;
                add_mem(a, L.itx_job);
                add_mem(a, L.itx_col_pref);
                add_int(a, L.itx_n);
                add_int(a, L.itx_cols);
                add_mem(a, tmp_);
                add_mem(a, frame_);
                add_int(a, maxv);
                if (!launch("av1g_k_itx_cols", L.itx_cols, a)) all = false;
                else ++launches;
              }
            }
            if (L.residual && (kind_mask & 8u)) {
              Launcher a;
              add_mem(a, L.res_job);
              add_mem(a, L.res_pref);
              add_int(a, L.res_n);
              add_int(a, L.res_total);
              add_mem(a, residual_);
              add_mem(a, frame_);
              add_int(a, maxv);
              if (!launch("av1g_k_add_residual", L.res_total, a)) all = false;
              else ++launches;
            }
          }
        }
      if (!all) return fail(error, "wave kernel launch failed");
    }
    /* A wave kernel that faults must fail this frame loudly: the sync here is
     * the single point where an out of bounds job array becomes visible. */
    const double t_launch_end = now_ms(); /* DEBUG */
    double gpu_ms = -1.0; /* DEBUG: true GPU side time of the wave kernels */
    if (ev1) {
      driver_.EventRecord(ev1, nullptr);
      driver_.EventSynchronize(ev1);
      float ms = 0;
      driver_.EventElapsedTime(&ms, ev0, ev1);
      gpu_ms = ms;
      driver_.EventDestroy(ev1);
      driver_.EventDestroy(ev0);
    }
    {
      const CUresult rc = driver_.CtxSynchronize();
      if (rc != cuda::CUDA_SUCCESS) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "wave kernel fault (sync = %d)", rc);
        return fail(error, buf);
      }
    }
    const double t_sync_end = now_ms(); /* DEBUG */

    t1 = now_ms();
    /* DEBUG: prep vs launch vs gpu wait split */
    {
      static double acc_prep = 0, acc_launch = 0, acc_sync = 0, acc_gpu = 0;
      static double worst_prep = 0, worst_launch = 0, worst_sync = 0;
      static long n2 = 0;
      const double d_prep = t_prep - t0, d_launch = t_launch_end - t_prep,
                   d_sync = t_sync_end - t_launch_end;
      const double d_prep_arena = t_arena_end - t0, d_prep_batch = t_batch_end - t_arena_end;
      acc_prep += d_prep; acc_launch += d_launch;
      acc_sync += d_sync; acc_gpu += gpu_ms > 0 ? gpu_ms : 0;
      static double acc_arena = 0, acc_batch = 0;
      acc_arena += d_prep_arena; acc_batch += d_prep_batch;
      if (d_prep > worst_prep) worst_prep = d_prep;
      if (d_launch > worst_launch) worst_launch = d_launch;
      if (d_sync > worst_sync) worst_sync = d_sync;
      ++n2;
      if (n2 % 50 == 0) {
        char msg[200];
        std::snprintf(msg, sizeof(msg),
                      "perf wave prep %.1f (w %.1f) [arena %.1f, batch %.1f], "
                      "launch %.1f (w %.1f), gpuwait %.1f "
                      "(w %.1f), gpu kernel %.1f ms over %ld",
                      acc_prep / n2, worst_prep, acc_arena / n2, acc_batch / n2,
                      acc_launch / n2, worst_launch,
                      acc_sync / n2, worst_sync, acc_gpu / n2, n2);
        char path[MAX_PATH];
        if (GetTempPathA(MAX_PATH, path)) {
          strncat_s(path, "av1gpu_ds.log", _TRUNCATE);
          if (FILE* f = std::fopen(path, "a")) {
            SYSTEMTIME st2; GetLocalTime(&st2);
            std::fprintf(f, "%02u:%02u:%02u.%03u pid %lu thr %lu  %s\n", st2.wHour,
                         st2.wMinute, st2.wSecond, st2.wMilliseconds,
                         GetCurrentProcessId(), GetCurrentThreadId(), msg);
            std::fclose(f);
          }
        }
        acc_prep = acc_launch = acc_sync = acc_gpu = 0;
        acc_arena = acc_batch = 0;
        worst_prep = worst_launch = worst_sync = 0;
        n2 = 0;
      }
    }
    } /* legacy/parallel wave path */

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

    /* The two lines either side of a stripe that restoration reads come from
     * the picture as it is *before* CDEF, so it has to be kept. */
    const bool uses_lr = !plan.restoration.empty();
    if (uses_lr &&
        driver_.MemcpyDtoD(lr_curr_, frame_, g.buffer_size * sizeof(uint16_t)) !=
            cuda::CUDA_SUCCESS)
      return fail(error, "restoration snapshot copy failed");
    if (!plan.cdef.empty()) {
      if (!ensure(&scratch_, &scratch_cap_, g.buffer_size * sizeof(uint16_t)))
        return fail(error, "out of device memory for the CDEF snapshot");
      if (driver_.MemcpyDtoD(scratch_, frame_, g.buffer_size * sizeof(uint16_t)) !=
          cuda::CUDA_SUCCESS)
        return fail(error, "CDEF snapshot copy failed");
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

    /* ---- superres ---------------------------------------------------------- */
    /* The upscaler reads a snapshot of the coded picture (the scratch buffer)
     * and writes the display picture into frame_: within one row a work item
     * reads samples ahead of the ones it writes, so the two must be different
     * buffers. */
    if (!plan.upscale.empty()) {
      if (driver_.MemcpyDtoD(scratch_, frame_, g.buffer_size * sizeof(uint16_t)) !=
          cuda::CUDA_SUCCESS)
        return fail(error, "superres snapshot copy failed");
      int total = 0;
      CUdeviceptr job_mem = 0, pref_mem = 0;
      if (!upload_jobs(plan.upscale, &job_mem, &pref_mem, &total,
                       [](const Av1gUpscaleJob& j) { return j.rows * j.upscaled_w; }))
        return fail(error, "upscale job upload failed");
      Launcher a;
      add_mem(a, job_mem);
      add_mem(a, pref_mem);
      add_int(a, static_cast<int>(plan.upscale.size()));
      add_int(a, total);
      add_mem(a, scratch_);
      add_mem(a, frame_);
      add_mem(a, filters_);
      add_int(a, maxv);
      if (!launch("av1g_k_upscale", total, a)) return fail(error, "av1g_k_upscale failed");
      ++launches;
      /* Restoration works on the upscaled picture, so the pre-CDEF snapshot is
       * resampled by the same jobs into a buffer of its own. */
      if (uses_lr) {
        Launcher a;
        add_mem(a, job_mem);
        add_mem(a, pref_mem);
        add_int(a, static_cast<int>(plan.upscale.size()));
        add_int(a, total);
        add_mem(a, lr_curr_);
        add_mem(a, lr_prev_up_);
        add_mem(a, filters_);
        add_int(a, maxv);
        if (!launch("av1g_k_upscale", total, a)) return fail(error, "av1g_k_upscale failed");
        ++launches;
      }
    }

    /* ---- loop restoration (7.17) ------------------------------------------- */
    /* Every job computes its samples from the two snapshots, never from samples
     * another unit has written, so the whole stage is one launch. */
    if (uses_lr) {
      /* the CDEF output frame_ holds is snapshot last: the upscaler above has
       * already replaced the coded picture with the display one */
      if (driver_.MemcpyDtoD(lr_cdef_, frame_, g.buffer_size * sizeof(uint16_t)) !=
          cuda::CUDA_SUCCESS)
        return fail(error, "restoration snapshot copy failed");
      int total = 0;
      CUdeviceptr job_mem = 0, pref_mem = 0;
      if (!upload_jobs(plan.restoration, &job_mem, &pref_mem, &total,
                       [](const Av1gLrJob& j) { return j.w * j.h; }))
        return fail(error, "restoration job upload failed");
      Launcher a;
      add_mem(a, job_mem);
      add_mem(a, pref_mem);
      add_int(a, static_cast<int>(plan.restoration.size()));
      add_int(a, total);
      add_mem(a, lr_cdef_);
      add_mem(a, plan.upscale.empty() ? lr_curr_ : lr_prev_up_);
      add_mem(a, frame_);
      add_int(a, maxv);
      if (!launch("av1g_k_lr", total, a)) return fail(error, "av1g_k_lr failed");
      ++launches;
    }

    /* ---- reference (7.20) -------------------------------------------------- */
    /* The reference frame update stores the decoded picture, which is the one
     * *before* the film grain: grain belongs to the output process (7.18.3), so
     * the reference copy is issued first and the grain kernels below change the
     * display picture only. */
    int slot = -1;
    if (keep_reference) {
      if (forced_slot >= 0) {
        /* an explicit slot reserved ahead of time by a pipelined caller */
        slot = forced_slot;
      } else {
        std::lock_guard<std::mutex> lk(slot_mutex_);
        slot = allocate_slot();
        if (slot < 0) return fail(error, "CUDA reference pool exhausted");
        slots_[slot] = true; /* mark atomically with the allocation */
      }
      cuda::CUresult rc = driver_.MemcpyDtoD(
          ref_pool_ + static_cast<size_t>(slot) * g.buffer_size * sizeof(uint16_t), frame_,
          g.buffer_size * sizeof(uint16_t));
      if (rc != cuda::CUDA_SUCCESS) {
        if (forced_slot < 0) {
          std::lock_guard<std::mutex> lk(slot_mutex_);
          slots_[slot] = false;
        }
        return fail(error, "reference copy failed");
      }
      if (driver_.CtxSynchronize() != cuda::CUDA_SUCCESS)
        return fail(error, "reference copy did not complete");
    }
    /* ---- film grain (7.18.3) ----------------------------------------------- */
    if (!plan.grain_chroma.empty() || !plan.grain_luma.empty()) {
      /* The noise image is built on the host (the sequential half of the
       * synthesis) and uploaded once; the two passes below only launch. */
      const size_t noise_size = plan.grain_noise.noise.size() * sizeof(short);
      if (!ensure(&grain_, &grain_cap_, noise_size) ||
          !ensure(&lut_, &lut_cap_, sizeof(plan.grain_noise.lut)))
        return fail(error, "out of device memory for the film grain tables");
      if (!upload_h2d(grain_, plan.grain_noise.noise.data(), noise_size) ||
          !upload_h2d(lut_, plan.grain_noise.lut, sizeof(plan.grain_noise.lut)))
        return fail(error, "film grain table upload failed");
      /* chroma first: its noise is scaled by the luma of the picture before any
       * grain was added, so the luma pass has to come after it */
      for (const std::vector<Av1gGrainJob>* pass : {&plan.grain_chroma, &plan.grain_luma}) {
        if (pass->empty()) continue;
        int total = 0;
        CUdeviceptr job_mem = 0, pref_mem = 0;
        if (!upload_jobs(*pass, &job_mem, &pref_mem, &total,
                         [](const Av1gGrainJob& j) { return j.w * j.h; }))
          return fail(error, "grain job upload failed");
        Launcher a;
        add_mem(a, job_mem);
        add_mem(a, pref_mem);
        add_int(a, static_cast<int>(pass->size()));
        add_int(a, total);
        add_mem(a, frame_);
        add_mem(a, frame_);
        add_mem(a, grain_);
        add_mem(a, lut_);
        add_int(a, maxv);
        if (!launch("av1g_k_grain", total, a)) return fail(error, "av1g_k_grain failed");
        ++launches;
      }
    }

    /* ---- the display copy of the slot's picture -------------------------- */
    /* The reference store above ran before film grain, because a reference is
     * the picture *without* it (7.18.3): the grain belongs to the output
     * process. A caller that packs an *older* slot's picture (Decoder::to_nv12
     * from outside the pipeline) wants the picture *with* the grain, so the
     * slot is refreshed with the finished display picture.
     *
     * It is off by default: the pipeline's pack runs on the decode stream right
     * after the decode, so it reads frame_ (the post-grain picture) directly and
     * this copy would be a second 38 MB device-to-device transfer a frame for
     * nobody. AV1GPU_SLOT_DISPLAY_COPY=1 turns it back on. */
    static const bool display_copy = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_SLOT_DISPLAY_COPY", f, sizeof(f)) > 0 &&
             f[0] != '0';
    }();
    if (display_copy && keep_reference && slot >= 0 &&
        driver_.MemcpyDtoD(
            ref_pool_ + static_cast<size_t>(slot) * g.buffer_size * sizeof(uint16_t), frame_,
            g.buffer_size * sizeof(uint16_t)) != cuda::CUDA_SUCCESS)
      return fail(error, "display copy failed");

    /* The single point where a device side fault becomes visible. OpenCL's
     * clFinish is checked the same way: a kernel that reads out of bounds - a
     * stale slot offset, an empty job array - must fail the frame loudly
     * instead of handing back whatever the device happened to leave behind. */
    {
      const cuda::CUresult rc = driver_.CtxSynchronize();
      if (rc != cuda::CUDA_SUCCESS) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "CUDA kernel execution failed (cuCtxSynchronize = %d)", rc);
        return fail(error, buf);
      }
    }
    const double t2 = now_ms();
    if (out) {
      if (!out->matches(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                        g.frame.ss_y))
        out->allocate(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                      g.frame.ss_y);
      /* The read back goes into the page locked buffer: a pageable device to
       * host copy of the whole picture runs at a third of the speed, and this
       * is several MB for every frame of the stream. */
      const size_t bytes = g.buffer_size * sizeof(uint16_t);
      uint16_t* staging = nullptr;
      if (ensure_pinned(&pinned_rb_, &pinned_rb_cap_, bytes)) {
        staging = static_cast<uint16_t*>(pinned_rb_);
      } else {
        if (staging_.size() < g.buffer_size) staging_.resize(g.buffer_size);
        staging = staging_.data();
      }
      if (driver_.MemcpyDtoH(staging, frame_, bytes) != cuda::CUDA_SUCCESS)
        return fail(error, "frame read back failed");
      for (int p = 0; p < 3; ++p) {
        const int w = g.out_w[p];
        const int h = g.out_h[p];
        for (int y = 0; y < h; ++y)
          std::memcpy(out->plane(p) + static_cast<size_t>(y) * out->stride[p],
                      staging + g.plane_off[p] + static_cast<size_t>(y) * g.stride[p],
                      static_cast<size_t>(w) * sizeof(uint16_t));
      }
    }
    const double t3 = now_ms();

    /* temporary profiling: how much of reconstruct was CPU prep vs GPU wait */
    {
      static double acc_sync = 0.0, acc_filter = 0.0, acc_rec = 0.0, acc_rb = 0.0;
      static long long acc_bytes = 0;
      static long n = 0;
      acc_sync += double(qf.QuadPart ? 0 : 0); /* no-op keeps qf referenced */
      acc_filter += t2 - t1;
      acc_rec += t1 - t0;
      acc_rb += t3 - t2;
      size_t waves = plan.waves.size();
      /* Real host to device volume this frame: measured in upload_h2d, so the
       * device dequant path (which sends levels instead of the raster pool)
       * shows up honestly. */
      acc_bytes += h2d_bytes;
      size_t job_count = 0;
      for (const WaveJobs& w : plan.waves)
        job_count += w.inter.size() + w.intra.size() + w.itx.size() + w.residual.size();
      job_count += plan.deblock_v.size() + plan.deblock_h.size() + plan.cdef.size();
      ++n;
      if (n % 50 == 0) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "perf backend: reconstruct %.1f, "
                      "waves %zu, filter(sync incl) %.1f, "
                      "readback %.1f, jobs %zu, uploads %lld KB "
                      "(coef %lld, dq %lld, wave %lld, batch %lld [inter %lld, itx %lld, res %lld], misc %lld), "
                      "coef %.1f MB, levels %.1f MB, res %.1f MB, %ld frames",
                      acc_rec / n, waves,
                      acc_filter / n, acc_rb / n, job_count, acc_bytes / 1024,
                      h2d_tag_bytes_[kH2dTagCoef] / 1024,
                      h2d_tag_bytes_[kH2dTagDequant] / 1024,
                      h2d_tag_bytes_[kH2dTagWave] / 1024,
                      h2d_tag_bytes_[kH2dTagBatch] / 1024,
                      h2d_tag_bytes_[kH2dTagInter] / 1024,
                      h2d_tag_bytes_[kH2dTagItx] / 1024,
                      h2d_tag_bytes_[kH2dTagRes] / 1024,
                      h2d_tag_bytes_[kH2dTagMisc] / 1024,
                      static_cast<double>(plan.coefficients.size()) * sizeof(int32_t) /
                          (1024.0 * 1024.0),
                      static_cast<double>(plan.levels.size()) * sizeof(int16_t) /
                          (1024.0 * 1024.0),
                      static_cast<double>(plan.residual.size()) * sizeof(int32_t) /
                          (1024.0 * 1024.0),
                      n);
        OutputDebugStringA("av1gpu: ");
        OutputDebugStringA(msg);
        OutputDebugStringA("\n");
        char path[MAX_PATH];
        if (GetTempPathA(MAX_PATH, path)) {
          strncat_s(path, "av1gpu_ds.log", _TRUNCATE);
          if (FILE* f = std::fopen(path, "a")) {
            SYSTEMTIME st2;
            GetLocalTime(&st2);
            std::fprintf(f, "%02u:%02u:%02u.%03u pid %lu thr %lu  %s\n", st2.wHour,
                         st2.wMinute, st2.wSecond, st2.wMilliseconds,
                         GetCurrentProcessId(), GetCurrentThreadId(), msg);
            std::fclose(f);
          }
        }
      }
    }


    if (out_slot) *out_slot = slot;
    {
      std::lock_guard<std::mutex> lk(slot_mutex_);
      last_slot_ = slot;
    }

    if (stats) {
      stats->blocks = plan.blocks;
      stats->waves = static_cast<int>(plan.waves.size());
      stats->kernel_launches = launches;
      stats->reconstruct_ms = t1 - t0;
      stats->filter_ms = t2 - t1;
      stats->readback_ms = t3 - t2;
      stats->total_ms = now_ms() - t0;
      /* a one-off stall is what a viewer sees as a hitch; log it on the spot
       * instead of waiting for the 50 frame summary */
      if (stats->filter_ms > 60.0 || stats->readback_ms > 60.0) {
        char path[MAX_PATH];
        if (GetTempPathA(MAX_PATH, path)) {
          strncat_s(path, "av1gpu_ds.log", _TRUNCATE);
          if (FILE* f = std::fopen(path, "a")) {
            SYSTEMTIME st2;
            GetLocalTime(&st2);
            std::fprintf(f, "%02u:%02u:%02u.%03u pid %lu thr %lu  SLOW backend rec %.1f "
                            "filt %.1f rb %.1f waves %d\n",
                         st2.wHour, st2.wMinute, st2.wSecond, st2.wMilliseconds,
                         GetCurrentProcessId(), GetCurrentThreadId(), stats->reconstruct_ms,
                         stats->filter_ms, stats->readback_ms, stats->waves);
            std::fclose(f);
          }
        }
      }
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
    /* The reference pool holds the picture *before* the film grain (7.20) and
     * frame_ the one after it, so a request for the frame that was just decoded
     * reads frame_ - the reference is only the fallback for an older slot. */
    CUdeviceptr source = frame_;
    int base = 0;
    if (slot >= 0) {
      std::lock_guard<std::mutex> lk(slot_mutex_);
      if (slot >= kReferenceSlots || !slots_[slot]) {
        *error = "invalid reference slot";
        return false;
      }
      if (slot != last_slot_) {
        source = ref_pool_;
        base = slot * static_cast<int>(g.buffer_size);
      }
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
    if (driver_.CtxSynchronize() != cuda::CUDA_SUCCESS) {
      *error = "av1g_k_color failed";
      return false;
    }
    rgb->resize(rgb_bytes);
    if (driver_.MemcpyDtoH(rgb->data(), rgb_, rgb_bytes) != cuda::CUDA_SUCCESS) {
      *error = "RGB read back failed";
      return false;
    }
    return true;
  }

  /* ---- the NV12 pack, in two halves ------------------------------------- *
   * The synchronous pack below does the work and then waits for it. The
   * asynchronous half does the same work but leaves the wait to another thread:
   * the kernel runs on the decode stream (so it is ordered after the decode it
   * packs, and it reads the slot's picture before any later frame can reuse
   * it), and the copy runs on a side stream into page locked memory, which the
   * decode of the *next* frame overlaps with. The output stage waits on the
   * copy's event and hands the bytes to its sink, so the worker's critical path
   * never pays for the copy at all.
   *
   * Each ring entry owns its own device output buffer: the copy of frame N and
   * the kernel of frame N+1 are both free to run, and they must not share one. */
  static constexpr int kNv12Ring = 3;
  struct Nv12Ring {
    CUstream stream = nullptr;   /* the side stream the copy runs on */
    CUevent kernel_done;         /* recorded on the decode stream */
    CUevent copy_done;           /* recorded on the side stream */
    CUdeviceptr out = 0;         /* this entry's own device output buffer */
    size_t out_cap = 0;
    void* pinned = nullptr;      /* its page locked destination */
    size_t pinned_cap = 0;
    size_t bytes = 0;
    bool busy = false;
  };
  Nv12Ring nv12_ring_[kNv12Ring];
  bool nv12_ring_broken_ = false; /* no async copy or no stream: stay sync */

  bool nv12_prepare(Nv12Ring& r, size_t bytes, std::string* error) {
    if (!driver_.MemAllocHost || !driver_.MemFreeHost) return false;
    if (r.pinned_cap < bytes) {
      if (r.pinned) driver_.MemFreeHost(r.pinned);
      r.pinned = nullptr;
      r.pinned_cap = 0;
      if (driver_.MemAllocHost(&r.pinned, bytes) != cuda::CUDA_SUCCESS || !r.pinned)
        return false;
      r.pinned_cap = bytes;
    }
    if (r.out_cap < bytes) {
      CUdeviceptr tmp = 0;
      if (driver_.MemAlloc(&tmp, bytes) != cuda::CUDA_SUCCESS) return false;
      if (r.out) driver_.MemFree(r.out);
      r.out = tmp;
      r.out_cap = bytes;
    }
    (void)error;
    return true;
  }

  int nv12_pack_begin(int slot, size_t pitch, std::string* error,
                      const PlanLayout* layout) override {
    std::lock_guard<std::mutex> driver_guard(driver_mutex_);
    error->clear();
    if (layout_.buffer_size == 0) {
      *error = "prepare() has not been called";
      return -1;
    }
    if (nv12_ring_broken_ || !driver_.MemcpyDtoHAsync || !driver_.StreamCreate ||
        !driver_.StreamWaitEvent || !driver_.EventRecord || !driver_.EventSynchronize)
      return -1; /* the caller packs synchronously instead */
    Nv12Ring* r = nullptr;
    for (int i = 0; i < kNv12Ring; ++i)
      if (!nv12_ring_[i].busy) {
        r = &nv12_ring_[i];
        break;
      }
    if (!r) return -1; /* every entry is in flight: pack synchronously */
    if (!r->stream) {
      /* non-blocking: the copy must not wait on the decode stream's work */
      if (driver_.StreamCreate(&r->stream, 1 /* CU_STREAM_NON_BLOCKING */) !=
              cuda::CUDA_SUCCESS ||
          !r->stream) {
        nv12_ring_broken_ = true;
        return -1;
      }
      if (driver_.EventCreate(&r->kernel_done, 0) != cuda::CUDA_SUCCESS ||
          driver_.EventCreate(&r->copy_done, 0) != cuda::CUDA_SUCCESS) {
        nv12_ring_broken_ = true;
        return -1;
      }
    }
    {
      std::lock_guard<std::mutex> lk(slot_mutex_);
      if (slot >= kReferenceSlots || !slots_[slot]) {
        *error = "invalid reference slot";
        return -1;
      }
    }
    /* The layout the slot's picture was decoded with: the engine records it
     * per slot, because a pipeline packs slot N while frame N+1 has already
     * been prepared. */
    const PlanLayout& g = layout ? *layout : layout_;
    CUdeviceptr source = frame_;
    int base = 0;
    if (slot != last_slot_) {
      source = ref_pool_;
      base = slot * static_cast<int>(g.buffer_size);
    }
    const size_t bytes = nv12_buffer_size(g.frame.width, g.frame.height, g.frame.ss_x,
                                          g.frame.ss_y, pitch);
    if (bytes == 0) {
      *error = "NV12 output needs a 4:2:0 frame and an even pitch at least as wide";
      return -1;
    }
    if (!nv12_prepare(*r, bytes, error)) {
      *error = "out of memory for the streamed NV12 pack";
      return -1;
    }
    Av1gNv12Job job;
    std::memset(&job, 0, sizeof(job));
    job.y_off = base + g.plane_off[0];
    job.y_stride = g.stride[0];
    job.u_off = base + g.plane_off[1];
    job.v_off = base + g.plane_off[2];
    job.c_stride = g.stride[1];
    job.w = g.frame.width;
    job.h = g.frame.height;
    job.pitch = static_cast<int>(pitch);
    job.ss_x = g.frame.ss_x;
    job.ss_y = g.frame.ss_y;
    job.bitdepth = g.frame.bit_depth;
    std::vector<Av1gNv12Job> jobs(1, job);
    CUdeviceptr job_mem = 0, pref_mem = 0;
    int total = 0;
    if (!upload_jobs(jobs, &job_mem, &pref_mem, &total, [](const Av1gNv12Job& j) {
          return j.pitch * (j.h + ((j.h + (1 << j.ss_y) - 1) >> j.ss_y));
        })) {
      *error = "NV12 job upload failed";
      return -1;
    }
    /* The pack kernel goes on the decode stream (active_stream_): it is then
     * ordered after this frame's kernels, and ahead of any later frame that
     * reuses the slot, so the picture it reads is this frame's. */
    Launcher a;
    add_mem(a, job_mem);
    add_mem(a, pref_mem);
    add_int(a, 1);
    add_int(a, total);
    add_mem(a, source);
    add_mem(a, source);
    add_mem(a, source);
    add_mem(a, r->out);
    if (!launch("av1g_k_nv12", total, a)) {
      *error = "av1g_k_nv12 failed";
      return -1;
    }
    /* The pack kernel and its copy both go on the decode stream (the legacy
     * stream the decode used): the kernel is then ordered after this frame's
     * decode and ahead of any later frame that reuses the slot, and the copy is
     * ordered after the kernel. Putting the copy on a non-blocking side stream
     * instead - and ordering it with an event recorded on the legacy stream -
     * does NOT work: an event recorded on the legacy stream does not order a
     * non-blocking stream's work (that is the legacy stream's whole point), so
     * the copy could read a half-written buffer. The cost is that the worker's
     * next context sync waits for the copy, which at ~12 MB is a fraction of
     * the ~38 MB picture read back it replaces. */
    if (driver_.MemcpyDtoHAsync(r->pinned, r->out, bytes, active_stream_) !=
            cuda::CUDA_SUCCESS ||
        driver_.EventRecord(r->copy_done, active_stream_) != cuda::CUDA_SUCCESS) {
      *error = "the streamed NV12 pack could not be started";
      nv12_ring_broken_ = true;
      return -1;
    }
    r->bytes = bytes;
    r->busy = true;
    return static_cast<int>(r - nv12_ring_);
  }

  /* Waits for a streamed pack's copy and, when `nv12` is not null, fills it.
   * The entry stays busy: a caller that keeps the bytes for a later re-show
   * (a show_existing_frame) waits here and releases afterwards, and one that
   * is done releases straight away. */
  bool nv12_pack_wait(int handle, std::vector<uint8_t>* nv12,
                      std::string* error) override {
    error->clear();
    if (handle < 0 || handle >= kNv12Ring) {
      *error = "not a streamed NV12 pack";
      return false;
    }
    Nv12Ring& r = nv12_ring_[handle];
    if (!r.busy) {
      *error = "the streamed NV12 pack is not in flight";
      return false;
    }
    if (driver_.EventSynchronize(r.copy_done) != cuda::CUDA_SUCCESS) {
      *error = "the streamed NV12 pack failed";
      return false;
    }
    if (nv12) {
      nv12->resize(r.bytes);
      std::memcpy(nv12->data(), r.pinned, r.bytes);
    }
    return true;
  }

  /* Hands a streamed pack's entry back. The copy reads the entry's page
   * locked buffer, so it is waited for first even when the bytes are dropped
   * (a flush). */
  void nv12_pack_release(int handle) override {
    if (handle < 0 || handle >= kNv12Ring) return;
    Nv12Ring& r = nv12_ring_[handle];
    if (!r.busy) return;
    driver_.EventSynchronize(r.copy_done);
    r.busy = false;
  }

  bool to_nv12(int slot, size_t pitch, std::vector<uint8_t>* nv12,
               std::string* error, const PlanLayout* layout) override {
    /* The same shared staging buffers as a decode go through, so the same lock
     * applies: a pipeline's output stage can call this while the worker decodes
     * the next frame. */
    std::lock_guard<std::mutex> driver_guard(driver_mutex_);
    error->clear();
    if (layout_.buffer_size == 0) {
      *error = "prepare() has not been called";
      return false;
    }
    /* The layout the slot's picture was decoded with, not the one the last
     * prepare() produced: a pipeline packs the picture of slot N on its output
     * stage while frame N+1 has already been prepared, and for a stream whose
     * frames do not share a geometry the two differ. */
    const PlanLayout& g = layout ? *layout : layout_;
    /* same source rule as to_rgb (see the CPU backend) */
    CUdeviceptr source = frame_;
    int base = 0;
    if (slot >= 0) {
      std::lock_guard<std::mutex> lk(slot_mutex_);
      if (slot >= kReferenceSlots || !slots_[slot]) {
        *error = "invalid reference slot";
        return false;
      }
      if (slot != last_slot_) {
        source = ref_pool_;
        base = slot * static_cast<int>(g.buffer_size);
      }
    }

    const size_t bytes = nv12_buffer_size(g.frame.width, g.frame.height, g.frame.ss_x,
                                          g.frame.ss_y, pitch);
    if (bytes == 0) {
      *error = "NV12 output needs a 4:2:0 frame and an even pitch at least as wide";
      return false;
    }

    Av1gNv12Job job;
    std::memset(&job, 0, sizeof(job));
    job.y_off = base + g.plane_off[0];
    job.y_stride = g.stride[0];
    job.u_off = base + g.plane_off[1];
    job.v_off = base + g.plane_off[2];
    job.c_stride = g.stride[1];
    job.w = g.frame.width;
    job.h = g.frame.height;
    job.pitch = static_cast<int>(pitch);
    job.ss_x = g.frame.ss_x;
    job.ss_y = g.frame.ss_y;
    job.bitdepth = g.frame.bit_depth;

    /* rgb_ is a byte buffer of the output size, so NV12 reuses it */
    if (!ensure(&rgb_, &rgb_cap_, bytes)) {
      *error = "out of device memory for NV12 output";
      return false;
    }
    std::vector<Av1gNv12Job> jobs(1, job);
    CUdeviceptr job_mem = 0, pref_mem = 0;
    int total = 0;
    if (!upload_jobs(jobs, &job_mem, &pref_mem, &total, [](const Av1gNv12Job& j) {
          return j.pitch * (j.h + ((j.h + (1 << j.ss_y) - 1) >> j.ss_y));
        })) {
      *error = "NV12 job upload failed";
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
    if (!launch("av1g_k_nv12", total, a)) {
      *error = "av1g_k_nv12 failed";
      return false;
    }
    if (driver_.CtxSynchronize() != cuda::CUDA_SUCCESS) {
      *error = "av1g_k_nv12 failed";
      return false;
    }
    nv12->resize(bytes);
    /* The copy goes through page locked memory: a direct device to host copy
     * into a pageable vector runs at a third of the speed, and this is several
     * MB every shown frame - the same rule the picture read back follows. */
    if (ensure_pinned(&pinned_nv12_, &pinned_nv12_cap_, bytes)) {
      if (driver_.MemcpyDtoH(pinned_nv12_, rgb_, bytes) != cuda::CUDA_SUCCESS) {
        *error = "NV12 read back failed";
        return false;
      }
      std::memcpy(nv12->data(), pinned_nv12_, bytes);
      return true;
    }
    if (driver_.MemcpyDtoH(nv12->data(), rgb_, bytes) != cuda::CUDA_SUCCESS) {
      *error = "NV12 read back failed";
      return false;
    }
    return true;
  }

  void reset() override {
    std::lock_guard<std::mutex> lk(slot_mutex_);
    for (int i = 0; i < kReferenceSlots; ++i) slots_[i] = false;
    last_slot_ = -1;
  }

 private:
  /* One pool slot per AV1 reference slot plus the pictures still in flight: a
   * pipelined caller keeps one slot reserved per queued frame, and a deeper
   * queue is what lets the decoder run ahead of the renderer far enough to
   * hide a keyframe's cost (see kPipeDepth in the decode core). The pool needs
   * headroom over 8 + kPipeDepth: a queued frame holds the slot it reserved and
   * the slots it displaced until its job has run, and when the pool runs out
   * the decode core drains the whole queue for one slot - a hitch. */
  static constexpr int kReferenceSlots = 32;
  /* the smallest device allocation ensure() hands out; see ensure() */
  static constexpr size_t kMinAllocation = 1024;

  void shutdown() {
    if (module_ && driver_.ModuleLoadDataEx) {
      /* no cuModuleUnload in the minimal set: the context teardown frees it */
    }
    /* every allocation: the palette map and the second prefix table were
     * missing from this list, so a backend that decoded a palette stream leaked
     * them when it was destroyed */
    CUdeviceptr buffers[] = {frame_,   scratch_,   coef_,   tmp_,    residual_,
                             filters_, ref_pool_,  grain_,  lut_,    rgb_,
                             job_,     pref_,      pref_b_, palette_,
                              lr_curr_, lr_cdef_,   lr_prev_up_,
                              job_arena_, pref_arena_, batch_dev_};
    for (CUdeviceptr b : buffers)
      if (b && driver_.MemFree) driver_.MemFree(b);
    frame_ = scratch_ = coef_ = tmp_ = residual_ = 0;
    filters_ = ref_pool_ = grain_ = 0;
    lut_ = rgb_ = job_ = pref_ = pref_b_ = palette_ = 0;
    batch_dev_ = 0;
    batch_dev_cap_ = 0;
    lr_curr_ = lr_cdef_ = lr_prev_up_ = 0;
    job_arena_ = pref_arena_ = 0;
    job_arena_cap_ = pref_arena_cap_ = 0;
    /* tear down any CUDA graph */
    if (graph_exec_ && driver_.GraphExecDestroy)
      driver_.GraphExecDestroy(graph_exec_);
    if (graph_ && driver_.GraphDestroy) driver_.GraphDestroy(graph_);
    graph_ = nullptr;
    graph_exec_ = nullptr;
    graph_built_ = false;
    graph_kind_mask_ = 0;
    graph_maxv_ = 0;
    palette_cap_ = 0;
    if (pinned_ && driver_.MemFreeHost) driver_.MemFreeHost(pinned_);
    pinned_ = nullptr;
    pinned_cap_ = 0;
    if (pinned_rb_ && driver_.MemFreeHost) driver_.MemFreeHost(pinned_rb_);
    pinned_rb_ = nullptr;
    pinned_rb_cap_ = 0;
    if (pinned_nv12_ && driver_.MemFreeHost) driver_.MemFreeHost(pinned_nv12_);
    pinned_nv12_ = nullptr;
    pinned_nv12_cap_ = 0;
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
    /* The intra kernel has two mappings: the flat one (a thread per row over
     * the whole wave) and the block one (a thread block per transform block,
     * with the edge shared through shared memory).  Which one wins depends on
     * the block sizes in the stream, so it is picked at run time and the
     * choice is part of the cache key. */
    static const int intra_block_per_job = [] {
      char f[8] = {0};
      if (GetEnvironmentVariableA("AV1GPU_INTRA_BLOCK", f, sizeof(f)) == 0) return 0;
      return atoi(f) != 0 ? 1 : 0;
    }();
    const std::string source =
        "#define AV1G_INTRA_BLOCK_PER_JOB " + std::to_string(intra_block_per_job) + "\n" +
        kernel_source::cuda();
    /* NVRTC compiles this kernel set in seconds and the driver's PTX JIT adds
     * several more, on every launch. The result is a pure function of the
     * source, the architecture and the toolkit, so cache it next to the user's
     * other data and load the cached binary when it is there. */
    const std::string cache_dir = kernel_cache_dir();
    std::string key;
    {
      uint64_t h = 1469598103934665603ull;
      const auto mix = [&](const char* p, size_t n) {
        for (size_t i = 0; i < n; ++i) {
          h ^= static_cast<unsigned char>(p[i]);
          h *= 1099511628211ull;
        }
      };
      mix(source.data(), source.size());
      char arch_key[64];
      std::snprintf(arch_key, sizeof(arch_key), "sm %d.%d", compute_major_, compute_minor_);
      mix(arch_key, std::strlen(arch_key));
      mix("v5", 2); /* bump when the cache format changes */
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
      key = buf;
    }
    const std::string cubin_path = cache_dir + "/av1gpu_" + key + ".cubin";
    const std::string ptx_path = cache_dir + "/av1gpu_" + key + ".ptx";

    /* a cached binary, if one is usable */
    for (const std::string* path : {&cubin_path, &ptx_path}) {
      std::vector<char> blob;
      if (!read_binary(*path, &blob) || blob.empty()) continue;
      if (driver_.ModuleLoadDataEx(&module_, blob.data(), 0, nullptr, nullptr) ==
          cuda::CUDA_SUCCESS) {
        if (options_.verbose) error->append("loaded cached kernels from " + *path + "\n");
        return true;
      }
      /* stale (a driver update can invalidate it): drop it and rebuild */
      std::remove(path->c_str());
    }

    /* NVRTC spells the minor version as one digit from 5 upwards (8.6 becomes
     * compute_86) and ignores it below that (6.0 becomes compute_60), so the
     * minor is only added when it is at least five - and "compute_100" for a
     * 10.0 part needs the full major printed, which is why this is not just two
     * characters. */
    if (compute_major_ <= 0) {
      *error = "the driver reported no compute capability (cuDeviceGetAttribute failed)";
      return false;
    }
    char arch[48];
    if (compute_minor_ >= 5)
      std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", compute_major_,
                    compute_minor_);
    else
      std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d0", compute_major_);

    void* program = nullptr;
    std::vector<char> binary;
    bool is_cubin = false;
    /* Prefer cubin (no driver JIT on load = faster startup). Graphs are
     * disabled by default on WDDM 13.1, so cubin is safe. Set
     * AV1GPU_FORCE_PTX=1 to force PTX (for graph experiments). */
    static const bool force_ptx = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_FORCE_PTX", f, sizeof(f)) > 0 &&
             f[0] == '1';
    }();
    if (!force_ptx && nvrtc_.GetCUBINSize && nvrtc_.GetCUBIN) {
      if (nvrtc_.CreateProgram(&program, source.c_str(), "av1gpu.cu", 0, nullptr, nullptr) == 0) {
        char sm_arch[48];
        if (compute_minor_ >= 5)
          std::snprintf(sm_arch, sizeof(sm_arch), "--gpu-architecture=sm_%d%d", compute_major_,
                        compute_minor_);
        else
          std::snprintf(sm_arch, sizeof(sm_arch), "--gpu-architecture=sm_%d0", compute_major_);
        const char* sm_options[] = {sm_arch, "-default-device"};
        if (nvrtc_.CompileProgram(program, 2, sm_options) == 0) {
          size_t cubin_size = 0;
          if (nvrtc_.GetCUBINSize(program, &cubin_size) == 0 && cubin_size > 0) {
            binary.assign(cubin_size, 0);
            if (nvrtc_.GetCUBIN(program, binary.data()) == 0) is_cubin = true;
          }
        }
        nvrtc_.DestroyProgram(&program);
        program = nullptr;
      }
    }

    if (!is_cubin) {
      /* no cubin support (or the sm_ build failed): compile the portable
       * compute_XX program and take its PTX, which the driver JITs on load */
      binary.clear();
      if (nvrtc_.CreateProgram(&program, source.c_str(), "av1gpu.cu", 0, nullptr, nullptr) != 0) {
        *error = "nvrtcCreateProgram failed";
        return false;
      }
      const char* options[] = {arch, "-default-device"};
      const int rc = nvrtc_.CompileProgram(program, 2, options);
      size_t log_size = 0;
      nvrtc_.GetProgramLogSize(program, &log_size);
      std::string log(log_size + 1, '\0');
      if (log_size) nvrtc_.GetProgramLog(program, &log[0]);
      if (rc != 0) {
        *error = std::string("NVRTC build failed (") + arch + "):\n" + log;
        nvrtc_.DestroyProgram(&program);
        return false;
      }
      if (options_.verbose) error->append(log);
      size_t ptx_size = 0;
      if (nvrtc_.GetPTXSize(program, &ptx_size) != 0 || ptx_size == 0) {
        nvrtc_.DestroyProgram(&program);
        *error = "NVRTC produced no PTX";
        return false;
      }
      binary.assign(ptx_size + 1, 0);
      if (nvrtc_.GetPTX(program, binary.data()) != 0) {
        nvrtc_.DestroyProgram(&program);
        *error = "NVRTC GetPTX failed";
        return false;
      }
      binary.resize(ptx_size);
      nvrtc_.DestroyProgram(&program);
      program = nullptr;
    }

    const cuda::CUresult load =
        driver_.ModuleLoadDataEx(&module_, binary.data(), 0, nullptr, nullptr);
    if (load != cuda::CUDA_SUCCESS) {
      *error = "cuModuleLoadDataEx failed (" + std::to_string(load) + ") for " + arch;
      return false;
    }
    /* write it for the next launch (a failure to write is not fatal) */
    if (!cache_dir.empty()) write_binary(is_cubin ? cubin_path : ptx_path, binary);
    return true;
  }

  /* cuMemAlloc rejects a request of zero bytes, and callers legitimately ask
   * for zero: a frame with no palette coded block has an empty palette map, and
   * passing that straight through failed the whole decode with "palette map
   * upload failed". Allocate a minimum block instead, so the buffer is always a
   * valid (if unused) device pointer. */
  bool ensure(CUdeviceptr* buffer, size_t* capacity, size_t bytes) {
    const size_t size = std::max<size_t>(bytes, kMinAllocation);
    if (*buffer && *capacity >= size) return true;
    if (*buffer) driver_.MemFree(*buffer);
    *buffer = 0;
    if (driver_.MemAlloc(buffer, size) != cuda::CUDA_SUCCESS) {
      *capacity = 0;
      return false;
    }
    *capacity = size;
    return true;
  }

  /* Large host to device copies go through one page-locked staging buffer:
   * pageable cuMemcpyHtoD runs at a fraction of the bus speed, and the job
   * arrays alone are tens of megabytes per 1080p frame. */
  /* Grows a page locked staging buffer so it holds `bytes`. */
  bool ensure_pinned(void** buffer, size_t* capacity, size_t bytes) {
    if (!driver_.MemAllocHost || !driver_.MemFreeHost) return false;
    if (*capacity >= bytes && *buffer) return true;
    if (*buffer) driver_.MemFreeHost(*buffer);
    *buffer = nullptr;
    *capacity = 0;
    const size_t want = bytes + bytes / 2; /* headroom for the next frames */
    if (driver_.MemAllocHost(buffer, want) != cuda::CUDA_SUCCESS || !*buffer) {
      *buffer = nullptr;
      *capacity = 0;
      return false;
    }
    *capacity = want;
    return true;
  }

  bool upload_h2d(CUdeviceptr dst, const void* src, size_t bytes) {
    if (bytes == 0) return true;
    h2d_bytes += static_cast<long long>(bytes); /* measured, for the log line */
    if (h2d_tag_ >= 0 && h2d_tag_ < 8)
      h2d_tag_bytes_[h2d_tag_] += static_cast<long long>(bytes);
    if (!driver_.MemAllocHost || !driver_.MemFreeHost)
      return driver_.MemcpyHtoD(dst, src, bytes) == cuda::CUDA_SUCCESS;
    if (!ensure_pinned(&pinned_, &pinned_cap_, bytes))
      return driver_.MemcpyHtoD(dst, src, bytes) == cuda::CUDA_SUCCESS;
    std::memcpy(pinned_, src, bytes);
    return driver_.MemcpyHtoD(dst, pinned_, bytes) == cuda::CUDA_SUCCESS;
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
    if (!upload_h2d(job_, jobs.data(), jobs.size() * sizeof(JobT))) return false;
    return upload_h2d(pref_, prefix.data(), prefix.size() * sizeof(int));
  }

  void add_mem(Launcher& a, CUdeviceptr value) { a.add_ptr(value); }
  void add_int(Launcher& a, int value) { a.add_int(value); }

  /* The kernel's CUfunction, cached: both the linear launch path and the graph
   * builder need it. */
  CUfunction function_for(const char* name) {
    auto it = functions_.find(name);
    if (it != functions_.end()) return it->second;
    CUfunction function = nullptr;
    if (driver_.ModuleGetFunction(&function, module_, name) != cuda::CUDA_SUCCESS)
      return nullptr;
    functions_[name] = function;
    return function;
  }

  /* launch on a specific stream (nullptr = default stream) */
  bool launch_on(CUstream stream, const char* name, int elements, Launcher& launcher) {
    if (elements <= 0) return true;
    if (!launcher.ok()) return false;
    CUfunction function = function_for(name);
    if (!function) return false;
    const unsigned block = 64;
    const unsigned grid = static_cast<unsigned>((elements + block - 1) / block);
    /* cuLaunchKernelEx is ~10x slower than cuLaunchKernel on WDDM 13.1
     * (measured 545ms vs 50ms), so always use the legacy entry point. */
    (void)driver_.LaunchKernelEx;
    const CUresult rc =
        driver_.LaunchKernel(function, grid, 1, 1, block, 1, 1, 0, stream,
                             launcher.values(), nullptr);
    if (rc == cuda::CUDA_SUCCESS && active_stream_)
      ++capture_nodes_;
    return rc == cuda::CUDA_SUCCESS;
  }

  bool launch(const char* name, int elements, Launcher& launcher) {
    return launch_on(active_stream_, name, elements, launcher);
  }

  /* launch with an explicit grid: the intra kernel is one thread block per
   * transform block, so its grid is the number of jobs, not the number of
   * rows. */
  bool launch_grid(const char* name, unsigned grid, unsigned block, Launcher& launcher) {
    if (grid == 0) return true;
    if (!launcher.ok()) return false;
    CUfunction function = function_for(name);
    if (!function) return false;
    const CUresult rc =
        driver_.LaunchKernel(function, grid, 1, 1, block, 1, 1, 0, active_stream_,
                             launcher.values(), nullptr);
    if (rc == cuda::CUDA_SUCCESS && active_stream_) ++capture_nodes_;
    return rc == cuda::CUDA_SUCCESS;
  }

  /* ---- stream capture ----------------------------------------------------
   * The frame's kernel launches are recorded into a CUDA graph and submitted
   * with a single cuGraphLaunch.  On WDDM every cuLaunchKernel costs tens of
   * microseconds of driver time, and a frame is hundreds of waves, so this is
   * where the frame time goes.  Unlike the manual cuGraphAddKernelNode path
   * this does not depend on the CUDA_KERNEL_NODE_PARAMS layout, which changed
   * between driver versions. */
  bool capture_begin() {
    if (!driver_.StreamBeginCapture || !driver_.StreamEndCapture ||
        !driver_.GraphInstantiate || !driver_.GraphLaunch)
      return false;
    if (!capture_stream_) {
      /* non-blocking: the capture stream must not wait on the default stream */
      if (driver_.StreamCreate(&capture_stream_, 1 /* CU_STREAM_NON_BLOCKING */) !=
              cuda::CUDA_SUCCESS ||
          !capture_stream_)
        return false;
    }
    if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS) return false;
    /* THREAD_LOCAL: only this thread's stream is captured, so the rest of the
     * program (the display thread) is unaffected. */
    if (driver_.StreamBeginCapture(capture_stream_, 1 /* THREAD_LOCAL */) !=
        cuda::CUDA_SUCCESS)
      return false;
    active_stream_ = capture_stream_;
    capture_nodes_ = 0;
    return true;
  }

  /* end the capture, instantiate the graph and submit it.  Returns false only
   * when nothing was executed, so the caller may run the plain launch path
   * instead. */
  bool capture_end_and_launch() {
    active_stream_ = nullptr;
    if (!capture_stream_) return false;
    CUgraph graph = nullptr;
    if (driver_.StreamEndCapture(capture_stream_, &graph) != cuda::CUDA_SUCCESS ||
        !graph) {
      /* the capture never completed: the stream may still be capturing, so
       * throw it away and make a fresh one next time */
      if (driver_.StreamDestroy) driver_.StreamDestroy(capture_stream_);
      capture_stream_ = nullptr;
      return false;
    }
    CUgraphExec exec = nullptr;
    CUresult rc = driver_.GraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (rc != cuda::CUDA_SUCCESS || !exec) {
      if (driver_.GraphDestroy) driver_.GraphDestroy(graph);
      return false;
    }
    /* launch on the default stream: the filter kernels and the device to
     * device copies that follow must run after the wave graph, and the
     * default stream orders them */
    rc = driver_.GraphLaunch(exec, nullptr);
    if (driver_.GraphExecDestroy) driver_.GraphExecDestroy(exec);
    if (driver_.GraphDestroy) driver_.GraphDestroy(graph);
    return rc == cuda::CUDA_SUCCESS;
  }

  /* Per-wave launch descriptors: one per wave, built during prep and consumed
   * by both per-wave launches and CUDA graph building. Defined at class scope
   * so launch_graph() can see it. */
  struct WaveLaunch {
    CUdeviceptr inter_job = 0, inter_pref = 0;
    int inter_n = 0, inter_total = 0;
    CUdeviceptr intra_job = 0, intra_pref = 0;
    int intra_n = 0, intra_total = 0;
    CUdeviceptr itx_job = 0, itx_row_pref = 0, itx_col_pref = 0;
    int itx_n = 0, itx_rows = 0, itx_cols = 0;
    CUdeviceptr res_job = 0, res_pref = 0;
    int res_n = 0, res_total = 0;
    bool inter = false, intra = false, itx = false, residual = false;
    /* Host offsets into host_job_/host_pref_ for batching: saved during prep so
     * we can concatenate all waves' jobs per kind into frame-level arrays. */
    size_t host_inter_job = 0, host_inter_pref = 0;
    size_t host_intra_job = 0, host_intra_pref = 0;
    size_t host_itx_job = 0, host_itx_row = 0, host_itx_col = 0;
    size_t host_res_job = 0, host_res_pref = 0;
  };

  /* Persistent storage for one graph node's kernel arguments. CUDA graphs need
   * the kernelParams pointer array to stay valid for the graph's lifetime, so
   * we copy the argument values here rather than keeping a stack Launcher. */
  struct PersistentNodeArgs {
    CUfunction func = nullptr;
    unsigned grid_x = 0;
    unsigned block_x = 0;
    int arg_count = 0;
    int value_kind[Launcher::kMaxArgs] = {0};   /* 0 = int, 1 = ptr */
    int int_vals[Launcher::kMaxArgs] = {0};
    CUdeviceptr ptr_vals[Launcher::kMaxArgs] = {0};
    void* slots[Launcher::kMaxArgs] = {nullptr};
    CUDA_KERNEL_NODE_PARAMS params{};
  };

  /* Persistent single-launch wave kernel: concatenates all waves' intra,
   * itx_cols, and residual jobs into frame-level arrays, builds wave boundary
   * tables, and launches ONE kernel that loops over waves with software
   * barriers. This eliminates the ~1800 per-wave launches that dominate 4K
   * keyframe time on WDDM. Inter and itx_rows are still batched separately
   * (they're wave-independent). */
  bool launch_wave_kernel(const std::vector<WaveLaunch>& prepped,
                          unsigned kind_mask, int maxv, size_t itx_job_off,
                          size_t itx_col_pref_off) {
    const size_t nw = prepped.size();
    if (nw == 0) return true;
    CUfunction fn = function_for("av1g_k_waves");
    if (!fn) return false;
    /* its own launch reads the inter jobs from the batch array, so nothing is
     * needed here; the arena upload below covers the wave kernel's kinds */
    (void)kind_mask;
    (void)maxv;

    /* Build concatenated arrays + wave boundaries */
    std::vector<Av1gIntraJob> w_intra;
    std::vector<int> w_intra_pref;
    std::vector<Av1gItxJob> w_itx;
    std::vector<int> w_itx_pref;
    std::vector<Av1gItxJob> w_res;
    std::vector<int> w_res_pref;
    std::vector<int> w_intra_begin(nw + 1), w_intra_end(nw + 1);
    std::vector<int> w_itx_begin(nw + 1), w_itx_end(nw + 1);
    std::vector<int> w_res_begin(nw + 1), w_res_end(nw + 1);

    const char* hjobs = reinterpret_cast<const char*>(host_job_.data());
    const char* hpref = reinterpret_cast<const char*>(host_pref_.data());

    int intra_sum = 0, itx_sum = 0, res_sum = 0;
    for (size_t i = 0; i < nw; ++i) {
      const WaveLaunch& L = prepped[i];
      w_intra_begin[i] = (int)w_intra.size();
      w_itx_begin[i] = (int)w_itx.size();
      w_res_begin[i] = (int)w_res.size();
      if (L.intra && (kind_mask & 2u)) {
        const Av1gIntraJob* src = reinterpret_cast<const Av1gIntraJob*>(hjobs + L.host_intra_job);
        const int* wp = reinterpret_cast<const int*>(hpref + L.host_intra_pref);
        for (size_t k = 0; k < (size_t)L.intra_n; ++k) {
          w_intra.push_back(src[k]);
          w_intra_pref.push_back(intra_sum + wp[k]);
        }
        intra_sum += wp[L.intra_n];
      }
      if (L.itx && (kind_mask & 4u)) {
        /* the itx jobs are in the frame-level batch buffer, and the per-wave
         * prefix is rebased to zero - exactly what a concatenation in wave
         * order wants: the running total below supplies the base. */
        const Av1gItxJob* src =
            reinterpret_cast<const Av1gItxJob*>(batch_host_.data() + itx_job_off) +
            L.host_itx_job;
        const int* cp =
            reinterpret_cast<const int*>(batch_host_.data() + itx_col_pref_off) +
            L.host_itx_col;
        for (size_t k = 0; k < (size_t)L.itx_n; ++k) {
          w_itx.push_back(src[k]);
          w_itx_pref.push_back(itx_sum + cp[k]);
        }
        itx_sum += cp[L.itx_n];
      }
      if (L.residual && (kind_mask & 8u)) {
        const Av1gItxJob* src = reinterpret_cast<const Av1gItxJob*>(hjobs + L.host_res_job);
        const int* wp = reinterpret_cast<const int*>(hpref + L.host_res_pref);
        for (size_t k = 0; k < (size_t)L.res_n; ++k) {
          w_res.push_back(src[k]);
          w_res_pref.push_back(res_sum + wp[k]);
        }
        res_sum += wp[L.res_n];
      }
      w_intra_end[i] = (int)w_intra.size();
      w_itx_end[i] = (int)w_itx.size();
      w_res_end[i] = (int)w_res.size();
    }
    w_intra_begin[nw] = w_intra_end[nw] = (int)w_intra.size();
    w_itx_begin[nw] = w_itx_end[nw] = (int)w_itx.size();
    w_res_begin[nw] = w_res_end[nw] = (int)w_res.size();
    /* append totals to pref arrays */
    w_intra_pref.push_back(intra_sum);
    w_intra_pref.push_back(intra_sum);
    w_itx_pref.push_back(itx_sum);
    w_itx_pref.push_back(itx_sum);
    w_res_pref.push_back(res_sum);
    w_res_pref.push_back(res_sum);

    /* Upload to device */
    auto ens = [&](CUdeviceptr& ptr, size_t& cap, const void* data, size_t bytes) -> bool {
      if (bytes == 0) return true;
      if (bytes > cap) {
        CUdeviceptr tmp = 0;
        if (driver_.MemAlloc(&tmp, bytes) != cuda::CUDA_SUCCESS) return false;
        if (ptr && driver_.MemFree(ptr) != cuda::CUDA_SUCCESS) {}
        ptr = tmp; cap = bytes;
      }
      return upload_h2d(ptr, data, bytes);
    };
    if (!ens(wave_intra_job_, wave_intra_job_cap_, w_intra.data(), w_intra.size() * sizeof(Av1gIntraJob))) return false;
    if (!ens(wave_intra_pref_, wave_intra_pref_cap_, w_intra_pref.data(), w_intra_pref.size() * sizeof(int))) return false;
    if (!ens(wave_itx_job_, wave_itx_job_cap_, w_itx.data(), w_itx.size() * sizeof(Av1gItxJob))) return false;
    if (!ens(wave_itx_col_pref_, wave_itx_col_pref_cap_, w_itx_pref.data(), w_itx_pref.size() * sizeof(int))) return false;
    if (!ens(wave_res_job_, wave_res_job_cap_, w_res.data(), w_res.size() * sizeof(Av1gItxJob))) return false;
    if (!ens(wave_res_pref_, wave_res_pref_cap_, w_res_pref.data(), w_res_pref.size() * sizeof(int))) return false;
    /* Pack all 6 boundary arrays into one buffer: [intra_begin, intra_end, itx_begin, itx_end, res_begin, res_end] each nw+1 ints */
    {
      std::vector<int> packed;
      packed.reserve(6 * (nw + 1));
      for (size_t i = 0; i <= nw; ++i) packed.push_back(w_intra_begin[i]);
      for (size_t i = 0; i <= nw; ++i) packed.push_back(w_intra_end[i]);
      for (size_t i = 0; i <= nw; ++i) packed.push_back(w_itx_begin[i]);
      for (size_t i = 0; i <= nw; ++i) packed.push_back(w_itx_end[i]);
      for (size_t i = 0; i <= nw; ++i) packed.push_back(w_res_begin[i]);
      for (size_t i = 0; i <= nw; ++i) packed.push_back(w_res_end[i]);
      if (!ens(wave_bounds_, wave_bounds_cap_, packed.data(), packed.size() * sizeof(int))) return false;
    }

    /* single barrier counter: zero it */
    if (!wave_barrier_) {
      if (driver_.MemAlloc(&wave_barrier_, sizeof(int)) != cuda::CUDA_SUCCESS) return false;
      wave_barrier_cap_ = sizeof(int);
    }
    {
      int zero = 0;
      if (!upload_h2d(wave_barrier_, &zero, sizeof(int))) return false;
    }

    /* grid-stride: few blocks to minimize barrier latency, 1024 threads each.
     * All blocks must be co-resident for the software barrier to work. */
    int sms = 0;
    driver_.DeviceGetAttribute(&sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, device_);
    if (sms <= 0) sms = 34;
    int nblocks = sms < 4 ? sms : 4; /* 4 blocks - fewer blocks = faster barrier */
    const unsigned block = 1024;
    const unsigned grid = (unsigned)nblocks;

    Launcher a;
    add_mem(a, wave_intra_job_);
    add_mem(a, wave_intra_pref_);
    add_mem(a, wave_itx_job_);
    add_mem(a, wave_itx_col_pref_);
    add_mem(a, wave_res_job_);
    add_mem(a, wave_res_pref_);
    add_mem(a, wave_bounds_); /* packed: [ib(nw+1), ie(nw+1), tb(nw+1), te(nw+1), rb(nw+1), re(nw+1)] */
    add_int(a, (int)nw);
    add_mem(a, wave_barrier_);
    add_mem(a, frame_);
    add_mem(a, palette_);
    add_mem(a, tmp_);
    add_mem(a, residual_);
    add_int(a, maxv);

    if (!a.ok()) return false;
    return driver_.LaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr,
                                a.values(), nullptr) == cuda::CUDA_SUCCESS;
  }

  /* Build (once per geometry) or relaunch a CUDA graph that batches every wave
   * kernel launch into a single driver submission.  The wave arena offsets are
   * fixed per geometry, so node arguments never change between frames:
   * build the graph once, then relaunch it per frame.  Returns false on
   * driver failure so the caller can fall back to per-wave launches. */
  bool launch_graph(const std::vector<WaveLaunch>& prepped,
                    unsigned kind_mask, int maxv) {
    if (!driver_.GraphCreate || !driver_.GraphAddKernelNode ||
        !driver_.GraphInstantiate || !driver_.GraphLaunch ||
        !driver_.GraphAddDependencies || !driver_.CtxSetCurrent)
      return false;

    /* Graph ops require the CUDA context to be current on this thread. */
    if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS)
      return false;

    /* Rebuild the graph every frame: the wave structure (node count, grid
     * sizes, njobs/total) changes per frame, so a cached graph from a
     * different frame would launch wrong grid sizes. Graph build is CPU-side
     * and cheap compared to per-wave kernel launches on WDDM. */
    if (graph_exec_ && driver_.GraphExecDestroy)
      driver_.GraphExecDestroy(graph_exec_);
    if (graph_ && driver_.GraphDestroy) driver_.GraphDestroy(graph_);
    graph_ = nullptr;
    graph_exec_ = nullptr;
    graph_built_ = false;
    graph_node_args_.clear();

    /* Chunked graphs: split waves into groups whose total node count stays
     * under the WDDM crash threshold (~64 nodes with real kernels). Each
     * chunk becomes one small graph, launched sequentially. This reduces
     * ~1800 per-wave launches to ~(n_waves/chunk_size) graph launches. */
    static const size_t max_nodes = [] {
      char f[16] = {0};
      if (GetEnvironmentVariableA("AV1GPU_MAX_GRAPH_NODES", f, sizeof(f)) > 0)
        return (size_t)atoi(f);
      return (size_t)16;
    }();

    bool ok = true;
    size_t w0 = 0;
    while (w0 < prepped.size() && ok) {
      /* find chunk end: accumulate nodes until limit */
      size_t w1 = w0;
      size_t nodes = 0;
      while (w1 < prepped.size()) {
        const WaveLaunch& L = prepped[w1];
        size_t n = 0;
        if (L.inter && (kind_mask & 1u)) ++n;
        if (L.intra && (kind_mask & 2u)) ++n;
        if (L.itx && (kind_mask & 4u)) {
          if (L.itx_rows > 0) ++n;
          if (L.itx_cols > 0) ++n;
        }
        if (L.residual && (kind_mask & 8u)) ++n;
        if (nodes + n > max_nodes && w1 > w0) break;
        nodes += n;
        ++w1;
      }
      if (w1 == w0) ++w1; /* ensure progress */
      /* build and launch graph for waves [w0, w1) */
      ok = launch_graph_chunk(prepped.data() + w0, w1 - w0, kind_mask, maxv);
      w0 = w1;
    }
    {
      static const bool dbg = GetEnvironmentVariableA("AV1GPU_DBG_GRAPH", nullptr, 0) > 0;
      if (dbg) std::fprintf(stderr, "GRAPH %s waves=%zu\n", ok ? "OK" : "FALLBACK", prepped.size());
    }
    if (!ok) {
      if (graph_exec_ && driver_.GraphExecDestroy) driver_.GraphExecDestroy(graph_exec_);
      if (graph_ && driver_.GraphDestroy) driver_.GraphDestroy(graph_);
      graph_ = nullptr;
      graph_exec_ = nullptr;
      graph_built_ = false;
      graph_node_args_.clear();
    }
    return ok;
  }

  /* SEH wrapper for one graph chunk. No C++ objects with destructors in
   * this function (C2712: __try + object unwinding is not allowed). The
   * chunk is a raw pointer + count into a vector owned by the caller. */
  bool launch_graph_chunk(const WaveLaunch* chunk, size_t chunk_n,
                          unsigned kind_mask, int maxv) {
#if defined(_WIN32)
    __try {
      return build_and_launch_graph_raw(chunk, chunk_n, kind_mask, maxv);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
      return false;
    }
#else
    return build_and_launch_graph_raw(chunk, chunk_n, kind_mask, maxv);
#endif
  }

  bool build_and_launch_graph_raw(const WaveLaunch* prepped, size_t prepped_n,
                                   unsigned kind_mask, int maxv) {
      /* ---- build the graph once per geometry ---------------------------- */
      CUgraph graph = nullptr;
      if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS) return false;
      if (driver_.GraphCreate(&graph, 0) != cuda::CUDA_SUCCESS)
        return false;

      /* Count total nodes needed */
      size_t total_nodes = 0;
      for (size_t wi = 0; wi < prepped_n; ++wi) {
        const WaveLaunch& L = prepped[wi];
        if (L.inter && (kind_mask & 1u)) ++total_nodes;
        if (L.intra && (kind_mask & 2u)) ++total_nodes;
        if (L.itx && (kind_mask & 4u)) {
          if (L.itx_rows > 0) ++total_nodes;
          if (L.itx_cols > 0) ++total_nodes;
        }
        if (L.residual && (kind_mask & 8u)) ++total_nodes;
      }
      graph_node_args_.resize(total_nodes);
      std::vector<CUgraphNode> nodes(total_nodes);
      CUresult rc = cuda::CUDA_SUCCESS;
      size_t ni = 0;

      for (size_t i = 0; i < prepped_n && rc == cuda::CUDA_SUCCESS; ++i) {
        const WaveLaunch& L = prepped[i];

        auto emit = [&](CUfunction fn, unsigned total, Launcher&& a) -> bool {
          if (!fn || total == 0) return true;
          auto& pa = graph_node_args_[ni];
          pa.func = fn;
          pa.grid_x = static_cast<unsigned>((total + 63) / 64);
          pa.block_x = 64;
          pa.arg_count = a.count();
          for (int k = 0; k < a.count(); ++k) {
            pa.value_kind[k] = a.value_kind(k);
            if (a.value_kind(k) == 0) {
              pa.int_vals[k] = a.int_val(k);
              pa.slots[k] = &pa.int_vals[k];
            } else {
              pa.ptr_vals[k] = a.ptr_val(k);
              pa.slots[k] = &pa.ptr_vals[k];
            }
          }
          pa.params.func = pa.func;
          pa.params.gridDimX = pa.grid_x;
          pa.params.gridDimY = 1;
          pa.params.gridDimZ = 1;
          pa.params.blockDimX = pa.block_x;
          pa.params.blockDimY = 1;
          pa.params.blockDimZ = 1;
          pa.params.sharedMemBytes = 0;
          pa.params.kernelParams = pa.slots;
          pa.params.extra = nullptr;
          CUgraphNode node = nullptr;
          if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS) return false;
          rc = driver_.GraphAddKernelNode(&node, graph, nullptr, 0, &pa.params);
          if (rc != cuda::CUDA_SUCCESS) return false;
          nodes[ni] = node;
          ++ni;
          return true;
        };

        if (L.inter && (kind_mask & 1u)) {
          CUfunction fn = function_for("av1g_k_inter");
          Launcher a;
          add_mem(a, L.inter_job);
          add_mem(a, L.inter_pref);
          add_int(a, L.inter_n);
          add_int(a, L.inter_total);
          add_mem(a, frame_);
          add_mem(a, ref_pool_);
          add_mem(a, frame_);
          add_mem(a, filters_);
          add_int(a, maxv);
          if (!emit(fn, L.inter_total, std::move(a))) return false;
        }
        if (L.intra && (kind_mask & 2u) && rc == cuda::CUDA_SUCCESS) {
          CUfunction fn = function_for("av1g_k_intra");
          Launcher a;
          add_mem(a, L.intra_job);
          add_mem(a, L.intra_pref);
          add_int(a, L.intra_n);
          add_int(a, L.intra_total);
          add_mem(a, frame_);
          add_mem(a, palette_);
          add_int(a, maxv);
          if (!emit(fn, L.intra_total, std::move(a))) return false;
        }
        if (L.itx && (kind_mask & 4u) && rc == cuda::CUDA_SUCCESS) {
          CUfunction fn = function_for("av1g_k_itx_rows");
          Launcher a;
          add_mem(a, L.itx_job);
          add_mem(a, L.itx_row_pref);
          add_int(a, L.itx_n);
          add_int(a, L.itx_rows);
          add_mem(a, coef_);
          add_mem(a, tmp_);
          if (!emit(fn, L.itx_rows, std::move(a))) return false;
          fn = function_for("av1g_k_itx_cols");
          Launcher b;
          add_mem(b, L.itx_job);
          add_mem(b, L.itx_col_pref);
          add_int(b, L.itx_n);
          add_int(b, L.itx_cols);
          add_mem(b, tmp_);
          add_mem(b, frame_);
          add_int(b, maxv);
          if (!emit(fn, L.itx_cols, std::move(b))) return false;
        }
        if (L.residual && (kind_mask & 8u) && rc == cuda::CUDA_SUCCESS) {
          CUfunction fn = function_for("av1g_k_add_residual");
          Launcher a;
          add_mem(a, L.res_job);
          add_mem(a, L.res_pref);
          add_int(a, L.res_n);
          add_int(a, L.res_total);
          add_mem(a, residual_);
          add_mem(a, frame_);
          add_int(a, maxv);
          if (!emit(fn, L.res_total, std::move(a))) return false;
        }
      }

      /* Chain nodes in order using GraphAddDependencies */
      if (rc == cuda::CUDA_SUCCESS && ni > 1) {
        std::vector<CUgraphNode> deps(ni - 1);
        for (size_t k = 0; k < ni - 1; ++k) deps[k] = nodes[k];
        if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS) return false;
        rc = driver_.GraphAddDependencies(graph, deps.data(), &deps[1], ni - 1);
      }

      CUgraphExec exec = nullptr;
      CUresult rc2 = driver_.GraphInstantiate(&exec, graph, nullptr, nullptr, 0);
      if (rc == cuda::CUDA_SUCCESS && rc2 == cuda::CUDA_SUCCESS) {
        graph_ = graph;
        graph_exec_ = exec;
        graph_built_ = true;
        graph_kind_mask_ = kind_mask;
        graph_maxv_ = maxv;
      } else {
        if (exec && driver_.GraphExecDestroy) driver_.GraphExecDestroy(exec);
        if (graph && driver_.GraphDestroy) driver_.GraphDestroy(graph);
        return false;
      }

    /* ---- launch the graph ---------------------------------------------- */
    if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS)
      return false;
    return driver_.GraphLaunch(graph_exec_, nullptr) == cuda::CUDA_SUCCESS;
  }
  static bool fail(std::string* error, const char* message) {
    *error = message;
    return false;
  }

  /* slots_/last_slot_ are the reference pool's bookkeeping. A pipelined
   * caller reserves from its own thread while the worker decodes into and
   * releases other slots, so every touch goes through slot_mutex_. */
  int allocate_slot() {
    for (int i = 0; i < kReferenceSlots; ++i)
      if (!slots_[i]) return i;
    return -1;
  }

  BackendOptions options_;
  DeviceInfo info_;
  /* page-locked host staging for the large per frame uploads */
  void* pinned_ = nullptr;
  size_t pinned_cap_ = 0;
  /* The read back has its own page-locked buffer: a host to device copy out
   * of page-locked memory is asynchronous, so the driver may still be reading
   * the coefficients out of `pinned_` when the read back would overwrite it. */
  void* pinned_rb_ = nullptr;
  size_t pinned_rb_cap_ = 0;
  /* The same page locked staging for the NV12 pack's read back (see to_nv12). */
  void* pinned_nv12_ = nullptr;
  size_t pinned_nv12_cap_ = 0;
  /* per frame arenas the parallel wave prep fills at fixed offsets */
  CUdeviceptr job_arena_ = 0;
  size_t job_arena_cap_ = 0;
  /* host side of the arena: built per frame, uploaded in one copy */
  std::vector<char> host_job_;
  std::vector<char> host_pref_;
  CUdeviceptr pref_arena_ = 0;
  size_t pref_arena_cap_ = 0;
  /* The frame-level batch arrays: every wave's inter and itx jobs, in wave
   * order, written by the prep and uploaded once - the inter jobs used to be
   * staged in an arena first and copied again here, which at 4K was ~7.6 MB of
   * writes and reads a frame for nothing.
   *
   * They live in ONE device buffer and ONE host buffer, laid out back to back
   * (see BatchLayout in decode_impl): seven separate uploads a frame cost
   * seven driver calls, which measured ~1.8 ms at 4K. The host side is a member
   * so its capacity survives from frame to frame, and the per-wave prefixes are
   * rebased to zero so a launch covering a single wave reads its own slice. */
  std::vector<char> batch_host_;
  CUdeviceptr batch_dev_ = 0;
  size_t batch_dev_cap_ = 0;
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
  CUdeviceptr lr_curr_ = 0, lr_cdef_ = 0, lr_prev_up_ = 0;
  size_t frame_cap_ = 0, scratch_cap_ = 0, coef_cap_ = 0, tmp_cap_ = 0;
  size_t residual_cap_ = 0, filters_cap_ = 0, ref_cap_ = 0;
  size_t grain_cap_ = 0, lut_cap_ = 0, rgb_cap_ = 0, palette_cap_ = 0;
  size_t job_cap_ = 0, pref_cap_ = 0, pref_b_cap_ = 0;
  size_t lr_curr_cap_ = 0, lr_cdef_cap_ = 0, lr_prev_up_cap_ = 0;
  bool alloc_failed_ = false;
  /* set once the first prepare() has sized the buffers for this geometry */
  bool ready_ = false;
  /* Two streams + events for pipelined per-wave launches: wave N goes on
   * stream[N%2], waits on event recorded after wave N-1 on the other stream.
   * This overlaps CPU-side launch overhead across consecutive waves on WDDM. */
  cuda::CUstream pipe_stream_[2] = {nullptr, nullptr};
  cuda::CUevent pipe_event_[2] = {nullptr, nullptr};
  /* Stream capture: the frame's kernel launches are recorded into a CUDA graph
   * (cuStreamBeginCapture/cuStreamEndCapture) and submitted once.  This is the
   * graph API without the versioned CUDA_KERNEL_NODE_PARAMS struct, so it does
   * not depend on the driver's struct ABI.  capture_stream_ is the stream the
   * capture runs on; active_stream_ is the stream every launch goes to (null
   * normally, capture_stream_ while a capture is open). */
  cuda::CUstream capture_stream_ = nullptr;
  cuda::CUstream active_stream_ = nullptr;
  /* how many launches the last capture recorded, for the perf log */
  int capture_nodes_ = 0;
  /* CUDA graph for batched wave kernel launches (one per geometry, reused
   * across frames). Built on first decode of a geometry; launched instead of
   * individual cuLaunchKernel calls per wave to eliminate per-wave driver
   * overhead on WDDM (keyframes have 500+ waves = thousands of launches). */
  cuda::CUgraph graph_ = nullptr;
  cuda::CUgraphExec graph_exec_ = nullptr;
  bool graph_built_ = false;
  /* cached kind_mask and maxv from the first decode so graph node arguments
   * stay valid across frames (wave arena addresses are fixed per geometry) */
  unsigned graph_kind_mask_ = 0;
  int graph_maxv_ = 0;
  /* Persistent storage for graph node kernel arguments: CUDA graphs require
   * the kernelParams pointer array to remain valid for the graph's lifetime,
   * so we keep the argument values and their pointer table here. */
  std::vector<PersistentNodeArgs> graph_node_args_;
  /* Kernel batching buffers: every wave's inter and itx jobs in one frame-level
   * array per kind, so a single kernel launch per kind per frame replaces the
   * thousands per 4K keyframe on WDDM (see the layout below). */
  /* Device dequantisation (AV1GPU_GPU_DEQUANT=1): the frame's coded levels and
   * its quantiser matrices, plus the three scan tables (constant, uploaded
   * once). av1g_k_dequant expands them into the raster pool the itx kernels
   * read, in place of the host doing it and uploading the whole pool. */
  CUdeviceptr dq_levels_ = 0, dq_qm_ = 0;
  CUdeviceptr dq_scan_2d_ = 0, dq_scan_row_ = 0, dq_scan_col_ = 0;
  size_t dq_levels_cap_ = 0, dq_qm_cap_ = 0;
  size_t dq_scan_2d_cap_ = 0, dq_scan_row_cap_ = 0, dq_scan_col_cap_ = 0;
  int dq_level_n_ = 0, dq_qm_n_ = 0;
  bool dq_tables_ready_ = false;
  /* Real host-to-device volume of the current frame, accumulated by
   * upload_h2d and reported by the per-50-frame log line (it used to print the
   * plan's pool size, which said nothing about what was actually sent).
   * h2d_tag_ attributes the bytes to a category (see the kH2dTag* constants):
   * with the device dequant the raster pool disappears from this list, and
   * the remaining volume is the per-wave job arenas. */
  long long h2d_bytes = 0;
  int h2d_tag_ = -1;
  long long h2d_tag_bytes_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  /* Persistent wave kernel buffers: concatenated intra/itx_cols/residual jobs
   * across ALL waves + wave boundary arrays + barrier counter. */
  CUdeviceptr wave_intra_job_ = 0, wave_intra_pref_ = 0;
  CUdeviceptr wave_itx_job_ = 0, wave_itx_col_pref_ = 0;
  CUdeviceptr wave_res_job_ = 0, wave_res_pref_ = 0;
  CUdeviceptr wave_bounds_ = 0; /* 6*(n_waves+1) ints: begin/end for intra,itx,res */
  CUdeviceptr wave_barrier_ = 0;
  size_t wave_barrier_cap_ = 0;
  size_t wave_intra_job_cap_ = 0, wave_intra_pref_cap_ = 0;
  size_t wave_itx_job_cap_ = 0, wave_itx_col_pref_cap_ = 0;
  size_t wave_res_job_cap_ = 0, wave_res_pref_cap_ = 0;
  size_t wave_bounds_cap_ = 0;
  std::vector<uint16_t> staging_;
  /* Serialises every driver side operation (decode_impl, to_nv12, to_rgb): the
   * staging buffers are shared, and a pipeline's output stage packs a frame on
   * its own thread while the worker decodes the next one. slot_mutex_ only
   * covers the slot bookkeeping, which never touches the driver. */
  std::mutex driver_mutex_;
  /* Guards slots_/last_slot_ (see allocate_slot). The decode paths run on the
   * pipeline worker while reserve/release can come from the demuxer thread. */
  mutable std::mutex slot_mutex_;
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
  /* the count must follow the array: the Windows list has one entry, not two,
   * and a hard coded bound reads past its end */
  *library = open_first(names, static_cast<int>(sizeof(names) / sizeof(names[0])));
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
  AV1GPU_CU(CtxSetCurrent, "cuCtxSetCurrent");
  AV1GPU_CU(CtxGetCurrent, "cuCtxGetCurrent");
  AV1GPU_CU(DriverGetVersion, "cuDriverGetVersion");
  AV1GPU_CU(EventCreate, "cuEventCreate");
  AV1GPU_CU(EventRecord, "cuEventRecord");
  AV1GPU_CU(EventSynchronize, "cuEventSynchronize");
  AV1GPU_CU(EventElapsedTime, "cuEventElapsedTime");
  AV1GPU_CU(EventDestroy, "cuEventDestroy");
  AV1GPU_CU(MemAlloc, "cuMemAlloc_v2");
  AV1GPU_CU(MemFree, "cuMemFree_v2");
  AV1GPU_CU(MemcpyHtoD, "cuMemcpyHtoD_v2");
  AV1GPU_CU(MemcpyDtoH, "cuMemcpyDtoH_v2");
  *(void**)(&driver->MemcpyDtoHAsync) = dlsym(*library, "cuMemcpyDtoHAsync_v2");
  AV1GPU_CU(MemcpyDtoD, "cuMemcpyDtoD_v2");
  AV1GPU_CU(MemAllocHost, "cuMemAllocHost_v2");
  AV1GPU_CU(MemFreeHost, "cuMemFreeHost");
  AV1GPU_CU(ModuleLoadDataEx, "cuModuleLoadDataEx");
  AV1GPU_CU(ModuleGetFunction, "cuModuleGetFunction");
  AV1GPU_CU(LaunchKernel, "cuLaunchKernel");
  /* optional: cuLaunchKernelEx (CUDA 11.8+) - may have lower WDDM overhead */
  *(void**)(&driver->LaunchKernelEx) = dlsym(*library, "cuLaunchKernelEx");
  /* optional: graphs, CUDA 10 and up. Missing entry points only mean the
   * launch-per-wave path stays in use. */
  *(void**)(&driver->GraphCreate) = dlsym(*library, "cuGraphCreate");
  *(void**)(&driver->GraphAddKernelNode) = dlsym(*library, "cuGraphAddKernelNode");
  *(void**)(&driver->GraphAddDependencies) = dlsym(*library, "cuGraphAddDependencies");
  *(void**)(&driver->GraphInstantiate) = dlsym(*library, "cuGraphInstantiate");
  *(void**)(&driver->GraphLaunch) = dlsym(*library, "cuGraphLaunch");
  *(void**)(&driver->GraphExecDestroy) = dlsym(*library, "cuGraphExecDestroy");
  *(void**)(&driver->GraphDestroy) = dlsym(*library, "cuGraphDestroy");
  *(void**)(&driver->StreamCreate) = dlsym(*library, "cuStreamCreate");
  *(void**)(&driver->StreamDestroy) = dlsym(*library, "cuStreamDestroy");
  *(void**)(&driver->StreamWaitEvent) = dlsym(*library, "cuStreamWaitEvent");
  *(void**)(&driver->StreamSynchronize) = dlsym(*library, "cuStreamSynchronize");
  /* optional: stream capture (CUDA 10.1+) - the graph API without the
   * versioned CUDA_KERNEL_NODE_PARAMS struct */
  *(void**)(&driver->StreamBeginCapture) = dlsym(*library, "cuStreamBeginCapture");
  *(void**)(&driver->StreamEndCapture) = dlsym(*library, "cuStreamEndCapture");
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
      /* CUDA 12.x keeps the 120_0 DLL name across all minors, 13.x is 130_0
       * and 11.x bumps per minor (11.2/11.8 = 112_0) */
      "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll",
      "nvrtc64_111_0.dll", "nvrtc64_110_0.dll", "nvrtc64_102_0.dll",
#else
      "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so.11.1", "libnvrtc.so.11.0",
      "libnvrtc.so.10.2", "libnvrtc.so",
#endif
  };
  *library = open_first(names, static_cast<int>(sizeof(names) / sizeof(names[0])));
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
  /* optional: the fast path when the installed NVRTC can assemble cubins */
  *(void**)(&nvrtc->GetCUBINSize) = dlsym(*library, "nvrtcGetCUBINSize");
  *(void**)(&nvrtc->GetCUBIN) = dlsym(*library, "nvrtcGetCUBIN");
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
  /* cuDeviceTotalMem needs a current context (the attribute queries do not), so
   * without this the listing used to report 0 bytes of device memory. Creating
   * a context is not free, but a device listing that lies about the size of the
   * device is worse than spending a few milliseconds. */
  cuda::CUcontext context = nullptr;
  driver.CtxCreate(&context, 0, 0);
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
  if (context) driver.CtxDestroy(context);
  cuda::close_library(library);
  return out;
}

} /* namespace internal */
} /* namespace av1gpu */























