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
    ensure_context(nullptr); /* prepare cannot report an error; decode will */
    const PlanLayout g = make_plan_layout(geometry, coded_width, 64);
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

  int reserve_reference() override {
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
    if (!ensure(&coef_, &coef_cap_, (plan.coefficients.size() + 1) * sizeof(int)) ||
        !ensure(&tmp_, &tmp_cap_, (plan.tmp_size + 1) * sizeof(int)))
      return fail(error, "out of device memory for the coefficient buffers");
    if (!plan.coefficients.empty() &&
        !upload_h2d(coef_, plan.coefficients.data(),
                    plan.coefficients.size() * sizeof(int)))
      return fail(error, "coefficient upload failed");
    if (!plan.residual.empty()) {
      if (!ensure(&residual_, &residual_cap_, (plan.residual.size() + 1) * sizeof(int)))
        return fail(error, "out of device memory for the residual buffer");
      if (!upload_h2d(residual_, plan.residual.data(),
                      plan.residual.size() * sizeof(int)))
        return fail(error, "residual upload failed");
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
          if (!launch("av1g_k_intra", total, a)) return fail(error, "av1g_k_intra failed");
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
    struct WaveLaunch {
      /* per kind pointers: one wave can carry several kinds, and each kernel
       * needs its own job/prefix slice of the arenas */
      CUdeviceptr inter_job = 0, inter_pref = 0;
      int inter_n = 0, inter_total = 0;
      CUdeviceptr intra_job = 0, intra_pref = 0;
      int intra_n = 0, intra_total = 0;
      CUdeviceptr itx_job = 0, itx_row_pref = 0, itx_col_pref = 0;
      int itx_n = 0, itx_rows = 0, itx_cols = 0;
      CUdeviceptr res_job = 0, res_pref = 0;
      int res_n = 0, res_total = 0;
      bool inter = false, intra = false, itx = false, residual = false;
    };
    std::vector<WaveLaunch> prepped(plan.waves.size());
    /* Per kind byte offsets into the two arenas. Every region starts on a 64
     * byte boundary: cumulative struct sizes alone can leave a kind's array
     * misaligned, and the kernels read the job structs with vector loads. */
    struct WaveOffsets {
      size_t inter_job = 0, intra_job = 0, itx_job = 0, res_job = 0;
      size_t inter_pref = 0, intra_pref = 0, itx_row = 0, itx_col = 0, res_pref = 0;
      size_t job_bytes = 0, pref_bytes = 0;
    };
    auto align64 = [](size_t x) { return (x + 63) & ~size_t(63); };
    std::vector<WaveOffsets> offs(plan.waves.size());
    size_t arena_job = 0, arena_pref = 0;
    for (size_t i = 0; i < plan.waves.size(); ++i) {
      const WaveJobs& w = plan.waves[i];
      WaveOffsets& o = offs[i];
      size_t job = align64(arena_job), pref = align64(arena_pref);
      if (!w.inter.empty()) {
        o.inter_job = job;
        o.inter_pref = pref;
        job += w.inter.size() * sizeof(Av1gInterJob);
        pref += (w.inter.size() + 2) * sizeof(int);
      }
      if (!w.intra.empty()) {
        o.intra_job = align64(job);
        o.intra_pref = align64(pref);
        job = o.intra_job + w.intra.size() * sizeof(Av1gIntraJob);
        pref = o.intra_pref + (w.intra.size() + 2) * sizeof(int);
      }
      if (!w.itx.empty()) {
        o.itx_job = align64(job);
        job = o.itx_job + w.itx.size() * sizeof(Av1gItxJob);
        o.itx_row = align64(pref);
        o.itx_col = o.itx_row + (w.itx.size() + 2) * sizeof(int);
        pref = o.itx_col + (w.itx.size() + 2) * sizeof(int);
      }
      if (!w.residual.empty()) {
        o.res_job = align64(job);
        job = o.res_job + w.residual.size() * sizeof(Av1gItxJob);
        o.res_pref = align64(pref);
        pref = o.res_pref + (w.residual.size() + 2) * sizeof(int);
      }
      o.job_bytes = job - align64(arena_job);
      o.pref_bytes = pref - align64(arena_pref);
      arena_job = job;
      arena_pref = pref;
    }
    if (!ensure(&job_arena_, &job_arena_cap_, arena_job) ||
        !ensure(&pref_arena_, &pref_arena_cap_, arena_pref * sizeof(int)))
      return fail(error, "out of device memory for the wave arena");

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
    const unsigned k_workers =
        plan.waves.size() >= 8 ? std::min<unsigned>(6, std::max(1u, hw - 1)) : 1;
    std::atomic<bool> prep_ok{true};

    /* one worker's slice: [id, id + K, ...) of the waves */
    auto prep_waves = [&](unsigned id) {
      if (k_workers > 1) {
        if (driver_.CtxSetCurrent(context_) != cuda::CUDA_SUCCESS) {
          prep_ok = false;
          return;
        }
      }
      void* pinned = nullptr; /* per worker staging for its uploads */
      size_t pinned_cap = 0;
      auto h2d = [&](CUdeviceptr dst, const void* src, size_t bytes) {
        if (bytes == 0) return true;
        if (!driver_.MemAllocHost || !driver_.MemFreeHost)
          return driver_.MemcpyHtoD(dst, src, bytes) == cuda::CUDA_SUCCESS;
        if (pinned_cap < bytes) {
          if (pinned) driver_.MemFreeHost(pinned);
          pinned = nullptr;
          size_t want = bytes + bytes / 2;
          if (driver_.MemAllocHost(&pinned, want) != cuda::CUDA_SUCCESS || !pinned) {
            pinned = nullptr;
            pinned_cap = 0;
            return driver_.MemcpyHtoD(dst, src, bytes) == cuda::CUDA_SUCCESS;
          }
          pinned_cap = want;
        }
        std::memcpy(pinned, src, bytes);
        return driver_.MemcpyHtoD(dst, pinned, bytes) == cuda::CUDA_SUCCESS;
      };
      for (size_t i = id; i < plan.waves.size() && prep_ok; i += k_workers) {
        const WaveJobs& w = plan.waves[i];
        WaveLaunch& L = prepped[i];
        const WaveOffsets& o = offs[i];
        if (!w.inter.empty()) {
          std::vector<Av1gInterJob> jobs = w.inter;
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
          std::vector<int> prefix(jobs.size() + 2, 0);
          int sum = 0;
          for (size_t k = 0; k < jobs.size(); ++k) {
            prefix[k] = sum;
            sum += jobs[k].w * jobs[k].h;
          }
          prefix[jobs.size()] = sum;
          prefix[jobs.size() + 1] = sum;
          if (!h2d(job_arena_ + o.inter_job, jobs.data(),
                   jobs.size() * sizeof(Av1gInterJob)) ||
              !h2d(pref_arena_ + o.inter_pref, prefix.data(),
                   prefix.size() * sizeof(int))) {
            prep_ok = false;
            break;
          }
          L.inter = true;
          L.inter_job = job_arena_ + o.inter_job;
          L.inter_pref = pref_arena_ + o.inter_pref;
          L.inter_n = static_cast<int>(jobs.size());
          L.inter_total = sum;
        }
        if (!w.intra.empty()) {
          std::vector<int> prefix(w.intra.size() + 2, 0);
          int sum = 0;
          for (size_t k = 0; k < w.intra.size(); ++k) {
            prefix[k] = sum;
            sum += w.intra[k].h;
          }
          prefix[w.intra.size()] = sum;
          prefix[w.intra.size() + 1] = sum;
          if (!h2d(job_arena_ + o.intra_job, w.intra.data(),
                   w.intra.size() * sizeof(Av1gIntraJob)) ||
              !h2d(pref_arena_ + o.intra_pref, prefix.data(),
                   prefix.size() * sizeof(int))) {
            prep_ok = false;
            break;
          }
          L.intra = true;
          L.intra_job = job_arena_ + o.intra_job;
          L.intra_pref = pref_arena_ + o.intra_pref;
          L.intra_n = static_cast<int>(w.intra.size());
          L.intra_total = sum;
        }
        if (!w.itx.empty()) {
          std::vector<int> row_prefix(w.itx.size() + 2, 0);
          std::vector<int> col_prefix(w.itx.size() + 2, 0);
          int rows_sum = 0, cols_sum = 0;
          for (size_t k = 0; k < w.itx.size(); ++k) {
            row_prefix[k] = rows_sum;
            rows_sum += w.itx[k].h;
            col_prefix[k] = cols_sum;
            cols_sum += w.itx[k].w;
          }
          row_prefix[w.itx.size()] = rows_sum;
          row_prefix[w.itx.size() + 1] = rows_sum;
          col_prefix[w.itx.size()] = cols_sum;
          col_prefix[w.itx.size() + 1] = cols_sum;
          if (!h2d(job_arena_ + o.itx_job, w.itx.data(),
                   w.itx.size() * sizeof(Av1gItxJob)) ||
              !h2d(pref_arena_ + o.itx_row, row_prefix.data(),
                   row_prefix.size() * sizeof(int)) ||
              !h2d(pref_arena_ + o.itx_col, col_prefix.data(),
                   col_prefix.size() * sizeof(int))) {
            prep_ok = false;
            break;
          }
          L.itx = true;
          L.itx_job = job_arena_ + o.itx_job;
          L.itx_row_pref = pref_arena_ + o.itx_row;
          L.itx_col_pref = pref_arena_ + o.itx_col;
          L.itx_n = static_cast<int>(w.itx.size());
          L.itx_rows = rows_sum;
          L.itx_cols = cols_sum;
        }
        if (!w.residual.empty()) {
          std::vector<int> prefix(w.residual.size() + 2, 0);
          int sum = 0;
          for (size_t k = 0; k < w.residual.size(); ++k) {
            prefix[k] = sum;
            sum += w.residual[k].w * w.residual[k].h;
          }
          prefix[w.residual.size()] = sum;
          prefix[w.residual.size() + 1] = sum;
          if (!h2d(job_arena_ + o.res_job, w.residual.data(),
                   w.residual.size() * sizeof(Av1gItxJob)) ||
              !h2d(pref_arena_ + o.res_pref, prefix.data(),
                   prefix.size() * sizeof(int))) {
            prep_ok = false;
            break;
          }
          L.residual = true;
          L.res_job = job_arena_ + o.res_job;
          L.res_pref = pref_arena_ + o.res_pref;
          L.res_n = static_cast<int>(w.residual.size());
          L.res_total = sum;
        }
      }
      if (pinned && driver_.MemFreeHost) driver_.MemFreeHost(pinned);
      if (k_workers > 1) driver_.CtxSetCurrent(nullptr);
    };

    if (k_workers <= 1) {
      prep_waves(0);
    } else {
      std::vector<std::thread> pool;
      pool.reserve(k_workers);
      for (unsigned id = 0; id < k_workers; ++id)
        pool.emplace_back(prep_waves, id);
      for (std::thread& t : pool) t.join();
    }
    if (!prep_ok) return fail(error, "wave job upload failed");
    const double t_prep = now_ms(); /* DEBUG */
    driver_.CtxSynchronize(); /* DEBUG: drain any queued prep work */

    /* launch phase: kernels must start in wave order */
    static const bool skip_launch = [] {
      char f[8] = {0};
      return GetEnvironmentVariableA("AV1GPU_NO_LAUNCH", f, sizeof(f)) > 0 &&
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
    void *ev0 = nullptr, *ev1 = nullptr; /* DEBUG: GPU side timing */
    if (driver_.EventCreate && !skip_launch) {
      driver_.EventCreate(&ev0, 0);
      driver_.EventCreate(&ev1, 0);
      if (ev0) driver_.EventRecord(ev0, nullptr);
    }
    for (size_t i = 0; !skip_launch && i < plan.waves.size(); ++i) {
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
        if (!launch("av1g_k_inter", L.inter_total, a))
          return fail(error, "av1g_k_inter failed");
        ++launches;
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
        if (!launch("av1g_k_intra", L.intra_total, a))
          return fail(error, "av1g_k_intra failed");
        ++launches;
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
          if (!launch("av1g_k_itx_rows", L.itx_rows, a))
            return fail(error, "av1g_k_itx_rows failed");
          ++launches;
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
          if (!launch("av1g_k_itx_cols", L.itx_cols, a))
            return fail(error, "av1g_k_itx_cols failed");
          ++launches;
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
        if (!launch("av1g_k_add_residual", L.res_total, a))
          return fail(error, "av1g_k_add_residual failed");
        ++launches;
      }
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
      static long n2 = 0;
      acc_prep += t_prep - t0; acc_launch += t_launch_end - t_prep;
      acc_sync += t_sync_end - t_launch_end; acc_gpu += gpu_ms > 0 ? gpu_ms : 0;
      ++n2;
      if (n2 % 50 == 0) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "perf wave prep %.1f, launch %.1f, gpuwait %.1f, "
                      "gpu kernel %.1f ms over %ld",
                      acc_prep / n2, acc_launch / n2, acc_sync / n2,
                      acc_gpu / n2, n2);
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
        slot = allocate_slot();
        if (slot < 0) return fail(error, "CUDA reference pool exhausted");
      }
      cuda::CUresult rc = driver_.MemcpyDtoD(
          ref_pool_ + static_cast<size_t>(slot) * g.buffer_size * sizeof(uint16_t), frame_,
          g.buffer_size * sizeof(uint16_t));
      if (rc != cuda::CUDA_SUCCESS) return fail(error, "reference copy failed");
      if (driver_.CtxSynchronize() != cuda::CUDA_SUCCESS)
        return fail(error, "reference copy did not complete");
      slots_[slot] = true;
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
    /* temporary profiling: how much of reconstruct was CPU prep vs GPU wait */
    {
      static double acc_sync = 0.0, acc_filter = 0.0, acc_rec = 0.0;
      static long long acc_bytes = 0;
      static long n = 0;
      acc_sync += double(qf.QuadPart ? 0 : 0); /* no-op keeps qf referenced */
      acc_filter += t2 - t1;
      acc_rec += t1 - t0;
      size_t waves = plan.waves.size();
      acc_bytes += static_cast<long long>(plan.coefficients.size() * 4) +
                   static_cast<long long>(plan.residual.size() * 4);
      size_t job_count = 0;
      for (const WaveJobs& w : plan.waves)
        job_count += w.inter.size() + w.intra.size() + w.itx.size() + w.residual.size();
      job_count += plan.deblock_v.size() + plan.deblock_h.size() + plan.cdef.size();
      ++n;
      if (n % 50 == 0) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "perf backend: reconstruct %.1f, "
                      "waves %zu, filter(sync incl) %.1f, "
                      "jobs %zu, uploads %lld KB, %ld frames",
                      acc_rec / n, waves,
                      acc_filter / n, job_count, acc_bytes / 1024, n);
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

    if (out) {
      if (!out->matches(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                        g.frame.ss_y))
        out->allocate(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                      g.frame.ss_y);
      if (staging_.size() < g.buffer_size) staging_.resize(g.buffer_size);
      if (driver_.MemcpyDtoH(staging_.data(), frame_, g.buffer_size * sizeof(uint16_t)) !=
          cuda::CUDA_SUCCESS)
        return fail(error, "frame read back failed");
      for (int p = 0; p < 3; ++p) {
        const int w = g.out_w[p];
        const int h = g.out_h[p];
        for (int y = 0; y < h; ++y)
          std::memcpy(out->plane(p) + static_cast<size_t>(y) * out->stride[p],
                      staging_.data() + g.plane_off[p] +
                          static_cast<size_t>(y) * g.stride[p],
                      static_cast<size_t>(w) * sizeof(uint16_t));
      }
    }
    const double t3 = now_ms();

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
    /* The reference pool holds the picture *before* the film grain (7.20) and
     * frame_ the one after it, so a request for the frame that was just decoded
     * reads frame_ - the reference is only the fallback for an older slot. */
    CUdeviceptr source = frame_;
    int base = 0;
    if (slot >= 0) {
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

  bool to_nv12(int slot, size_t pitch, std::vector<uint8_t>* nv12,
               std::string* error) override {
    error->clear();
    if (layout_.buffer_size == 0) {
      *error = "prepare() has not been called";
      return false;
    }
    const PlanLayout& g = layout_;
    /* same source rule as to_rgb (see the CPU backend) */
    CUdeviceptr source = frame_;
    int base = 0;
    if (slot >= 0) {
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
    if (driver_.MemcpyDtoH(nv12->data(), rgb_, bytes) != cuda::CUDA_SUCCESS) {
      *error = "NV12 read back failed";
      return false;
    }
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
                             job_arena_, pref_arena_};
    for (CUdeviceptr b : buffers)
      if (b && driver_.MemFree) driver_.MemFree(b);
    frame_ = scratch_ = coef_ = tmp_ = residual_ = 0;
    filters_ = ref_pool_ = grain_ = 0;
    lut_ = rgb_ = job_ = pref_ = pref_b_ = palette_ = 0;
    lr_curr_ = lr_cdef_ = lr_prev_up_ = 0;
    job_arena_ = pref_arena_ = 0;
    job_arena_cap_ = pref_arena_cap_ = 0;
    palette_cap_ = 0;
    if (pinned_ && driver_.MemFreeHost) driver_.MemFreeHost(pinned_);
    pinned_ = nullptr;
    pinned_cap_ = 0;
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
      mix("v2", 2); /* bump when the cache format changes */
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
    /* Prefer NVRTC's assembled cubin: the driver loads machine code as it is,
     * while a PTX module costs its JIT on every launch - the several seconds
     * the cache exists to avoid. A cubin targets sm_XX, so it needs its own
     * NVRTC run; the PTX build is only done when there is no cubin to be had. */
    if (nvrtc_.GetCUBINSize && nvrtc_.GetCUBIN) {
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
  bool upload_h2d(CUdeviceptr dst, const void* src, size_t bytes) {
    if (bytes == 0) return true;
    if (!driver_.MemAllocHost || !driver_.MemFreeHost)
      return driver_.MemcpyHtoD(dst, src, bytes) == cuda::CUDA_SUCCESS;
    if (pinned_cap_ < bytes) {
      if (pinned_) driver_.MemFreeHost(pinned_);
      pinned_ = nullptr;
      size_t want = bytes + bytes / 2; /* headroom for the next frames */
      if (driver_.MemAllocHost(&pinned_, want) != cuda::CUDA_SUCCESS || !pinned_) {
        pinned_ = nullptr;
        pinned_cap_ = 0;
        return driver_.MemcpyHtoD(dst, src, bytes) == cuda::CUDA_SUCCESS;
      }
      pinned_cap_ = want;
    }
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

  bool launch(const char* name, int elements, Launcher& launcher) {
    if (elements <= 0) return true;
    if (!launcher.ok()) return false;
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
  /* page-locked host staging for the large per frame uploads */
  void* pinned_ = nullptr;
  size_t pinned_cap_ = 0;
  /* per frame arenas the parallel wave prep fills at fixed offsets */
  CUdeviceptr job_arena_ = 0;
  size_t job_arena_cap_ = 0;
  CUdeviceptr pref_arena_ = 0;
  size_t pref_arena_cap_ = 0;
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
  AV1GPU_CU(EventCreate, "cuEventCreate");
  AV1GPU_CU(EventRecord, "cuEventRecord");
  AV1GPU_CU(EventSynchronize, "cuEventSynchronize");
  AV1GPU_CU(EventElapsedTime, "cuEventElapsedTime");
  AV1GPU_CU(EventDestroy, "cuEventDestroy");
  AV1GPU_CU(MemAlloc, "cuMemAlloc_v2");
  AV1GPU_CU(MemFree, "cuMemFree_v2");
  AV1GPU_CU(MemcpyHtoD, "cuMemcpyHtoD_v2");
  AV1GPU_CU(MemcpyDtoH, "cuMemcpyDtoH_v2");
  AV1GPU_CU(MemcpyDtoD, "cuMemcpyDtoD_v2");
  AV1GPU_CU(MemAllocHost, "cuMemAllocHost_v2");
  AV1GPU_CU(MemFreeHost, "cuMemFreeHost");
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
