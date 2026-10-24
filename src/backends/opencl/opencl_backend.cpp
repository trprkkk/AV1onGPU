/*
 * AV1onGPU - OpenCL backend
 *
 * Runs the whole reconstruction chain as OpenCL kernels, which is what makes it
 * usable on GPUs that have no AV1 (or even VP9) fixed function decoder: the
 * shaders do the work. It targets OpenCL 2.0 because the shared kernel header
 * relies on generic address spaces.
 *
 * Buffer strategy
 *   - one flat `frame` buffer holds the frame being decoded (3 planes back to
 *     back, offsets come from the plan),
 *   - one `ref` pool holds the reference frames, so a decoded frame can become
 *     a reference without ever touching host memory,
 *   - the coefficient pool, the transform scratch and the job arrays are
 *     uploaded once per frame (or per wave).
 */
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "backends/opencl/cl_min.hpp"
#include "core/backend.hpp"
#include "core/kernel_source.hpp"

namespace av1gpu {
namespace internal {

using namespace ::av1gpu::opencl;

namespace {

double now_ms() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point start = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

inline size_t round_up(size_t v, size_t m) { return ((v + m - 1) / m) * m; }

/* ------------------------------------------------------------------------- */

class OpenCLBackend final : public Backend {
 public:
  explicit OpenCLBackend(const BackendOptions& options) : options_(options) {}
  ~OpenCLBackend() override { shutdown(); }

  const DeviceInfo& info() const override { return info_; }
  const BackendOptions& options() const override { return options_; }

  bool init(std::string* error) {
    if (!opencl::load(&api_, &library_, error)) return false;

    cl_uint nplat = 0;
    if (api_.GetPlatformIDs(0, nullptr, &nplat) != opencl::CL_SUCCESS || nplat == 0) {
      *error = "no OpenCL platform";
      return false;
    }
    std::vector<cl_platform_id> platforms(nplat);
    api_.GetPlatformIDs(nplat, platforms.data(), nullptr);

    /* pick the device with the requested index across all platforms */
    int seen = 0;
    cl_device_id chosen = nullptr;
    cl_platform_id chosen_platform = nullptr;
    for (cl_uint p = 0; p < nplat && !chosen; ++p) {
      cl_uint ndev = 0;
      if (api_.GetDeviceIDs(platforms[p], opencl::CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev) !=
              opencl::CL_SUCCESS ||
          ndev == 0)
        continue;
      std::vector<cl_device_id> devices(ndev);
      api_.GetDeviceIDs(platforms[p], opencl::CL_DEVICE_TYPE_ALL, ndev, devices.data(),
                        nullptr);
      for (cl_uint d = 0; d < ndev; ++d) {
        if (seen++ == options_.device_index) {
          chosen = devices[d];
          chosen_platform = platforms[p];
          break;
        }
      }
    }
    if (!chosen) {
      *error = "OpenCL device index out of range";
      return false;
    }
    platform_ = chosen_platform;
    device_ = chosen;

    char name[256] = {0};
    char vendor[256] = {0};
    char version[256] = {0};
    char driver[256] = {0};
    cl_ulong mem = 0;
    cl_uint cus = 0;
    cl_uint max_wg = 0;
    api_.GetDeviceInfo(device_, opencl::CL_DEVICE_NAME, sizeof(name), name, nullptr);
    api_.GetDeviceInfo(device_, opencl::CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    api_.GetDeviceInfo(device_, opencl::CL_DEVICE_VERSION, sizeof(version), version, nullptr);
    api_.GetDeviceInfo(device_, opencl::CL_DEVICE_DRIVER_VERSION, sizeof(driver), driver,
                       nullptr);
    api_.GetDeviceInfo(device_, opencl::CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    api_.GetDeviceInfo(device_, opencl::CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    api_.GetDeviceInfo(device_, opencl::CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg,
                       nullptr);
    info_.kind = BackendKind::OpenCL;
    info_.name = name;
    info_.vendor = vendor;
    info_.version = version;
    info_.detail = std::string("driver ") + driver + ", " + std::to_string(cus) +
                   " compute units, max workgroup " + std::to_string(max_wg);
    info_.global_mem = mem;
    info_.compute_units = static_cast<int>(cus);
    info_.available = true;
    max_local_ = static_cast<int>(max_wg);
    if (max_wg < 64) {
      *error = "device workgroup limit too small";
      return false;
    }

    cl_int rc = 0;
    const opencl::cl_context_properties props[] = {0};
    context_ = api_.CreateContext(props, 1, &device_, nullptr, nullptr, &rc);
    if (!context_ || rc != opencl::CL_SUCCESS) {
      *error = "clCreateContext failed";
      return false;
    }
    queue_ = api_.CreateCommandQueue(context_, device_, 0, &rc);
    if (!queue_ || rc != opencl::CL_SUCCESS) {
      *error = "clCreateCommandQueue failed";
      return false;
    }
    return build_program(options_.verbose ? error : nullptr, error);
  }

  PlanLayout prepare(const FrameGeometry& geometry) override {
    PlanLayout g;
    g.frame = geometry;
    g.plane_w[0] = geometry.width;
    g.plane_h[0] = geometry.height;
    g.plane_w[1] = g.plane_w[2] =
        (geometry.width + (1 << geometry.ss_x) - 1) >> geometry.ss_x;
    g.plane_h[1] = g.plane_h[2] =
        (geometry.height + (1 << geometry.ss_y) - 1) >> geometry.ss_y;
    size_t size = 0;
    for (int p = 0; p < 3; ++p) {
      /* 64 sample aligned rows keep the row accesses coalesced */
      g.stride[p] = (g.plane_w[p] + 63) & ~63;
      g.plane_off[p] = static_cast<int>(size);
      size += static_cast<size_t>(g.stride[p]) * g.plane_h[p];
    }
    g.buffer_size = size;
    layout_ = g;
    ensure(&frame_, &frame_cap_, size * sizeof(uint16_t));
    ensure(&scratch_, &scratch_cap_, size * sizeof(uint16_t));
    ensure_ref_pool(size);
    ensure(&filters_, &filters_cap_, tables::filters().size() * sizeof(int), true);
    return g;
  }

  int upload_reference(const Frame& frame, std::string* error) override {
    if (layout_.buffer_size == 0) {
      if (error) *error = "upload_reference before prepare";
      return -1;
    }
    const int slot = allocate_slot();
    if (slot < 0) {
      if (error) *error = "OpenCL reference pool exhausted";
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
    if (api_.EnqueueWriteBuffer(queue_, ref_pool_, opencl::CL_TRUE,
                                static_cast<size_t>(slot) * size * sizeof(uint16_t),
                                size * sizeof(uint16_t), packed.data(), 0, nullptr,
                                nullptr) != opencl::CL_SUCCESS) {
      if (error) *error = "clEnqueueWriteBuffer failed";
      slots_[slot] = false;
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

    /* ---- static tables ---------------------------------------------------- */
    if (!tables_uploaded_) {
      if (tables::filters().size())
        api_.EnqueueWriteBuffer(queue_, filters_, opencl::CL_TRUE, 0,
                                tables::filters().size() * sizeof(int),
                                tables::filters().data(), 0, nullptr, nullptr);
      tables_uploaded_ = true;
    }

    /* ---- coefficient pool -------------------------------------------------- */
    ensure(&coef_, &coef_cap_, (plan.coefficients.size() + 1) * sizeof(int));
    if (!plan.coefficients.empty())
      api_.EnqueueWriteBuffer(queue_, coef_, opencl::CL_TRUE, 0,
                              plan.coefficients.size() * sizeof(int),
                              plan.coefficients.data(), 0, nullptr, nullptr);
    ensure(&tmp_, &tmp_cap_, (plan.coefficients.size() + 1) * sizeof(int));
    if (!plan.residual.empty()) {
      ensure(&residual_, &residual_cap_, (plan.residual.size() + 1) * sizeof(int));
      api_.EnqueueWriteBuffer(queue_, residual_, opencl::CL_TRUE, 0,
                              plan.residual.size() * sizeof(int), plan.residual.data(), 0,
                              nullptr, nullptr);
    }

    /* ---- waves ------------------------------------------------------------- */
    for (const WaveJobs& wave : plan.waves) {
      if (!wave.inter.empty()) {
        std::vector<Av1gInterJob> jobs = wave.inter;
        for (Av1gInterJob& job : jobs) {
          if (job.ref0_slot >= 0)
            job.ref0_off += job.ref0_slot * static_cast<int>(g.buffer_size);
          if (job.ref1_slot >= 0)
            job.ref1_off += job.ref1_slot * static_cast<int>(g.buffer_size);
        }
        int total = 0;
        cl_mem job_mem = 0, pref_mem = 0;
        if (!upload_jobs(jobs, &job_mem, &pref_mem, &total,
                         [](const Av1gInterJob& j) { return j.w * j.h; }))
          return fail(error, "inter job upload failed");
        cl_kernel k = kernel("av1g_k_inter");
        cl_mem ref_pool = ref_pool_;
        set_mem(k, 0, job_mem);
        set_mem(k, 1, pref_mem);
        set_int(k, 2, static_cast<int>(jobs.size()));
        set_int(k, 3, total);
        set_mem(k, 4, ref_pool);
        set_mem(k, 5, ref_pool);
        set_mem(k, 6, frame_);
        set_mem(k, 7, filters_);
        set_int(k, 8, maxv);
        if (!enqueue(k, total, 64)) return fail(error, "av1g_k_inter failed");
        ++launches;
      }
      if (!wave.intra.empty()) {
        int n = static_cast<int>(wave.intra.size());
        cl_mem job_mem = 0, unused = 0;
        int total = 0;
        if (!upload_jobs(wave.intra, &job_mem, &unused, &total,
                         [](const Av1gIntraJob&) { return 1; }))
          return fail(error, "intra job upload failed");
        (void)total;
        /* one work item per block: intra prediction is a whole block operation */
        cl_kernel k = kernel("av1g_k_intra");
        set_mem(k, 0, job_mem);
        set_int(k, 1, n);
        set_mem(k, 2, frame_);
        set_int(k, 3, maxv);
        if (!enqueue(k, n, 64)) return fail(error, "av1g_k_intra failed");
        ++launches;
      }
      if (!wave.itx.empty()) {
        int rows = 0, cols = 0;
        cl_mem job_mem = 0, row_pref = 0, col_pref = 0;
        if (!upload_jobs(wave.itx, &job_mem, &row_pref, &rows,
                         [](const Av1gItxJob& j) { return j.h; }))
          return fail(error, "itx job upload failed");
        if (!upload_prefix(wave.itx, &col_pref, &cols,
                           [](const Av1gItxJob& j) { return j.w; }))
          return fail(error, "itx prefix upload failed");
        cl_kernel kr = kernel("av1g_k_itx_rows");
        set_mem(kr, 0, job_mem);
        set_mem(kr, 1, row_pref);
        set_int(kr, 2, static_cast<int>(wave.itx.size()));
        set_int(kr, 3, rows);
        set_mem(kr, 4, coef_);
        set_mem(kr, 5, tmp_);
        if (!enqueue(kr, rows, 64)) return fail(error, "av1g_k_itx_rows failed");
        ++launches;

        cl_kernel kc = kernel("av1g_k_itx_cols");
        set_mem(kc, 0, job_mem);
        set_mem(kc, 1, col_pref);
        set_int(kc, 2, static_cast<int>(wave.itx.size()));
        set_int(kc, 3, cols);
        set_mem(kc, 4, tmp_);
        set_mem(kc, 5, frame_);
        set_int(kc, 6, maxv);
        if (!enqueue(kc, cols, 64)) return fail(error, "av1g_k_itx_cols failed");
        ++launches;
      }
      if (!wave.residual.empty()) {
        int total = 0;
        cl_mem job_mem = 0, pref_mem = 0;
        if (!upload_jobs(wave.residual, &job_mem, &pref_mem, &total,
                         [](const Av1gItxJob& j) { return j.w * j.h; }))
          return fail(error, "residual job upload failed");
        cl_kernel k = kernel("av1g_k_add_residual");
        set_mem(k, 0, job_mem);
        set_mem(k, 1, pref_mem);
        set_int(k, 2, static_cast<int>(wave.residual.size()));
        set_int(k, 3, total);
        set_mem(k, 4, residual_);
        set_mem(k, 5, frame_);
        set_int(k, 6, maxv);
        if (!enqueue(k, total, 64)) return fail(error, "av1g_k_add_residual failed");
        ++launches;
      }
    }

    const double t1 = now_ms();

    /* ---- deblock ----------------------------------------------------------- */
    /* vertical edges first, then horizontal ones: the two groups cross */
    for (const std::vector<Av1gDeblockJob>* pass : {&plan.deblock_v, &plan.deblock_h}) {
      if (pass->empty()) continue;
      int total = 0;
      cl_mem job_mem = 0, pref_mem = 0;
      if (!upload_jobs(*pass, &job_mem, &pref_mem, &total,
                       [](const Av1gDeblockJob& j) { return j.len / 4; }))
        return fail(error, "deblock job upload failed");
      cl_kernel k = kernel("av1g_k_deblock");
      set_mem(k, 0, job_mem);
      set_mem(k, 1, pref_mem);
      set_int(k, 2, static_cast<int>(pass->size()));
      set_int(k, 3, total);
      set_mem(k, 4, frame_);
      set_int(k, 5, maxv);
      if (!enqueue(k, total, 64)) return fail(error, "av1g_k_deblock failed");
      ++launches;
    }

    /* ---- CDEF (needs a snapshot of the frame) ------------------------------ */
    if (!plan.cdef.empty()) {
      if (!ensure(&scratch_, &scratch_cap_, g.buffer_size * sizeof(uint16_t)))
        return fail(error, "out of device memory for the CDEF snapshot");
      if (api_.EnqueueCopyBuffer(queue_, frame_, scratch_, 0, 0,
                                 g.buffer_size * sizeof(uint16_t), 0, nullptr,
                                 nullptr) != opencl::CL_SUCCESS)
        return fail(error, "CDEF snapshot copy failed");
      int total = 0;
      cl_mem job_mem = 0, pref_mem = 0;
      if (!upload_jobs(plan.cdef, &job_mem, &pref_mem, &total,
                       [](const Av1gCdefJob&) { return AV1G_CDEF_DIM * AV1G_CDEF_DIM; }))
        return fail(error, "CDEF job upload failed");
      cl_kernel k = kernel("av1g_k_cdef");
      set_mem(k, 0, job_mem);
      set_mem(k, 1, pref_mem);
      set_int(k, 2, static_cast<int>(plan.cdef.size()));
      set_int(k, 3, total);
      set_mem(k, 4, scratch_);
      set_mem(k, 5, frame_);
      set_int(k, 6, maxv);
      if (!enqueue(k, total, 64)) return fail(error, "av1g_k_cdef failed");
      ++launches;
    }

    /* ---- film grain -------------------------------------------------------- */
    if (!plan.grain.empty()) {
      ensure(&grain_, &grain_cap_, plan.grain_template.size());
      api_.EnqueueWriteBuffer(queue_, grain_, opencl::CL_TRUE, 0,
                              plan.grain_template.size(), plan.grain_template.data(), 0,
                              nullptr, nullptr);
      ensure(&lut_, &lut_cap_, sizeof(plan.grain_lut));
      api_.EnqueueWriteBuffer(queue_, lut_, opencl::CL_TRUE, 0, sizeof(plan.grain_lut),
                              plan.grain_lut, 0, nullptr, nullptr);
      int total = 0;
      cl_mem job_mem = 0, pref_mem = 0;
      if (!upload_jobs(plan.grain, &job_mem, &pref_mem, &total,
                       [](const Av1gGrainJob& j) { return j.w * j.h; }))
        return fail(error, "grain job upload failed");
      cl_kernel k = kernel("av1g_k_grain");
      set_mem(k, 0, job_mem);
      set_mem(k, 1, pref_mem);
      set_int(k, 2, static_cast<int>(plan.grain.size()));
      set_int(k, 3, total);
      set_mem(k, 4, frame_);
      set_mem(k, 5, grain_);
      set_mem(k, 6, lut_);
      set_int(k, 7, maxv);
      if (!enqueue(k, total, 64)) return fail(error, "av1g_k_grain failed");
      ++launches;
    }

    if (api_.Finish(queue_) != opencl::CL_SUCCESS) return fail(error, "clFinish failed");
    const double t2 = now_ms();

    /* ---- read back --------------------------------------------------------- */
    if (out) {
      if (!out->matches(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                        g.frame.ss_y))
        out->allocate(g.frame.width, g.frame.height, g.frame.bit_depth, g.frame.ss_x,
                      g.frame.ss_y);
      if (!ensure_staging(g.buffer_size)) return fail(error, "readback staging failed");
      if (api_.EnqueueReadBuffer(queue_, frame_, opencl::CL_TRUE, 0,
                                 g.buffer_size * sizeof(uint16_t), staging_.data(), 0,
                                 nullptr, nullptr) != opencl::CL_SUCCESS)
        return fail(error, "clEnqueueReadBuffer failed");
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

    /* ---- keep as reference (device to device) ------------------------------ */
    int slot = -1;
    if (keep_reference) {
      slot = allocate_slot();
      if (slot < 0) return fail(error, "OpenCL reference pool exhausted");
      if (api_.EnqueueCopyBuffer(queue_, frame_, ref_pool_,
                                 static_cast<size_t>(slot) * g.buffer_size *
                                     sizeof(uint16_t),
                                 0, g.buffer_size * sizeof(uint16_t), 0, nullptr,
                                 nullptr) != opencl::CL_SUCCESS)
        return fail(error, "reference copy failed");
      api_.Finish(queue_);
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
    cl_mem source = frame_;
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
    cl_mem job_mem = 0, pref_mem = 0;
    std::vector<Av1gColorJob> jobs(1, job);
    int total = 0;
    if (!upload_jobs(jobs, &job_mem, &pref_mem, &total,
                     [](const Av1gColorJob& j) { return j.w * j.h; })) {
      *error = "RGB job upload failed";
      return false;
    }
    cl_kernel k = kernel("av1g_k_color");
    set_mem(k, 0, job_mem);
    set_mem(k, 1, pref_mem);
    set_int(k, 2, 1);
    set_int(k, 3, total);
    set_mem(k, 4, source);
    set_mem(k, 5, source);
    set_mem(k, 6, source);
    set_mem(k, 7, rgb_);
    if (!enqueue(k, total, 64)) {
      *error = "av1g_k_color failed";
      return false;
    }
    rgb->resize(rgb_bytes);
    if (api_.EnqueueReadBuffer(queue_, rgb_, opencl::CL_TRUE, 0, rgb_bytes, rgb->data(), 0,
                               nullptr, nullptr) != opencl::CL_SUCCESS) {
      *error = "RGB readback failed";
      return false;
    }
    return true;
  }

  void reset() override {
    for (int i = 0; i < kReferenceSlots; ++i) slots_[i] = false;
    last_slot_ = -1;
  }

  int allocate_slot() {
    for (int i = 0; i < kReferenceSlots; ++i)
      if (!slots_[i]) return i;
    return -1;
  }

 private:
  static constexpr int kReferenceSlots = 6;

  void shutdown() {
    for (auto& entry : kernels_) {
      if (entry.second && api_.ReleaseKernel) api_.ReleaseKernel(entry.second);
    }
    kernels_.clear();
    cl_mem buffers[] = {frame_, scratch_, coef_,    tmp_,     residual_,
                        filters_, ref_pool_, grain_,   lut_,     rgb_,
                        job_a_, job_b_,   pref_a_,  pref_b_};
    for (cl_mem m : buffers)
      if (m && api_.ReleaseMemObject) api_.ReleaseMemObject(m);
    frame_ = scratch_ = coef_ = tmp_ = residual_ = filters_ = nullptr;
    ref_pool_ = grain_ = lut_ = rgb_ = job_a_ = job_b_ = nullptr;
    pref_a_ = pref_b_ = nullptr;
    if (program_ && api_.ReleaseProgram) api_.ReleaseProgram(program_);
    program_ = nullptr;
    if (queue_ && api_.ReleaseCommandQueue) api_.ReleaseCommandQueue(queue_);
    queue_ = nullptr;
    if (context_ && api_.ReleaseContext) api_.ReleaseContext(context_);
    context_ = nullptr;
    if (library_) {
#if defined(_WIN32)
      FreeLibrary(static_cast<HMODULE>(library_));
#else
      dlclose(library_);
#endif
      library_ = nullptr;
    }
  }

  bool build_program(std::string* log_out, std::string* error) {
    const std::string& source = kernel_source::opencl();
    const char* src = source.c_str();
    const size_t len = source.size();
    cl_int rc = 0;
    program_ = api_.CreateProgramWithSource(context_, 1, &src, &len, &rc);
    if (!program_ || rc != opencl::CL_SUCCESS) {
      *error = "clCreateProgramWithSource failed";
      return false;
    }
    const char* options = "-cl-std=CL2.0 -cl-mad-enable";
    rc = api_.BuildProgram(program_, 1, &device_, options, nullptr, nullptr);
    size_t log_size = 0;
    api_.GetProgramBuildInfo(program_, device_, opencl::CL_PROGRAM_BUILD_LOG, 0, nullptr,
                             &log_size);
    std::string build_log(log_size + 1, '\0');
    if (log_size)
      api_.GetProgramBuildInfo(program_, device_, opencl::CL_PROGRAM_BUILD_LOG, log_size,
                               &build_log[0], nullptr);
    if (log_out) *log_out += build_log;
    if (rc != opencl::CL_SUCCESS) {
      *error = "OpenCL kernel build failed:\n" + build_log;
      return false;
    }
    return true;
  }

  cl_kernel kernel(const char* name) {
    auto it = kernels_.find(name);
    if (it != kernels_.end()) return it->second;
    cl_int rc = 0;
    cl_kernel k = api_.CreateKernel(program_, name, &rc);
    kernels_[name] = k;
    return k;
  }

  bool ensure(cl_mem* buffer, size_t* capacity, size_t bytes, bool initial = false) {
    (void)initial;
    if (*buffer && *capacity >= bytes) return true;
    if (*buffer) api_.ReleaseMemObject(*buffer);
    cl_int rc = 0;
    const size_t size = std::max<size_t>(bytes, 1024);
    *buffer = api_.CreateBuffer(context_, opencl::CL_MEM_READ_WRITE, size, nullptr, &rc);
    if (!*buffer || rc != opencl::CL_SUCCESS) {
      *buffer = nullptr;
      *capacity = 0;
      return false;
    }
    *capacity = size;
    return true;
  }

  bool ensure_staging(size_t samples) {
    if (staging_.size() < samples) staging_.resize(samples);
    return true;
  }

  bool ensure_ref_pool(size_t slot_samples) {
    const size_t need = slot_samples * kReferenceSlots * sizeof(uint16_t);
    if (ref_pool_ && ref_cap_ >= need) return true;
    if (ref_pool_) api_.ReleaseMemObject(ref_pool_);
    cl_int rc = 0;
    ref_pool_ = api_.CreateBuffer(context_, opencl::CL_MEM_READ_WRITE, need, nullptr, &rc);
    if (!ref_pool_ || rc != opencl::CL_SUCCESS) {
      ref_pool_ = nullptr;
      ref_cap_ = 0;
      return false;
    }
    ref_cap_ = need;
    return true;
  }

  /* Uploads only the prefix-sum table. The transform passes need two of them:
   * the row pass has one work item per row (job.h) and the column pass one per
   * column (job.w), and both read the same job array. */
  template <typename JobT, typename Fn>
  bool upload_prefix(const std::vector<JobT>& jobs, cl_mem* pref_mem, int* total,
                     Fn element_count) {
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
    return api_.EnqueueWriteBuffer(queue_, pref_b_, opencl::CL_TRUE, 0,
                                   prefix.size() * sizeof(int), prefix.data(), 0,
                                   nullptr, nullptr) == opencl::CL_SUCCESS;
  }

  /* Uploads a job array plus its prefix sum table; returns the element total. */
  template <typename JobT, typename Fn>
  bool upload_jobs(const std::vector<JobT>& jobs, cl_mem* job_mem, cl_mem* pref_mem,
                   int* total, Fn element_count) {
    if (!ensure(&job_a_, &job_a_cap_, jobs.size() * sizeof(JobT))) return false;
    if (!ensure(&pref_a_, &pref_a_cap_, (jobs.size() + 2) * sizeof(int))) return false;
    *job_mem = job_a_;
    *pref_mem = pref_a_;
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
    if (api_.EnqueueWriteBuffer(queue_, job_a_, opencl::CL_TRUE, 0,
                                jobs.size() * sizeof(JobT), jobs.data(), 0, nullptr,
                                nullptr) != opencl::CL_SUCCESS)
      return false;
    if (api_.EnqueueWriteBuffer(queue_, pref_a_, opencl::CL_TRUE, 0, prefix.size() * sizeof(int),
                                prefix.data(), 0, nullptr, nullptr) != opencl::CL_SUCCESS)
      return false;
    return true;
  }

  void set_mem(cl_kernel k, cl_uint index, cl_mem mem) {
    api_.SetKernelArg(k, index, sizeof(cl_mem), &mem);
  }
  void set_int(cl_kernel k, cl_uint index, int value) {
    api_.SetKernelArg(k, index, sizeof(int), &value);
  }

  /* Launches `elements` work items. The preferred workgroup size is only a
   * hint: the per kernel limit can be lower than the device maximum (the
   * deblocking kernel needs more registers, for instance), so a rejected
   * launch is retried with a smaller group and finally without one. */
  bool enqueue(cl_kernel k, int elements, int local) {
    if (elements <= 0) return true;
    if (max_local_ > 0 && local > max_local_) local = max_local_;
    for (int attempt = 0; attempt < 4; ++attempt) {
      const size_t local_size = static_cast<size_t>(local);
      const size_t global = round_up(static_cast<size_t>(elements), local_size);
      const cl_int rc = api_.EnqueueNDRangeKernel(queue_, k, 1, nullptr, &global,
                                                  &local_size, 0, nullptr, nullptr);
      if (rc == opencl::CL_SUCCESS) return true;
      if (rc != opencl::CL_INVALID_WORK_GROUP_SIZE) return false;
      local = local > 8 ? local / 8 : 0;
      if (local <= 0) {
        /* no workgroup at all: the runtime picks one */
        const size_t plain = static_cast<size_t>(elements);
        return api_.EnqueueNDRangeKernel(queue_, k, 1, nullptr, &plain, nullptr, 0,
                                         nullptr, nullptr) == opencl::CL_SUCCESS;
      }
    }
    return false;
  }

  static bool fail(std::string* error, const char* message) {
    *error = message;
    return false;
  }

  BackendOptions options_;
  DeviceInfo info_;
  opencl::Api api_{};
  void* library_ = nullptr;
  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  std::map<std::string, cl_kernel> kernels_;

  PlanLayout layout_;
  bool tables_uploaded_ = false;

  cl_mem frame_ = nullptr, scratch_ = nullptr, coef_ = nullptr, tmp_ = nullptr;
  cl_mem residual_ = nullptr, filters_ = nullptr;
  cl_mem ref_pool_ = nullptr, grain_ = nullptr, lut_ = nullptr;
  cl_mem rgb_ = nullptr, job_a_ = nullptr, job_b_ = nullptr;
  cl_mem pref_a_ = nullptr, pref_b_ = nullptr;
  size_t frame_cap_ = 0, scratch_cap_ = 0, coef_cap_ = 0, tmp_cap_ = 0;
  size_t residual_cap_ = 0, filters_cap_ = 0, ref_cap_ = 0;
  size_t grain_cap_ = 0, lut_cap_ = 0, rgb_cap_ = 0;
  size_t job_a_cap_ = 0, pref_a_cap_ = 0, pref_b_cap_ = 0;
  int max_local_ = 0;
  std::vector<uint16_t> staging_;
  bool slots_[kReferenceSlots] = {false, false, false, false, false, false};
  int last_slot_ = -1;
};

} /* namespace */

} /* namespace internal */

/* ------------------------------------------------------------------------- */

namespace opencl {

bool load(Api* api, void** library, std::string* error) {
  const char* candidates[] = {
#if defined(_WIN32)
      "OpenCL.dll",
#elif defined(__APPLE__)
      "/System/Library/Frameworks/OpenCL.framework/OpenCL",
      "libOpenCL.dylib",
#else
      "libOpenCL.so.1", "libOpenCL.so", "libOpenCL.so.2",
#endif
  };
  for (const char* name : candidates) {
#if defined(_WIN32)
    *library = LoadLibraryA(name);
#else
    *library = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
#endif
    if (*library) break;
  }
  if (!*library) {
    *error =
        "no OpenCL loader found (install the ICD loader, e.g. ocl-icd-libopencl1)";
    return false;
  }

  std::string missing;
#define AV1GPU_CL_SYM(field, name)                                            \
  do {                                                                       \
    *(void**)(&api->field) = dlsym(*library, name);                           \
    if (!api->field) missing += std::string(missing.empty() ? "" : ", ") + name; \
  } while (0)
  AV1GPU_CL_SYM(GetPlatformIDs, "clGetPlatformIDs");
  AV1GPU_CL_SYM(GetPlatformInfo, "clGetPlatformInfo");
  AV1GPU_CL_SYM(GetDeviceIDs, "clGetDeviceIDs");
  AV1GPU_CL_SYM(GetDeviceInfo, "clGetDeviceInfo");
  AV1GPU_CL_SYM(CreateContext, "clCreateContext");
  AV1GPU_CL_SYM(ReleaseContext, "clReleaseContext");
  AV1GPU_CL_SYM(CreateCommandQueue, "clCreateCommandQueue");
  AV1GPU_CL_SYM(ReleaseCommandQueue, "clReleaseCommandQueue");
  AV1GPU_CL_SYM(CreateProgramWithSource, "clCreateProgramWithSource");
  AV1GPU_CL_SYM(BuildProgram, "clBuildProgram");
  AV1GPU_CL_SYM(GetProgramBuildInfo, "clGetProgramBuildInfo");
  AV1GPU_CL_SYM(ReleaseProgram, "clReleaseProgram");
  AV1GPU_CL_SYM(CreateKernel, "clCreateKernel");
  AV1GPU_CL_SYM(ReleaseKernel, "clReleaseKernel");
  AV1GPU_CL_SYM(SetKernelArg, "clSetKernelArg");
  AV1GPU_CL_SYM(CreateBuffer, "clCreateBuffer");
  AV1GPU_CL_SYM(ReleaseMemObject, "clReleaseMemObject");
  AV1GPU_CL_SYM(EnqueueWriteBuffer, "clEnqueueWriteBuffer");
  AV1GPU_CL_SYM(EnqueueReadBuffer, "clEnqueueReadBuffer");
  AV1GPU_CL_SYM(EnqueueCopyBuffer, "clEnqueueCopyBuffer");
  AV1GPU_CL_SYM(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
  AV1GPU_CL_SYM(Finish, "clFinish");
  AV1GPU_CL_SYM(Flush, "clFlush");
#undef AV1GPU_CL_SYM
  if (!missing.empty()) {
    *error = "OpenCL loader is missing entry points: " + missing;
    return false;
  }
  return true;
}

} /* namespace opencl */

namespace internal {

std::unique_ptr<Backend> create_opencl_backend(const BackendOptions& options,
                                              std::string* error) {
  std::string local;
  if (!error) error = &local;
  std::unique_ptr<OpenCLBackend> backend(new OpenCLBackend(options));
  if (!backend->init(error)) return nullptr;
  return std::unique_ptr<Backend>(backend.release());
}

std::vector<DeviceInfo> list_opencl_devices() {
  std::vector<DeviceInfo> out;
  opencl::Api api{};
  void* library = nullptr;
  std::string error;
  if (!opencl::load(&api, &library, &error)) return out;
  cl_uint nplat = 0;
  if (api.GetPlatformIDs(0, nullptr, &nplat) != opencl::CL_SUCCESS) {
#if !defined(_WIN32)
    dlclose(library);
#else
    FreeLibrary(static_cast<HMODULE>(library));
#endif
    return out;
  }
  std::vector<cl_platform_id> platforms(nplat);
  api.GetPlatformIDs(nplat, platforms.data(), nullptr);
  for (cl_uint p = 0; p < nplat; ++p) {
    cl_uint ndev = 0;
    if (api.GetDeviceIDs(platforms[p], opencl::CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev) !=
        opencl::CL_SUCCESS)
      continue;
    std::vector<cl_device_id> devices(ndev);
    api.GetDeviceIDs(platforms[p], opencl::CL_DEVICE_TYPE_ALL, ndev, devices.data(),
                     nullptr);
    for (cl_uint d = 0; d < ndev; ++d) {
      DeviceInfo info;
      info.kind = BackendKind::OpenCL;
      char name[256] = {0};
      char vendor[256] = {0};
      char version[256] = {0};
      cl_ulong mem = 0;
      cl_uint cus = 0;
      api.GetDeviceInfo(devices[d], opencl::CL_DEVICE_NAME, sizeof(name), name, nullptr);
      api.GetDeviceInfo(devices[d], opencl::CL_DEVICE_VENDOR, sizeof(vendor), vendor,
                        nullptr);
      api.GetDeviceInfo(devices[d], opencl::CL_DEVICE_VERSION, sizeof(version), version,
                        nullptr);
      api.GetDeviceInfo(devices[d], opencl::CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem,
                        nullptr);
      api.GetDeviceInfo(devices[d], opencl::CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus,
                        nullptr);
      info.name = name;
      info.vendor = vendor;
      info.version = version;
      info.global_mem = mem;
      info.compute_units = static_cast<int>(cus);
      info.available = true;
      out.push_back(info);
    }
  }
#if !defined(_WIN32)
  dlclose(library);
#else
  FreeLibrary(static_cast<HMODULE>(library));
#endif
  return out;
}

} /* namespace internal */
} /* namespace av1gpu */
