/*
 * AV1onGPU - device enumeration and public helpers
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/backend.hpp"

namespace av1gpu {

const char* to_string(BackendKind kind) {
  switch (kind) {
    case BackendKind::Cpu:
      return "cpu";
    case BackendKind::OpenCL:
      return "opencl";
    case BackendKind::CUDA:
      return "cuda";
    case BackendKind::Auto:
    default:
      return "auto";
  }
}

BackendKind backend_kind_from_string(const std::string& name) {
  std::string lower;
  lower.reserve(name.size());
  for (char c : name) lower.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
  if (lower == "cpu" || lower == "reference") return BackendKind::Cpu;
  if (lower == "opencl" || lower == "cl" || lower == "gpu") return BackendKind::OpenCL;
  if (lower == "cuda" || lower == "nvidia") return BackendKind::CUDA;
  return BackendKind::Auto;
}

std::vector<DeviceInfo> enumerate_devices() {
  std::vector<DeviceInfo> devices;

  DeviceInfo cpu;
  cpu.kind = BackendKind::Cpu;
  cpu.name = "CPU reference";
  cpu.vendor = "AV1onGPU";
  cpu.version = "portable";
  cpu.available = true;
  devices.push_back(cpu);

  for (const DeviceInfo& info : internal::list_cuda_devices()) devices.push_back(info);
  for (const DeviceInfo& info : internal::list_opencl_devices()) devices.push_back(info);
  return devices;
}

std::string build_info() {
  std::string out;
  out += "AV1onGPU reconstruction library\n";
#if defined(AV1GPU_HAVE_OPENCL)
  out += "  OpenCL backend: enabled (kernels compiled at run time)\n";
#else
  out += "  OpenCL backend: disabled at build time\n";
#endif
#if defined(AV1GPU_HAVE_CUDA)
  out += "  CUDA backend:   enabled (kernels compiled with NVRTC at run time)\n";
#else
  out += "  CUDA backend:   disabled at build time\n";
#endif
#if defined(AV1GPU_HAVE_OPENMP)
  out += "  CPU backend:    OpenMP enabled\n";
#else
  out += "  CPU backend:    single threaded\n";
#endif
  out += "  reconstruction stages: intra, inter, inverse transform, deblock, CDEF, "
         "film grain, YUV->RGB\n";
  return out;
}

/* ------------------------------------------------------------------------- */

int Frame::plane_width(int plane) const {
  if (plane == 0) return width;
  return ss_x ? (width + 1) / 2 : width;
}

int Frame::plane_height(int plane) const {
  if (plane == 0) return height;
  return ss_y ? (height + 1) / 2 : height;
}

void Frame::allocate(int w, int h, int bd, int sx, int sy) {
  width = w;
  height = h;
  bit_depth = bd;
  ss_x = sx;
  ss_y = sy;
  for (int p = 0; p < 3; ++p) {
    const int pw = (p == 0) ? w : (sx ? (w + 1) / 2 : w);
    const int ph = (p == 0) ? h : (sy ? (h + 1) / 2 : h);
    stride[p] = (pw + 31) & ~31;
    /* A few rows of border: the loop filters read up to three samples past an
     * edge, and an edge can sit on the last row band of a plane whose size is
     * not a multiple of four (`av1g_deblock_px` reads q3). A decoded picture
     * lives in the plan's padded buffer anyway; this is what keeps a frame that
     * is allocated on its own - the synthetic encoder's, a test's - inside its
     * storage. */
    const int rows = (ph + 7) & ~7;
    storage[p].assign(static_cast<size_t>(stride[p]) * rows, 0);
  }
}

namespace internal {

PlanLayout make_plan_layout(const FrameGeometry& geometry, int coded_width, int stride_align) {
  PlanLayout g;
  g.frame = geometry;
  if (coded_width <= 0 || coded_width > geometry.width) coded_width = geometry.width;
  /* the display geometry, which the output frame and the reference pool use */
  g.out_width = geometry.width;
  g.out_height = geometry.height;
  g.out_w[0] = geometry.width;
  g.out_h[0] = geometry.height;
  g.out_w[1] = g.out_w[2] =
      (geometry.width + (1 << geometry.ss_x) - 1) >> geometry.ss_x;
  g.out_h[1] = g.out_h[2] =
      (geometry.height + (1 << geometry.ss_y) - 1) >> geometry.ss_y;
  /* the coded geometry: what the tiles code and the reconstruction produces */
  g.plane_w[0] = coded_width;
  g.plane_h[0] = geometry.height;
  g.plane_w[1] = g.plane_w[2] = (coded_width + (1 << geometry.ss_x) - 1) >> geometry.ss_x;
  g.plane_h[1] = g.plane_h[2] =
      (geometry.height + (1 << geometry.ss_y) - 1) >> geometry.ss_y;
  size_t size = 0;
  for (int p = 0; p < 3; ++p) {
    /* The border is in luma samples, so a subsampled plane gets it subsampled
     * too - rounding up, because 64 is even and a chroma block needs the same
     * room relative to its own samples. */
    const int pad = (p == 0) ? kPlanePadding
                             : (kPlanePadding >> (geometry.ss_x ? 1 : 0));
    /* The stride follows the *display* width: the buffer holds the coded
     * picture and then the upscaled one, and both have to address their rows
     * the way a job does. */
    const int pitch = g.out_w[p] + pad;
    g.stride[p] = (pitch + stride_align - 1) & ~(stride_align - 1);
    g.plane_off[p] = static_cast<int>(size);
    /* Rows below the picture are padding as well: the flat buffer simply holds
     * them, and nothing reads them back. */
    size += static_cast<size_t>(g.stride[p]) * (g.out_h[p] + pad);
  }
  g.buffer_size = size;
  return g;
}

} /* namespace internal */

bool Frame::matches(int w, int h, int bd, int sx, int sy) const {
  if (width != w || height != h || bit_depth != bd || ss_x != sx || ss_y != sy)
    return false;
  for (int p = 0; p < 3; ++p) {
    const int pw = (p == 0) ? w : (sx ? (w + 1) / 2 : w);
    const int ph = (p == 0) ? h : (sy ? (h + 1) / 2 : h);
    if (storage[p].size() < static_cast<size_t>(stride[p]) * ph) return false;
    (void)pw;
  }
  return true;
}

Psnr frame_psnr(const Frame& a, const Frame& b) {
  Psnr out;
  if (a.width != b.width || a.height != b.height || a.bit_depth != b.bit_depth) return out;
  const double peak = static_cast<double>((1 << a.bit_depth) - 1);
  double result[3] = {0, 0, 0};
  for (int p = 0; p < 3; ++p) {
    const int w = a.plane_width(p);
    const int h = a.plane_height(p);
    long double sse = 0;
    for (int y = 0; y < h; ++y) {
      const uint16_t* ra = a.plane(p) + static_cast<size_t>(y) * a.stride[p];
      const uint16_t* rb = b.plane(p) + static_cast<size_t>(y) * b.stride[p];
      for (int x = 0; x < w; ++x) {
        const double d = static_cast<double>(ra[x]) - static_cast<double>(rb[x]);
        sse += d * d;
      }
    }
    const double mse = static_cast<double>(sse) / (static_cast<double>(w) * h);
    result[p] = mse <= 1e-12 ? 99.0 : 10.0 * std::log10(peak * peak / mse);
  }
  out.y = result[0];
  out.u = result[1];
  out.v = result[2];
  return out;
}

void frame_to_yuv16(const Frame& frame, std::vector<uint8_t>* out) {
  const int w = frame.width;
  const int h = frame.height;
  size_t samples = static_cast<size_t>(w) * h;
  samples += 2 * static_cast<size_t>(frame.plane_width(1)) * frame.plane_height(1);
  out->resize(samples * 2);
  size_t pos = 0;
  for (int p = 0; p < 3; ++p) {
    const int pw = frame.plane_width(p);
    const int ph = frame.plane_height(p);
    for (int y = 0; y < ph; ++y) {
      const uint16_t* row = frame.plane(p) + static_cast<size_t>(y) * frame.stride[p];
      for (int x = 0; x < pw; ++x) {
        (*out)[pos++] = static_cast<uint8_t>(row[x] & 0xff);
        (*out)[pos++] = static_cast<uint8_t>(row[x] >> 8);
      }
    }
  }
}

void frame_to_yuv8(const Frame& frame, std::vector<uint8_t>* out) {
  /* The size comes from the planes, not from `width * height * 3 / 2`: a 4:4:4
   * frame's chroma planes are as large as its luma one, so the 4:2:0 formula
   * leaves the last two thirds of the picture writing past the buffer. */
  size_t samples = 0;
  for (int p = 0; p < 3; ++p)
    samples += static_cast<size_t>(frame.plane_width(p)) * frame.plane_height(p);
  out->resize(samples);
  const int shift = frame.bit_depth - 8;
  size_t pos = 0;
  for (int p = 0; p < 3; ++p) {
    const int pw = frame.plane_width(p);
    const int ph = frame.plane_height(p);
    for (int y = 0; y < ph; ++y) {
      const uint16_t* row = frame.plane(p) + static_cast<size_t>(y) * frame.stride[p];
      for (int x = 0; x < pw; ++x) (*out)[pos++] = static_cast<uint8_t>(row[x] >> shift);
    }
  }
}

Decoder::~Decoder() = default;

} /* namespace av1gpu */
