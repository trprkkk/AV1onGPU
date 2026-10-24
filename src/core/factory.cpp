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
    storage[p].assign(static_cast<size_t>(stride[p]) * ph, 0);
  }
}

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

void frame_to_yuv8(const Frame& frame, std::vector<uint8_t>* out) {
  const int w = frame.width;
  const int h = frame.height;
  out->resize(static_cast<size_t>(w) * h * 3 / 2);
  const int shift = frame.bit_depth - 8;
  size_t pos = 0;
  for (int y = 0; y < h; ++y) {
    const uint16_t* row = frame.plane(0) + static_cast<size_t>(y) * frame.stride[0];
    for (int x = 0; x < w; ++x) (*out)[pos++] = static_cast<uint8_t>(row[x] >> shift);
  }
  for (int p = 1; p < 3; ++p) {
    const int cw = frame.plane_width(p);
    const int ch = frame.plane_height(p);
    for (int y = 0; y < ch; ++y) {
      const uint16_t* row = frame.plane(p) + static_cast<size_t>(y) * frame.stride[p];
      for (int x = 0; x < cw; ++x) (*out)[pos++] = static_cast<uint8_t>(row[x] >> shift);
    }
  }
}

Decoder::~Decoder() = default;

} /* namespace av1gpu */
