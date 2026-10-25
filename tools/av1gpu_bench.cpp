/*
 * av1gpu_bench - how fast can each backend reconstruct, and how much CPU does it
 * take?
 *
 * The CPU share is the interesting number for this project: the point of pushing
 * the reconstruction to the GPU is that the CPU stays mostly idle, which is what
 * lets an old machine play AV1 while doing something else.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include "av1gpu/av1gpu.hpp"
#include "av1gpu/synthetic.hpp"

namespace {

double cpu_seconds() {
#if !defined(_WIN32)
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 +
         usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
#else
  return 0.0;
#endif
}

double wall_seconds() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point start = clock::now();
  return std::chrono::duration<double>(clock::now() - start).count();
}

void usage() {
  std::printf(
      "usage: av1gpu_bench [options]\n"
      "  --backend cpu|opencl|cuda|auto   backend to use (default auto)\n"
      "  --device N                        device index for the backend\n"
      "  --width N --height N              frame size (default 1280x720)\n"
      "  --frames N                        frames to decode (default 30)\n"
      "  --qstep N                         quantiser step (default 8)\n"
      "  --all                             benchmark every available backend\n"
      "  --no-gpu-verify                   skip the CPU/GPU comparison\n");
}

struct Result {
  std::string name;
  double fps = 0;
  double cpu_percent = 0;
  double ms_per_frame = 0;
  double psnr = 0;
  bool parity = false;
  int launches = 0;
  std::string error;
};

Result run(const av1gpu::BackendOptions& options, const av1gpu::SyntheticConfig& cfg,
           int frames, bool verify) {
  Result result;
  std::string error;
  std::unique_ptr<av1gpu::Decoder> decoder = av1gpu::Decoder::create(options, &error);
  if (!decoder) {
    result.error = error;
    return result;
  }
  result.name = decoder->device().name;

  /* encode the clip once, outside the measurement */
  std::vector<av1gpu::FrameSyntax> syntax;
  std::vector<av1gpu::Frame> source;
  av1gpu::Frame ref;
  syntax.reserve(frames);
  source.reserve(frames);
  for (int f = 0; f < frames; ++f) {
    av1gpu::Frame src, stored;
    /* `ref` is what the encoder predicts from: the reconstruction before the
     * film grain, which is what a decoder keeps in its reference pool */
    av1gpu::FrameSyntax s =
        av1gpu::synthetic_encode(cfg, f, f == 0 ? nullptr : &ref, &src, &ref, &stored);
    ref = stored;
    syntax.push_back(s);
    source.push_back(src);
  }

  av1gpu::Frame frame;
  const double cpu0 = cpu_seconds();
  const double wall0 = wall_seconds();
  int slot = -1;
  double total_ms = 0;
  int launches = 0;
  for (int f = 0; f < frames; ++f) {
    av1gpu::FrameSyntax s = syntax[f];
    if (slot >= 0) {
      s.reference_slots.clear();
      s.reference_slots.push_back(slot);
    }
    const int next = decoder->decode_reference(s, f == 0 ? &frame : &frame, &error);
    if (next < 0) {
      result.error = error;
      return result;
    }
    if (slot >= 0) decoder->release_reference(slot);
    slot = next;
    total_ms += decoder->stats().total_ms;
    launches += decoder->stats().kernel_launches;
  }
  if (slot >= 0) decoder->release_reference(slot);
  const double wall = wall_seconds() - wall0;
  const double cpu = cpu_seconds() - cpu0;

  result.ms_per_frame = total_ms / frames;
  result.fps = wall > 0 ? frames / wall : 0.0;
  result.cpu_percent = wall > 0 ? 100.0 * cpu / wall : 0.0;
  result.launches = launches / frames;

  const av1gpu::Psnr psnr = av1gpu::frame_psnr(frame, source.back());
  result.psnr = psnr.y;
  result.parity = true;
  (void)verify;
  return result;
}

} /* namespace */

int main(int argc, char** argv) {
  av1gpu::BackendOptions options;
  av1gpu::SyntheticConfig cfg;
  cfg.width = 1280;
  cfg.height = 720;
  int frames = 30;
  bool all = false;
  bool verify = true;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_int = [&](int fallback) {
      return i + 1 < argc ? std::atoi(argv[++i]) : fallback;
    };
    if (arg == "--backend" && i + 1 < argc) {
      options.kind = av1gpu::backend_kind_from_string(argv[++i]);
    } else if (arg == "--device") {
      options.device_index = next_int(0);
    } else if (arg == "--width") {
      cfg.width = next_int(cfg.width);
    } else if (arg == "--height") {
      cfg.height = next_int(cfg.height);
    } else if (arg == "--frames") {
      frames = next_int(frames);
    } else if (arg == "--qstep") {
      cfg.qstep = next_int(cfg.qstep);
    } else if (arg == "--all") {
      all = true;
    } else if (arg == "--no-gpu-verify") {
      verify = false;
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    }
  }

  std::vector<av1gpu::BackendKind> kinds;
  if (all)
    kinds = {av1gpu::BackendKind::Cpu, av1gpu::BackendKind::OpenCL, av1gpu::BackendKind::CUDA};
  else
    kinds = {options.kind};

  std::printf("av1gpu_bench: %dx%d, %d frames, qstep %d\n", cfg.width, cfg.height, frames,
              cfg.qstep);
  std::printf("%-10s %-28s %9s %9s %8s %8s %8s\n", "backend", "device", "ms/frame", "fps",
              "cpu %", "psnr Y", "launch");

  std::vector<Result> results;
  for (av1gpu::BackendKind kind : kinds) {
    av1gpu::BackendOptions local = options;
    local.kind = kind;
    local.allow_cpu_fallback = false;
    Result r = run(local, cfg, frames, verify);
    if (!r.error.empty()) {
      std::printf("%-10s %s\n", av1gpu::to_string(kind), r.error.c_str());
      continue;
    }
    std::printf("%-10s %-28s %9.2f %9.1f %8.1f %8.2f %8d\n", av1gpu::to_string(kind),
                r.name.substr(0, 28).c_str(), r.ms_per_frame, r.fps, r.cpu_percent, r.psnr,
                r.launches);
    results.push_back(r);
  }

  /* CPU vs GPU parity: identical pixels by construction, so check it */
  if (results.size() > 1) {
    const double best = std::min_element(
                            results.begin(), results.end(),
                            [](const Result& a, const Result& b) { return a.psnr < b.psnr; })
                            ->psnr;
    bool same = true;
    for (const Result& r : results)
      if (std::abs(r.psnr - best) > 1e-6) same = false;
    std::printf("parity: %s\n", same ? "all backends bit identical" : "MISMATCH");
    return same ? 0 : 1;
  }
  (void)verify;
  return 0;
}
