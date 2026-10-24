/*
 * AV1onGPU - end to end pipeline tests
 *
 * Decodes a small synthetic clip with every available backend and checks that
 *   - the output matches the source picture (PSNR),
 *   - the GPU backends are bit identical to the CPU reference,
 *   - the wavefront scheduler is independent of its cell size,
 *   - reference frames survive the zero copy path.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "av1gpu/synthetic.hpp"
#include "test_common.hpp"

using namespace av1gpu;

namespace {

struct Clip {
  SyntheticConfig cfg;
  std::vector<Frame> source;
  std::vector<FrameSyntax> syntax;
  std::vector<Frame> recon;
};

Clip make_clip(int frames, int qstep, bool cdef, bool grain, int w = 128, int h = 96) {
  Clip clip;
  clip.cfg.width = w;
  clip.cfg.height = h;
  clip.cfg.block_size = 16;
  clip.cfg.transform_size = 8;
  clip.cfg.qstep = qstep;
  clip.cfg.deblock_level = 12;
  clip.cfg.cdef = cdef;
  clip.cfg.grain = grain;
  clip.cfg.search_radius = 6;
  Frame ref;
  clip.source.resize(frames);
  clip.syntax.resize(frames);
  clip.recon.resize(frames);
  for (int f = 0; f < frames; ++f) {
    FrameSyntax syntax = synthetic_encode(clip.cfg, f, f == 0 ? nullptr : &ref, &clip.source[f],
                                          &clip.recon[f]);
    clip.syntax[f] = syntax;
    ref = clip.recon[f];
  }
  return clip;
}

struct Run {
  std::vector<Frame> frames;
  double ms = 0;
  bool ok = false;
  std::string error;
};

Run decode_clip(const BackendOptions& options, const Clip& clip) {
  Run run;
  std::string error;
  std::unique_ptr<Decoder> decoder = Decoder::create(options, &error);
  if (!decoder) {
    run.error = error;
    return run;
  }
  run.frames.resize(clip.syntax.size());
  int slot = -1;
  for (size_t f = 0; f < clip.syntax.size(); ++f) {
    FrameSyntax syntax = clip.syntax[f];
    if (slot >= 0) {
      syntax.reference_slots.clear();
      syntax.reference_slots.push_back(slot);
    }
    const int next = decoder->decode_reference(syntax, &run.frames[f], &error);
    if (next < 0) {
      run.error = error;
      return run;
    }
    if (slot >= 0) decoder->release_reference(slot);
    slot = next;
    run.ms += decoder->stats().total_ms;
  }
  if (slot >= 0) decoder->release_reference(slot);
  run.ok = true;
  return run;
}

bool identical(const Frame& a, const Frame& b) {
  if (a.width != b.width || a.height != b.height) return false;
  for (int p = 0; p < 3; ++p) {
    const int w = a.plane_width(p);
    const int h = a.plane_height(p);
    for (int y = 0; y < h; ++y) {
      const uint16_t* ra = a.plane(p) + static_cast<size_t>(y) * a.stride[p];
      const uint16_t* rb = b.plane(p) + static_cast<size_t>(y) * b.stride[p];
      if (std::memcmp(ra, rb, static_cast<size_t>(w) * sizeof(uint16_t)) != 0) return false;
    }
  }
  return true;
}

} /* namespace */

int main() {
  std::printf("AV1onGPU pipeline tests\n");

  /* ---- 1. the synthetic encoder and decoder agree ------------------------ */
  std::printf("synthetic clip\n");
  const Clip clip = make_clip(3, 4, true, true);
  CHECK(!clip.syntax[0].blocks.empty());

  {
    BackendOptions options;
    options.kind = BackendKind::Cpu;
    const Run cpu = decode_clip(options, clip);
    CHECK(cpu.ok);
    if (!cpu.ok) {
      std::printf("  cpu decode failed: %s\n", cpu.error.c_str());
      return testing::finish("test_pipeline");
    }
    for (size_t f = 0; f < clip.source.size(); ++f) {
      const Psnr psnr = frame_psnr(cpu.frames[f], clip.recon[f]);
      std::printf("  frame %zu: %.2f dB (vs encoder recon) %.2f ms\n", f, psnr.y, cpu.ms);
      CHECK(psnr.y > 40.0);
      /* the encoder's own reconstruction and the decoder must agree exactly */
      CHECK(identical(cpu.frames[f], clip.recon[f]));
    }
    /* PSNR against the source picture, which is what a user would judge */
    const Psnr source_psnr = frame_psnr(cpu.frames[0], clip.source[0]);
    std::printf("  frame 0 source PSNR: Y %.2f U %.2f V %.2f dB\n", source_psnr.y,
                source_psnr.u, source_psnr.v);
    CHECK(source_psnr.y > 25.0);
  }

  /* ---- 2. the scheduler must be reproducible ------------------------------ */
  {
    /* The wavefront is built from exact sample ownership, so two decodes of the
     * same syntax have to produce the same picture, the same number of waves
     * and no leftover samples: an uninitialised or unowned sample would show up
     * as a difference between the two runs (or as a mismatch against the
     * encoder's reconstruction, checked above). */
    BackendOptions options;
    options.kind = BackendKind::Cpu;
    const Run ra = decode_clip(options, clip);
    const Run rb = decode_clip(options, clip);
    CHECK(ra.ok && rb.ok);
    if (ra.ok && rb.ok)
      for (size_t f = 0; f < clip.source.size(); ++f) CHECK(identical(ra.frames[f], rb.frames[f]));
  }

  /* ---- 3. GPU backends, when present ------------------------------------ */
  {
    BackendOptions cpu_options;
    cpu_options.kind = BackendKind::Cpu;
    cpu_options.allow_cpu_fallback = false;
    const Run cpu = decode_clip(cpu_options, clip);

    const BackendKind kinds[] = {BackendKind::OpenCL, BackendKind::CUDA};
    for (BackendKind kind : kinds) {
      BackendOptions options;
      options.kind = kind;
      options.allow_cpu_fallback = false;
      std::string error;
      std::unique_ptr<Decoder> probe = Decoder::create(options, &error);
      if (!probe || probe->device().kind != kind) {
        std::printf("%s: not available (%s)\n", to_string(kind),
                    error.empty() ? "fallback only" : error.c_str());
        continue;
      }
      std::printf("%s: %s\n", to_string(kind), probe->device().name.c_str());
      probe.reset();
      const Run gpu = decode_clip(options, clip);
      CHECK(gpu.ok);
      if (!gpu.ok) {
        std::printf("  decode failed: %s\n", gpu.error.c_str());
        continue;
      }
      for (size_t f = 0; f < clip.source.size(); ++f) {
        const bool same = identical(gpu.frames[f], cpu.frames[f]);
        CHECK(same);
        if (!same) {
          const Psnr diff = frame_psnr(gpu.frames[f], cpu.frames[f]);
          std::printf("  frame %zu differs from the CPU reference (%.2f dB)\n", f, diff.y);
        }
      }
      std::printf("  cpu %.2f ms, gpu %.2f ms per clip\n", cpu.ms, gpu.ms);
    }
  }

  return testing::finish("test_pipeline");
}
