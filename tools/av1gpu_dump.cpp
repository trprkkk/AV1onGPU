/*
 * AV1onGPU - dump decode plans for external validation.
 *
 * The GPU backends can only be exercised on a machine with a working driver.
 * This tool closes that gap: it encodes a synthetic clip, builds the *real*
 * decode plan for every frame (the same function the engine calls), writes the
 * plans out in a flat binary format, and writes the CPU reference decode next
 * to them. tests/opencl_pipeline.py then replays each plan through the real
 * OpenCL kernels with pyopencl and compares against the reference, which
 * validates the job arrays, prefix tables, launch order and element counts
 * without needing an ICD loader in the C++ binary.
 *
 *   av1gpu_dump --width 160 --height 96 --frames 3 --output /tmp/plans
 *
 * File layout (little endian, no padding beyond the POD structs):
 *
 *   frame_<n>.plan
 *     char magic[8] = "AV1GPLAN"
 *     int32 version
 *     int32 width, height, bit_depth, ss_x, ss_y
 *     int32 maxv, blocks
 *     int32 plane_w[3], plane_h[3], stride[3], plane_off[3]
 *     int64 buffer_size
 *     int64 tmp_size
 *     int32 ref_slot[2]
 *     int32 grain_stride, grain_h
 *     int32 grain_lut[64]
 *     int32 coef_count, residual_count, grain_template_count, palette_count
 *     int32 nwaves
 *     for each wave: int32 n_intra, n_inter, n_itx, n_residual, then the arrays
 *     int32 n_deblock_v, n_deblock_h
 *     then the deblock arrays
 *     int32 n_cdef, then the cdef array
 *     int32 n_grain, then the grain array
 *     coefficients[], residual[], grain_template[]
 *     palette_map[]           (palette_count bytes)
 *
 * Version 2 added the palette colour map. It is a byte per sample, one map per
 * plane per palette coded block, concatenated at the offsets the intra jobs
 * carry in `palette_off`; without it a palette block cannot be replayed at all,
 * which is the whole reason it is in the file.
 *
 * Version 3 added tmp_size, the size in ints of the inverse transform scratch
 * buffer that `Av1gItxJob::tmp_off` indexes. A job needs w * h ints of scratch,
 * which is *not* the size of its coefficient area (that one is clipped to 32
 * samples wide for a 64 point transform, and the scratch region is carved from
 * one arena for the whole frame), so a replay that sizes the scratch buffer
 * from the coefficient pool writes past the end of it. Take the size from here
 * and check it against the jobs rather than guessing.
 *
 *   frame_<n>.yuv16
 *     three planes, 16 bit samples, rows of `plane_w` (stride is dropped)
 *
 *   manifest.txt
 *     one line per frame: "frame <n> refs <slot0> <slot1> out <slot>".  The
 *     reference slots say which entries of the reference pool the plan reads;
 *     `out` says where the decoded frame has to be stored so the next plan can
 *     use it as its reference.  A backend keeps that pool on the device, which
 *     is why the replay in tests/opencl_pipeline.py needs it spelled out.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "av1gpu/synthetic.hpp"
#include "core/backend.hpp"
#include "plan_dump.hpp"

namespace {

using namespace av1gpu;

void usage() {
  std::printf(
      "usage: av1gpu_dump [options]\n"
      "  --width N --height N    frame size (default 160x96)\n"
      "  --frames N              frames to dump (default 3)\n"
      "  --qstep N               quantiser step (default 6)\n"
      "  --cdef --grain          enable the optional loop filters\n"
      "  --raw                   drop the loop filters (isolates reconstruction)\n"
      "  --output DIR            directory for frame_<n>.plan / frame_<n>.yuv16\n");
}

} /* namespace */

int main(int argc, char** argv) {
  int width = 160, height = 96, frames = 3, qstep = 6;
  bool cdef = false, grain = false, raw = false;
  std::string output = ".";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_int = [&](int fallback) { return i + 1 < argc ? std::atoi(argv[++i]) : fallback; };
    if (arg == "--width") width = next_int(width);
    else if (arg == "--height") height = next_int(height);
    else if (arg == "--frames") frames = next_int(frames);
    else if (arg == "--qstep") qstep = next_int(qstep);
    else if (arg == "--cdef") cdef = true;
    else if (arg == "--grain") grain = true;
    else if (arg == "--raw") raw = true;
    else if (arg == "--output" && i + 1 < argc) output = argv[++i];
    else if (arg == "--help" || arg == "-h") { usage(); return 0; }
  }

  SyntheticConfig cfg;
  cfg.width = width;
  cfg.height = height;
  cfg.qstep = qstep;
  cfg.cdef = cdef;
  cfg.grain = grain;

  /* The reference backend gives us the layout the jobs are built against. */
  std::string error;
  std::unique_ptr<internal::Backend> backend = internal::create_cpu_backend({}, &error);
  if (!backend) {
    std::fprintf(stderr, "av1gpu_dump: %s\n", error.c_str());
    return 1;
  }

  BackendOptions options;
  options.kind = BackendKind::Cpu;
  options.allow_cpu_fallback = false;
  std::unique_ptr<Decoder> decoder = Decoder::create(options, &error);
  if (!decoder) {
    std::fprintf(stderr, "av1gpu_dump: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::string> manifest;
  Frame reference;   /* encoder side reference */
  Frame decoded;     /* decoder side output */
  int slot = -1;

  for (int f = 0; f < frames; ++f) {
    Frame source, recon;
    FrameSyntax syntax = synthetic_encode(cfg, f, f == 0 ? nullptr : &reference, &source, &recon);
    reference = recon;
    if (raw) {
      /* reconstruction only: the loop filters are dropped from the syntax, so
         the CPU reference and the plan both stop after the waves. */
      syntax.deblock.clear();
      syntax.cdef.clear();
      syntax.apply_grain = false;
    }
    if (slot >= 0) {
      syntax.reference_slots.clear();
      syntax.reference_slots.push_back(slot);
    }

    /* plan, exactly as the engine builds it for a backend */
    internal::FrameGeometry geom;
    geom.width = syntax.width;
    geom.height = syntax.height;
    geom.bit_depth = syntax.bit_depth;
    geom.ss_x = syntax.ss_x;
    geom.ss_y = syntax.ss_y;
    internal::FramePlan plan;
    if (!internal::build_frame_plan(syntax, backend->prepare(geom), options,
                                    backend->reference_capacity(), &plan, &error)) {
      std::fprintf(stderr, "av1gpu_dump: plan failed: %s\n", error.c_str());
      return 1;
    }

    const std::string plan_path = output + "/frame_" + std::to_string(f) + ".plan";
    if (!plan_dump::write_plan(plan_path, plan)) {
      std::fprintf(stderr, "av1gpu_dump: cannot write %s\n", plan_path.c_str());
      return 1;
    }

    /* reference decode with the CPU backend */
    const int next = decoder->decode_reference(syntax, &decoded, &error);
    if (next < 0) {
      std::fprintf(stderr, "av1gpu_dump: decode failed: %s\n", error.c_str());
      return 1;
    }
    if (slot >= 0) decoder->release_reference(slot);
    slot = next;

    manifest.push_back("frame " + std::to_string(f) + " refs " +
                       std::to_string(plan.ref_slot[0]) + " " +
                       std::to_string(plan.ref_slot[1]) + " out " + std::to_string(slot));

    const std::string frame_path = output + "/frame_" + std::to_string(f) + ".yuv16";
    if (!plan_dump::write_frame(frame_path, decoded)) {
      std::fprintf(stderr, "av1gpu_dump: cannot write %s\n", frame_path.c_str());
      return 1;
    }
    std::printf("frame %d: %zu waves, %d blocks -> %s\n", f, plan.waves.size(), plan.blocks,
                plan_path.c_str());
  }
  if (slot >= 0) decoder->release_reference(slot);

  const std::string manifest_path = output + "/manifest.txt";
  std::FILE* manifest_file = std::fopen(manifest_path.c_str(), "wb");
  if (!manifest_file) {
    std::fprintf(stderr, "av1gpu_dump: cannot write %s\n", manifest_path.c_str());
    return 1;
  }
  for (const std::string& line : manifest) std::fprintf(manifest_file, "%s\n", line.c_str());
  std::fclose(manifest_file);
  return 0;
}
