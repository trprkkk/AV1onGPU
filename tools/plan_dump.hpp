/*
 * AV1onGPU - write decode plans for external validation
 *
 * The GPU backends can only be exercised on a machine with a working driver.
 * The plan file closes that gap: it holds everything a backend is handed for
 * one frame - the job arrays with their prefix tables, the wave order, the
 * coefficient and residual pools, the loop filter setup, the palette colour map
 * and the reference slots the frame reads - so a driverless machine can replay
 * the frame through the real device kernels (tests/opencl_pipeline.py) and
 * compare the result with the CPU reference decode.
 *
 * The format is documented in av1gpu_dump.cpp, which is the tool that writes it
 * for synthetic clips; av1gpu_decode writes the same files for a real stream
 * with --dump-plans. Both go through this header so the two can never drift.
 *
 *   frame_<n>.plan     the plan (see av1gpu_dump.cpp for the field list)
 *   frame_<n>.yuv16    the CPU reference decode of that frame, three planes,
 *                      16 bit samples, rows of plane_w (the stride is dropped)
 *   manifest.txt       one line per frame: "frame <n> refs <s0> <s1> out <s>"
 */
#ifndef AV1GPU_TOOLS_PLAN_DUMP_HPP
#define AV1GPU_TOOLS_PLAN_DUMP_HPP

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "core/backend.hpp"

namespace plan_dump {

inline bool write_all(std::FILE* file, const void* data, size_t size) {
  return std::fwrite(data, 1, size, file) == size;
}

template <typename T>
inline bool write_vector(std::FILE* file, const std::vector<T>& values) {
  const int32_t count = static_cast<int32_t>(values.size());
  if (!write_all(file, &count, sizeof(count))) return false;
  return values.empty() || write_all(file, values.data(), values.size() * sizeof(T));
}

inline bool write_plan(const std::string& path, const av1gpu::internal::FramePlan& plan) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) return false;
  const av1gpu::internal::PlanLayout& g = plan.layout;

  const char magic[8] = {'A', 'V', '1', 'G', 'P', 'L', 'A', 'N'};
  /* 6: the film grain noise image is one 16 bit sample per entry (a stream
   *    deeper than 8 bit has grain outside the signed char range)
   * 5: the film grain noise image and its two passes replace the noise
   * template, 4 added the display geometry and the upscale jobs, 3 tmp_size, 2
   * the palette colour map */
  /* 7: the loop restoration jobs (7.17), so a stream that codes them can be
   *    replayed through the device kernels at all
   * 8: an inter job carries its plane, which is what tells a 4:4:4 chroma job
   *    that its difference weighted mask still comes from the luma samples */
  const int32_t version = 8;
  const int32_t geom[5] = {g.frame.width, g.frame.height, g.frame.bit_depth,
                           g.frame.ss_x, g.frame.ss_y};
  bool ok = write_all(file, magic, sizeof(magic)) &&
            write_all(file, &version, sizeof(version)) &&
            write_all(file, geom, sizeof(geom)) &&
            write_all(file, &plan.maxv, sizeof(plan.maxv)) &&
            write_all(file, &plan.blocks, sizeof(plan.blocks)) &&
            write_all(file, g.plane_w, sizeof(g.plane_w)) &&
            write_all(file, g.plane_h, sizeof(g.plane_h)) &&
            write_all(file, g.stride, sizeof(g.stride)) &&
            write_all(file, g.plane_off, sizeof(g.plane_off)) &&
            /* the display geometry: a superres frame is coded narrower than it
             * is displayed, and the output frame and the reference pool hold the
             * displayed one, so a replay has to know both sizes */
            write_all(file, &g.out_width, sizeof(g.out_width)) &&
            write_all(file, &g.out_height, sizeof(g.out_height)) &&
            write_all(file, g.out_w, sizeof(g.out_w)) &&
            write_all(file, g.out_h, sizeof(g.out_h));
  const int64_t buffer_size = static_cast<int64_t>(g.buffer_size);
  /* the inverse transform scratch buffer is indexed by Av1gItxJob::tmp_off and
   * a job needs w * h ints there, which is more than its coefficient area, so
   * the size has to travel with the plan */
  const int64_t tmp_size = static_cast<int64_t>(plan.tmp_size);
  ok = ok && write_all(file, &buffer_size, sizeof(buffer_size)) &&
       write_all(file, &tmp_size, sizeof(tmp_size)) &&
       write_all(file, plan.ref_slot, sizeof(plan.ref_slot)) &&
       /* the film grain noise image: the host builds it (the synthesis is
        * sequential) and the jobs index it per plane */
       write_all(file, plan.grain_noise.noise_off, sizeof(plan.grain_noise.noise_off)) &&
       write_all(file, plan.grain_noise.noise_stride, sizeof(plan.grain_noise.noise_stride)) &&
       write_all(file, plan.grain_noise.lut, sizeof(plan.grain_noise.lut));

  const int32_t counts[4] = {static_cast<int32_t>(plan.coefficients.size()),
                             static_cast<int32_t>(plan.residual.size()),
                             static_cast<int32_t>(plan.grain_noise.noise.size()),
                             static_cast<int32_t>(plan.palette_map.size())};
  ok = ok && write_all(file, counts, sizeof(counts));

  const int32_t nwaves = static_cast<int32_t>(plan.waves.size());
  ok = ok && write_all(file, &nwaves, sizeof(nwaves));
  for (const av1gpu::internal::WaveJobs& wave : plan.waves) {
    if (!ok) break;
    ok = write_vector(file, wave.intra) && write_vector(file, wave.inter) &&
         write_vector(file, wave.itx) && write_vector(file, wave.residual);
  }
  ok = ok && write_vector(file, plan.deblock_v) && write_vector(file, plan.deblock_h) &&
       write_vector(file, plan.cdef) && write_vector(file, plan.upscale) &&
       write_vector(file, plan.restoration) &&
       write_vector(file, plan.grain_chroma) && write_vector(file, plan.grain_luma);

  if (ok && !plan.coefficients.empty())
    ok = write_all(file, plan.coefficients.data(),
                   plan.coefficients.size() * sizeof(int32_t));
  if (ok && !plan.residual.empty())
    ok = write_all(file, plan.residual.data(), plan.residual.size() * sizeof(int32_t));
  if (ok && !plan.grain_noise.noise.empty())
    ok = write_all(file, plan.grain_noise.noise.data(),
                   plan.grain_noise.noise.size() * sizeof(plan.grain_noise.noise[0]));
  if (ok && !plan.palette_map.empty())
    ok = write_all(file, plan.palette_map.data(), plan.palette_map.size());

  std::fclose(file);
  return ok;
}

inline bool write_frame(const std::string& path, const av1gpu::Frame& frame) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) return false;
  bool ok = true;
  for (int p = 0; p < 3 && ok; ++p) {
    const int w = frame.plane_width(p);
    for (int y = 0; y < frame.plane_height(p) && ok; ++y)
      ok = write_all(file, frame.plane(p) + static_cast<size_t>(y) * frame.stride[p],
                     static_cast<size_t>(w) * sizeof(uint16_t));
  }
  std::fclose(file);
  return ok;
}

} /* namespace plan_dump */

#endif /* AV1GPU_TOOLS_PLAN_DUMP_HPP */
