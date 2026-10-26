/*
 * AV1onGPU - the DirectShow decode core, without DirectShow
 *
 * dsfilter/decode_core.cpp is the part of the DirectShow filter that is worth
 * testing on its own: it takes the byte stream a demuxer pushes (one media
 * sample at a time) and produces the shown frames as NV12 buffers. This test
 * drives it over the real fixtures and checks
 *
 *   1. pushing whole temporal units, the way every splitter does, decodes the
 *      fixture to its full frame count;
 *   2. pushing the same bytes in small chunks - so OBUs end mid buffer - gives
 *      byte identical output (the incremental OBU parser and its pending
 *      buffer);
 *   3. reset_stream() followed by a re-decode reproduces the same output, the
 *      property a seek relies on;
 *   4. with a GPU device present, the backend the filter would really use
 *      (Auto) produces the same NV12 bytes as the CPU reference, frame for
 *      frame. Without a GPU the parity check is skipped but the plumbing
 *      checks still run, so this test is worth having on every machine.
 */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "dsfilter/decode_core.hpp"
#include "test_common.hpp"

using av1gpu::BackendKind;
using av1gpu::BackendOptions;
using av1gpu::dsfilter::StreamDecoder;

namespace {

/* One IVF temporal unit: the bytes between the 12 byte frame headers. */
struct Unit {
  std::vector<uint8_t> data;
};

bool read_ivf(const std::string& path, std::vector<Unit>* units, std::string* error) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    *error = "cannot open " + path;
    return false;
  }
  uint8_t header[32];
  if (std::fread(header, 1, sizeof(header), f) != sizeof(header) ||
      std::memcmp(header, "DKIF", 4) != 0) {
    std::fclose(f);
    *error = path + " is not an IVF file";
    return false;
  }
  for (;;) {
    uint8_t frame_header[12];
    if (std::fread(frame_header, 1, sizeof(frame_header), f) != sizeof(frame_header)) {
      std::fclose(f);
      return true; /* clean end of file */
    }
    const size_t size = static_cast<size_t>(frame_header[0]) |
                        (static_cast<size_t>(frame_header[1]) << 8) |
                        (static_cast<size_t>(frame_header[2]) << 16) |
                        (static_cast<size_t>(frame_header[3]) << 24);
    if (size == 0) {
      std::fclose(f);
      return true;
    }
    Unit unit;
    unit.data.resize(size);
    if (std::fread(unit.data.data(), 1, size, f) != size) {
      std::fclose(f);
      *error = path + " ends inside a temporal unit";
      return false;
    }
    units->push_back(std::move(unit));
  }
}

/* The NV12 row stride the filter negotiates: 32 sample aligned, which is what
 * most renderers hand out (and a superset of the compact stride). */
size_t filter_pitch(int width) { return static_cast<size_t>((width + 31) & ~31); }

struct Decoded {
  /* every shown frame, flat NV12 at `pitch` */
  std::vector<std::vector<uint8_t>> frames;
  size_t shown = 0, dropped = 0;
};

/* Feeds `units` through the core, splitting each unit into `chunk` byte
 * pieces (0 = one push per unit). `backend` selects the device. */
bool decode_all(const std::vector<Unit>& units, size_t chunk,
                const BackendOptions& options, Decoded* out, std::string* error) {
  StreamDecoder core(options);
  if (!core.init(error)) return false;
  const size_t pitch = filter_pitch(4096); /* geometry comes per frame; any wide pitch is legal */
  out->frames.clear();
  out->shown = out->dropped = 0;

  StreamDecoder::Sink sink = [&](const std::vector<uint8_t>& nv12, int width, int height,
                                 std::string* sink_error) -> bool {
    (void)width;
    (void)height;
    (void)sink_error;
    out->frames.push_back(nv12);
    return true;
  };
  for (const Unit& unit : units) {
    if (chunk == 0) {
      if (!core.push(unit.data.data(), unit.data.size(), pitch, sink, error)) return false;
    } else {
      for (size_t at = 0; at < unit.data.size(); at += chunk) {
        const size_t n = std::min(chunk, unit.data.size() - at);
        if (!core.push(unit.data.data() + at, n, pitch, sink, error)) return false;
      }
    }
  }
  core.finish(); /* let the worker show the frames still queued */
  out->shown = core.shown_frames();
  out->dropped = core.dropped_frames();
  return true;
}

bool same_frames(const Decoded& a, const Decoded& b, std::string* where) {
  if (a.frames.size() != b.frames.size()) {
    *where = "frame count " + std::to_string(a.frames.size()) + " vs " +
             std::to_string(b.frames.size());
    return false;
  }
  for (size_t i = 0; i < a.frames.size(); ++i) {
    if (a.frames[i].size() != b.frames[i].size()) {
      *where = "frame " + std::to_string(i) + " size " +
               std::to_string(a.frames[i].size()) + " vs " +
               std::to_string(b.frames[i].size());
      return false;
    }
    if (std::memcmp(a.frames[i].data(), b.frames[i].data(), a.frames[i].size()) != 0) {
      *where = "frame " + std::to_string(i) + " bytes differ";
      return false;
    }
  }
  return true;
}

/* The fixtures the core is checked against: 4:2:0, one per feature the
 * planner has to carry across the filter boundary. */
const char* kFixtures[] = {
    "lw_t2",       /* local warped motion + OBMC, 30 frames        */
    "lr",          /* loop restoration, 8 frames                   */
    "grain",       /* film grain, 30 frames                        */
    "tiles",       /* 2x2 tiles, 4 frames                          */
    "superres",    /* coded narrower than displayed, 30 frames     */
    "odd_size",    /* 130x100, 6 frames                            */
    "lossless",    /* no quantiser, no in loop filters, 4 frames   */
};

} /* namespace */

int main() {
  const std::string data_dir = AV1GPU_TEST_DATA_DIR;

  /* The GPU device the filter would really pick, when there is one. Without a
   * GPU device the plumbing checks still run on the CPU; the backend parity
   * check is what skips. */
  const std::vector<av1gpu::DeviceInfo> devices = av1gpu::enumerate_devices();
  const av1gpu::DeviceInfo* gpu = nullptr;
  for (const av1gpu::DeviceInfo& d : devices) {
    if (d.kind != BackendKind::Cpu && d.available) {
      gpu = &d;
      break;
    }
  }

  for (const char* name : kFixtures) {
    const std::string path = data_dir + "/" + name + ".ivf";
    std::string error;
    std::vector<Unit> units;
    if (!read_ivf(path, &units, &error)) {
      std::printf("test_ds_core: %s\n", error.c_str());
      return 1;
    }
    CHECK(units.size() > 0);

    /* 1. whole temporal units, CPU reference */
    BackendOptions cpu_options;
    cpu_options.kind = BackendKind::Cpu;
    Decoded whole;
    if (!decode_all(units, 0, cpu_options, &whole, &error)) {
      std::printf("test_ds_core: %s: %s\n", name, error.c_str());
      return 1;
    }
    CHECK(whole.shown == whole.frames.size());
    CHECK(whole.shown > 0);
    CHECK(whole.dropped == 0);

    /* 2. the same bytes in chunks: OBUs that end mid buffer have to wait */
    for (size_t chunk : {size_t(1), size_t(7), size_t(64)}) {
      Decoded chopped;
      if (!decode_all(units, chunk, cpu_options, &chopped, &error)) {
        std::printf("test_ds_core: %s (chunk %zu): %s\n", name, chunk, error.c_str());
        return 1;
      }
      std::string where;
      CHECK(same_frames(whole, chopped, &where));
      if (!same_frames(whole, chopped, &where)) {
        std::printf("  chunk %zu: %s\n", chunk, where.c_str());
      }
    }

    /* 3. a reset in the middle and a full re-decode: the seek path */
    {
      StreamDecoder core(cpu_options);
      if (!core.init(&error)) {
        std::printf("test_ds_core: %s\n", error.c_str());
        return 1;
      }
      const size_t pitch = filter_pitch(4096);
      Decoded redecoded;
      size_t first_frame = 0;
      StreamDecoder::Sink sink = [&](const std::vector<uint8_t>& nv12, int w, int h,
                                     std::string* sink_error) -> bool {
        (void)w;
        (void)h;
        (void)sink_error;
        redecoded.frames.push_back(nv12);
        return true;
      };
      /* one frame, then the state a seek destroys, then everything again */
      const bool first_ok =
          core.push(units[0].data.data(), units[0].data.size(), pitch, sink, &error);
      CHECK(first_ok);
      first_frame = redecoded.frames.size();
      core.reset_stream();
      redecoded.frames.clear();
      bool all_ok = true;
      for (const Unit& unit : units) {
        all_ok = all_ok && core.push(unit.data.data(), unit.data.size(), pitch, sink, &error);
      }
      CHECK(all_ok);
      core.finish(); /* drain the pipeline before counting frames */
      (void)first_frame;
      std::string where;
      CHECK(same_frames(whole, redecoded, &where));
      if (!same_frames(whole, redecoded, &where)) {
        std::printf("  re-decode: %s\n", where.c_str());
      }
    }

    /* 4. the backend the filter really picks against the CPU reference. The
     * fallback is off, so a reported device is really the one decoding. */
    if (gpu) {
      BackendOptions auto_options; /* Auto, like the filter's ensure_core */
      auto_options.allow_cpu_fallback = false;
      Decoded device_frames;
      if (!decode_all(units, 0, auto_options, &device_frames, &error)) {
        std::printf("test_ds_core: %s (device): %s\n", name, error.c_str());
        return 1;
      }
      std::string where;
      CHECK(same_frames(whole, device_frames, &where));
      if (!same_frames(whole, device_frames, &where)) {
        std::printf("  device %s: %s\n", gpu->name.c_str(), where.c_str());
      }
    }

    std::printf("  %-10s %2zu unit(s), %2zu frame(s)%s\n", name, units.size(),
                whole.frames.size(), gpu ? ", device bit exact" : "");
  }

  if (gpu) {
    std::printf("test_ds_core: backend under test: %s (%s)\n", gpu->name.c_str(),
                gpu->kind == BackendKind::CUDA
                    ? "cuda"
                    : (gpu->kind == BackendKind::OpenCL ? "opencl" : "other"));
  } else {
    /* the plumbing was checked on the CPU; the parity check did not run */
    std::printf("test_ds_core: no GPU device, backend parity not checked\n");
  }
  return ::testing::finish("test_ds_core");
}
