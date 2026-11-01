/*
 * av1gpu_pipebench - how fast does the filter's decode *pipeline* run?
 *
 * The DirectShow filter feeds temporal units to StreamDecoder, which parses and
 * plans them on the calling thread while the worker submits the previous frame
 * to the backend. That pipeline is what decides playback smoothness, and this
 * tool drives exactly it over a real IVF stream - no DirectShow, no sink work
 * beyond counting frames - so a change to the core shows up as a number here.
 *
 *   av1gpu_pipebench clip.ivf [--frames N] [--backend cuda]
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#ifdef _DEBUG
#include <crtdbg.h>
#endif

#include "av1gpu/av1gpu.hpp"
#include "dsfilter/decode_core.hpp"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

bool read_file(const std::string& path, std::vector<uint8_t>* out) {
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
  return got == out->size();
}

uint32_t le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

} /* namespace */

int main(int argc, char** argv) {
#ifdef _DEBUG
  /* AV1GPU_CRT_DEBUG=1 (Debug build only): verify the heap on every allocation
   * and report damage with the offending block's allocation site. */
  if (std::getenv("AV1GPU_CRT_DEBUG")) {
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_DELAY_FREE_MEM_DF);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  }
#endif
  std::string path;
  int frames = 0;
  size_t pitch = 4096;
  std::string backend = "auto";
  int device = 0;
  bool copy_sink = false;
  bool cadence = false;
  int seek_at = -1; /* reset the stream (a seek) after this many units */
  int seek_every = 0; /* and then again every N units */
  int seek_race_ms = 0; /* a second thread calls reset_stream() every N ms */
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* fallback) -> const char* {
      return i + 1 < argc ? argv[++i] : fallback;
    };
    if (arg == "--frames") {
      frames = std::atoi(next("0"));
    } else if (arg == "--backend") {
      backend = next("auto");
    } else if (arg == "--device") {
      device = std::atoi(next("0"));
    } else if (arg == "--pitch") {
      pitch = static_cast<size_t>(std::atoi(next("4096")));
    } else if (arg == "--copy-sink") {
      copy_sink = true;
    } else if (arg == "--cadence") {
      cadence = true;
    } else if (arg == "--seek-at") {
      seek_at = std::atoi(next("-1"));
    } else if (arg == "--seek-every") {
      seek_every = std::atoi(next("0"));
    } else if (arg == "--seek-race") {
      seek_race_ms = std::atoi(next("0"));
    } else if (arg.size() && arg[0] == '-') {
      std::fprintf(stderr, "av1gpu_pipebench: unknown option %s\n", arg.c_str());
      return 2;
    } else {
      path = arg;
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, "usage: av1gpu_pipebench CLIP.ivf [--frames N] "
                         "[--backend B] [--device N] [--pitch N] [--copy-sink] [--seek-at N] "
                         "[--seek-every N] [--seek-race MS]\n");
    return 2;
  }

  std::vector<uint8_t> data;
  if (!read_file(path, &data) || data.size() < 32) {
    std::fprintf(stderr, "av1gpu_pipebench: cannot read %s\n", path.c_str());
    return 1;
  }
  if (std::memcmp(data.data(), "DKIF", 4) != 0) {
    std::fprintf(stderr, "av1gpu_pipebench: %s is not an IVF file\n", path.c_str());
    return 1;
  }

  /* the IVF frame headers: 12 bytes (4 size, 8 timestamp) after the 32 byte
   * file header, each followed by one temporal unit */
  std::vector<const uint8_t*> units;
  std::vector<size_t> sizes;
  size_t pos = 32;
  while (pos + 12 <= data.size()) {
    const uint32_t size = le32(data.data() + pos);
    pos += 12;
    if (size == 0 || pos + size > data.size()) break;
    units.push_back(data.data() + pos);
    sizes.push_back(size);
    pos += size;
    if (frames > 0 && units.size() >= static_cast<size_t>(frames)) break;
  }
  if (units.empty()) {
    std::fprintf(stderr, "av1gpu_pipebench: no frames in %s\n", path.c_str());
    return 1;
  }

  /* A player seeking does not continue mid-GOP: the splitter restarts delivery
   * at the next keyframe. Mark the keyframe units (an OBU_FRAME/_HEADER whose
   * frame header starts with show_existing=0, frame_type=KEY) so a synthetic
   * seek can jump like a real one. */
  std::vector<char> keyframe(units.size(), 0);
  for (size_t i = 0; i < units.size(); ++i) {
    const uint8_t* u = units[i];
    const size_t n = sizes[i];
    size_t p = 0;
    while (p + 1 < n) {
      const uint8_t h = u[p];
      const int type = (h >> 3) & 0xf;
      const bool has_ext = (h >> 2) & 1;
      const bool has_size = (h >> 1) & 1;
      size_t body = p + 1 + (has_ext ? 1 : 0);
      size_t len = n - body;
      if (has_size) {
        size_t v = 0;
        int shift = 0;
        for (int k = 0; k < 8 && body < n; ++k, ++body) {
          v |= static_cast<size_t>(u[body] & 0x7f) << shift;
          shift += 7;
          if (!(u[body] & 0x80)) break;
        }
        len = v;
      }
      if ((type == 6 || type == 3) && body < n) {
        const uint8_t b = u[body];
        if ((b & 0x80) == 0 && ((b >> 5) & 3) == 0) { keyframe[i] = 1; break; }
      }
      if (body + len > n) break;
      p = body + len;
    }
  }

  av1gpu::BackendOptions options;
  options.kind = av1gpu::backend_kind_from_string(backend);
  options.device_index = device;
  options.allow_cpu_fallback = true;

  std::string error;
  av1gpu::dsfilter::StreamDecoder decoder(options);
  if (!decoder.init(&error)) {
    std::fprintf(stderr, "av1gpu_pipebench: no backend: %s\n", error.c_str());
    return 1;
  }
  std::printf("backend: %s (%s)\n", av1gpu::to_string(decoder.device().kind),
              decoder.device().name.c_str());

  size_t shown = 0;
  /* A renderer does not just count frames: it copies the NV12 into its own
   * allocator buffer. --copy-sink adds that copy, so the number is closer to
   * what a player sees. */
  std::vector<uint8_t> renderer;
  /* --cadence: the interval between deliveries is what a player presents; a
   * spike over the frame time is a visible hitch even when the average is
   * fine, so the tail of that distribution is worth a number of its own. */
  std::vector<double> intervals;
  std::ofstream gap_log;
  if (cadence) gap_log.open("av1gpu_gaps.txt");
  bool have_prev = false;
  std::chrono::steady_clock::time_point prev{};
  const auto t0 = std::chrono::steady_clock::now();
  const auto since0 = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
  };
  av1gpu::dsfilter::StreamDecoder::Sink sink =
      [&](const std::vector<uint8_t>& nv12, int w, int h, std::string* sink_error) -> bool {
    (void)w;
    (void)h;
    (void)sink_error;
    if (copy_sink) {
      renderer.resize(nv12.size());
      std::memcpy(renderer.data(), nv12.data(), nv12.size());
    }
    if (cadence) {
      const auto now = std::chrono::steady_clock::now();
      if (have_prev) {
        const double gap = std::chrono::duration<double, std::milli>(now - prev).count();
        intervals.push_back(gap);
        if (gap_log.is_open())
          gap_log << "sink " << shown << " t " << since0() << " gap " << gap << " decoded "
                  << decoder.decoded_frames() << "\n";
      }
      prev = now;
      have_prev = true;
    }
    ++shown;
    return true;
  };

  /* --seek-race MS: a second thread calls reset_stream() every MS
   * milliseconds, the way MPC-HC's GUI thread does EndFlush/NewSegment while
   * the streaming thread is still inside Receive/push. */
  std::atomic<bool> race_stop{false};
  std::thread race_thread;
  if (seek_race_ms > 0) {
    race_thread = std::thread([&] {
      while (!race_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(seek_race_ms));
        decoder.reset_stream();
      }
    });
  }

  std::ofstream push_log;
  if (cadence) push_log.open("av1gpu_pushes.txt");
  for (size_t i = 0; i < units.size();) {
    const int unit = static_cast<int>(i);
    const bool do_seek = (seek_at >= 0 && unit == seek_at) ||
                         (seek_every > 0 && unit > (seek_at < 0 ? 0 : seek_at) &&
                          (unit - (seek_at < 0 ? 0 : seek_at)) % seek_every == 0);
    if (do_seek) {
      decoder.reset_stream(); /* what EndFlush/NewSegment do on a seek */
      if (push_log.is_open()) push_log << "SEEK at unit " << i << "\n";
      /* like a real seek: resume at the next keyframe unit */
      size_t j = i + 1;
      while (j < units.size() && !keyframe[j]) ++j;
      if (j >= units.size()) break;
      if (push_log.is_open()) push_log << "  -> resume at keyframe unit " << j << "\n";
      i = j;
      continue;
    }
    if (!decoder.push(units[i], sizes[i], pitch, sink, &error)) {
      std::fprintf(stderr, "av1gpu_pipebench: unit %zu: %s\n", i, error.c_str());
      /* in race mode a reset can land mid-GOP and the next unit may name a
       * reference the reset dropped; the real filter just keeps going */
      if (seek_race_ms <= 0) return 1;
      error.clear();
    }
    if (push_log.is_open()) push_log << "push " << i << " t " << since0() << "\n";
    if (cadence) {
      LARGE_INTEGER q1, qf;
      QueryPerformanceCounter(&q1);
      static LARGE_INTEGER q0{}; /* previous push end */
      static bool have_q0 = false;
      QueryPerformanceFrequency(&qf);
      if (have_q0) {
        const double d =
            double(q1.QuadPart - q0.QuadPart) * 1000.0 / double(qf.QuadPart);
        if (d > 60.0 && push_log.is_open())
          push_log << "SLOW push " << i << " took " << d << " ms\n";
      }
      q0 = q1;
      have_q0 = true;
    }
    ++i;
  }
  race_stop.store(true);
  if (race_thread.joinable()) race_thread.join();
  decoder.finish();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const size_t decoded = decoder.decoded_frames();
  std::printf("%zu unit(s) -> %zu frame(s) shown, %zu decoded in %.2f s = %.2f fps "
              "(%.2f ms/frame)\n",
              units.size(), shown, decoded, elapsed,
              decoded > 0 ? static_cast<double>(decoded) / elapsed : 0.0,
              decoded > 0 ? elapsed * 1000.0 / static_cast<double>(decoded) : 0.0);
  if (cadence && !intervals.empty()) {
    std::vector<double> sorted = intervals;
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double p) {
      return sorted[static_cast<size_t>(p * (sorted.size() - 1))];
    };
    double sum = 0;
    for (double v : intervals) sum += v;
    size_t over = 0;
    for (double v : intervals)
      if (v > 41.7) ++over; /* 23.976 fps budget */
    std::printf("cadence: avg %.2f ms, p50 %.2f, p90 %.2f, p99 %.2f, max %.2f, "
                "over 41.7 ms: %zu/%zu\n",
                sum / static_cast<double>(intervals.size()), pct(0.50), pct(0.90),
                pct(0.99), sorted.back(), over, intervals.size());
  }
  return 0;
}
