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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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
  std::string path;
  int frames = 0;
  size_t pitch = 4096;
  std::string backend = "auto";
  int device = 0;
  bool copy_sink = false;
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
    } else if (arg.size() && arg[0] == '-') {
      std::fprintf(stderr, "av1gpu_pipebench: unknown option %s\n", arg.c_str());
      return 2;
    } else {
      path = arg;
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, "usage: av1gpu_pipebench CLIP.ivf [--frames N] "
                         "[--backend B] [--device N] [--pitch N] [--copy-sink]\n");
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
  av1gpu::dsfilter::StreamDecoder::Sink sink =
      [&](const std::vector<uint8_t>& nv12, int w, int h, std::string* sink_error) -> bool {
    (void)w;
    (void)h;
    (void)sink_error;
    if (copy_sink) {
      renderer.resize(nv12.size());
      std::memcpy(renderer.data(), nv12.data(), nv12.size());
    }
    ++shown;
    return true;
  };

  const auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < units.size(); ++i) {
    if (!decoder.push(units[i], sizes[i], pitch, sink, &error)) {
      std::fprintf(stderr, "av1gpu_pipebench: unit %zu: %s\n", i, error.c_str());
      return 1;
    }
  }
  decoder.finish();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const size_t decoded = decoder.decoded_frames();
  std::printf("%zu unit(s) -> %zu frame(s) shown, %zu decoded in %.2f s = %.2f fps "
              "(%.2f ms/frame)\n",
              units.size(), shown, decoded, elapsed,
              decoded > 0 ? static_cast<double>(decoded) / elapsed : 0.0,
              decoded > 0 ? elapsed * 1000.0 / static_cast<double>(decoded) : 0.0);
  return 0;
}
