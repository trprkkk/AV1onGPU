/*
 * av1gpu_play - decode a clip on the GPU and write it out for playback.
 *
 * Writes raw YUV420p (or RGB24 with --rgb) to a file or to stdout, so it can be
 * piped straight into a player:
 *
 *   av1gpu_play --width 1280 --height 720 --frames 300 --output - |
 *     ffplay -f rawvideo -pixel_format yuv420p -video_size 1280x720 -
 *
 * The frames come from the built-in synthetic generator (see
 * include/av1gpu/synthetic.hpp) because this tree does not include an AV1
 * entropy decoder yet. Everything after the entropy stage - prediction,
 * transforms, deblocking, CDEF, film grain and colour conversion - is the real
 * pipeline that a front-end would feed.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "av1gpu/synthetic.hpp"

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

void usage() {
  std::printf(
      "usage: av1gpu_play [options]\n"
      "  --backend cpu|opencl|cuda|auto   backend (default auto)\n"
      "  --device N                        device index\n"
      "  --width N --height N              frame size (default 640x360)\n"
      "  --frames N                        number of frames (default 120)\n"
      "  --qstep N                         quantiser step (default 6)\n"
      "  --fps N                           pace the output (0 = as fast as possible)\n"
      "  --output FILE                     raw output, '-' for stdout\n"
      "  --rgb                             write RGB24 instead of YUV420p\n"
      "  --cdef --grain                    enable CDEF and film grain\n"
      "  --stats                           print per frame statistics\n");
}

bool write_all(std::FILE* file, const void* data, size_t size) {
  return std::fwrite(data, 1, size, file) == size;
}

} /* namespace */

int main(int argc, char** argv) {
  av1gpu::BackendOptions options;
  av1gpu::SyntheticConfig cfg;
  cfg.width = 640;
  cfg.height = 360;
  int frames = 120;
  int fps = 0;
  bool rgb = false;
  bool stats = false;
  std::string output = "-";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_int = [&](int fallback) { return i + 1 < argc ? std::atoi(argv[++i]) : fallback; };
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
    } else if (arg == "--fps") {
      fps = next_int(fps);
    } else if (arg == "--output" && i + 1 < argc) {
      output = argv[++i];
    } else if (arg == "--rgb") {
      rgb = true;
    } else if (arg == "--cdef") {
      cfg.cdef = true;
    } else if (arg == "--grain") {
      cfg.grain = true;
    } else if (arg == "--stats") {
      stats = true;
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    }
  }

  std::string error;
  std::unique_ptr<av1gpu::Decoder> decoder = av1gpu::Decoder::create(options, &error);
  if (!decoder) {
    std::fprintf(stderr, "av1gpu_play: %s\n", error.c_str());
    return 1;
  }
  std::fprintf(stderr, "av1gpu_play: %s via %s, %dx%d, %d frames\n",
               decoder->device().name.c_str(), av1gpu::to_string(decoder->device().kind),
               cfg.width, cfg.height, frames);

  std::FILE* file = stdout;
  if (output != "-") {
    file = std::fopen(output.c_str(), "wb");
    if (!file) {
      std::fprintf(stderr, "av1gpu_play: cannot open %s\n", output.c_str());
      return 1;
    }
  }

  av1gpu::Frame reference;
  av1gpu::Frame frame;
  std::vector<uint8_t> buffer;
  int slot = -1;
  double total_ms = 0;
  int written = 0;

  for (int f = 0; f < frames; ++f) {
    av1gpu::Frame stored;
    av1gpu::FrameSyntax syntax =
        av1gpu::synthetic_encode(cfg, f, f == 0 ? nullptr : &reference, nullptr, &reference, &stored);
    /* the next frame predicts from the picture before the film grain */
    reference = stored;
    if (slot >= 0) {
      syntax.reference_slots.clear();
      syntax.reference_slots.push_back(slot);
    }
    const int next = decoder->decode_reference(syntax, &frame, &error);
    if (next < 0) {
      std::fprintf(stderr, "av1gpu_play: decode failed: %s\n", error.c_str());
      break;
    }
    if (slot >= 0) decoder->release_reference(slot);
    slot = next;
    total_ms += decoder->stats().total_ms;

    if (rgb) {
      if (!decoder->to_rgb(slot, av1gpu::ColorMatrix::Bt709, av1gpu::ColorRange::Limited,
                           &buffer, &error)) {
        std::fprintf(stderr, "av1gpu_play: colour conversion failed: %s\n", error.c_str());
        break;
      }
    } else {
      av1gpu::frame_to_yuv8(frame, &buffer);
    }
    if (!write_all(file, buffer.data(), buffer.size())) {
      std::fprintf(stderr, "av1gpu_play: write failed\n");
      break;
    }
    ++written;

    if (stats) {
      std::fprintf(stderr, "  frame %4d  %6.2f ms  %3d launches  %d waves\n", f,
                   decoder->stats().total_ms, decoder->stats().kernel_launches,
                   decoder->stats().waves);
    }
    if (fps > 0) {
      /* pace the output so that a player can consume it in real time */
      const double target = 1000.0 / fps;
      const double used = decoder->stats().total_ms;
      if (used < target) {
        const long usec = static_cast<long>((target - used) * 1000.0);
        if (usec > 0) {
#if defined(_WIN32)
          Sleep(static_cast<DWORD>(usec / 1000));
#else
          usleep(static_cast<useconds_t>(usec));
#endif
        }
      }
    }
  }

  if (file != stdout) std::fclose(file);
  else std::fflush(stdout);
  std::fprintf(stderr, "av1gpu_play: %d frames, %.2f ms/frame average (%.1f fps)\n", written,
               written ? total_ms / written : 0.0,
               total_ms > 0 ? 1000.0 * written / total_ms : 0.0);
  return 0;
}
