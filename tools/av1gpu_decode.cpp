/*
 * AV1onGPU - decode a real AV1 file
 *
 * This is the tool that turns the front end into pictures: it reads an .ivf,
 * walks the OBUs, decodes the tile of each intra frame, plans the frame for the
 * engine and lets a backend reconstruct it. Until inter prediction lands, only
 * key frames and intra-only frames are decodable, which is why every other
 * frame is reported and skipped rather than guessed at.
 *
 *   av1gpu_decode clip.ivf -o out.yuv
 *   av1gpu_decode clip.ivf -o out.yuv --compare reference.yuv
 *   av1gpu_decode clip.ivf --backend cpu --frames 1 -o frame0.yuv
 *
 * `--compare` reads a planar 8 bit YUV file of the same geometry and reports
 * the per plane PSNR and the first sample that differs, with its coordinates.
 * That is the check that matters: the reference is dav1d with its loop filter
 * disabled (AV1GPU_NOLF), so a difference is a reconstruction difference and
 * not a filter difference.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "frontend/dequant.hpp"
#include "frontend/headers.hpp"
#include "frontend/ivf.hpp"
#include "frontend/obu.hpp"
#include "frontend/reconstruct.hpp"
#include "frontend/tile_decoder.hpp"
#include "frontend/tiles.hpp"

namespace {

using namespace av1gpu;
using namespace av1gpu::frontend;

struct Options {
  std::string input;
  std::string output;
  std::string compare;
  std::string backend = "auto";
  int frames = -1;
  bool verbose = false;
};

void usage() {
  std::printf(
      "usage: av1gpu_decode [options] FILE\n"
      "  FILE              an .ivf file or a raw OBU stream\n"
      "  -o, --output F    write the decoded frames to F as planar 8 bit YUV\n"
      "  --compare F       compare against a planar 8 bit YUV file\n"
      "  --backend B       auto (default), cpu, opencl or cuda\n"
      "  --frames N        stop after N temporal units\n"
      "  -v, --verbose     print every OBU and every frame\n"
      "\n"
      "debugging, when a frame does not match the reference decoder:\n"
      "  AV1GPU_DUMP_BLOCKS=N  print the first N planned blocks (plane, position,\n"
      "                        mode, neighbours, EOB, transform type, coefficients)\n"
      "  AV1GPU_DUMP_COEF=N    print the dequantised coefficients of block N in\n"
      "                        raster order, to compare with a reference decoder\n"
      "  AV1GPU_NO_RESIDUAL=1  decode without any residual, to look at the\n"
      "                        predictions alone\n");
}

bool read_file(const std::string& path, std::vector<uint8_t>* data) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) return false;
  uint8_t chunk[65536];
  size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
    data->insert(data->end(), chunk, chunk + got);
  const bool failed = std::ferror(file) != 0;
  std::fclose(file);
  return !failed;
}

BackendKind parse_backend(const std::string& name, bool* ok) {
  *ok = true;
  if (name == "auto") return BackendKind::Auto;
  if (name == "cpu") return BackendKind::Cpu;
  if (name == "opencl") return BackendKind::OpenCL;
  if (name == "cuda") return BackendKind::CUDA;
  *ok = false;
  return BackendKind::Auto;
}

/* One plane of one frame, so the comparison can name the sample that differs
 * instead of saying "the files are not equal". */
void report_difference(const Frame& ours, const Frame& ref, const char* label) {
  int planes = ref.ss_x == 1 && ref.ss_y == 1 ? 3 : 3;
  const int ss[3][2] = {{0, 0}, {ref.ss_x, ref.ss_y}, {ref.ss_x, ref.ss_y}};
  bool identical = true;
  for (int p = 0; p < planes; ++p) {
    const int w = ref.width >> ss[p][0], h = ref.height >> ss[p][1];
    double sse = 0.0;
    int first_x = -1, first_y = -1, first_a = 0, first_b = 0;
    for (int y = 0; y < h; ++y) {
      const uint16_t* a = ours.plane(p) + static_cast<size_t>(y) * ours.stride[p];
      const uint16_t* b = ref.plane(p) + static_cast<size_t>(y) * ref.stride[p];
      for (int x = 0; x < w; ++x) {
        const int d = static_cast<int>(a[x]) - static_cast<int>(b[x]);
        sse += static_cast<double>(d) * d;
        if (d != 0 && first_x < 0) {
          first_x = x;
          first_y = y;
          first_a = a[x];
          first_b = b[x];
        }
      }
    }
    const double mse = sse / (static_cast<double>(w) * h);
    const double psnr = mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
    std::printf("  %s plane %d: %.2f dB, %s", label, p, psnr,
                first_x < 0 ? "identical" : "differs");
    if (first_x >= 0) {
      identical = false;
      std::printf(" at (%d,%d): ours %d, reference %d", first_x, first_y, first_a, first_b);
    }
    std::printf("\n");
  }
  if (!identical) std::printf("  %s: not bit exact\n", label);
}

} /* namespace */

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    }
    if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
      opt.output = argv[++i];
      continue;
    }
    if (arg == "--compare" && i + 1 < argc) {
      opt.compare = argv[++i];
      continue;
    }
    if (arg == "--backend" && i + 1 < argc) {
      opt.backend = argv[++i];
      continue;
    }
    if (arg == "--frames" && i + 1 < argc) {
      opt.frames = std::atoi(argv[++i]);
      continue;
    }
    if (arg == "-v" || arg == "--verbose") {
      opt.verbose = true;
      continue;
    }
    if (!opt.input.empty()) {
      usage();
      return 2;
    }
    opt.input = arg;
  }
  if (opt.input.empty()) {
    usage();
    return 2;
  }

  std::vector<uint8_t> data;
  if (!read_file(opt.input, &data)) {
    std::fprintf(stderr, "av1gpu_decode: cannot read %s\n", opt.input.c_str());
    return 2;
  }

  bool backend_ok = false;
  const BackendKind kind = parse_backend(opt.backend, &backend_ok);
  if (!backend_ok) {
    std::fprintf(stderr, "av1gpu_decode: unknown backend '%s'\n", opt.backend.c_str());
    return 2;
  }

  std::string error;
  BackendOptions backend_options;
  backend_options.kind = kind;
  std::unique_ptr<Decoder> decoder = Decoder::create(backend_options, &error);
  if (!decoder) {
    std::fprintf(stderr, "av1gpu_decode: no backend: %s\n", error.c_str());
    return 1;
  }
  std::printf("backend: %s\n", decoder->device().name.c_str());

  std::vector<uint8_t> reference;
  if (!opt.compare.empty() && !read_file(opt.compare, &reference)) {
    std::fprintf(stderr, "av1gpu_decode: cannot read %s\n", opt.compare.c_str());
    return 2;
  }

  std::FILE* out_file = nullptr;
  if (!opt.output.empty()) {
    out_file = std::fopen(opt.output.c_str(), "wb");
    if (!out_file) {
      std::fprintf(stderr, "av1gpu_decode: cannot write %s\n", opt.output.c_str());
      return 2;
    }
  }

  std::vector<IvfFrame> units;
  int status = 0;
  if (data.size() >= 4 && std::memcmp(data.data(), "DKIF", 4) == 0) {
    IvfReader reader;
    if (!reader.open(data, &error)) {
      std::fprintf(stderr, "av1gpu_decode: %s\n", error.c_str());
      return 1;
    }
    IvfFrame unit;
    while (reader.next(&unit, &error)) {
      units.push_back(unit);
      if (opt.frames >= 0 && static_cast<int>(units.size()) >= opt.frames) break;
    }
    if (!error.empty()) {
      std::fprintf(stderr, "av1gpu_decode: %s\n", error.c_str());
      return 1;
    }
  } else {
    IvfFrame unit;
    unit.payload = data;
    units.push_back(unit);
  }

  SequenceHeader seq;
  bool have_seq = false;
  RefPool refs;
  size_t decoded_frames = 0;
  size_t reference_offset = 0;
  int comparisons = 0, exact = 0;

  for (size_t index = 0; index < units.size(); ++index) {
    std::vector<Obu> obus;
    if (!split_obus(units[index].payload.data(), units[index].payload.size(), &obus, &error)) {
      std::fprintf(stderr, "av1gpu_decode: temporal unit %zu: %s\n", index, error.c_str());
      status = 1;
      break;
    }
    for (const Obu& obu : obus) {
      if (obu.type == ObuType::SequenceHeader) {
        if (parse_sequence_header(obu.data, obu.size, &seq, &error) == ParseStatus::Invalid) {
          std::fprintf(stderr, "av1gpu_decode: %s\n", error.c_str());
          status = 1;
          break;
        }
        have_seq = true;
        continue;
      }
      if (obu.type != ObuType::FrameHeader && obu.type != ObuType::Frame &&
          obu.type != ObuType::RedundantFrameHeader)
        continue;
      if (!have_seq) {
        std::fprintf(stderr, "av1gpu_decode: frame before any sequence header\n");
        status = 1;
        break;
      }
      FrameHeader frame;
      const ParseStatus parsed = parse_frame_header(seq, obu.data, obu.size, &frame, &error, &refs);
      if (parsed == ParseStatus::Invalid) {
        std::fprintf(stderr, "av1gpu_decode: %s\n", error.c_str());
        status = 1;
        break;
      }
      if (parsed == ParseStatus::Unsupported) {
        std::printf("frame %zu: %s\n", index, error.c_str());
        refresh_ref_pool(frame, &refs);
        continue;
      }
      if (frame.show_existing_frame) {
        std::printf("frame %zu: show_existing_frame %d, skipped\n", index,
                    frame.existing_frame_idx);
        refresh_ref_pool(frame, &refs);
        continue;
      }
      if (frame.type != FrameType::Key && frame.type != FrameType::IntraOnly) {
        std::printf("frame %zu: %s, inter prediction is not implemented\n", index,
                    frame_type_name(frame.type));
        refresh_ref_pool(frame, &refs);
        continue;
      }

      std::string tile_error;
      TileGroup group;
      if (parse_tile_group(seq, frame, obu.data, obu.size, &group, &tile_error) !=
          ParseStatus::Ok) {
        std::printf("frame %zu: %s\n", index, tile_error.c_str());
        status = 1;
        break;
      }
      if (group.tiles.size() != 1) {
        std::printf("frame %zu: %zu tiles, multi-tile decoding is not implemented\n", index,
                    group.tiles.size());
        status = 1;
        break;
      }

      const TileExtent& tile = group.tiles[0];
      TileDecodeResult decoded;
      const ParseStatus tile_status =
          decode_intra_tile(seq, frame, obu.data + tile.offset, tile.size, &decoded, &tile_error);
      if (tile_status != ParseStatus::Ok) {
        std::printf("frame %zu: tile: %s\n", index,
                    decoded.stop_reason.empty() ? tile_error.c_str()
                                                : decoded.stop_reason.c_str());
        status = 1;
        break;
      }

      DequantisedTile quant;
      dequantise_tile(frame, seq.bit_depth, decoded, &quant);

      FrameSyntax syntax;
      if (!plan_from_tile(seq, frame, decoded, quant, &syntax, &error)) {
        std::printf("frame %zu: %s\n", index, error.c_str());
        status = 1;
        break;
      }

      if (const char* dump = std::getenv("AV1GPU_DUMP_BLOCKS")) {
        const int limit = std::atoi(dump);
        for (int i = 0; i < limit && i < static_cast<int>(syntax.blocks.size()); ++i) {
          const BlockSyntax& b = syntax.blocks[i];
          std::printf("  block %d: plane %d at (%d,%d) %dx%d mode %d angle %d cfl %d "
                      "L%d A%d AR%d BL%d ft %d eob %d tx %d off %d\n",
                      i, b.plane, b.x, b.y, b.w, b.h, static_cast<int>(b.mode), b.angle_delta,
                      b.cfl_alpha, b.have_left ? 1 : 0, b.have_above ? 1 : 0,
                      b.have_above_right, b.have_below_left, b.filter_type,
                      b.transforms.empty() ? -1 : b.transforms[0].eob,
                      b.transforms.empty() ? -1 : static_cast<int>(b.transforms[0].tx_type),
                      b.transforms.empty() ? -1 : b.transforms[0].coef_offset);
        }
      }
      if (const char* which = std::getenv("AV1GPU_DUMP_COEF")) {
        const int want = std::atoi(which);
        if (want >= 0 && want < static_cast<int>(syntax.blocks.size())) {
          const BlockSyntax& b = syntax.blocks[want];
          const TransformBlock& tx = b.transforms[0];
          std::printf("  coef block %d: %d x %d\n", want, tx.w, tx.h);
          for (int y = 0; y < tx.h; ++y) {
            std::printf("   ");
            for (int x = 0; x < tx.w; ++x)
              std::printf(" %d", syntax.coefficients[tx.coef_offset + y * tx.w + x]);
            std::printf("\n");
          }
        }
      }
      if (std::getenv("AV1GPU_NO_RESIDUAL")) {
        for (BlockSyntax& b : syntax.blocks)
          for (TransformBlock& tx : b.transforms) tx.eob = 0;
      }

      Frame picture;
      if (!decoder->decode(syntax, &picture, &error)) {
        std::printf("frame %zu: %s\n", index, error.c_str());
        status = 1;
        break;
      }

      std::printf("frame %zu: %dx%d, %zu block(s), %d transform block(s), "
                  "%d wave(s), %.2f ms\n",
                  index, frame.width, frame.height, syntax.blocks.size(),
                  decoded.transform_blocks, decoder->stats().waves,
                  decoder->stats().total_ms);
      ++decoded_frames;

      std::vector<uint8_t> yuv;
      frame_to_yuv8(picture, &yuv);
      if (out_file && !yuv.empty()) std::fwrite(yuv.data(), 1, yuv.size(), out_file);

      if (!reference.empty()) {
        Frame expected(frame.width, frame.height, seq.bit_depth, syntax.ss_x, syntax.ss_y);
        const size_t frame_size = static_cast<size_t>(frame.width) * frame.height +
                                  ((syntax.ss_x ? (frame.width + 1) / 2 : frame.width) *
                                   (syntax.ss_y ? (frame.height + 1) / 2 : frame.height)) * 2;
        if (reference_offset + frame_size > reference.size()) {
          std::printf("  reference is too short for another frame\n");
        } else {
          const uint8_t* src = reference.data() + reference_offset;
          const int ss[3][2] = {{0, 0}, {syntax.ss_x, syntax.ss_y}, {syntax.ss_x, syntax.ss_y}};
          size_t at = 0;
          for (int p = 0; p < 3; ++p) {
            const int w = frame.width >> ss[p][0], h = frame.height >> ss[p][1];
            for (int y = 0; y < h; ++y) {
              uint16_t* dst = expected.plane(p) + static_cast<size_t>(y) * expected.stride[p];
              for (int x = 0; x < w; ++x) dst[x] = src[at++];
            }
          }
          report_difference(picture, expected, "reference");
          ++comparisons;
          const Psnr psnr = frame_psnr(picture, expected);
          if (psnr.y >= 99.0 && psnr.u >= 99.0 && psnr.v >= 99.0) ++exact;
          reference_offset += frame_size;
        }
      }
      refresh_ref_pool(frame, &refs);
    }
    if (status != 0) break;
  }

  if (out_file) std::fclose(out_file);
  if (comparisons > 0)
    std::printf("%d of %d decoded frame(s) bit exact\n", exact, comparisons);
  std::printf("%zu frame(s) decoded\n", decoded_frames);
  return status;
}
