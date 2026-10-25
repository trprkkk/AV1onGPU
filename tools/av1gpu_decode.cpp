/*
 * AV1onGPU - decode a real AV1 file
 *
 * This is the tool that turns the front end into pictures: it reads an .ivf,
 * walks the OBUs, decodes each frame's tile, plans the frame for the engine and
 * lets a backend reconstruct it. Frames are decoded in output order, so the
 * picture of a frame stays in the backend's reference pool for the frames that
 * name it; `show_existing_frame` outputs a picture that is already there.
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
 *
 * `--hash-out F` writes one FNV-1a 64 hash of each shown frame's 8 bit planar
 * output, in output order; `--hash-check F` decodes and compares against such a
 * file, naming the first frame that differs. A hash list is a couple of
 * kilobytes per stream instead of a few megabytes of YUV, which is what makes a
 * bit exactness regression test cheap enough to keep in the repository.
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
  std::string hash_out;
  std::string hash_check;
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
      "  --hash-out F      write one hash per shown frame to F\n"
      "  --hash-check F    compare the output against the hashes in F\n"
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

/* FNV-1a 64 over the frame's bytes. It is used on the 8 bit planar output, which
 * is exactly what `--compare` reads, so a hash mismatch means a picture
 * difference and not a bit depth or layout difference. */
uint64_t frame_hash(const std::vector<uint8_t>& bytes) {
  uint64_t hash = 0xcbf29ce484222325ull;
  for (uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 0x100000001b3ull;
  }
  return hash;
}

/* A hash list is one `frame N: 0x...` line per shown frame, in output order, as
 * `--hash-out` writes it. `#` comments and blank lines are skipped, and the
 * first field has to be `frame`, so a comment that happens to contain `0x` (a
 * `320x240` resolution, say) is not read as a hash. */
bool read_hashes(const std::string& path, std::vector<uint64_t>* hashes) {
  std::FILE* file = std::fopen(path.c_str(), "r");
  if (!file) return false;
  char line[512];
  while (std::fgets(line, sizeof(line), file)) {
    const char* at = line;
    while (*at == ' ' || *at == '\t') ++at;
    if (*at == '#' || *at == '\n' || *at == '\0') continue;
    unsigned long long index = 0, hash = 0;
    if (std::sscanf(at, "frame %llu: 0x%llx", &index, &hash) != 2) continue;
    hashes->push_back(static_cast<uint64_t>(hash));
  }
  std::fclose(file);
  return true;
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
    if (arg == "--hash-out" && i + 1 < argc) {
      opt.hash_out = argv[++i];
      continue;
    }
    if (arg == "--hash-check" && i + 1 < argc) {
      opt.hash_check = argv[++i];
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

  std::vector<uint64_t> expected_hashes;
  if (!opt.hash_check.empty() && !read_hashes(opt.hash_check, &expected_hashes)) {
    std::fprintf(stderr, "av1gpu_decode: cannot read %s\n", opt.hash_check.c_str());
    return 2;
  }
  std::FILE* hash_file = nullptr;
  if (!opt.hash_out.empty()) {
    hash_file = std::fopen(opt.hash_out.c_str(), "w");
    if (!hash_file) {
      std::fprintf(stderr, "av1gpu_decode: cannot write %s\n", opt.hash_out.c_str());
      return 2;
    }
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
  size_t hashes_written = 0, hashes_checked = 0, hash_mismatches = 0;

  /* The decoder's own reference pool. AV1 has eight reference slots; each holds
   * a backend pool slot, and - because a show_existing_frame has to output a
   * picture that was decoded earlier - a copy of the picture. A stream may
   * refresh several AV1 slots with one picture, so a backend slot is only
   * released when no AV1 slot names it any more. */
  int pool_slot[8];
  Frame pool_picture[8];
  bool pool_picture_valid[8] = {false, false, false, false, false, false, false, false};
  for (int i = 0; i < 8; ++i) pool_slot[i] = -1;

  auto release_if_unused = [&](int slot) {
    if (slot < 0) return;
    for (int i = 0; i < 8; ++i)
      if (pool_slot[i] == slot) return;
    decoder->release_reference(slot);
  };

  /* Writes one picture to the output and compares it with the reference, if
   * there is one. A frame that is shown twice (show_existing_frame) goes
   * through here twice, which is what the reference decoder's output has. */
  auto emit_picture = [&](const Frame& picture, int ss_x, int ss_y) {
    std::vector<uint8_t> yuv;
    frame_to_yuv8(picture, &yuv);
    if (hash_file && !yuv.empty()) {
      std::fprintf(hash_file, "frame %zu: 0x%016llx\n", hashes_written++,
                   static_cast<unsigned long long>(frame_hash(yuv)));
    }
    if (!expected_hashes.empty() && !yuv.empty()) {
      const size_t at = hashes_checked++;
      const uint64_t got = frame_hash(yuv);
      if (at >= expected_hashes.size()) {
        std::printf("frame %zu: hash 0x%016llx, but the hash list ends at %zu frame(s)\n", at,
                    static_cast<unsigned long long>(got), expected_hashes.size());
        ++hash_mismatches;
      } else if (expected_hashes[at] != got) {
        std::printf("frame %zu: hash 0x%016llx, expected 0x%016llx\n", at,
                    static_cast<unsigned long long>(got),
                    static_cast<unsigned long long>(expected_hashes[at]));
        ++hash_mismatches;
      }
    }
    if (out_file && !yuv.empty()) std::fwrite(yuv.data(), 1, yuv.size(), out_file);
    if (reference.empty()) return;
    Frame expected(picture.width, picture.height, picture.bit_depth, ss_x, ss_y);
    const size_t frame_size = static_cast<size_t>(picture.width) * picture.height +
                              ((ss_x ? (picture.width + 1) / 2 : picture.width) *
                               (ss_y ? (picture.height + 1) / 2 : picture.height)) * 2;
    if (reference_offset + frame_size > reference.size()) {
      std::printf("  reference is too short for another frame\n");
      return;
    }
    const uint8_t* src = reference.data() + reference_offset;
    const int ss[3][2] = {{0, 0}, {ss_x, ss_y}, {ss_x, ss_y}};
    size_t at = 0;
    for (int p = 0; p < 3; ++p) {
      const int w = picture.width >> ss[p][0], h = picture.height >> ss[p][1];
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
  };

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
      if (std::getenv("AV1GPU_FRAMES"))
        std::fprintf(stderr,
                     "HDR unit=%zu type=%s show=%d showable=%d oh=%d refresh=0x%02x existing=%d "
                     "primary=%d refs=%d,%d,%d,%d,%d,%d,%d\n",
                     index, frame_type_name(frame.type), frame.show_frame ? 1 : 0,
                     frame.showable_frame ? 1 : 0, frame.order_hint,
                     frame.refresh_frame_flags, frame.existing_frame_idx,
                     frame.primary_ref_frame, frame.ref_frame_idx[0], frame.ref_frame_idx[1],
                     frame.ref_frame_idx[2], frame.ref_frame_idx[3], frame.ref_frame_idx[4],
                     frame.ref_frame_idx[5], frame.ref_frame_idx[6]);
      if (frame.show_existing_frame) {
        const int which = frame.existing_frame_idx;
        if (which < 0 || which >= 8 || !pool_picture_valid[which]) {
          std::printf("frame %zu: show_existing_frame %d names a slot that was never"
                      " decoded\n", index, which);
          status = 1;
          break;
        }
        std::printf("frame %zu: show_existing_frame %d\n", index, which);
        const Frame& shown = pool_picture[which];
        emit_picture(shown, shown.ss_x, shown.ss_y);
        ++decoded_frames;
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
      const ParseStatus tile_status = decode_tile(seq, frame, &refs, obu.data + tile.offset,
                                                  tile.size, &decoded, &tile_error);
      if (tile_status != ParseStatus::Ok) {
        std::printf("frame %zu: tile: %s\n", index,
                    decoded.stop_reason.empty() ? tile_error.c_str()
                                                : decoded.stop_reason.c_str());
        status = 1;
        break;
      }

      DequantisedTile quant;
      dequantise_tile(frame, seq.bit_depth, decoded, &quant);

      /* Which backend slot each of the seven reference types names. The frame
       * header resolves the type to one of the eight AV1 slots (`ref_frame_idx`)
       * and the pool says what that slot holds. */
      ReferenceInfo reference_info;
      for (int i = 0; i < 7; ++i) {
        const int which = frame.ref_frame_idx[i];
        if (which < 0 || which >= 8 || !refs.slots[which].valid) continue;
        reference_info.slot[i] = pool_slot[which];
        reference_info.width[i] = refs.slots[which].width;
        reference_info.height[i] = refs.slots[which].height;
      }

      FrameSyntax syntax;
      if (!plan_from_tile(seq, frame, decoded, quant, &syntax, &error, &reference_info)) {
        std::printf("frame %zu: %s\n", index, error.c_str());
        status = 1;
        break;
      }

      if (const char* dump = std::getenv("AV1GPU_DUMP_BLOCKS")) {
        const int limit = std::atoi(dump);
        for (int i = 0; i < limit && i < static_cast<int>(syntax.blocks.size()); ++i) {
          const BlockSyntax& b = syntax.blocks[i];
          std::printf("  block %d: plane %d at (%d,%d) %dx%d intra %d mode %d angle %d cfl %d "
                      "L%d A%d AR%d BL%d ft %d eob %d tx %d off %d",
                      i, b.plane, b.x, b.y, b.w, b.h, b.is_intra ? 1 : 0,
                      static_cast<int>(b.mode), b.angle_delta,
                      b.cfl_alpha, b.have_left ? 1 : 0, b.have_above ? 1 : 0,
                      b.have_above_right, b.have_below_left, b.filter_type,
                      b.transforms.empty() ? -1 : b.transforms[0].eob,
                      b.transforms.empty() ? -1 : static_cast<int>(b.transforms[0].tx_type),
                      b.transforms.empty() ? -1 : b.transforms[0].coef_offset);
          if (!b.is_intra)
            std::printf(" obmc %d mm %d refs %d,%d mv %d,%d %d,%d filt %d/%d comp %d blend %d,%d",
                        static_cast<int>(b.obmc.size()), b.motion_mode,
                        b.ref[0], b.ref[1], b.mv[0][0], b.mv[0][1], b.mv[1][0], b.mv[1][1],
                        static_cast<int>(b.filter), static_cast<int>(b.filter_v),
                        static_cast<int>(b.compound), b.blend[0], b.blend[1]);
          std::printf("\n");
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
      if (std::getenv("AV1GPU_NO_RESIDUAL") &&
          (std::getenv("AV1GPU_NO_RESIDUAL")[0] != '2' ||
           frame.type == FrameType::Inter)) {
        for (BlockSyntax& b : syntax.blocks)
          for (TransformBlock& tx : b.transforms) tx.eob = 0;
      }

      /* Free the slots this frame refreshes but does not predict from before
       * it is decoded. A frame is allowed to refresh a slot it reads, so the
       * picture being decoded needs a buffer of its own in that case; letting
       * the other refresh targets go early keeps the pool from filling up on a
       * stream that refreshes every slot it can. */
      for (int i = 0; i < 8; ++i) {
        if (!(frame.refresh_frame_flags & (1 << i))) continue;
        bool referenced = false;
        for (int j = 0; j < 7; ++j) {
          if (frame.ref_frame_idx[j] == i) {
            referenced = true;
            break;
          }
        }
        if (referenced) continue;
        const int freed = pool_slot[i];
        pool_slot[i] = -1;
        pool_picture_valid[i] = false;
        release_if_unused(freed);
      }

      Frame picture;
      const int slot = decoder->decode_reference(syntax, &picture, &error);
      if (slot < 0) {
        std::printf("frame %zu: %s\n", index, error.c_str());
        status = 1;
        break;
      }

      std::printf("frame %zu: %dx%d, %zu block(s), %d transform block(s), "
                  "%d wave(s), %.2f ms%s\n",
                  index, frame.width, frame.height, syntax.blocks.size(),
                  decoded.transform_blocks, decoder->stats().waves,
                  decoder->stats().total_ms, frame.show_frame ? "" : " (not shown)");
      ++decoded_frames;

      /* A frame that is not shown is still a reference for the frames that name
       * it; it just never reaches the output. Its picture is displayed later
       * through a show_existing_frame, which is what keeps the output in display
       * order while the decoder runs in coding order. */
      if (frame.show_frame) emit_picture(picture, syntax.ss_x, syntax.ss_y);

      /* Hand the picture to the AV1 slots this frame refreshes. */
      bool kept = false;
      for (int i = 0; i < 8; ++i) {
        if (!(frame.refresh_frame_flags & (1 << i))) continue;
        const int previous = pool_slot[i];
        pool_slot[i] = -1;
        release_if_unused(previous);
        pool_slot[i] = slot;
        pool_picture[i] = picture;
        pool_picture_valid[i] = true;
        kept = true;
      }
      if (!kept) release_if_unused(slot);

      /* the tile adapted the probability tables; the slots this frame refreshes
       * hand them on to the frames that name it as their primary reference */
      refresh_ref_pool(frame, &refs, decoded.ref_mvs, decoded.cdf);
    }
    if (status != 0) break;
  }

  if (out_file) std::fclose(out_file);
  if (hash_file) std::fclose(hash_file);
  if (comparisons > 0)
    std::printf("%d of %d decoded frame(s) bit exact\n", exact, comparisons);
  if (!expected_hashes.empty()) {
    const size_t matched = hashes_checked - hash_mismatches;
    std::printf("%zu of %zu decoded frame(s) match the recorded hashes\n", matched,
                expected_hashes.size());
    /* A short output is a difference too: a stream that stops early must not pass
     * just because every frame it did produce matches. */
    if (hash_mismatches != 0 || hashes_checked != expected_hashes.size()) status = 1;
  }
  std::printf("%zu frame(s) decoded\n", decoded_frames);
  return status;
}
