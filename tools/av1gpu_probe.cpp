/*
 * AV1onGPU - what is in this file?
 *
 * The front-end parses AV1 headers but not yet the arithmetic coded tile data,
 * so this tool is the honest answer to "does the parser work on a real file?".
 * It reads an .ivf or a raw .obu stream, walks the OBUs, and prints the
 * sequence and frame header fields the decoder would use, plus exactly where
 * decoding would start.
 *
 *   av1gpu_probe clip.ivf
 *   av1gpu_probe --frames 3 clip.ivf
 *   av1gpu_probe stream.obu
 *
 * Exit status: 0 the stream parsed (unsupported features are reported but are
 * not an error), 1 the stream is invalid, 2 the file could not be read.
 */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "frontend/headers.hpp"
#include "frontend/ivf.hpp"
#include "frontend/obu.hpp"

namespace {

using namespace av1gpu::frontend;

int g_frames = -1;

void print_sequence(const SequenceHeader& seq) {
  std::printf("sequence header\n");
  std::printf("  profile %d, %d bit, %s, %dx%d max\n", seq.profile, seq.bit_depth,
              seq.monochrome ? "monochrome"
                             : (seq.is_420() ? "4:2:0" : (seq.ss_x ? "4:2:2" : "4:4:4")),
              seq.max_frame_width, seq.max_frame_height);
  std::printf("  superblock %d, cdef %s, restoration %s, superres %s\n",
              seq.use_128x128_superblock ? 128 : 64, seq.enable_cdef ? "yes" : "no",
              seq.enable_restoration ? "yes" : "no", seq.enable_superres ? "yes" : "no");
  if (!seq.reduced_still_picture_header) {
    std::printf("  order hint %s, warped motion %s, dual filter %s, masked compound %s\n",
                seq.enable_order_hint ? "yes" : "no",
                seq.enable_warped_motion ? "yes" : "no",
                seq.enable_dual_filter ? "yes" : "no",
                seq.enable_masked_compound ? "yes" : "no");
  }
  if (seq.timing.present) {
    std::printf("  timing %u/%u, %u operating point(s)\n",
                seq.timing.num_units_in_display_tick, seq.timing.time_scale,
                static_cast<unsigned>(seq.operating_points.size()));
  }
  if (seq.color_description_present) {
    std::printf("  colour: primaries %d, transfer %d, matrix %d, %s range\n",
                seq.color_primaries, seq.transfer_characteristics,
                seq.matrix_coefficients, seq.color_range ? "full" : "limited");
  }
  std::printf("  film grain parameters %s\n",
              seq.film_grain_params_present ? "present" : "absent");
}

void print_frame(const FrameHeader& hdr, size_t payload_size) {
  const size_t tile_bytes = payload_size > hdr.tile_data_offset
                                ? payload_size - hdr.tile_data_offset
                                : 0;
  std::printf("  %s frame%s, %dx%d, base_q_idx %d, %d tile(s) (%dx%d)\n",
              frame_type_name(hdr.type), hdr.show_frame ? "" : " (not shown)",
              hdr.width, hdr.height, hdr.base_q_idx, hdr.tiles.cols * hdr.tiles.rows,
              hdr.tiles.cols, hdr.tiles.rows);
  std::printf("  deblock %d/%d luma, %d/%d chroma, sharpness %d; cdef damping %d",
              hdr.loop_filter.level_y[0], hdr.loop_filter.level_y[1],
              hdr.loop_filter.level_u, hdr.loop_filter.level_v,
              hdr.loop_filter.sharpness, hdr.cdef.damping);
  if (!hdr.all_lossless) {
    std::printf(", y strengths");
    const int count = 1 << hdr.cdef.bits;
    for (int i = 0; i < count && i < 8; ++i) std::printf(" %d", hdr.cdef.y_strength[i]);
  }
  std::printf("\n");
  if (hdr.frame_id_present) std::printf("  frame id %d\n", hdr.frame_id);
  if (hdr.reduced_tx_set) std::printf("  reduced transform set\n");
  if (tile_bytes == 0) {
    std::printf("  no tile data in this OBU; decoding is not implemented yet\n");
  } else {
    std::printf("  tile data: %zu bytes at offset %zu; decoding is not implemented yet\n",
                tile_bytes, hdr.tile_data_offset);
  }
}

/* Returns 0 on success, 1 on an invalid stream. */
int probe_temporal_unit(const std::vector<uint8_t>& unit, size_t index,
                        SequenceHeader* seq, bool* have_seq) {
  std::vector<Obu> obus;
  std::string error;
  if (!split_obus(unit.data(), unit.size(), &obus, &error)) {
    std::printf("temporal unit %zu: %s\n", index, error.c_str());
    return 1;
  }
  int status = 0;
  for (const Obu& obu : obus) {
    std::printf("  obu %-20s %zu bytes\n", obu_name(obu.type), obu.size);
    if (obu.type == ObuType::SequenceHeader) {
      const ParseStatus parsed = parse_sequence_header(obu.data, obu.size, seq, &error);
      if (parsed == ParseStatus::Invalid) {
        std::printf("  sequence header invalid: %s\n", error.c_str());
        return 1;
      }
      *have_seq = true;
      print_sequence(*seq);
    } else if (obu.type == ObuType::FrameHeader || obu.type == ObuType::Frame ||
               obu.type == ObuType::RedundantFrameHeader) {
      if (!*have_seq) {
        std::printf("  frame before any sequence header\n");
        return 1;
      }
      FrameHeader hdr;
      const ParseStatus parsed = parse_frame_header(*seq, obu.data, obu.size, &hdr, &error);
      if (parsed == ParseStatus::Invalid) {
        std::printf("  frame header invalid: %s\n", error.c_str());
        return 1;
      }
      print_frame(hdr, obu.size);
      if (parsed == ParseStatus::Unsupported) {
        std::printf("  front-end: %s\n", error.c_str());
      }
    }
  }
  return status;
}

void usage() {
  std::printf(
      "usage: av1gpu_probe [--frames N] FILE\n"
      "  FILE   an .ivf file or a raw OBU stream\n"
      "  --frames N   stop after N temporal units\n");
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

} /* namespace */

int main(int argc, char** argv) {
  std::string path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    }
    if (arg == "--frames" && i + 1 < argc) {
      g_frames = std::atoi(argv[++i]);
      continue;
    }
    if (!path.empty()) {
      usage();
      return 2;
    }
    path = arg;
  }
  if (path.empty()) {
    usage();
    return 2;
  }

  std::vector<uint8_t> data;
  if (!read_file(path, &data)) {
    std::fprintf(stderr, "av1gpu_probe: cannot read %s\n", path.c_str());
    return 2;
  }

  SequenceHeader seq;
  bool have_seq = false;
  int status = 0;

  if (data.size() >= 4 && std::memcmp(data.data(), "DKIF", 4) == 0) {
    IvfReader reader;
    std::string error;
    if (!reader.open(data, &error)) {
      std::fprintf(stderr, "av1gpu_probe: %s\n", error.c_str());
      return 1;
    }
    const IvfHeader& h = reader.header();
    std::printf("%s: IVF, %dx%d, %u frame(s), timebase %u/%u\n", path.c_str(), h.width,
                h.height, h.frame_count, h.timebase_num, h.timebase_den);
    size_t index = 0;
    while (g_frames < 0 || static_cast<int>(index) < g_frames) {
      IvfFrame frame;
      if (!reader.next(&frame, &error)) {
        if (!reader.at_end() || !error.empty()) {
          std::fprintf(stderr, "av1gpu_probe: %s\n", error.c_str());
          status = 1;
        }
        break;
      }
      std::printf("temporal unit %zu (%zu bytes)\n", index, frame.payload.size());
      if (probe_temporal_unit(frame.payload, index, &seq, &have_seq) != 0) {
        status = 1;
        break;
      }
      ++index;
    }
    std::printf("parsed %zu temporal unit(s)\n", reader.frames_read());
  } else {
    std::printf("%s: raw OBU stream, %zu bytes\n", path.c_str(), data.size());
    status = probe_temporal_unit(data, 0, &seq, &have_seq);
  }

  if (!have_seq) {
    std::fprintf(stderr, "av1gpu_probe: no sequence header found\n");
    return 1;
  }
  return status;
}
