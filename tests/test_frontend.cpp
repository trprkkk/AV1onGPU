/*
 * AV1onGPU - front-end tests
 *
 * The bitstream layer is tested without any external encoder: the test writes
 * the same syntax the parser reads (a small bit writer, which is exactly what
 * the other tests do with the synthetic encoder), then checks field by field
 * that it comes back, and checks that broken or unsupported input is reported
 * as such instead of being misparsed.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "frontend/bitreader.hpp"
#include "frontend/cdef.hpp"
#include "frontend/coef_tables.hpp"
#include "frontend/dequant.hpp"
#include "frontend/headers.hpp"
#include "frontend/ivf.hpp"
#include "frontend/obu.hpp"
#include "frontend/reconstruct.hpp"
#include "frontend/tile_decoder.hpp"
#include "frontend/tiles.hpp"
#include "frontend/warp.hpp"
#include "test_common.hpp"

namespace {

using namespace av1gpu::frontend;

class BitWriter {
 public:
  void bit(bool value) {
    if ((bits_ % 8) == 0) bytes_.push_back(0);
    if (value) bytes_.back() |= static_cast<uint8_t>(1u << (7 - (bits_ % 8)));
    ++bits_;
  }

  void bits(uint32_t value, int count) {
    for (int i = count - 1; i >= 0; --i) bit(((value >> i) & 1u) != 0);
  }

  /* su(n): value in [-2^(n-1), 2^(n-1)-1] as a two's complement pattern. */
  void signed_bits(int value, int count) {
    bits(static_cast<uint32_t>(value) & ((1u << count) - 1u), count);
  }

  void uleb128(uint64_t value) {
    do {
      uint8_t byte = static_cast<uint8_t>(value & 0x7f);
      value >>= 7;
      if (value) byte |= 0x80;
      bits(byte, 8);
    } while (value);
  }

  void uvlc(uint32_t value) {
    const uint32_t v = value + 1;
    int zeros = 0;
    while ((v >> (zeros + 1)) != 0) ++zeros;
    for (int i = 0; i < zeros; ++i) bit(false);
    bit(true); /* stop bit */
    bits(v - (1u << zeros), zeros);
  }

  void align() {
    while ((bits_ % 8) != 0) bit(false);
  }

  const std::vector<uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
  size_t bits_ = 0;
};

/* ---- syntax writers: the mirror image of the parser ---------------------- */

struct SequenceSpec {
  int profile = 0;
  int width = 640;
  int height = 360;
  bool cdef = true;
  bool film_grain = false;
  bool order_hint = false;
  /* frame_id_numbers_present: the sequence carries frame ids, so every frame
   * header names its id and every reference names how far back it is. */
  bool frame_ids = false;
  int delta_frame_id_minus_2 = 1;
  int additional_frame_id_length_minus_1 = 0;

  /* idLen, the number of bits a frame id occupies */
  int frame_id_n_bits() const {
    return additional_frame_id_length_minus_1 + delta_frame_id_minus_2 + 3;
  }
  /* deltaFrameIdLength: the width of the "how far back" field */
  int delta_frame_id_n_bits() const { return delta_frame_id_minus_2 + 2; }
};

std::vector<uint8_t> write_sequence_header(const SequenceSpec& spec) {
  BitWriter w;
  w.bits(static_cast<uint32_t>(spec.profile), 3);
  w.bit(true);  /* still_picture */
  w.bit(false); /* reduced_still_picture_header */
  w.bit(false); /* timing_info_present */
  w.bit(false); /* display_model_info_present */
  w.bits(0, 5); /* num_operating_points_minus_1 */
  w.bits(0, 12); /* operating point idc */
  w.bits(0, 3);  /* seq_level_idx (level 2, so no seq_tier bit follows) */
  w.bits(0, 2);  /* seq_level_idx minor part */
  /* width/height: 10 bits each */
  w.bits(9, 4);  /* frame_width_bits_minus_1 */
  w.bits(9, 4);  /* frame_height_bits_minus_1 */
  w.bits(static_cast<uint32_t>(spec.width - 1), 10);
  w.bits(static_cast<uint32_t>(spec.height - 1), 10);
  w.bit(spec.frame_ids); /* frame_id_numbers_present */
  if (spec.frame_ids) {
    w.bits(static_cast<uint32_t>(spec.delta_frame_id_minus_2), 4);
    w.bits(static_cast<uint32_t>(spec.additional_frame_id_length_minus_1), 3);
  }
  w.bit(false); /* use_128x128_superblock */
  w.bit(false); /* enable_filter_intra */
  w.bit(false); /* enable_intra_edge_filter */
  w.bit(false); /* enable_interintra_compound */
  w.bit(false); /* enable_masked_compound */
  w.bit(false); /* enable_warped_motion */
  w.bit(false); /* enable_dual_filter */
  w.bit(spec.order_hint);
  if (spec.order_hint) {
    w.bit(false); /* enable_jnt_comp */
    w.bit(false); /* enable_ref_frame_mvs */
  }
  w.bit(true);  /* seq_choose_screen_content_tools */
  w.bit(true);  /* seq_choose_integer_mv */
  if (spec.order_hint) w.bits(2, 3); /* order_hint_bits_minus_1 */
  w.bit(false);           /* enable_superres */
  w.bit(spec.cdef);       /* enable_cdef */
  w.bit(false);           /* enable_restoration */
  w.bit(false);           /* high_bitdepth */
  w.bit(false);           /* monochrome */
  w.bit(false);           /* color_description_present */
  w.bit(false);           /* color_range */
  w.bits(0, 2);           /* chroma_sample_position */
  w.bit(false);           /* separate_uv_delta_q */
  w.bit(spec.film_grain); /* film_grain_params_present */
  w.bit(true);            /* trailing_bits(): the one bit */
  w.align();
  return w.bytes();
}

struct KeyFrameSpec {
  int width = 640;
  int height = 360;
  int base_q_idx = 60;
  int log2_cols = 0;
  bool show = true;
  /* only written when the sequence carries frame ids */
  int frame_id = 0;
  int order_hint = 0;
};

std::vector<uint8_t> write_key_frame_header(const SequenceSpec& seq_spec,
                                            const KeyFrameSpec& spec) {
  BitWriter w;
  w.bit(false);            /* show_existing_frame */
  w.bits(0, 2);            /* frame_type = KEY */
  w.bit(spec.show);
  w.bit(false);            /* disable_cdf_update */
  w.bit(false);            /* allow_screen_content_tools */
  if (seq_spec.frame_ids) w.bits(static_cast<uint32_t>(spec.frame_id),
                                 seq_spec.frame_id_n_bits());
  w.bit(false);            /* frame_size_override_flag */
  if (seq_spec.order_hint) w.bits(static_cast<uint32_t>(spec.order_hint), 3);
  w.bit(false);            /* render_and_frame_size_different */
  w.bit(true);             /* refresh_frame_context (inverted on the wire) */

  /* tile_info() */
  const int sbw = (spec.width + 63) / 64;
  const int sbh = (spec.height + 63) / 64;
  w.bit(true); /* uniform_tile_spacing_flag */
  for (int i = 0; i < spec.log2_cols; ++i) w.bit(true);
  if (spec.log2_cols < 4 && (1 << spec.log2_cols) < sbw) w.bit(false);
  for (int i = 0; i < 4; ++i) {
    if ((spec.height + 63) / 64 <= (1 << i)) break;
    w.bit(false); /* log2_rows: stop immediately */
    break;
  }
  (void)sbh;
  if (spec.log2_cols) {
    w.bits(0, spec.log2_cols); /* context_update_tile_id */
    w.bits(0, 2);              /* tile_size_bytes_minus_1 */
  }

  /* quantisation */
  w.bits(static_cast<uint32_t>(spec.base_q_idx), 8);
  w.bit(false); /* delta_q_y_dc present */
  w.bit(false); /* delta_q_u_dc present */
  w.bit(false); /* delta_q_u_ac present */
  w.bit(false); /* using_qmatrix */
  w.bit(false); /* segmentation_enabled */
  w.bit(false); /* delta_q_present */
  w.bits(24, 6);/* loop filter level_y[0] */
  w.bits(24, 6);/* loop filter level_y[1] */
  w.bits(12, 6);/* level_u */
  w.bits(12, 6);/* level_v */
  w.bits(0, 3); /* sharpness */
  w.bit(false); /* mode_ref_delta_enabled */
  w.bits(0, 2); /* cdef damping_minus_3 */
  w.bits(0, 2); /* cdef bits */
  w.bits(4, 6); /* cdef y strength[0] */
  w.bits(2, 6); /* cdef uv strength[0] */
  w.bit(true);  /* tx_mode_select */
  w.bit(false); /* reduced_tx_set */
  if (seq_spec.film_grain) w.bit(false); /* film grain present */
  w.align();
  return w.bytes();
}

std::vector<uint8_t> make_obu(ObuType type, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> out;
  out.push_back(static_cast<uint8_t>((static_cast<int>(type) << 3) | 0x02)); /* has size */
  uint64_t size = payload.size();
  do {
    uint8_t byte = static_cast<uint8_t>(size & 0x7f);
    size >>= 7;
    if (size) byte |= 0x80;
    out.push_back(byte);
  } while (size);
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

std::vector<uint8_t> make_ivf(int width, int height, const std::vector<uint8_t>& unit) {
  std::vector<uint8_t> layout;
  auto u16 = [&](uint16_t v) {
    layout.push_back(static_cast<uint8_t>(v & 0xff));
    layout.push_back(static_cast<uint8_t>(v >> 8));
  };
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) layout.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
  };
  layout.insert(layout.end(), {'D', 'K', 'I', 'F'});
  u16(0);  /* version */
  u16(32); /* header size */
  layout.insert(layout.end(), {'A', 'V', '0', '1'});
  u16(static_cast<uint16_t>(width));
  u16(static_cast<uint16_t>(height));
  u32(30); /* timebase num */
  u32(1);
  u32(1); /* frame count */
  u32(0); /* unused */
  u32(static_cast<uint32_t>(unit.size())); /* frame size */
  u32(0);  /* timestamp low */
  u32(0);  /* timestamp high */
  layout.insert(layout.end(), unit.begin(), unit.end());
  return layout;
}

/* ---- tests --------------------------------------------------------------- */

void test_bit_reader() {
  std::printf("bit reader\n");
  const uint8_t bytes[] = {0xb3, 0x40}; /* 1011 0011 0100 0000 */
  BitReader r(bytes, sizeof(bytes));
  CHECK_EQ(r.read(3), 5u);
  CHECK_EQ(r.read_bit(), true);
  CHECK_EQ(r.read(2), 0u);
  CHECK_EQ(r.read(4), 13u);
  CHECK_EQ(r.read_bit(), false);
  CHECK_EQ(r.bits_read(), 11u);

  const uint8_t signed_byte[] = {0xe0};
  BitReader s(signed_byte, sizeof(signed_byte));
  CHECK_EQ(s.read_signed(4), -2);
  CHECK_EQ(s.read_signed(4), 0);

  /* ns(3): w = 2, m = 1, codes 0, 1 0, 1 1 */
  const uint8_t zero[] = {0x00};
  BitReader u1(zero, sizeof(zero));
  CHECK_EQ(u1.read_uniform(3), 0u);
  const uint8_t one[] = {0x80};
  BitReader u2(one, sizeof(one));
  CHECK_EQ(u2.read_uniform(3), 1u);
  const uint8_t two[] = {0xc0};
  BitReader u3(two, sizeof(two));
  CHECK_EQ(u3.read_uniform(3), 2u);
  CHECK_EQ(u3.read_uniform(1), 0u); /* ns(1) consumes nothing */

  /* uvlc(3): two leading zeros, stop bit, then two zero bits */
  const uint8_t vlc[] = {0x20};
  BitReader v(vlc, sizeof(vlc));
  CHECK_EQ(v.read_uvlc(), 3u);

  /* leb128: 300 = 0xac 0x02 */
  const uint8_t leb[] = {0xac, 0x02};
  size_t pos = 0;
  uint64_t value = 0;
  std::string error;
  CHECK(read_leb128(leb, sizeof(leb), &pos, &value, &error));
  CHECK_EQ(value, 300ull);
  CHECK_EQ(pos, 2u);

  const uint8_t overrun[] = {0xff};
  BitReader o(overrun, sizeof(overrun));
  o.read(8);
  CHECK_EQ(o.overrun(), false);
  o.read(8);
  CHECK_EQ(o.overrun(), true);
}

void test_obu_framing() {
  std::printf("OBU framing\n");
  std::vector<uint8_t> payload(300, 0x5a);
  const std::vector<uint8_t> stream = make_obu(ObuType::Frame, payload);

  std::vector<Obu> obus;
  std::string error;
  CHECK(split_obus(stream.data(), stream.size(), &obus, &error));
  CHECK_EQ(obus.size(), 1u);
  if (obus.size() == 1) {
    CHECK(obus[0].type == ObuType::Frame);
    CHECK_EQ(obus[0].size, 300u);
    CHECK_EQ(obus[0].data[0], 0x5a);
  }
  CHECK(std::string(obu_name(ObuType::TileGroup)) == "tile_group");

  /* A truncated size field must be an error, not a short OBU. */
  const uint8_t broken[] = {0x1a, 0x80};
  CHECK(!split_obus(broken, sizeof(broken), &obus, &error));
  CHECK(!error.empty());

  /* Sizes that run past the end of the buffer are rejected. */
  const uint8_t too_long[] = {0x1a, 0x7f, 0x00};
  CHECK(!split_obus(too_long, sizeof(too_long), &obus, &error));
  CHECK(!error.empty());

  /* Two OBUs back to back, one with an extension header. */
  std::vector<uint8_t> two = make_obu(ObuType::TemporalDelimiter, {});
  two.push_back(static_cast<uint8_t>((static_cast<int>(ObuType::Padding) << 3) | 0x06));
  two.push_back(0x00); /* temporal + spatial id */
  two.push_back(0x01); /* size: 1 byte of padding */
  two.push_back(0x00);
  CHECK(split_obus(two.data(), two.size(), &obus, &error));
  CHECK_EQ(obus.size(), 2u);
  if (obus.size() == 2) {
    CHECK(obus[0].type == ObuType::TemporalDelimiter);
    CHECK_EQ(obus[1].size, 1u);
  }

  /* The forbidden bit and reserved types are refused. */
  const uint8_t forbidden[] = {0x9a, 0x00};
  CHECK(!split_obus(forbidden, sizeof(forbidden), &obus, &error));
  const uint8_t reserved[] = {0x4a, 0x00};
  CHECK(!split_obus(reserved, sizeof(reserved), &obus, &error));
}

void test_ivf() {
  std::printf("IVF container\n");
  SequenceSpec spec;
  const std::vector<uint8_t> seq = write_sequence_header(spec);
  const std::vector<uint8_t> frame = write_key_frame_header(spec, KeyFrameSpec());
  std::vector<uint8_t> unit = make_obu(ObuType::TemporalDelimiter, {});
  const std::vector<uint8_t> seq_obu = make_obu(ObuType::SequenceHeader, seq);
  const std::vector<uint8_t> frame_obu = make_obu(ObuType::Frame, frame);
  unit.insert(unit.end(), seq_obu.begin(), seq_obu.end());
  unit.insert(unit.end(), frame_obu.begin(), frame_obu.end());

  const std::vector<uint8_t> ivf = make_ivf(spec.width, spec.height, unit);
  IvfReader reader;
  std::string error;
  CHECK(reader.open(ivf, &error));
  CHECK_EQ(reader.header().width, 640);
  CHECK_EQ(reader.header().height, 360);
  CHECK_EQ(reader.header().frame_count, 1u);
  CHECK(std::string(reader.header().fourcc) == "AV01");

  IvfFrame got;
  CHECK(reader.next(&got, &error));
  CHECK_EQ(got.payload.size(), unit.size());
  CHECK(reader.at_end());
  CHECK(!reader.next(&got, &error));

  /* Wrong fourcc, wrong signature, truncated frame header. */
  std::vector<uint8_t> other = ivf;
  other[8] = 'V'; /* VP90 -> AV01 becomes VV01 */
  CHECK(!reader.open(other, &error));
  CHECK(!error.empty());

  /* A valid container around a truncated temporal unit: the header is fine,
   * reading the frame is what fails. */
  std::vector<uint8_t> truncated(ivf.begin(), ivf.begin() + 40);
  CHECK(reader.open(truncated, &error));
  CHECK(!reader.next(&got, &error));
  CHECK(!error.empty());

  /* And through the whole pipeline: OBUs -> sequence -> frame header. */
  std::vector<Obu> obus;
  CHECK(split_obus(unit.data(), unit.size(), &obus, &error));
  CHECK_EQ(obus.size(), 3u);
  SequenceHeader sequence;
  CHECK(parse_sequence_header(obus[1].data, obus[1].size, &sequence, &error) ==
        ParseStatus::Ok);
  FrameHeader header;
  CHECK(parse_frame_header(sequence, obus[2].data, obus[2].size, &header, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(header.width, 640);
}

void test_sequence_header() {
  std::printf("sequence header\n");
  SequenceSpec spec;
  const std::vector<uint8_t> bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(bytes.data(), bytes.size(), &seq, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(seq.profile, 0);
  CHECK_EQ(seq.bit_depth, 8);
  CHECK_EQ(seq.max_frame_width, 640);
  CHECK_EQ(seq.max_frame_height, 360);
  CHECK_EQ(seq.ss_x, 1);
  CHECK_EQ(seq.ss_y, 1);
  CHECK_EQ(seq.monochrome, false);
  CHECK_EQ(seq.enable_cdef, true);
  CHECK_EQ(seq.enable_restoration, false);
  CHECK_EQ(seq.film_grain_params_present, false);
  CHECK_EQ(seq.enable_order_hint, false);
  CHECK_EQ(seq.reduced_still_picture_header, false);
  CHECK_EQ(seq.width_n_bits, 10);
  CHECK_EQ(seq.seq_force_screen_content_tools, 2);

  /* 10 bit 4:4:4 is a different colour_config branch. */
  SequenceSpec hbd;
  hbd.profile = 1;
  BitWriter w;
  w.bits(1, 3);          /* profile */
  w.bit(true);           /* still_picture */
  w.bit(false);          /* reduced_still_picture_header */
  w.bit(false);          /* timing_info_present */
  w.bit(false);          /* display_model_info_present */
  w.bits(0, 5);          /* one operating point */
  w.bits(0, 12);
  w.bits(0, 3);
  w.bits(0, 2);
  w.bits(9, 4);
  w.bits(9, 4);
  w.bits(1279 & 0x3ff, 10);
  w.bits(719, 10);
  w.bit(false); /* frame_id_numbers_present */
  w.bit(false); /* use_128x128_superblock */
  w.bit(true);  /* enable_filter_intra */
  w.bit(true);  /* enable_intra_edge_filter */
  w.bit(false); /* interintra */
  w.bit(false); /* masked compound */
  w.bit(false); /* warped motion */
  w.bit(false); /* dual filter */
  w.bit(false); /* order hint */
  w.bit(true);  /* choose screen content tools */
  w.bit(true);  /* choose integer mv */
  w.bit(false); /* superres */
  w.bit(true);  /* cdef */
  w.bit(true);  /* restoration */
  w.bit(true);  /* high_bitdepth: 10 bit */
  /* profile 1 has no monochrome bit */
  w.bit(true);  /* color_description_present */
  w.bits(1, 8);  /* BT.709 */
  w.bits(13, 8); /* sRGB */
  w.bits(0, 8);  /* identity matrix: 4:4:4 full range, no extra bits */
  w.bit(false);  /* separate_uv_delta_q */
  w.bit(false);  /* film_grain_params_present */
  w.bit(true);   /* trailing_bits() */
  w.align();
  const std::vector<uint8_t> rgb = w.bytes();
  const ParseStatus rgb_status = parse_sequence_header(rgb.data(), rgb.size(), &seq, &error);
  if (rgb_status != ParseStatus::Ok) std::printf("  identity header: %s\n", error.c_str());
  CHECK(rgb_status == ParseStatus::Ok);
  CHECK_EQ(seq.profile, 1);
  CHECK_EQ(seq.bit_depth, 10);
  CHECK_EQ(seq.ss_x, 0);
  CHECK_EQ(seq.ss_y, 0);

  /* Broken input: profile 3, truncated, and non-zero padding. */
  const uint8_t bad_profile[] = {0xe0, 0x00, 0x00};
  CHECK(parse_sequence_header(bad_profile, sizeof(bad_profile), &seq, &error) ==
        ParseStatus::Invalid);
  CHECK(parse_sequence_header(bytes.data(), 2, &seq, &error) != ParseStatus::Ok);

  std::vector<uint8_t> padded = bytes;
  padded.push_back(0x01);
  CHECK(parse_sequence_header(padded.data(), padded.size(), &seq, &error) ==
        ParseStatus::Invalid);
  CHECK(!error.empty());
}

void test_frame_header() {
  std::printf("frame header\n");
  SequenceSpec spec;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);

  KeyFrameSpec frame_spec;
  const std::vector<uint8_t> bytes = write_key_frame_header(spec, frame_spec);
  FrameHeader header;
  CHECK(parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error) ==
        ParseStatus::Ok);
  CHECK(header.type == FrameType::Key);
  CHECK_EQ(header.show_frame, true);
  CHECK_EQ(header.error_resilient_mode, true);
  CHECK_EQ(header.refresh_frame_flags, 0xff);
  CHECK_EQ(header.width, 640);
  CHECK_EQ(header.height, 360);
  CHECK_EQ(header.render_width, 640);
  CHECK_EQ(header.base_q_idx, 60);
  CHECK_EQ(header.tiles.cols, 1);
  CHECK_EQ(header.tiles.rows, 1);
  CHECK_EQ(header.tiles.log2_cols, 0);
  CHECK_EQ(header.tiles.tile_size_bytes, 1);
  CHECK_EQ(header.loop_filter.level_y[0], 24);
  CHECK_EQ(header.loop_filter.level_u, 12);
  CHECK_EQ(header.loop_filter.sharpness, 0);
  CHECK_EQ(header.cdef.damping, 3);
  CHECK_EQ(header.cdef.y_strength[0], 4);
  CHECK_EQ(header.cdef.uv_strength[0], 2);
  CHECK_EQ(header.all_lossless, false);
  CHECK_EQ(header.tx_mode_select, true);
  CHECK_EQ(header.tile_data_offset, bytes.size());
  CHECK_EQ(static_cast<int>(header.primary_ref_frame), kPrimaryRefNone);

  /* Two tile columns: exercises the uniform tile loop and its extra fields. */
  KeyFrameSpec tiles;
  tiles.log2_cols = 1;
  const std::vector<uint8_t> tile_bytes = write_key_frame_header(spec, tiles);
  CHECK(parse_frame_header(seq, tile_bytes.data(), tile_bytes.size(), &header, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(header.tiles.log2_cols, 1);
  CHECK_EQ(header.tiles.cols, 2);
  CHECK_EQ(header.tiles.rows, 1);
  CHECK_EQ(header.tiles.context_update_tile_id, 0);

  /* Lossless key frame: the deblocking and CDEF bits disappear. */
  BitWriter w;
  w.bit(false); /* show_existing_frame */
  w.bits(0, 2); /* KEY */
  w.bit(true);  /* show_frame */
  w.bit(false); /* disable_cdf_update */
  w.bit(false); /* allow_screen_content_tools */
  w.bit(false); /* frame_size_override */
  w.bit(false); /* render size */
  w.bit(true);  /* refresh_frame_context */
  w.bit(true);  /* uniform tiles */
  w.bit(false); /* stop: one column */
  w.bit(false); /* stop: one row */
  w.bits(0, 8); /* base_q_idx = 0 */
  w.bit(false); /* delta_q_y_dc */
  w.bit(false); /* delta_q_u_dc */
  w.bit(false); /* delta_q_u_ac */
  w.bit(false); /* using_qmatrix */
  w.bit(false); /* segmentation */
  w.bit(false); /* reduced_tx_set */
  w.align();
  const std::vector<uint8_t> lossless = w.bytes();
  CHECK(parse_frame_header(seq, lossless.data(), lossless.size(), &header, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(header.all_lossless, true);
  CHECK_EQ(header.loop_filter.mode_ref_delta_enabled, true);
  CHECK_EQ(header.cdef.damping, 3);

  /* A truncated frame header is invalid, not a short frame. */
  CHECK(parse_frame_header(seq, bytes.data(), 8, &header, &error) ==
        ParseStatus::Invalid);
}

/* One inter frame header referencing slot 0, following the specification order.
 * Every field the parser expects is written here, so a reordering in the parser
 * shows up as a mismatch rather than as silently shifted values. */
std::vector<uint8_t> write_inter_frame_header(const SequenceSpec& spec, int order_hint,
                                              int refresh_flags) {
  BitWriter w;
  w.bit(false);     /* show_existing_frame */
  w.bits(1, 2);     /* INTER */
  w.bit(true);      /* show_frame */
  w.bit(false);     /* error_resilient_mode */
  w.bit(false);     /* disable_cdf_update */
  w.bit(false);     /* allow_screen_content_tools */
  w.bit(false);     /* frame_size_override: inherit the reference size */
  if (spec.order_hint) w.bits(static_cast<uint32_t>(order_hint), 3);
  w.bits(0, 3);     /* primary_ref_frame: slot 0 */
  w.bits(static_cast<uint32_t>(refresh_flags), 8);
  w.bit(false);     /* frame_refs_short_signaling */
  for (int i = 0; i < 7; ++i) w.bits(0, 3); /* ref_frame_idx: all slot 0 */
  w.bit(false);     /* render_and_frame_size_different */
  w.bit(false);     /* allow_high_precision_mv */
  w.bit(true);      /* is_filter_switchable */
  w.bit(false);     /* is_motion_mode_switchable */
  w.bit(false);     /* disable_frame_end_update_cdf */
  w.bit(true);      /* uniform_tile_spacing_flag */
  w.bit(false);     /* stop: one tile column */
  w.bit(false);     /* stop: one tile row */
  w.bits(100, 8);   /* base_q_idx */
  w.bit(false);     /* delta_q_y_dc */
  w.bit(false);     /* delta_q_u_dc */
  w.bit(false);     /* delta_q_u_ac */
  w.bit(false);     /* using_qmatrix */
  w.bit(false);     /* segmentation_enabled */
  w.bit(false);     /* delta_q_present */
  w.bits(0, 6);     /* loop_filter_level[0] */
  w.bits(0, 6);     /* loop_filter_level[1] */
  w.bits(0, 3);     /* loop_filter_sharpness */
  w.bit(false);     /* loop_filter_delta_enabled */
  w.bits(0, 2);     /* cdef_damping_minus_3 */
  w.bits(0, 2);     /* cdef_bits */
  w.bits(0, 6);     /* cdef y strength */
  w.bits(0, 6);     /* cdef uv strength */
  w.bit(false);     /* tx_mode_select */
  w.bit(true);      /* reference_select */
  w.bit(false);     /* reduced_tx_set */
  for (int i = 1; i < 8; ++i) w.bit(false); /* is_global */
  w.align();
  return w.bytes();
}

/* show_existing_frame: no tile data, just the slot to display - and, once the
 * sequence carries frame ids, the id that frame was stored with. */
std::vector<uint8_t> write_show_existing_frame(const SequenceSpec& spec, int slot,
                                               int display_frame_id) {
  BitWriter w;
  w.bit(true);  /* show_existing_frame */
  w.bits(static_cast<uint32_t>(slot), 3);
  if (spec.frame_ids) w.bits(static_cast<uint32_t>(display_frame_id), spec.frame_id_n_bits());
  w.align();
  return w.bytes();
}

/* An inter frame header written field by field for a sequence that carries
 * order hints and/or frame ids - the two things `write_inter_frame_header`
 * above leaves out, because its sequence has neither. `slots` is the reference
 * list the frame codes (ignored when `short_signaling` is set) and `deltas` the
 * "how far back" value for each of the seven references. */
struct InterHeaderSpec {
  int order_hint = 4;
  int frame_id = 0;
  int refresh_flags = 0x01;
  bool short_signaling = false;
  int last_frame_idx = 0;
  int gold_frame_idx = 0;
  int slots[7] = {0, 0, 0, 0, 0, 0, 0};
  int deltas[7] = {1, 1, 1, 1, 1, 1, 1};
  bool error_resilient = false;
  int ref_hints[8] = {0, 0, 0, 0, 0, 0, 0, 0}; /* only coded when error_resilient */
  /* frame_size_override_flag: with it the frame codes a size - its own, or the
   * index of the reference whose size it copies (found_ref) when it is not
   * error resilient. */
  bool size_override = false;
  int found_ref = -1; /* -1: frame_size(), otherwise frame_size_with_refs() */
  int width = 640;
  int height = 360;
  bool reference_select = false;
};

std::vector<uint8_t> write_inter_frame_header_ids(const SequenceSpec& spec,
                                                  const InterHeaderSpec& in) {
  BitWriter w;
  w.bit(false);        /* show_existing_frame */
  w.bits(1, 2);        /* frame_type = INTER */
  w.bit(true);         /* show_frame */
  w.bit(in.error_resilient);
  w.bit(false);        /* disable_cdf_update */
  w.bit(false);        /* allow_screen_content_tools */
  if (spec.frame_ids) w.bits(static_cast<uint32_t>(in.frame_id), spec.frame_id_n_bits());
  w.bit(in.size_override);
  if (spec.order_hint) w.bits(static_cast<uint32_t>(in.order_hint), 3);
  if (!in.error_resilient) w.bits(0, 3); /* primary_ref_frame: slot 0 */
  w.bits(static_cast<uint32_t>(in.refresh_flags), 8);
  if (in.error_resilient && spec.order_hint) {
    /* an error resilient frame re-states the order hint of every slot */
    for (int i = 0; i < 8; ++i) w.bits(static_cast<uint32_t>(in.ref_hints[i]), 3);
  }
  w.bit(in.short_signaling);
  if (in.short_signaling) {
    w.bits(static_cast<uint32_t>(in.last_frame_idx), 3);
    w.bits(static_cast<uint32_t>(in.gold_frame_idx), 3);
  } else {
    for (int i = 0; i < 7; ++i) w.bits(static_cast<uint32_t>(in.slots[i]), 3);
  }
  if (spec.frame_ids) {
    for (int i = 0; i < 7; ++i) {
      /* the field is delta_frame_id_minus_1: one less than the distance */
      w.bits(static_cast<uint32_t>(in.deltas[i] - 1), spec.delta_frame_id_n_bits());
    }
  }
  /* frame_size_with_refs() is reached with the override on and error resilient
   * off: one found_ref bit per reference, and the bits after the first one that
   * is set are not coded. If none is set the frame codes its size as usual. */
  const bool with_refs = in.size_override && !in.error_resilient;
  bool inherited = false;
  if (with_refs) {
    for (int i = 0; i < 7; ++i) {
      w.bit(in.found_ref == i);
      if (in.found_ref == i) {
        inherited = true;
        break;
      }
    }
  }
  if (in.size_override && !inherited) {
    w.bits(static_cast<uint32_t>(in.width - 1), 10);  /* width_n_bits */
    w.bits(static_cast<uint32_t>(in.height - 1), 10); /* height_n_bits */
  }
  /* render_size() is skipped when the size came from a reference */
  if (!inherited) w.bit(false); /* render_and_frame_size_different */
  w.bit(false);        /* allow_high_precision_mv */
  w.bit(true);         /* is_filter_switchable */
  w.bit(false);        /* is_motion_mode_switchable */
  w.bit(false);        /* disable_frame_end_update_cdf */
  w.bit(true);         /* uniform_tile_spacing_flag */
  w.bit(false);        /* stop: one tile column */
  w.bit(false);        /* stop: one tile row */
  w.bits(100, 8);      /* base_q_idx */
  w.bit(false);        /* delta_q_y_dc */
  w.bit(false);        /* delta_q_u_dc */
  w.bit(false);        /* delta_q_u_ac */
  w.bit(false);        /* using_qmatrix */
  w.bit(false);        /* segmentation_enabled */
  w.bit(false);        /* delta_q_present */
  w.bits(0, 6);        /* loop_filter_level[0] */
  w.bits(0, 6);        /* loop_filter_level[1] */
  w.bits(0, 3);        /* loop_filter_sharpness */
  w.bit(false);        /* loop_filter_delta_enabled */
  w.bits(0, 2);        /* cdef_damping_minus_3 */
  w.bits(0, 2);        /* cdef_bits */
  w.bits(0, 6);        /* cdef y strength */
  w.bits(0, 6);        /* cdef uv strength */
  w.bit(false);        /* tx_mode_select */
  w.bit(in.reference_select);
  w.bit(false);        /* reduced_tx_set */
  for (int ref = 1; ref < 8; ++ref) w.bit(false); /* is_global */
  w.align();
  return w.bytes();
}

void test_inter_frame() {
  std::printf("inter frame header\n");
  SequenceSpec spec;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);

  /* A key frame first, so the reference pool has a valid slot to point at. */
  const std::vector<uint8_t> key = write_key_frame_header(spec, KeyFrameSpec());
  FrameHeader header;
  RefPool pool;
  CHECK(parse_frame_header(seq, key.data(), key.size(), &header, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(header.refresh_frame_flags, 0xff);
  refresh_ref_pool(header, &pool);
  CHECK_EQ(pool.slots[0].valid, true);
  CHECK_EQ(pool.slots[0].width, 640);

  const std::vector<uint8_t> inter = write_inter_frame_header(spec, 0, 0x01);
  const ParseStatus inter_status =
      parse_frame_header(seq, inter.data(), inter.size(), &header, &error, &pool);
  if (inter_status != ParseStatus::Ok) std::printf("  inter header: %s\n", error.c_str());
  CHECK(inter_status == ParseStatus::Ok);
  CHECK(header.type == FrameType::Inter);
  CHECK_EQ(header.show_frame, true);
  /* frame_size_override is off, so the frame is the sequence's maximum size and
   * no size bits were coded - not the reference's size (they are equal here,
   * but that is a property of this stream, not of the syntax). */
  CHECK_EQ(header.size_from_reference, false);
  CHECK_EQ(header.width, 640);
  CHECK_EQ(header.height, 360);
  CHECK_EQ(header.refresh_frame_flags, 1);
  CHECK_EQ(header.ref_frame_idx[0], 0);
  CHECK_EQ(header.base_q_idx, 100);
  CHECK_EQ(header.reference_select, true);
  CHECK_EQ(header.skip_mode_allowed, false); /* order hints are off in this seq */
  CHECK_EQ(header.tiles.cols, 1);
  CHECK_EQ(header.tiles.rows, 1);
  CHECK_EQ(header.tile_data_offset, inter.size());
  refresh_ref_pool(header, &pool);

  /* Inter frames need the pool: without it the parser says so instead of
   * guessing a size. */
  CHECK(parse_frame_header(seq, inter.data(), inter.size(), &header, &error) ==
        ParseStatus::Unsupported);
  CHECK(!error.empty());

  /* An inter frame pointing at an empty slot is invalid. */
  RefPool empty;
  CHECK(parse_frame_header(seq, inter.data(), inter.size(), &header, &error, &empty) ==
        ParseStatus::Invalid);

  /* show_existing_frame: index into the pool, no tile data. */
  BitWriter e;
  e.bit(true);  /* show_existing_frame */
  e.bits(0, 3); /* existing_frame_idx */
  e.align();
  const std::vector<uint8_t> existing = e.bytes();
  CHECK(parse_frame_header(seq, existing.data(), existing.size(), &header, &error,
                           &pool) == ParseStatus::Ok);
  CHECK_EQ(header.show_existing_frame, true);
  CHECK_EQ(header.existing_frame_idx, 0);
  CHECK_EQ(header.width, 640);
  CHECK_EQ(header.height, 360);

  /* The same header with an empty pool is invalid. */
  CHECK(parse_frame_header(seq, existing.data(), existing.size(), &header, &error,
                           &empty) == ParseStatus::Invalid);
}

/* set_frame_refs (7.8): a frame that codes only LAST_FRAME and GOLDEN_FRAME
 * still ends up with all seven references, derived from the order hints the
 * slots carry. The pools here are hand built, because the interesting cases
 * (nothing in the future, two slots with the same hint) are ones an encoder
 * rarely produces and a real stream therefore cannot test. */
void test_set_frame_refs() {
  std::printf("set frame refs\n");
  SequenceSpec spec;
  spec.order_hint = true;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(seq.order_hint_n_bits, 3);

  auto make_pool = [](const int (&hints)[8]) {
    RefPool pool;
    for (int i = 0; i < 8; ++i) {
      pool.slots[i].valid = true;
      pool.slots[i].order_hint = hints[i];
      pool.slots[i].frame_id = -1;
    }
    return pool;
  };

  /* Hints 1 and 3 are coded as LAST and GOLDEN, 6 and 7 wait in the future,
   * and two slots share hint 4. curFrameHint is 1 << (3 - 1) = 4, so a frame
   * three hints back sits at 1 and one hint ahead at 5. */
  {
    const int hints[8] = {1, 2, 3, 6, 7, 4, 4, 3};
    const RefPool pool = make_pool(hints);
    FrameHeader h;
    h.order_hint = 5;
    h.last_frame_idx = 0;
    h.gold_frame_idx = 2;
    set_frame_refs(seq, &h, pool);
    CHECK_EQ(h.ref_frame_idx[kRefLast - 1], 0);
    CHECK_EQ(h.ref_frame_idx[kRefGolden - 1], 2);
    /* ALTREF is the frame with the highest output order, BWDREF the closest of
     * the remaining ones - so the two future frames land in that order. */
    CHECK_EQ(h.ref_frame_idx[kRefAltref - 1], 4);
    CHECK_EQ(h.ref_frame_idx[kRefBwdref - 1], 3);
    /* no future frame is left for ALTREF2, so it falls through to a forward
     * one: the latest unused, which is hint 3 at slot 7. */
    CHECK_EQ(h.ref_frame_idx[kRefAltref2 - 1], 7);
    /* LAST2 and LAST3 take the two slots sharing hint 4, later index first */
    CHECK_EQ(h.ref_frame_idx[kRefLast2 - 1], 6);
    CHECK_EQ(h.ref_frame_idx[kRefLast3 - 1], 5);
    /* all seven references resolve to seven different slots */
    for (int i = 0; i < 7; ++i)
      for (int j = i + 1; j < 7; ++j) CHECK(h.ref_frame_idx[i] != h.ref_frame_idx[j]);
  }

  /* Nothing in the future at all: the backward scans come up empty, and every
   * reference comes from the forward list, deepest first. */
  {
    const int hints[8] = {1, 2, 3, 4, 4, 3, 2, 1};
    const RefPool pool = make_pool(hints);
    FrameHeader h;
    h.order_hint = 5;
    h.last_frame_idx = 3; /* hint 4, the nearest */
    h.gold_frame_idx = 0; /* hint 1, the furthest */
    set_frame_refs(seq, &h, pool);
    CHECK_EQ(h.ref_frame_idx[kRefLast - 1], 3);
    CHECK_EQ(h.ref_frame_idx[kRefGolden - 1], 0);
    CHECK_EQ(h.ref_frame_idx[kRefLast2 - 1], 4);  /* hint 4, the other slot */
    CHECK_EQ(h.ref_frame_idx[kRefLast3 - 1], 5);  /* hint 3, later slot on the tie */
    CHECK_EQ(h.ref_frame_idx[kRefBwdref - 1], 2); /* hint 3 as well */
    CHECK_EQ(h.ref_frame_idx[kRefAltref2 - 1], 6);
    CHECK_EQ(h.ref_frame_idx[kRefAltref - 1], 1);
    for (int i = 0; i < 7; ++i) CHECK(h.ref_frame_idx[i] >= 0 && h.ref_frame_idx[i] < 8);
    /* and none of them is a frame from the future */
    for (int i = 0; i < 7; ++i) {
      const int slot = h.ref_frame_idx[i];
      CHECK(get_relative_dist(seq, pool.slots[slot].order_hint, h.order_hint) < 0);
    }
  }

  /* A pool with a single usable slot: every scan finds nothing, so all five
   * derived references land on the one frame there is. Wrong, but defined -
   * a decoder that left them at -1 would index out of bounds instead. */
  {
    const int hints[8] = {2, 0, 0, 0, 0, 0, 0, 0};
    RefPool pool = make_pool(hints);
    for (int i = 1; i < 8; ++i) pool.slots[i].valid = false;
    FrameHeader h;
    h.order_hint = 3;
    h.last_frame_idx = 0;
    h.gold_frame_idx = 0;
    set_frame_refs(seq, &h, pool);
    for (int i = 0; i < 7; ++i) CHECK_EQ(h.ref_frame_idx[i], 0);
  }
}

/* The scans above have several places where the order of two candidates - and
 * what happens when they are equal - decides which frame a reference names. No
 * encoder we have emits short signaling, so a stream cannot settle those cases;
 * the reference decoder can. dav1d resolves the same list from the same order
 * hints with a different structure (plain signed differences and a "used"
 * marker instead of shifted hints and a used array), so its result is an
 * independent answer to compare ours against. The transcription below is
 * dav1d's src/obu.c, keeping its sentinels so the comparison is literal. */
std::vector<int> dav1d_set_frame_refs(const SequenceHeader& seq, const int (&hints)[8], int cur,
                                      int last, int gold) {
  const int kUsed = INT32_MIN;
  int off[8];
  for (int i = 0; i < 8; ++i) off[i] = get_relative_dist(seq, hints[i], cur);
  int earliest_ref = -1;
  for (int i = 0, earliest = INT32_MAX; i < 8; ++i) {
    if (off[i] < earliest) {
      earliest = off[i];
      earliest_ref = i;
    }
  }
  off[last] = kUsed;
  off[gold] = kUsed;

  std::vector<int> refs(7, -1);
  refs[0] = last;
  refs[3] = gold;
  int refidx = -1;
  for (int i = 0, latest = 0; i < 8; ++i) {
    if (off[i] >= latest) {
      latest = off[i];
      refidx = i;
    }
  }
  if (refidx >= 0) off[refidx] = kUsed;
  refs[6] = refidx;
  for (int i = 4; i < 6; ++i) {
    unsigned earliest = 255; /* UINT8_MAX: only non-negative offsets qualify */
    refidx = -1;
    for (int j = 0; j < 8; ++j) {
      const unsigned hint = static_cast<unsigned>(off[j]);
      if (hint < earliest) {
        earliest = hint;
        refidx = j;
      }
    }
    if (refidx >= 0) off[refidx] = kUsed;
    refs[i] = refidx;
  }
  for (int i = 1; i < 7; ++i) {
    if (refs[i] < 0) {
      unsigned latest = ~255u; /* negative offsets, largest first */
      refidx = -1;
      for (int j = 0; j < 8; ++j) {
        const unsigned hint = static_cast<unsigned>(off[j]);
        if (hint >= latest) {
          latest = hint;
          refidx = j;
        }
      }
      if (refidx >= 0) off[refidx] = kUsed;
      refs[i] = refidx >= 0 ? refidx : earliest_ref;
    }
  }
  return refs;
}

void test_set_frame_refs_matches_reference() {
  std::printf("set frame refs against the reference decoder\n");
  SequenceSpec spec;
  spec.order_hint = true;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);

  std::mt19937 rng(20251025u);
  int ties = 0, future = 0;
  for (int trial = 0; trial < 400; ++trial) {
    int hints[8];
    for (int& hint : hints) hint = static_cast<int>(rng() % 8u);
    const int cur = static_cast<int>(rng() % 8u);
    const int last = static_cast<int>(rng() % 8u);
    const int gold = static_cast<int>(rng() % 8u);

    RefPool pool;
    for (int i = 0; i < 8; ++i) {
      pool.slots[i].valid = true;
      pool.slots[i].order_hint = hints[i];
      pool.slots[i].frame_id = -1;
    }
    FrameHeader h;
    h.order_hint = cur;
    h.last_frame_idx = last;
    h.gold_frame_idx = gold;
    set_frame_refs(seq, &h, pool);

    const std::vector<int> want = dav1d_set_frame_refs(seq, hints, cur, last, gold);
    for (int i = 0; i < 7; ++i) {
      if (h.ref_frame_idx[i] != want[i]) {
        std::printf("  trial %d: reference %d is slot %d, dav1d says %d "
                    "(cur %d, last %d, gold %d, hints %d %d %d %d %d %d %d %d)\n",
                    trial, i, h.ref_frame_idx[i], want[i], cur, last, gold, hints[0], hints[1],
                    hints[2], hints[3], hints[4], hints[5], hints[6], hints[7]);
      }
      CHECK_EQ(h.ref_frame_idx[i], want[i]);
    }
    for (int i = 0; i < 8; ++i)
      for (int j = i + 1; j < 8; ++j)
        if (get_relative_dist(seq, hints[i], cur) == get_relative_dist(seq, hints[j], cur)) {
          ++ties;
          i = 8;
          break;
        }
    if (want[6] >= 0) ++future;
  }
  /* the trials are only worth much if they reach both interesting shapes */
  CHECK(ties > 0);
  CHECK(future > 0);
}

/* Frame ids (5.9.2 and mark_ref_frames): the header names its own id and, for
 * every reference, how far back that reference's id has to be. Two things come
 * out of that - a reference whose id does not match is rejected instead of
 * being predicted from, and a slot whose id has drifted out of the window is
 * marked empty, which is how a decoder notices a lost frame. */
void test_frame_ids() {
  std::printf("frame ids\n");
  SequenceSpec spec;
  spec.frame_ids = true;  /* idLen 4, deltaFrameIdLength 3 */
  spec.order_hint = true;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(seq.frame_id_n_bits, 4);
  CHECK_EQ(seq.delta_frame_id_n_bits, 3);

  RefPool pool;
  FrameHeader header;
  KeyFrameSpec key;
  key.frame_id = 1;
  key.order_hint = 1;
  const std::vector<uint8_t> key_bytes = write_key_frame_header(spec, key);
  CHECK(parse_frame_header(seq, key_bytes.data(), key_bytes.size(), &header, &error,
                           &pool) == ParseStatus::Ok);
  CHECK_EQ(header.frame_id_present, true);
  CHECK_EQ(header.frame_id, 1);
  refresh_ref_pool(header, &pool);
  for (int i = 0; i < 8; ++i) {
    CHECK_EQ(pool.slots[i].valid, true);
    CHECK_EQ(pool.slots[i].frame_id, 1);
  }

  /* An inter frame at id 5 refreshing only slot 1. Its reference is the key
   * frame in slot 0, four ids back. */
  InterHeaderSpec in;
  in.frame_id = 5;
  in.order_hint = 5;
  in.refresh_flags = 0x02;
  for (int i = 0; i < 7; ++i) in.slots[i] = 0;
  for (int i = 0; i < 7; ++i) in.deltas[i] = 4;
  std::vector<uint8_t> bytes = write_inter_frame_header_ids(spec, in);
  const ParseStatus first = parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error,
                                               &pool);
  if (first != ParseStatus::Ok) std::printf("  inter frame: %s\n", error.c_str());
  CHECK(first == ParseStatus::Ok);
  CHECK_EQ(header.ref_frame_idx[0], 0);
  refresh_ref_pool(header, &pool);
  CHECK_EQ(pool.slots[1].frame_id, 5);
  CHECK_EQ(pool.slots[0].frame_id, 1);

  /* A wrong distance is caught: id 5 with delta 1 expects a frame at id 4, and
   * slot 0 holds the key frame (id 1). */
  {
    InterHeaderSpec bad = in;
    for (int i = 0; i < 7; ++i) bad.deltas[i] = 1;
    const std::vector<uint8_t> bad_bytes = write_inter_frame_header_ids(spec, bad);
    CHECK(parse_frame_header(seq, bad_bytes.data(), bad_bytes.size(), &header, &error,
                             &pool) == ParseStatus::Invalid);
    CHECK(error.find("frame id") != std::string::npos);
  }

  /* Now id 12 - seven ids later, still inside the delta window - which marks
   * every slot older than 12 - 8 = 4 empty. Slot 0 holds id 1 and is dropped;
   * slot 1 holds id 5 and survives. */
  {
    InterHeaderSpec late;
    late.frame_id = 12;
    late.order_hint = 0;
    late.refresh_flags = 0x04; /* refresh slot 2 only */
    for (int i = 0; i < 7; ++i) late.slots[i] = 0; /* the expired slot */
    for (int i = 0; i < 7; ++i) late.deltas[i] = 7;
    const std::vector<uint8_t> late_bytes = write_inter_frame_header_ids(spec, late);
    CHECK(parse_frame_header(seq, late_bytes.data(), late_bytes.size(), &header, &error,
                             &pool) == ParseStatus::Invalid);
    CHECK_EQ(pool.slots[0].valid, false);
    CHECK_EQ(pool.slots[1].valid, true); /* 12 - 5 = 7 <= 8, still in the window */

    /* The slot that does survive can be referenced: id 5 is seven ids back. */
    InterHeaderSpec good;
    good.frame_id = 12;
    good.order_hint = 0;
    good.refresh_flags = 0x04;
    for (int i = 0; i < 7; ++i) good.slots[i] = 1;
    for (int i = 0; i < 7; ++i) good.deltas[i] = 7;
    const std::vector<uint8_t> good_bytes = write_inter_frame_header_ids(spec, good);
    const ParseStatus status =
        parse_frame_header(seq, good_bytes.data(), good_bytes.size(), &header, &error, &pool);
    if (status != ParseStatus::Ok) std::printf("  late inter frame: %s\n", error.c_str());
    CHECK(status == ParseStatus::Ok);
    CHECK_EQ(header.ref_frame_idx[0], 1);
  }

  /* show_existing_frame names a slot and re-states the id it was stored with;
   * the two have to agree, otherwise the decoder is displaying a frame the
   * encoder did not mean. */
  {
    const std::vector<uint8_t> show = write_show_existing_frame(spec, 1, 5);
    CHECK(parse_frame_header(seq, show.data(), show.size(), &header, &error, &pool) ==
          ParseStatus::Ok);
    CHECK_EQ(header.show_existing_frame, true);
    CHECK_EQ(header.existing_frame_idx, 1);
    CHECK_EQ(header.type, FrameType::Inter); /* taken from the slot, not coded */
    CHECK_EQ(header.order_hint, 5);          /* the slot's own hint */

    const std::vector<uint8_t> wrong = write_show_existing_frame(spec, 1, 4);
    CHECK(parse_frame_header(seq, wrong.data(), wrong.size(), &header, &error, &pool) ==
          ParseStatus::Invalid);

    const std::vector<uint8_t> empty = write_show_existing_frame(spec, 3, 5);
    CHECK(parse_frame_header(seq, empty.data(), empty.size(), &header, &error, &pool) ==
          ParseStatus::Invalid);
  }

  /* An error resilient frame re-states the order hint of every slot. A slot
   * whose hint does not come back the same is empty as far as this frame is
   * concerned: the decoder's pool went stale while frames were being lost. */
  {
    RefPool er_pool;
    const int ids[3] = {7, 12, 10};
    const int hints[3] = {3, 0, 2};
    for (int i = 0; i < 3; ++i) {
      er_pool.slots[i].valid = true;
      er_pool.slots[i].frame_id = ids[i];
      er_pool.slots[i].order_hint = hints[i];
      er_pool.slots[i].width = 320;
      er_pool.slots[i].height = 240;
    }
    InterHeaderSpec er;
    er.error_resilient = true;
    er.frame_id = 13;
    er.order_hint = 0;
    er.refresh_flags = 0x08; /* refresh slot 3 only */
    for (int i = 0; i < 7; ++i) er.slots[i] = 1; /* the one frame it predicts from */
    for (int i = 0; i < 7; ++i) er.deltas[i] = 1; /* id 12, one id back */
    er.ref_hints[0] = 3; /* agrees with the pool */
    er.ref_hints[1] = 0; /* agrees */
    er.ref_hints[2] = 7; /* slot 2 holds hint 2 - stale, so it is dropped */
    const std::vector<uint8_t> er_bytes = write_inter_frame_header_ids(spec, er);
    const ParseStatus status =
        parse_frame_header(seq, er_bytes.data(), er_bytes.size(), &header, &error, &er_pool);
    if (status != ParseStatus::Ok) std::printf("  error resilient frame: %s\n", error.c_str());
    CHECK(status == ParseStatus::Ok);
    CHECK_EQ(er_pool.slots[0].valid, true);
    CHECK_EQ(er_pool.slots[1].valid, true);
    CHECK_EQ(er_pool.slots[2].valid, false); /* the restatement dropped it */
    refresh_ref_pool(header, &er_pool);
    CHECK_EQ(er_pool.slots[3].frame_id, 13);
    CHECK_EQ(er_pool.slots[3].order_hint, 0);
  }
}

/* frame_refs_short_signaling: the header codes two slot indices and the rest
 * come from set_frame_refs. Chained after a key frame, so the pool the derived
 * list points into is a real one. */
void test_short_ref_signaling() {
  std::printf("frame_refs_short_signaling\n");
  SequenceSpec spec;
  spec.order_hint = true;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);

  RefPool pool;
  FrameHeader header;
  KeyFrameSpec key;
  key.order_hint = 4;
  const std::vector<uint8_t> key_bytes = write_key_frame_header(spec, key);
  CHECK(parse_frame_header(seq, key_bytes.data(), key_bytes.size(), &header, &error,
                           &pool) == ParseStatus::Ok);
  refresh_ref_pool(header, &pool);
  /* give the other seven slots distinct, older hints */
  for (int i = 1; i < 8; ++i) {
    pool.slots[i].valid = true;
    pool.slots[i].order_hint = i - 1;
    pool.slots[i].frame_id = -1;
  }

  InterHeaderSpec in;
  in.order_hint = 4;
  in.short_signaling = true;
  in.last_frame_idx = 3; /* hint 2 */
  in.gold_frame_idx = 1; /* hint 0 */
  const std::vector<uint8_t> bytes = write_inter_frame_header_ids(spec, in);
  const ParseStatus status =
      parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error, &pool);
  if (status != ParseStatus::Ok) std::printf("  short signaling: %s\n", error.c_str());
  CHECK(status == ParseStatus::Ok);
  CHECK_EQ(header.last_frame_idx, 3);
  CHECK_EQ(header.gold_frame_idx, 1);
  /* and the seven references were derived, not read: all of them name slots */
  for (int i = 0; i < 7; ++i) CHECK(header.ref_frame_idx[i] >= 0 && header.ref_frame_idx[i] < 8);
  CHECK_EQ(header.ref_frame_idx[kRefLast - 1], 3);
  CHECK_EQ(header.ref_frame_idx[kRefGolden - 1], 1);
  /* the derived list is what the frame size and everything after it are read
   * against, so it has to be right and not merely present */
  CHECK_EQ(header.width, 640);
  CHECK_EQ(header.height, 360);
}

/* frame_size_with_refs (5.9.4): a frame that codes frame_size_override may take
 * its whole geometry from one of its references instead of coding a size. The
 * found_ref bits come one per reference and stop at the first set one, and the
 * size, the render size and the superres decision all follow from that frame -
 * which is how an encoder scales a frame relative to the frame it predicts
 * from, and why a decoder that read the bits as an explicit size would lose the
 * bitstream from here on. */
void test_frame_size_with_refs() {
  std::printf("frame size with refs\n");
  SequenceSpec spec; /* 640x360 maximum */
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);

  RefPool pool;
  pool.slots[0].valid = true;
  pool.slots[0].upscaled_width = 640;
  pool.slots[0].width = 640;
  pool.slots[0].height = 360;
  pool.slots[0].render_width = 640;
  pool.slots[0].render_height = 360;
  /* a frame that was coded at half size and is displayed upscaled */
  pool.slots[1].valid = true;
  pool.slots[1].upscaled_width = 320;
  pool.slots[1].width = 320;
  pool.slots[1].height = 240;
  pool.slots[1].render_width = 512;
  pool.slots[1].render_height = 384;
  pool.slots[1].type = FrameType::Inter;

  FrameHeader header;
  {
    InterHeaderSpec in;
    in.size_override = true;
    in.found_ref = 1;
    in.refresh_flags = 0x04;
    for (int i = 0; i < 7; ++i) in.slots[i] = 1;
    const std::vector<uint8_t> bytes = write_inter_frame_header_ids(spec, in);
    const ParseStatus status =
        parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error, &pool);
    if (status != ParseStatus::Ok) std::printf("  found_ref: %s\n", error.c_str());
    CHECK(status == ParseStatus::Ok);
    CHECK_EQ(header.size_from_reference, true);
    CHECK_EQ(header.width, 320);
    CHECK_EQ(header.height, 240);
    /* the render size came with it, and was not coded */
    CHECK_EQ(header.render_width, 512);
    CHECK_EQ(header.render_height, 384);
    /* the tile grid follows the inherited size: 320x240 with 64x64 superblocks
     * is one tile of five by four */
    CHECK_EQ(header.tiles.cols, 1);
    CHECK_EQ(header.tiles.rows, 1);
  }

  /* The bits stop at the first set one: a frame that names its fifth reference
   * and leaves the sixth alone has "found" it, not "found two". */
  {
    InterHeaderSpec in;
    in.size_override = true;
    in.found_ref = 4;
    in.refresh_flags = 0x04;
    for (int i = 0; i < 7; ++i) in.slots[i] = (i == 4) ? 0 : 1;
    const std::vector<uint8_t> bytes = write_inter_frame_header_ids(spec, in);
    const ParseStatus status =
        parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error, &pool);
    if (status != ParseStatus::Ok) std::printf("  found_ref 4: %s\n", error.c_str());
    CHECK(status == ParseStatus::Ok);
    CHECK_EQ(header.size_from_reference, true);
    CHECK_EQ(header.width, 640); /* slot 0, the full size frame */
    CHECK_EQ(header.height, 360);
    CHECK_EQ(header.render_height, 360);
  }

  /* No found_ref bit set at all: the frame falls through to frame_size() and
   * codes its own size. */
  {
    InterHeaderSpec in;
    in.size_override = true;
    in.found_ref = -1;
    in.width = 480;
    in.height = 270;
    in.refresh_flags = 0x04;
    for (int i = 0; i < 7; ++i) in.slots[i] = 1;
    const std::vector<uint8_t> bytes = write_inter_frame_header_ids(spec, in);
    const ParseStatus status =
        parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error, &pool);
    if (status != ParseStatus::Ok) std::printf("  explicit size: %s\n", error.c_str());
    CHECK(status == ParseStatus::Ok);
    CHECK_EQ(header.size_from_reference, false);
    CHECK_EQ(header.width, 480);
    CHECK_EQ(header.height, 270);
    CHECK_EQ(header.render_width, 480); /* render_size() defaults to the frame */
  }

  /* An error resilient frame codes no found_ref bits even with the override on:
   * it cannot rely on the pool it is re-stating, so it says its size outright. */
  {
    InterHeaderSpec in;
    in.error_resilient = true;
    in.size_override = true;
    in.width = 320;
    in.height = 240;
    in.refresh_flags = 0x04;
    for (int i = 0; i < 7; ++i) in.slots[i] = 1;
    const std::vector<uint8_t> bytes = write_inter_frame_header_ids(spec, in);
    const ParseStatus status =
        parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error, &pool);
    if (status != ParseStatus::Ok) std::printf("  resilient size: %s\n", error.c_str());
    CHECK(status == ParseStatus::Ok);
    CHECK_EQ(header.size_from_reference, false);
    CHECK_EQ(header.width, 320);
    CHECK_EQ(header.height, 240);
  }

  /* found_ref naming a slot the pool does not hold is an error, not a size of
   * zero that would propagate into every buffer this frame touches. */
  {
    InterHeaderSpec in;
    in.size_override = true;
    in.found_ref = 2;
    in.refresh_flags = 0x04;
    for (int i = 0; i < 7; ++i) in.slots[i] = 1;
    in.slots[2] = 5; /* an empty slot */
    const std::vector<uint8_t> bytes = write_inter_frame_header_ids(spec, in);
    CHECK(parse_frame_header(seq, bytes.data(), bytes.size(), &header, &error, &pool) ==
          ParseStatus::Invalid);
  }
}

/* Segmentation is part of the header, not a reason to stop: the flags, the per
 * segment features, and the three values derived from them - which segments are
 * lossless, the last active segment, and whether the segment id has to be known
 * before the skip flag - all come back so that the tile decoder can use them. */
void test_segmentation() {
  std::printf("segmentation\n");
  SequenceSpec spec;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);

  /* A key frame header up to segmentation_params(). base_q_idx is non zero, so
   * the frame is not lossless and the filter and transform mode fields come
   * after the segmentation ones, exactly as in write_key_frame_header(). */
  auto write_prefix = [](BitWriter* w, int base_q) {
    w->bit(false); /* show_existing_frame */
    w->bits(0, 2); /* KEY */
    w->bit(true);  /* show_frame */
    w->bit(false); /* disable_cdf_update */
    w->bit(false); /* allow_screen_content_tools */
    w->bit(false); /* frame_size_override */
    w->bit(false); /* render_and_frame_size_different */
    w->bit(true);  /* refresh_frame_context (inverted on the wire) */
    w->bit(true);  /* uniform_tile_spacing_flag */
    w->bit(false); /* stop: one column */
    w->bit(false); /* stop: one row */
    w->bits(static_cast<uint32_t>(base_q), 8);
    w->bit(false); /* delta_q_y_dc */
    w->bit(false); /* delta_q_u_dc */
    w->bit(false); /* delta_q_u_ac */
    w->bit(false); /* using_qmatrix */
  };
  auto write_suffix = [](BitWriter* w) {
    w->bit(false);  /* delta_q_present */
    w->bits(0, 6);  /* loop_filter_level[0] */
    w->bits(0, 6);  /* loop_filter_level[1] */
    w->bits(0, 3);  /* loop_filter_sharpness */
    w->bit(false);  /* loop_filter_delta_enabled */
    w->bits(0, 2);  /* cdef_damping_minus_3 */
    w->bits(0, 2);  /* cdef_bits */
    w->bits(0, 6);  /* cdef y strength */
    w->bits(0, 6);  /* cdef uv strength */
    w->bit(true);   /* tx_mode_select */
    w->bit(false);  /* reduced_tx_set */
    w->align();
  };
  /* feature_enabled[i][j] = (i, j) in `enabled`, with `value` coded as
   * su(1 + bits) for a signed feature and f(bits) for an unsigned one. */
  auto write_features = [](BitWriter* w, const bool (&enabled)[kMaxSegments][kSegFeatureCount],
                           const int (&value)[kMaxSegments][kSegFeatureCount]) {
    static const int bits[kSegFeatureCount] = {8, 6, 6, 6, 6, 3, 0, 0};
    static const bool is_signed[kSegFeatureCount] = {true, true, true, true, true,
                                                     false, false, false};
    for (int i = 0; i < kMaxSegments; ++i) {
      for (int j = 0; j < kSegFeatureCount; ++j) {
        w->bit(enabled[i][j]);
        if (!enabled[i][j]) continue;
        if (is_signed[j]) {
          w->signed_bits(value[i][j], bits[j] + 1);
        } else {
          w->bits(static_cast<uint32_t>(value[i][j]), bits[j]);
        }
      }
    }
  };

  /* Segmentation on and nothing else: a key frame has primary_ref_frame equal to
   * PRIMARY_REF_NONE, so update_map / temporal_update / update_data are not on
   * the wire at all - they are 1 / 0 / 1 by definition. */
  bool no_features[kMaxSegments][kSegFeatureCount] = {};
  int no_values[kMaxSegments][kSegFeatureCount] = {};
  BitWriter plain;
  write_prefix(&plain, 60);
  plain.bit(true); /* segmentation_enabled */
  write_features(&plain, no_features, no_values);
  write_suffix(&plain);
  const std::vector<uint8_t> plain_bytes = plain.bytes();
  FrameHeader header;
  CHECK(parse_frame_header(seq, plain_bytes.data(), plain_bytes.size(), &header, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(header.segmentation.enabled, true);
  CHECK_EQ(header.segmentation.update_map, true);
  CHECK_EQ(header.segmentation.temporal_update, false);
  CHECK_EQ(header.segmentation.update_data, true);
  CHECK_EQ(header.segmentation.last_active_seg_id, 0);
  CHECK_EQ(header.segmentation.seg_id_pre_skip, false);
  CHECK_EQ(header.all_lossless, false);
  for (int i = 0; i < kMaxSegments; ++i) CHECK_EQ(header.segmentation.lossless[i], false);

  /* Feature data. Every feature that libaom and SVT-AV1 do not emit is here
   * too, because no stream we can encode uses them: the alternate loop filter
   * offsets, the per segment reference frame, skip and global motion. */
  bool enabled[kMaxSegments][kSegFeatureCount] = {};
  int values[kMaxSegments][kSegFeatureCount] = {};
  const auto set = [&](int segment, int feature, int value) {
    enabled[segment][feature] = true;
    values[segment][feature] = value;
  };
  set(0, kSegAltQ, -256);     /* su(9) can code -256; the limit clips it to -255 */
  set(1, kSegAltQ, -60);      /* 60 - 60 = 0, so segment 1 is lossless */
  set(2, kSegAltQ, 18);
  set(3, kSegAltLfYV, -64);   /* su(7) can code -64; the limit clips it to -63 */
  set(4, kSegAltLfYH, 63);
  set(5, kSegRefFrame, 3);    /* f(3): segment 5 always predicts from GOLDEN */
  set(6, kSegSkip, 0);        /* no value: enabling the feature is the signal */
  set(7, kSegGlobalMv, 0);
  BitWriter data;
  write_prefix(&data, 60);
  data.bit(true); /* segmentation_enabled */
  write_features(&data, enabled, values);
  write_suffix(&data);
  const std::vector<uint8_t> data_bytes = data.bytes();
  CHECK(parse_frame_header(seq, data_bytes.data(), data_bytes.size(), &header, &error) ==
        ParseStatus::Ok);
  const SegmentationParams& seg = header.segmentation;
  CHECK_EQ(seg.update_data, true);
  CHECK_EQ(seg.feature_enabled[0][kSegAltQ], true);
  CHECK_EQ(seg.feature_data[0][kSegAltQ], -255);
  CHECK_EQ(seg.feature_data[1][kSegAltQ], -60);
  CHECK_EQ(seg.feature_data[2][kSegAltQ], 18);
  CHECK_EQ(seg.feature_data[3][kSegAltLfYV], -63);
  CHECK_EQ(seg.feature_data[4][kSegAltLfYH], 63);
  CHECK_EQ(seg.feature_enabled[5][kSegRefFrame], true);
  CHECK_EQ(seg.feature_data[5][kSegRefFrame], 3);
  CHECK_EQ(seg.feature_enabled[6][kSegSkip], true);
  CHECK_EQ(seg.feature_data[6][kSegSkip], 0);
  CHECK_EQ(seg.feature_enabled[7][kSegGlobalMv], true);
  CHECK_EQ(seg.feature_data[7][kSegGlobalMv], 0);
  CHECK_EQ(seg.feature_enabled[2][kSegAltLfYV], false);
  /* The last segment that carries a feature at all, and whether a feature that
   * is read before the skip flag is among them. */
  CHECK_EQ(seg.last_active_seg_id, 7);
  CHECK_EQ(seg.seg_id_pre_skip, true);
  /* The quantiser is per segment: 60 - 60 = 0, and segment 0's clipped -255
   * takes it below zero and is clipped at 0 again, so both are lossless; that is
   * a property of the segment, decided here and used by the tile decoder for the
   * transform size and for the transform type. No delta_q is in play, so the
   * frame as a whole is not lossless. */
  CHECK_EQ(seg.lossless[1], true);
  CHECK_EQ(seg.lossless[0], true);
  CHECK_EQ(seg.lossless[2], false);
  CHECK_EQ(seg.lossless[3], false);
  CHECK_EQ(header.all_lossless, false);
  CHECK_EQ(segment_qindex(header, 0), 0);
  CHECK_EQ(segment_qindex(header, 2), 78);

  /* segmentation_update_data == 0 on a frame that has a primary reference: the
   * features are inherited from that reference and only the map is on the wire. */
  const std::vector<uint8_t> key_bytes = write_key_frame_header(spec, KeyFrameSpec());
  RefPool pool;
  CHECK(parse_frame_header(seq, key_bytes.data(), key_bytes.size(), &header, &error,
                           &pool) == ParseStatus::Ok);
  refresh_ref_pool(header, &pool);

  SegmentationParams inherited;
  inherited.enabled = true;
  inherited.update_data = true;
  inherited.feature_enabled[3][kSegAltQ] = true;
  inherited.feature_data[3][kSegAltQ] = -12;
  inherited.last_active_seg_id = 3;
  pool.slots[0].segmentation = inherited;

  BitWriter keep;
  keep.bit(false); /* show_existing_frame */
  keep.bits(1, 2); /* INTER */
  keep.bit(true);  /* show_frame */
  keep.bit(false); /* error_resilient_mode */
  keep.bit(false); /* disable_cdf_update */
  keep.bit(false); /* allow_screen_content_tools */
  keep.bit(false); /* frame_size_override: inherit the reference size */
  keep.bits(0, 3); /* primary_ref_frame: ref_frame_idx[0] = slot 0 */
  keep.bits(0, 8); /* refresh_frame_flags */
  keep.bit(false); /* frame_refs_short_signaling */
  for (int i = 0; i < 7; ++i) keep.bits(0, 3); /* ref_frame_idx: slot 0 */
  keep.bit(false); /* render_and_frame_size_different */
  keep.bit(false); /* allow_high_precision_mv */
  keep.bit(true);  /* is_filter_switchable */
  keep.bit(false); /* is_motion_mode_switchable */
  keep.bit(false); /* disable_frame_end_update_cdf */
  keep.bit(true);  /* uniform_tile_spacing_flag */
  keep.bit(false); /* stop: one column */
  keep.bit(false); /* stop: one row */
  keep.bits(100, 8);
  keep.bit(false); /* delta_q_y_dc */
  keep.bit(false); /* delta_q_u_dc */
  keep.bit(false); /* delta_q_u_ac */
  keep.bit(false); /* using_qmatrix */
  keep.bit(true);  /* segmentation_enabled */
  keep.bit(false); /* update_map */
  keep.bit(false); /* update_data: keep the features of the primary reference */
  keep.bit(false); /* delta_q_present */
  keep.bits(0, 6); /* loop_filter_level[0] */
  keep.bits(0, 6); /* loop_filter_level[1] */
  keep.bits(0, 3); /* loop_filter_sharpness */
  keep.bit(false); /* loop_filter_delta_enabled */
  keep.bits(0, 2); /* cdef_damping_minus_3 */
  keep.bits(0, 2); /* cdef_bits */
  keep.bits(0, 6); /* cdef y strength */
  keep.bits(0, 6); /* cdef uv strength */
  keep.bit(false); /* tx_mode_select */
  keep.bit(false); /* reference_select */
  keep.bit(false); /* reduced_tx_set */
  for (int i = 1; i < 8; ++i) keep.bit(false); /* is_global */
  keep.align();
  const std::vector<uint8_t> keep_bytes = keep.bytes();
  CHECK(parse_frame_header(seq, keep_bytes.data(), keep_bytes.size(), &header, &error,
                           &pool) == ParseStatus::Ok);
  CHECK_EQ(header.segmentation.enabled, true);
  CHECK_EQ(header.segmentation.update_map, false);
  CHECK_EQ(header.segmentation.update_data, false);
  CHECK_EQ(header.segmentation.feature_enabled[3][kSegAltQ], true);
  CHECK_EQ(header.segmentation.feature_data[3][kSegAltQ], -12);
}

/* The segment's alternative loop filter level is added on top of the plane's
 * level by deblock_level_selection(), which is the only place in the loop filter
 * that knows about segmentation. */
void test_segment_deblock_level() {
  std::printf("segmentation (loop filter level)\n");
  FrameHeader frame;
  frame.loop_filter.level_y[0] = 8;
  frame.loop_filter.level_y[1] = 4;
  frame.loop_filter.level_u = 6;
  frame.loop_filter.level_v = 6;
  frame.loop_filter.mode_ref_delta_enabled = false;
  frame.segmentation.enabled = true;
  frame.segmentation.feature_enabled[2][kSegAltLfYV] = true;
  frame.segmentation.feature_data[2][kSegAltLfYV] = -3;
  frame.segmentation.feature_enabled[2][kSegAltLfYH] = true;
  frame.segmentation.feature_data[2][kSegAltLfYH] = 2;
  frame.segmentation.feature_enabled[2][kSegAltLfU] = true;
  frame.segmentation.feature_data[2][kSegAltLfU] = -6;

  DeblockGrid grid;
  grid.mi_cols = 2;
  grid.mi_rows = 1;
  grid.size.assign(2, kBlock4x4);
  grid.skip.assign(2, 0);
  grid.segment.assign(2, 0);
  grid.ref_frame.assign(2, 0);
  grid.mode.assign(2, 0);
  grid.delta_lf.assign(8, 0);
  grid.segment[0] = 0; /* no feature: the plane's level comes back unchanged */
  grid.segment[1] = 2;

  CHECK_EQ(deblock_level_selection(frame, grid, 0, 0, 0, 0), 8);
  CHECK_EQ(deblock_level_selection(frame, grid, 0, 0, 0, 1), 4);
  CHECK_EQ(deblock_level_selection(frame, grid, 0, 1, 0, 0), 5);  /* 8 - 3 */
  CHECK_EQ(deblock_level_selection(frame, grid, 0, 1, 0, 1), 6);  /* 4 + 2 */
  CHECK_EQ(deblock_level_selection(frame, grid, 0, 1, 1, 0), 0);  /* 6 - 6, and clipped at 0 */
  CHECK_EQ(deblock_level_selection(frame, grid, 0, 1, 2, 0), 6);  /* plane V has no feature */

  /* A feature that is disabled is ignored even when the frame segments. */
  frame.segmentation.feature_enabled[2][kSegAltLfYV] = false;
  CHECK_EQ(deblock_level_selection(frame, grid, 0, 1, 0, 0), 8);
}

void test_real_stream() {
  std::printf("real stream (SVT-AV1)\n");
  const std::string path = std::string(AV1GPU_TEST_DATA_DIR) + "/svt_av1_320x240.ivf";
  std::vector<uint8_t> data;
  {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    CHECK(file != nullptr);
    if (!file) return;
    uint8_t chunk[16384];
    size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
      data.insert(data.end(), chunk, chunk + got);
    std::fclose(file);
  }

  IvfReader reader;
  std::string error;
  CHECK(reader.open(data, &error));
  CHECK_EQ(reader.header().frame_count, 6u);
  CHECK_EQ(reader.header().width, 320);

  SequenceHeader seq;
  RefPool pool;
  bool have_seq = false;
  struct Seen {
    int frames = 0;
    int keys = 0;
    int inters = 0;
    int existing = 0;
    int ok = 0;
  } seen;
  int key_q = -1;
  int key_deblock = -1;
  size_t key_tile_offset = 0;
  int order_hints[8] = {0};

  IvfFrame frame;
  while (reader.next(&frame, &error)) {
    std::vector<Obu> obus;
    CHECK(split_obus(frame.payload.data(), frame.payload.size(), &obus, &error));
    for (const Obu& obu : obus) {
      if (obu.type == ObuType::SequenceHeader) {
        CHECK(parse_sequence_header(obu.data, obu.size, &seq, &error) == ParseStatus::Ok);
        have_seq = true;
      } else if (obu.type == ObuType::Frame || obu.type == ObuType::FrameHeader) {
        CHECK(have_seq);
        FrameHeader h;
        CHECK(parse_frame_header(seq, obu.data, obu.size, &h, &error, &pool) ==
              ParseStatus::Ok);
        seen.ok++;
        if (h.show_existing_frame) {
          seen.existing++;
          CHECK_EQ(h.width, 320);
          CHECK_EQ(h.height, 240);
          continue;
        }
        seen.frames++;
        if (h.type == FrameType::Key) {
          seen.keys++;
          key_q = h.base_q_idx;
          key_deblock = h.loop_filter.level_y[0];
          key_tile_offset = h.tile_data_offset;
          CHECK_EQ(h.width, 320);
          CHECK_EQ(h.height, 240);
        } else {
          seen.inters++;
          /* every frame of this stream is coded at the sequence's maximum size
           * with frame_size_override off, so no frame inherits a size and no
           * found_ref bit is coded */
          CHECK_EQ(h.size_from_reference, false);
          CHECK_EQ(h.width, 320);
          CHECK_EQ(h.height, 240);
          order_hints[seen.inters] = h.order_hint;
        }
        CHECK(h.tiles.cols == 1 && h.tiles.rows == 1);
        CHECK(h.tile_data_offset < obu.size);
        refresh_ref_pool(h, &pool);
      }
    }
  }

  CHECK(reader.at_end());
  CHECK_EQ(seen.frames, 6);
  CHECK_EQ(seen.keys, 1);
  CHECK_EQ(seen.inters, 5);
  CHECK_EQ(seen.existing, 2);
  CHECK_EQ(seen.ok, 8);
  CHECK_EQ(key_q, 51);
  CHECK_EQ(key_deblock, 2);
  CHECK_EQ(key_tile_offset, 11u);
  CHECK_EQ(order_hints[1], 4);
  CHECK_EQ(order_hints[2], 2);
  CHECK_EQ(order_hints[3], 1);
  CHECK_EQ(order_hints[4], 3);
  CHECK_EQ(order_hints[5], 5);
}


/* Skip mode, from the header down (7.10.1 skip_mode_params and the
 * SkipModeFrame pair). A frame that codes it names two references - one on each
 * side of the frame in display order - and the pair is what a skipped block
 * predicts from: both references, both vectors, no mode symbols and no
 * residual. The pair is sorted, so index 0 is always one of the forward
 * references (LAST..LAST3) and index 1 one of the backward ones
 * (BWDREF..ALTREF).
 *
 * The block layer is checked by the `stream_*` tests: tests/data/inter_intra.ivf
 * codes 1033 skip-mode blocks, lw_t2 2220, lw_t3 2042 and svt_av1_320x240 43,
 * and every frame of all four is bit exact against dav1d - a skip-mode block
 * that read a mode symbol, or took the wrong pair of references, would show up
 * immediately. */
void test_skip_mode_header() {
  std::printf("skip mode (frame header)\n");
  const std::string path = std::string(AV1GPU_TEST_DATA_DIR) + "/inter_intra.ivf";
  std::vector<uint8_t> data;
  std::FILE* file = std::fopen(path.c_str(), "rb");
  CHECK(file != nullptr);
  if (!file) return;
  uint8_t chunk[16384];
  size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
    data.insert(data.end(), chunk, chunk + got);
  std::fclose(file);

  IvfReader reader;
  std::string error;
  CHECK(reader.open(data, &error));

  SequenceHeader seq;
  RefPool pool;
  int present = 0, allowed = 0, frames = 0, both_past = 0;
  IvfFrame frame;
  while (reader.next(&frame, &error)) {
    std::vector<Obu> obus;
    CHECK(split_obus(frame.payload.data(), frame.payload.size(), &obus, &error));
    for (const Obu& obu : obus) {
      if (obu.type == ObuType::SequenceHeader) {
        CHECK(parse_sequence_header(obu.data, obu.size, &seq, &error) == ParseStatus::Ok);
      } else if (obu.type == ObuType::Frame || obu.type == ObuType::FrameHeader) {
        FrameHeader h;
        CHECK(parse_frame_header(seq, obu.data, obu.size, &h, &error, &pool) ==
              ParseStatus::Ok);
        if (!h.show_existing_frame) {
          ++frames;
          if (h.skip_mode_allowed) ++allowed;
          if (h.skip_mode_present) {
            ++present;
            /* Two different references, sorted by reference type. Usually they
             * straddle this frame - one forward, one backward - but a frame
             * whose references all lie in the past pairs the two nearest ones
             * instead (skip_mode_params), so both members being forward is
             * legal and this fixture does code such frames. */
            CHECK(h.skip_mode_frame[0] >= kRefLast);
            CHECK(h.skip_mode_frame[1] <= kRefAltref);
            CHECK(h.skip_mode_frame[0] < h.skip_mode_frame[1]);
            /* and both name frames the header actually references */
            CHECK(h.ref_frame_idx[h.skip_mode_frame[0] - 1] >= 0);
            CHECK(h.ref_frame_idx[h.skip_mode_frame[1] - 1] >= 0);
            if (h.skip_mode_frame[1] < kRefBwdref) ++both_past;
          }
        }
        refresh_ref_pool(h, &pool);
      }
    }
  }
  CHECK(frames > 0);
  /* skip mode is only allowed when the frame has a reference on each side, and
   * the fixture's encoder switches it on for most of its inter frames - a
   * regression that stopped allowing it would show up here */
  CHECK(allowed > frames / 2);
  CHECK(present > 0);
  /* the fixture codes both kinds of pair, which is what makes it worth
   * checking the fallback as well as the straddling case */
  CHECK(both_past > 0);
  CHECK(both_past < present);
}

/* The tile group is where the payload stops being plain bits. A single tile
 * that runs to the end of the payload is the common case; the group can also
 * carry a subset of the tiles, and the last tile of the group is never given a
 * size because it always takes the rest of the OBU. */
void test_tile_group() {
  std::printf("tile group\n");
  FrameHeader h;
  h.type = FrameType::Key;
  h.upscaled_width = 640;
  h.width = 640;
  h.height = 240;
  h.tiles.cols = 2;
  h.tiles.rows = 1;
  h.tiles.log2_cols = 1;
  h.tiles.log2_rows = 0;
  h.tiles.col_start_sb = {0, 5, 10};
  h.tiles.row_start_sb = {0, 4};
  h.tiles.tile_size_bytes = 2;
  h.tile_data_offset = 0;

  SequenceHeader seq; /* 64x64 superblocks */
  TileGroup group;
  std::string error;

  /* one group covering both tiles: flag 0, byte align, then tile 0's size */
  std::vector<uint8_t> payload = {0x00, 0x03, 0x00, 0xaa, 0xbb, 0xcc, 0xdd,
                                  0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  CHECK(parse_tile_group(seq, h, payload.data(), payload.size(), &group, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(group.start, 0);
  CHECK_EQ(group.end, 1);
  CHECK_EQ(group.tiles.size(), 2u);
  if (group.tiles.size() == 2) {
    CHECK_EQ(group.tiles[0].offset, 3u);
    CHECK_EQ(group.tiles[0].size, 4u);
    CHECK_EQ(group.tiles[0].x, 0);
    CHECK_EQ(group.tiles[0].w, 320);
    CHECK_EQ(group.tiles[0].h, 240);
    CHECK_EQ(group.tiles[1].offset, 7u);
    CHECK_EQ(group.tiles[1].size, 6u);
    CHECK_EQ(group.tiles[1].x, 320);
    CHECK_EQ(group.tiles[1].w, 320);
  }
  CHECK(tile_group_is_complete(h, group));

  /* a group covering only tile 1: 1 (present) 1 (start) 1 (end), then tile 1 */
  std::vector<uint8_t> subset = {0xe0, 0x11, 0x22, 0x33, 0x44};
  CHECK(parse_tile_group(seq, h, subset.data(), subset.size(), &group, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(group.start, 1);
  CHECK_EQ(group.end, 1);
  CHECK_EQ(group.tiles.size(), 1u);
  if (group.tiles.size() == 1) {
    CHECK_EQ(group.tiles[0].index, 1);
    CHECK_EQ(group.tiles[0].offset, 1u);
    CHECK_EQ(group.tiles[0].size, 4u);
  }
  /* a subset is a valid group but not the whole frame */
  CHECK(!tile_group_is_complete(h, group));

  /* a size that runs past the payload is reported, not silently clipped */
  std::vector<uint8_t> broken = {0x00, 0xff, 0x00, 0xaa};
  CHECK(parse_tile_group(seq, h, broken.data(), broken.size(), &group, &error) ==
        ParseStatus::Invalid);
  CHECK(!error.empty());

  /* an end tile outside the grid is invalid too */
  std::vector<uint8_t> bad_end = {0xc0, 0x00}; /* start 1, end 0 */
  CHECK(parse_tile_group(seq, h, bad_end.data(), bad_end.size(), &group, &error) ==
        ParseStatus::Invalid);
}


/* Block sizes, partitions and transform sizes are all measured in 4x4 units,
 * and every table in the syntax layer is indexed by those numbers. The helpers
 * are small, so they are worth pinning directly: a wrong block level or a
 * wrong quadrant size does not fail loudly, it decodes a plausible picture of
 * the wrong shape. */
void test_block_geometry() {
  std::printf("block geometry\n");
  /* the four legacy partitions of an 8x8 block, and the 4:1 sizes that only
   * larger blocks can use */
  CHECK(partition_subsize(kPartitionNone, kBlock8x8) == kBlock8x8);
  CHECK(partition_subsize(kPartitionHorizontal, kBlock8x8) == kBlock8x4);
  CHECK(partition_subsize(kPartitionVertical, kBlock8x8) == kBlock4x8);
  CHECK(partition_subsize(kPartitionSplit, kBlock8x8) == kBlock4x4);
  CHECK(partition_subsize(kPartitionHorizontal4, kBlock64x64) == kBlock64x16);
  CHECK(partition_subsize(kPartitionVertical4, kBlock32x32) == kBlock8x32);
  /* the T shapes: Partition_Subsize gives the largest of the two sub-blocks,
   * which is the long one - the parent's width for HORZ_A/B, its height for
   * VERT_A/B - and only the square halves come from the split size */
  CHECK(partition_subsize(kPartitionHorizontalA, kBlock16x16) == kBlock16x8);
  CHECK(partition_subsize(kPartitionHorizontalB, kBlock16x16) == kBlock16x8);
  CHECK(partition_subsize(kPartitionVerticalA, kBlock16x16) == kBlock8x16);
  CHECK(partition_subsize(kPartitionVerticalB, kBlock16x16) == kBlock8x16);
  CHECK(partition_subsize(kPartitionHorizontalA, kBlock32x32) == kBlock32x16);
  CHECK(partition_subsize(kPartitionVerticalA, kBlock32x32) == kBlock16x32);
  CHECK(partition_subsize(kPartitionHorizontal4, kBlock16x16) == kBlock16x4);

  /* block levels are counted from the largest block down, because that is the
   * order of the partition tables */
  CHECK_EQ(block_level_of(kBlock128x128), 0);
  CHECK_EQ(block_level_of(kBlock8x8), 4);
  CHECK(square_block_size(4) == kBlock8x8);
  CHECK(block_size_from_dims(8, 8) == kBlock32x32);
  CHECK(block_size_from_dims(16, 4) == kBlock64x16);
  CHECK(block_size_from_dims(3, 3) == kBlockSizeCount);

  /* transforms: the block's transform is capped at 64x64, and a split halves
   * the longer side, so 64x64 goes to 32x64 and then 32x32 */
  CHECK(max_tx_size(8, 8) == (TxSize{3, 3}));
  CHECK(max_tx_size(32, 32) == (TxSize{4, 4}));
  CHECK(max_tx_size(4, 16) == (TxSize{2, 4}));
  CHECK(split_tx_size(TxSize{4, 4}) == (TxSize{3, 3}));
  CHECK(split_tx_size(TxSize{3, 4}) == (TxSize{3, 3}));
  CHECK(split_tx_size(TxSize{1, 2}) == (TxSize{1, 1}));
  CHECK(split_tx_size(TxSize{0, 2}) == (TxSize{0, 1}));
  CHECK_EQ(max_tx_depth(TxSize{4, 4}), 4);
  CHECK_EQ(tx_square_class(TxSize{2, 3}), 3);
  CHECK_EQ(tx_min_class(TxSize{2, 3}), 2);
  /* 4:2:0 chroma halves both sides, and never exceeds 32x32 */
  /* a chroma plane's transform comes from the block size, not the luma one */
  CHECK(plane_tx_size(kBlock64x64, 1, TxSize{4, 4}, true, true) == (TxSize{3, 3}));
  CHECK(plane_tx_size(kBlock64x64, 1, TxSize{4, 1}, true, true) == (TxSize{3, 3}));
  CHECK(plane_tx_size(kBlock4x4, 1, TxSize{0, 0}, true, true) == (TxSize{0, 0}));
  CHECK(plane_tx_size(kBlock16x16, 1, TxSize{0, 0}, true, true) == (TxSize{1, 1}));
  /* 4:2:0 turns 4x16 into a 4x8 chroma residual, 16x4 into 8x4 */
  CHECK(plane_tx_size(kBlock4x16, 1, TxSize{0, 0}, true, true) == (TxSize{0, 1}));
  CHECK(plane_tx_size(kBlock16x4, 1, TxSize{0, 0}, true, false) == (TxSize{1, 0}));
}

/* Temporal (projected) motion vectors: the arithmetic that moves a stored
 * vector from the frame that saved it onto a later frame, and the reference
 * store that carries it there.
 *
 * The projection is a rational scaling by the ratio of two order hint
 * distances, and its rounding decides which cells a vector lands on - which
 * decides which candidates a block sees, which decides a dynamic reference list
 * index, which is a bit in the stream. It is pinned here because a stream can
 * only show that *a* choice disagrees, not which step did it. */
void test_projected_motion_vectors() {
  std::printf("projected motion vectors\n");
  /* a ratio of one is the identity */
  CHECK_EQ(project_mv_component(1234, 1, 1), 1234);
  CHECK_EQ(project_mv_component(-1234, 1, 1), -1234);
  CHECK_EQ(project_mv_component(0, 7, 3), 0);
  /* projecting outwards multiplies exactly */
  CHECK_EQ(project_mv_component(1000, 3, 1), 3000);
  CHECK_EQ(project_mv_component(-1000, 3, 1), -3000);
  /* inwards it rounds away from zero in both directions, and a fraction of a
   * step lands on zero rather than on plus or minus one */
  CHECK_EQ(project_mv_component(3, 1, 3), 1);
  CHECK_EQ(project_mv_component(-3, 1, 3), -1);
  CHECK_EQ(project_mv_component(1, 1, 3), 0);
  CHECK_EQ(project_mv_component(-1, 1, 3), 0);
  /* and the result is clipped to the 14 bit range a projected vector spans;
   * 4095 is the largest vector a frame can store (see save_temporal_mvs) */
  CHECK_EQ(project_mv_component(4095, 31, 1), 0x3fff);
  CHECK_EQ(project_mv_component(-4095, 31, 1), -0x3fff);
  CHECK_EQ(project_mv_component(4095, 1, 1), 4095);
  /* a denominator the projection cannot produce contributes nothing instead of
   * dividing by zero */
  CHECK_EQ(project_mv_component(1000, 1, 0), 0);
  CHECK_EQ(project_mv_component(1000, 1, 32), 0);

  /* The reference store: an inter frame hands its motion vector grid and the
   * order hints *it* referred to to every slot it refreshes; an intra frame
   * clears both, because a frame that stores no vectors must not leave the
   * previous occupant's in place for the next frame to project. */
  RefPool pool;
  FrameHeader inter;
  inter.type = FrameType::Inter;
  inter.width = 320;
  inter.height = 240;
  inter.order_hint = 5;
  inter.refresh_frame_flags = 0x02 | 0x20;
  for (int i = 0; i < 7; ++i) inter.ref_order_hint[i] = i + 1;
  auto store = std::make_shared<RefMvs>();
  store->resize(40, 30);
  store->at(3, 4).ref = 1;
  store->at(3, 4).x = 64;
  store->at(3, 4).y = -32;
  refresh_ref_pool(inter, &pool, store);
  CHECK(pool.slots[1].ref_mvs == store);
  CHECK(pool.slots[5].ref_mvs == store);
  CHECK(pool.slots[0].ref_mvs == nullptr);
  CHECK_EQ(pool.slots[1].order_hint, 5);
  CHECK_EQ(pool.slots[1].ref_poc[6], 7);
  CHECK_EQ(pool.slots[1].ref_mvs->at(3, 4).x, 64);

  FrameHeader key;
  key.type = FrameType::Key;
  key.width = 320;
  key.height = 240;
  key.refresh_frame_flags = 0xff;
  refresh_ref_pool(key, &pool, store);
  CHECK(pool.slots[1].valid);
  CHECK(pool.slots[1].ref_mvs == nullptr);

  /* The probability context rides along with the slot: a frame that refreshes a
   * slot also stores the tables it ended with, and a later frame whose
   * primary_ref_frame names that slot starts from them. */
  auto cdf = std::make_shared<CdfContext>();
  init_default_cdf(cdf.get(), 100);
  FrameHeader coder = inter;
  refresh_ref_pool(coder, &pool, nullptr, cdf);
  CHECK(pool.slots[1].cdf == cdf);
  CHECK(pool.slots[5].cdf == cdf);
  CHECK(pool.slots[0].cdf == nullptr);
}

/* Local warped motion: the least squares fit of 7.11.3.8.
 *
 * The samples are pinned here rather than in a stream because a stream can only
 * show that a model disagrees, not which of the fit's steps did it: the
 * positions are in 1/8 luma samples, scaled by 2^-2 inside the normal
 * equations, and the divisor is resolved at 64 bit. What the fit must at least
 * do is reproduce a translation exactly when every neighbour agrees on one, and
 * refuse a configuration with no solution at all.
 */
void test_local_warp() {
  std::printf("local warp\n");

  /* the divisor, as 7.11.3.7 defines it: a power of two is the table's first
   * entry, and the sign rides on the factor */
  int shift = 0;
  int64_t factor = 0;
  resolve_divisor_64(1 << 20, &shift, &factor);
  CHECK_EQ(shift, 20 + kDivLutPrecBits);
  CHECK_EQ(factor, static_cast<int64_t>(kDivLut[0]));
  resolve_divisor_64(-(1 << 20), &shift, &factor);
  CHECK_EQ(factor, -static_cast<int64_t>(kDivLut[0]));
  /* Round2Signed rounds a negative input towards zero, and a shift of zero has
   * no rounding term */
  CHECK_EQ(round2s_64(static_cast<int64_t>(1) << 34, 34), 1);
  CHECK_EQ(round2s_64(-(static_cast<int64_t>(1) << 34), 34), -1);
  CHECK_EQ(round2s_64(static_cast<int64_t>(1) << 33, 34), 1);
  CHECK_EQ(round2s_64(-(static_cast<int64_t>(1) << 33), 34), -1);
  CHECK_EQ(round2s_64(-7, 0), -7);

  /* A 16x16 block at mi (8, 4): its centre is (39, 23) luma samples. Three
   * neighbours above and to the left carry the same vector, (16, -8) in 1/8
   * samples, so the model that fits them is that translation and no shear. */
  const int w4 = 4, h4 = 4;
  const int row = 4, col = 8;
  const int mid_x = col * 4 + w4 * 2 - 1;
  const int mid_y = row * 4 + h4 * 2 - 1;
  const int mv_x = 16, mv_y = -8;
  int cand[kWarpSamplesMax][4] = {};
  const int nx[3] = {mid_x - 40, mid_x - 8, mid_x + 24};
  const int ny[3] = {mid_y - 16, mid_y - 40, mid_y - 16};
  for (int i = 0; i < 3; ++i) {
    cand[i][0] = ny[i] * 8;
    cand[i][1] = nx[i] * 8;
    cand[i][2] = ny[i] * 8 + mv_y;
    cand[i][3] = nx[i] * 8 + mv_x;
  }
  int params[6] = {0, 0, 0, 0, 0, 0};
  CHECK(warp_estimation(cand, 3, mid_x, mid_y, mv_x, mv_y, params));
  /* The linear part is the identity: the divisor keeps eight bits of mantissa,
   * so the fit lands within a fraction of a percent of it rather than on it. */
  CHECK(std::abs(params[2] - kGmIdentityParam) <= 128);
  CHECK(std::abs(params[5] - kGmIdentityParam) <= 128);
  CHECK(std::abs(params[3]) <= 128);
  CHECK(std::abs(params[4]) <= 128);
  /* What must hold exactly is that the block's own centre lands on its own
   * vector: the translation is derived from the linear part rather than
   * fitted, so a slightly wrong scale is compensated instead of showing up as a
   * shift. The model takes luma positions and produces them at
   * WARPEDMODEL_PREC_BITS of a luma sample, so the centre in 1/8 samples is
   * (mid_x * 8 + mv_x) * 2^(PREC_BITS - 3) before the shift. */
  CHECK_EQ(params[2] * mid_x + params[3] * mid_y + params[0],
           (mid_x * 8 + mv_x) * (1 << (kWarpmodelPrecBits - 3)));
  CHECK_EQ(params[4] * mid_x + params[5] * mid_y + params[1],
           (mid_y * 8 + mv_y) * (1 << (kWarpmodelPrecBits - 3)));

  /* No sample at all leaves the normal equations empty, and their determinant
   * zero: the specification's LocalValid = 0, the block is predicted by its
   * vector. This is the only way the fit itself can fail - the rounding of the
   * diagonal terms keeps the determinant non-zero for any single sample. */
  CHECK(!warp_estimation(cand, 0, mid_x, mid_y, mv_x, mv_y, params));
}

/* The two halves of handing a probability context to a later frame: the tables
 * travel with the reference slot, and what a later frame starts from is those
 * tables with every adaptation counter back at zero - the probabilities the
 * encoder and decoder agreed on, adapted at the rate of a fresh context.
 *
 * Keeping the counters would look harmless (the probabilities still match) and
 * only show up frames later, as a symbol decoded against a probability that has
 * drifted: the counters decide how far each observation moves a probability, so
 * a decoder that keeps them adapts more slowly than the encoder did. */
void test_cdf_inheritance() {
  std::printf("probability context inheritance\n");
  CdfContext cdf;
  init_default_cdf(&cdf, 100);
  /* the counter of a row sits just past its probabilities: skip has one
   * probability, the chroma mode table thirteen when chroma from luma is
   * allowed and twelve when it is not */
  CHECK_EQ(cdf.skip.table[1][1], 0);
  CHECK_EQ(cdf.inter.skip_mode[2][1], 0);
  CHECK_EQ(cdf.mode.uv_mode[0][5][12], 0);
  CHECK_EQ(cdf.mode.uv_mode[1][5][13], 0);

  /* pretend a frame adapted these two tables a few thousand symbols' worth */
  cdf.skip.table[1][0] -= 1000;
  cdf.skip.table[1][1] = 32;
  cdf.inter.skip_mode[2][0] += 500;
  cdf.inter.skip_mode[2][1] = 17;
  const uint16_t skip_probability = cdf.skip.table[1][0];
  const uint16_t mode_probability = cdf.inter.skip_mode[2][0];

  reset_cdf_counts(&cdf);
  /* the probabilities survive, the counters do not */
  CHECK_EQ(cdf.skip.table[1][0], skip_probability);
  CHECK_EQ(cdf.inter.skip_mode[2][0], mode_probability);
  CHECK_EQ(cdf.skip.table[1][1], 0);
  CHECK_EQ(cdf.inter.skip_mode[2][1], 0);

  /* every table, not just the ones above: an unadapted table with a counter
   * that survived would silently keep adapting slowly */
  cdf.coef.base_tok[0][0][0][3] = 9;
  cdf.coef.eob_bin_64[1][0][6] = 9;
  cdf.mode.uv_mode[0][3][12] = 9;
  cdf.mode.uv_mode[1][3][13] = 9;
  cdf.mode.txsz[0][0][1] = 9;
  cdf.mode.txsz[3][0][2] = 9;
  cdf.partition.table[4][0][3] = 9;
  cdf.inter.mv_class0_fr[0][1][1][3] = 9;
  cdf.inter.mv_joint[0][3] = 9;
  reset_cdf_counts(&cdf);
  CHECK_EQ(cdf.coef.base_tok[0][0][0][3], 0);
  CHECK_EQ(cdf.coef.eob_bin_64[1][0][6], 0);
  CHECK_EQ(cdf.mode.uv_mode[0][3][12], 0);
  CHECK_EQ(cdf.mode.uv_mode[1][3][13], 0);
  CHECK_EQ(cdf.mode.txsz[0][0][1], 0);
  CHECK_EQ(cdf.mode.txsz[3][0][2], 0);
  CHECK_EQ(cdf.partition.table[4][0][3], 0);
  CHECK_EQ(cdf.inter.mv_class0_fr[0][1][1][3], 0);
  CHECK_EQ(cdf.inter.mv_joint[0][3], 0);
}

/* The first block of a real key frame, symbol for symbol. dav1d's trace of this
 * frame (see tests/data/tile_trace.hpp) starts with:
 *
 *   partition  64x64 -> split        (so a 32x32 block comes next)
 *   partition  32x32 -> none
 *   skip       0
 *   delta_q    2                     (51 + 2 = 53, the frame's base_q_idx is 51)
 *   y_mode     SMOOTH_VERTICAL (10)
 *   uv_mode    SMOOTH_VERTICAL (10)
 *   tx_depth   0                     (so the transform stays at 32x32)
 *
 * and then the block's coefficients. This checks that the block tree and mode
 * info layer produces exactly that, and that it stops where the coefficient
 * syntax starts instead of guessing at it. */
/* Reads one whole IVF frame record, for the tests that inspect a stream. */
bool read_ivf_frame(const std::string& path, size_t index, std::vector<uint8_t>* payload) {
  std::vector<uint8_t> data;
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) return false;
  uint8_t chunk[16384];
  size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
    data.insert(data.end(), chunk, chunk + got);
  std::fclose(file);
  IvfReader reader;
  std::string error;
  if (!reader.open(data, &error)) return false;
  IvfFrame frame;
  for (size_t i = 0; i <= index; ++i) {
    if (!reader.next(&frame, &error)) return false;
  }
  *payload = frame.payload;
  return true;
}

/* Intra block copy in a real stream (tests/data/intrabc.ivf). The tool is an
 * inter block inside an intra frame, so both halves of that sentence are worth
 * checking without pixels: the frame header has to say allow_intrabc, a block
 * has to set use_intrabc and end up `is_inter` with everything the syntax fixes
 * (DC_PRED, SIMPLE, bilinear, no palette), and the vector it ends up with has to
 * obey 7.11.5 - a whole number of samples, pointing at a rectangle that is
 * inside the tile and, since the picture is still being decoded, clear of the
 * superblock the block itself is in. */
void test_intrabc_frame() {
  std::printf("intra block copy (real key frame)\n");
  const std::string path = std::string(AV1GPU_TEST_DATA_DIR) + "/intrabc.ivf";
  std::vector<uint8_t> payload;
  CHECK(read_ivf_frame(path, 0, &payload));
  if (payload.empty()) return;

  SequenceHeader seq;
  FrameHeader header;
  TileGroup group;
  const uint8_t* tile = nullptr;
  size_t tile_size = 0;
  std::string error;
  std::vector<Obu> obus;
  CHECK(split_obus(payload.data(), payload.size(), &obus, &error));
  for (const Obu& obu : obus) {
    if (obu.type == ObuType::SequenceHeader) {
      CHECK(parse_sequence_header(obu.data, obu.size, &seq, &error) == ParseStatus::Ok);
    } else if (obu.type == ObuType::Frame || obu.type == ObuType::FrameHeader) {
      CHECK(parse_frame_header(seq, obu.data, obu.size, &header, &error) == ParseStatus::Ok);
      CHECK(parse_tile_group(seq, header, obu.data, obu.size, &group, &error) ==
            ParseStatus::Ok);
      if (group.tiles.size() == 1) {
        tile = obu.data + group.tiles[0].offset;
        tile_size = group.tiles[0].size;
      }
    }
  }
  CHECK(tile != nullptr);
  if (!tile) return;

  /* the frame is all intra, and it allows the tool - otherwise no block could
   * code one and the rest of this test would be vacuous */
  CHECK(header.type == FrameType::Key);
  CHECK(header.allow_screen_content_tools);
  CHECK(header.allow_intrabc);
  /* 7.11.5 needs whole sample vectors, and allow_intrabc implies it */
  CHECK(header.force_integer_mv);

  TileDecodeResult result;
  CHECK(decode_intra_tile(seq, header, tile, tile_size, &result, &error) == ParseStatus::Ok);
  CHECK(result.complete);

  int copies = 0;
  for (const TileBlock& block : result.blocks) {
    if (!block.intrabc) {
      /* every other block of this frame is an ordinary intra block */
      CHECK(!block.is_inter);
      continue;
    }
    ++copies;
    CHECK(block.is_inter);
    CHECK(block.y_mode == kModeDc);
    CHECK(block.uv_mode == kModeDc);
    CHECK(block.motion_mode == kMotionTranslation);
    CHECK(block.compound_type == kCompoundNone);
    CHECK_EQ(block.palette_size[0], 0);
    CHECK_EQ(block.palette_size[1], 0);
    CHECK_EQ(block.interp_filter[0], 3); /* BILINEAR */
    CHECK_EQ(block.interp_filter[1], 3);
    CHECK_EQ(block.ref_frame[0], kRefIntra);

    const int bw4 = block_width4(block.size), bh4 = block_height4(block.size);
    /* a whole sample vector: the lowest three bits are the fractional part */
    CHECK_EQ(block.mv[0][0] & 7, 0);
    CHECK_EQ(block.mv[0][1] & 7, 0);
    CHECK(block.mv[0][0] != 0 || block.mv[0][1] != 0);
    const int src_col = (block.mi_col * 4 + (block.mv[0][0] >> 3)) / 4;
    const int src_row = (block.mi_row * 4 + (block.mv[0][1] >> 3)) / 4;
    /* inside the tile ... */
    CHECK(src_col >= 0);
    CHECK(src_row >= 0);
    CHECK(src_col + bw4 <= (header.width + 3) >> 2);
    CHECK(src_row + bh4 <= (header.height + 3) >> 2);
    /* ... and clear of the superblock the block is in, one way or the other */
    const int sb4 = seq.use_128x128_superblock ? 32 : 16;
    const int sb_col = (block.mi_col / sb4) * sb4;
    const int sb_row = (block.mi_row / sb4) * sb4;
    const bool above = src_row + bh4 <= sb_row;
    const bool left = src_col + bw4 <= sb_col;
    if (!above && !left) {
      std::printf("  block %d,%d copies from %d,%d (superblock %d,%d)\n", block.mi_col,
                  block.mi_row, src_col, src_row, sb_col, sb_row);
      CHECK(false);
    }
  }
  /* the fixture codes 239 of them over 16 frames; frame 0 alone has a handful,
   * so anything close to zero means the flag was not read at all */
  CHECK(copies > 3);
}

void test_tile_decoder() {
  std::printf("tile decoder (real key frame)\n");
  const std::string path = std::string(AV1GPU_TEST_DATA_DIR) + "/svt_av1_320x240.ivf";
  std::vector<uint8_t> data;
  {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    CHECK(file != nullptr);
    if (!file) return;
    uint8_t chunk[16384];
    size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
      data.insert(data.end(), chunk, chunk + got);
    std::fclose(file);
  }

  IvfReader reader;
  std::string error;
  CHECK(reader.open(data, &error));
  IvfFrame frame;
  CHECK(reader.next(&frame, &error));

  SequenceHeader seq;
  FrameHeader header;
  TileGroup group;
  const uint8_t* tile = nullptr;
  size_t tile_size = 0;
  bool found = false;
  std::vector<Obu> obus;
  CHECK(split_obus(frame.payload.data(), frame.payload.size(), &obus, &error));
  for (const Obu& obu : obus) {
    if (obu.type == ObuType::SequenceHeader) {
      CHECK(parse_sequence_header(obu.data, obu.size, &seq, &error) == ParseStatus::Ok);
    } else if (obu.type == ObuType::Frame || obu.type == ObuType::FrameHeader) {
      CHECK(parse_frame_header(seq, obu.data, obu.size, &header, &error) == ParseStatus::Ok);
      CHECK(parse_tile_group(seq, header, obu.data, obu.size, &group, &error) ==
            ParseStatus::Ok);
      if (group.tiles.size() == 1) {
        tile = obu.data + group.tiles[0].offset;
        tile_size = group.tiles[0].size;
        found = true;
      }
    }
  }
  CHECK(found);
  if (!found) return;
  CHECK_EQ(header.base_q_idx, 51);
  CHECK(header.tx_mode_select);
  CHECK(!header.allow_screen_content_tools);

  TileDecodeResult result;
  const ParseStatus status = decode_intra_tile(seq, header, tile, tile_size, &result, &error);
  CHECK(status == ParseStatus::Ok);
  CHECK(result.complete);
  CHECK(result.stop_reason.empty());
  CHECK(result.blocks.size() > 1u);
  if (!result.blocks.empty()) {
    const TileBlock& block = result.blocks[0];
    CHECK_EQ(block.mi_row, 0);
    CHECK_EQ(block.mi_col, 0);
    CHECK(block.size == kBlock32x32);
    CHECK(block.partition == kPartitionNone);
    CHECK_EQ(block.block_level, 2);
    CHECK(!block.skip);
    CHECK_EQ(block.y_mode, kModeSmoothVertical);
    CHECK_EQ(block.uv_mode, kModeSmoothVertical);
    CHECK_EQ(block.y_angle, 0);
    CHECK_EQ(block.qindex, 53);
    CHECK(block.tx_size == (TxSize{3, 3}));
    CHECK(block.has_chroma);

    /* One transform block per plane, and the coefficients dav1d read for
     * them: a 32x32 luma block with 1014 coded coefficients and two 16x16
     * chroma blocks with 3 each. */
    CHECK_EQ(block.transforms.size(), 3u);
    if (block.transforms.size() == 3) {
      const TileTransform& y = block.transforms[0];
      CHECK_EQ(y.plane, 0);
      CHECK(y.size == (TxSize{3, 3}));
      CHECK_EQ(y.eob, 1014);
      CHECK_EQ(y.tx_type, coef_tables::kDctDct);
      CHECK_EQ(y.level_offset, 0u);
      for (int plane = 1; plane < 3; ++plane) {
        const TileTransform& uv = block.transforms[plane];
        CHECK_EQ(uv.plane, plane);
        CHECK(uv.size == (TxSize{2, 2}));
        CHECK_EQ(uv.eob, 3);
        CHECK_EQ(uv.tx_type, coef_tables::kAdstDct);
      }
    }
  }
  /* the split that produced that block was read before it, so the parser had
   * to place the 32x32 block on the grid, not just read symbols in a row */
  CHECK(result.bytes_used > 0);
  CHECK(result.bytes_used <= tile_size);
  /* The whole frame, as dav1d decodes it: 306 blocks and 64301 coded
   * coefficients in 1063 transform blocks. The tile's bytes are exactly
   * consumed, which is the end to end check that every symbol was read with the
   * right table.
   *
   * The frame has 1075 transform blocks in total: the three skipped 8x8 blocks
   * still have the specification's residual structure (residual() runs for
   * every block; a skipped one just carries no coefficients), so they add 12
   * transform blocks with EOB 0 - one of them splits its luma into four 4x4
   * transforms. Predicting such a block as one rectangle instead would leave
   * the parts it does not cover untouched, and every neighbour that predicts
   * from those samples would inherit the error. */
  CHECK_EQ(result.blocks.size(), 306u);
  CHECK_EQ(result.transform_blocks, 1075);
  CHECK_EQ(result.coefficients, 64301u);
  CHECK_EQ(result.levels.size(), 64301u);
  CHECK_EQ(result.bytes_used, tile_size);

  /* Dequantisation: the levels expanded back into raster order at the frame's
   * step sizes. dav1d's own dequantised coefficients for this frame sum to
   * 3329521 with 26102 non-zero values, which pins the quantiser tables, the
   * per plane step sizes, the transform size shift and the scan inversion all
   * at once. */
  DequantisedTile dequantised;
  dequantise_tile(header, seq.bit_depth, result, &dequantised);
  CHECK_EQ(dequantised.bit_depth, 8);
  CHECK_EQ(dequantised.offsets.size(), static_cast<size_t>(result.transform_blocks));
  long long magnitude_sum = 0;
  int non_zero = 0;
  for (int32_t value : dequantised.coefficients) {
    if (value == 0) continue;
    magnitude_sum += value < 0 ? -value : value;
    ++non_zero;
  }
  CHECK_EQ(non_zero, 26102);
  CHECK_EQ(magnitude_sum, 3329521LL);

  /* ---- the plan: every sample has exactly one owner ----------------------- */
  using av1gpu::BlockSyntax;
  using av1gpu::FrameSyntax;
  FrameSyntax syntax;
  CHECK(plan_from_tile(seq, header, result, dequantised, &syntax, &error));
  CHECK_EQ(syntax.blocks.size(), static_cast<size_t>(result.transform_blocks));
  CHECK_EQ(syntax.ss_x, 1);
  CHECK_EQ(syntax.ss_y, 1);

  /* One prediction block per transform block, and all of them together tile the
   * frame. A sample no block covers is never written - it stays at zero and
   * every neighbour that predicts from it inherits the error - and a sample two
   * blocks claim is reconstructed twice in an order the syntax does not define.
   * Both are silent pixel errors, so they are worth an explicit check on a real
   * frame: the skipped 8x8 block at (32,136) is coded with 4x4 transforms and
   * was exactly the hole in this frame. */
  {
    const int w = syntax.width, h = syntax.height;
    std::vector<int> owner(static_cast<size_t>(w) * h, -1);
    int overlaps = 0;
    for (size_t i = 0; i < syntax.blocks.size(); ++i) {
      const BlockSyntax& b = syntax.blocks[i];
      if (b.plane != 0) continue;
      for (int y = b.y; y < b.y + b.h && y < h; ++y) {
        for (int x = b.x; x < b.x + b.w && x < w; ++x) {
          int& o = owner[static_cast<size_t>(y) * w + x];
          if (o >= 0) ++overlaps;
          o = static_cast<int>(i);
        }
      }
    }
    int uncovered = 0;
    for (int o : owner) uncovered += o < 0 ? 1 : 0;
    CHECK_EQ(uncovered, 0);
    CHECK_EQ(overlaps, 0);
    /* the frame really does exercise chroma from luma */
    int cfl_blocks = 0;
    for (const BlockSyntax& b : syntax.blocks)
      if (b.mode == av1gpu::PredictionMode::ChromaFromLuma) ++cfl_blocks;
    CHECK(cfl_blocks > 0);

    /* ---- the deblock plan -------------------------------------------------- */
    /* The edge plan comes out of the front end rather than the backend, so it
     * can be checked here without a device: one edge per four samples of every
     * filtering pass of every plane, levels bounded by the frame header, and
     * nothing outside of the planes. */
    CHECK(!syntax.deblock.empty());
    CHECK_EQ(syntax.sharpness, 0);
    int deblock_planes[3] = {0, 0, 0};
    int deblock_filter_sizes[3] = {0, 0, 0};
    for (const av1gpu::DeblockEdge& e : syntax.deblock) {
      CHECK(e.plane >= 0 && e.plane < 3);
      CHECK(e.level > 0 && e.level <= 63);
      CHECK(e.filter_size == 4 || e.filter_size == 8 || e.filter_size == 16);
      CHECK(e.length > 0 && e.length % 4 == 0);
      const int pw = e.plane == 0 ? w : (w + 1) >> 1;
      const int ph = e.plane == 0 ? h : (h + 1) >> 1;
      if (e.vertical) {
        CHECK(e.x >= 0 && e.x < pw);
        CHECK(e.y >= 0 && e.y + e.length <= ph);
      } else {
        CHECK(e.y >= 0 && e.y < ph);
        CHECK(e.x >= 0 && e.x + e.length <= pw);
      }
      ++deblock_planes[e.plane];
      if (e.filter_size == 16) ++deblock_filter_sizes[e.plane];
    }
    CHECK(deblock_planes[0] > 0 && deblock_planes[1] > 0 && deblock_planes[2] > 0);
    /* Frame 0 signals level 2 for both luma passes and level 1 for each chroma
     * plane, with no segment or delta-lf adjustment anywhere in the frame. */
    for (const av1gpu::DeblockEdge& e : syntax.deblock)
      CHECK_EQ(e.level, e.plane == 0 ? 2 : 1);
    std::printf("  %d deblock edge(s), %d/%d/%d per plane\n",
                static_cast<int>(syntax.deblock.size()), deblock_planes[0],
                deblock_planes[1], deblock_planes[2]);

    /* ---- the CDEF plan ----------------------------------------------------- */
    /* Frame 0 signals luma strength 0 and chroma strength 60, so every entry in
     * the plan is chroma: one per eight by eight luma block, which at 4:2:0 is a
     * four by four block of each chroma plane, that is not entirely skipped. A plane whose strength is zero has nothing to filter,
     * which is why the luma plane contributes nothing here, and a block whose
     * four 4x4 cells are all skipped is dropped, so the plan is not simply the
     * whole grid. The strengths stay at their base value: the variance that
     * scales them and the direction the kernel needs both come from the
     * deblocked samples, which only the device has. */
    CHECK(!syntax.cdef.empty());
    int cdef_planes[3] = {0, 0, 0};
    const int chroma_w = (w + 1) >> 1, chroma_h = (h + 1) >> 1;
    for (const av1gpu::CdefBlock& c : syntax.cdef) {
      CHECK(c.plane == 1 || c.plane == 2);
      CHECK_EQ(c.x % 4, 0);
      CHECK_EQ(c.y % 4, 0);
      CHECK_EQ(c.pri_strength, av1gpu::frontend::cdef_primary_strength(60));
      CHECK_EQ(c.sec_strength, av1gpu::frontend::cdef_secondary_strength(60));
      CHECK_EQ(c.damping, 3);
      CHECK_EQ(c.dir, -1); /* the kernel detects the direction */
      /* the block may be cut short by an odd frame size, so only its origin has
       * to be inside the plane; both sizes here happen to be multiples of four */
      CHECK(c.x >= 0 && c.x < chroma_w);
      CHECK(c.y >= 0 && c.y < chroma_h);
      ++cdef_planes[c.plane];
    }
    CHECK(cdef_planes[1] > 0 && cdef_planes[2] > 0);
    /* chroma blocks are planned in pairs, so the two planes always agree */
    CHECK_EQ(cdef_planes[1], cdef_planes[2]);
    std::printf("  %d cdef block(s), %d/%d/%d per plane\n",
                static_cast<int>(syntax.cdef.size()), cdef_planes[0],
                cdef_planes[1], cdef_planes[2]);

    /* ---- the reconstruction ------------------------------------------------ */
    /* The engine's own arithmetic, on the syntax of a real frame, compared with
     * dav1d's reference picture for it. dav1d was run with every in-loop filter
     * it has (this stream signals restoration off, so that is deblocking plus
     * CDEF) and the two pictures are bit identical; the hash below is that
     * picture, so any change in parsing, prediction, chroma from luma, the
     * inverse transforms, the deblocking filter or CDEF that costs a single
     * least significant bit fails here. */
    av1gpu::BackendOptions options;
    options.kind = av1gpu::BackendKind::Cpu;
    std::string decode_error;
    std::unique_ptr<av1gpu::Decoder> decoder = av1gpu::Decoder::create(options, &decode_error);
    CHECK(decoder != nullptr);
    if (!decoder) return;
    av1gpu::Frame picture;
    CHECK(decoder->decode(syntax, &picture, &decode_error));
    if (!decode_error.empty()) std::printf("  decode failed: %s\n", decode_error.c_str());
    CHECK(picture.matches(w, h, 8, 1, 1));

    uint64_t hash = 1469598103934665603ull; /* FNV-1a, 64 bit */
    for (int p = 0; p < 3; ++p) {
      const int pw = picture.plane_width(p), ph = picture.plane_height(p);
      for (int y = 0; y < ph; ++y) {
        const uint16_t* row = picture.plane(p) + static_cast<size_t>(y) * picture.stride[p];
        for (int x = 0; x < pw; ++x) {
          hash = (hash ^ static_cast<uint64_t>(row[x] & 0xff)) * 1099511628211ull;
        }
      }
    }
    std::printf("  frame hash %016llx (%d chroma-from-luma blocks)\n",
                static_cast<unsigned long long>(hash), cfl_blocks);
    /* dav1d's picture, plane by plane, 8 bits per sample, is this frame:
     * 320x240 luma then 160x120 of each chroma plane. */
    CHECK_EQ(hash, 0xc7affd541d516823ULL);

    /* ---- a much stronger filter, from the same tile data ------------------- */
    /* The levels are the only thing frame 0's levels of 2 and 1 leave out: at
     * level 2 the filter barely moves 204 samples of this picture. The levels
     * are uncompressed header syntax, six bits each, so they can be patched in
     * place without shifting a single tile byte - and the tile decode above
     * stays valid, because nothing in the tile depends on them. Each variant's
     * hash is dav1d's output for the patched file with all of its in-loop
     * filters on (the CDEF parameters are untouched, so only the deblocking
     * levels differ between the variants), so the wide filters, the flat masks
     * and the per pass, per plane level selection are all compared sample for
     * sample against a reference decoder. */
    auto decode_with_levels = [&](const int levels[4], FrameSyntax* out_syntax,
                                  av1gpu::Frame* out_picture, uint64_t* out_hash) {
      std::vector<uint8_t> tu = frame.payload;
      size_t obu_start = 0;
      for (const Obu& obu : obus)
        if (obu.type == ObuType::Frame || obu.type == ObuType::FrameHeader) {
          obu_start = static_cast<size_t>(obu.data - frame.payload.data());
          break;
        }
      /* loop_filter_level_y0 sits 36 bits into this stream's frame header. */
      const size_t base = obu_start * 8 + 36;
      for (int i = 0; i < 4; ++i)
        for (int bit = 0; bit < 6; ++bit) {
          const size_t pos = base + static_cast<size_t>(i) * 6 + static_cast<size_t>(bit);
          const uint8_t mask = static_cast<uint8_t>(1u << (7 - (pos & 7)));
          if ((levels[i] >> (5 - bit)) & 1) {
            tu[pos >> 3] = static_cast<uint8_t>(tu[pos >> 3] | mask);
          } else {
            tu[pos >> 3] = static_cast<uint8_t>(tu[pos >> 3] & ~mask);
          }
        }
      std::vector<Obu> patched;
      if (!split_obus(tu.data(), tu.size(), &patched, &error)) return false;
      FrameHeader patched_header;
      bool have_header = false;
      for (const Obu& obu : patched)
        if (obu.type == ObuType::Frame || obu.type == ObuType::FrameHeader) {
          if (parse_frame_header(seq, obu.data, obu.size, &patched_header, &error) !=
              ParseStatus::Ok)
            return false;
          have_header = true;
          break;
        }
      if (!have_header) return false;
      /* a wrong offset would land in another field, so prove the patch landed */
      if (patched_header.loop_filter.level_y[0] != levels[0] ||
          patched_header.loop_filter.level_y[1] != levels[1] ||
          patched_header.loop_filter.level_u != levels[2] ||
          patched_header.loop_filter.level_v != levels[3])
        return false;
      if (!plan_from_tile(seq, patched_header, result, dequantised, out_syntax, &error))
        return false;
      if (!decoder->decode(*out_syntax, out_picture, &decode_error)) return false;
      uint64_t h = 1469598103934665603ull;
      for (int p = 0; p < 3; ++p) {
        const int pw = out_picture->plane_width(p), ph = out_picture->plane_height(p);
        for (int y = 0; y < ph; ++y) {
          const uint16_t* row = out_picture->plane(p) + static_cast<size_t>(y) * out_picture->stride[p];
          for (int x = 0; x < pw; ++x)
            h = (h ^ static_cast<uint64_t>(row[x] & 0xff)) * 1099511628211ull;
        }
      }
      *out_hash = h;
      return true;
    };

    const int level_63[4] = {63, 63, 63, 63};
    FrameSyntax strong_syntax;
    av1gpu::Frame strong_picture;
    uint64_t strong_hash = 0;
    CHECK(decode_with_levels(level_63, &strong_syntax, &strong_picture, &strong_hash));
    if (strong_hash) {
      /* 15666 of the luma samples move here, up to 11 away from the unfiltered
       * picture, so this covers far more of the filter than level 2 does. */
      CHECK_EQ(strong_hash, 0x1b7bc234e2ac9e1cULL);
      for (const av1gpu::DeblockEdge& e : strong_syntax.deblock) CHECK_EQ(e.level, 63);
    }

    /* A different level for every pass and plane pins the index the selection
     * uses: luma vertical, luma horizontal, U, V. */
    const int mixed[4] = {40, 12, 24, 51};
    FrameSyntax mixed_syntax;
    av1gpu::Frame mixed_picture;
    uint64_t mixed_hash = 0;
    CHECK(decode_with_levels(mixed, &mixed_syntax, &mixed_picture, &mixed_hash));
    if (mixed_hash) {
      CHECK_EQ(mixed_hash, 0xc5f375c546ed4a6dULL);
      for (const av1gpu::DeblockEdge& e : mixed_syntax.deblock) {
        const int want = e.plane == 0 ? (e.vertical ? mixed[0] : mixed[1])
                                      : (e.plane == 1 ? mixed[2] : mixed[3]);
        CHECK_EQ(e.level, want);
      }
    }
  }

  /* inter frames are refused rather than misparsed */
  TileDecodeResult other;
  FrameHeader inter = header;
  inter.type = FrameType::Inter;
  CHECK(decode_intra_tile(seq, inter, tile, tile_size, &other, &error) ==
        ParseStatus::Unsupported);
  CHECK(!error.empty());
}

} /* namespace */

int main() {
  test_bit_reader();
  test_obu_framing();
  test_ivf();
  test_sequence_header();
  test_frame_header();
  test_inter_frame();
  test_frame_size_with_refs();
  test_set_frame_refs();
  test_set_frame_refs_matches_reference();
  test_frame_ids();
  test_short_ref_signaling();
  test_segmentation();
  test_segment_deblock_level();
  test_block_geometry();
  test_projected_motion_vectors();
  test_local_warp();
  test_cdf_inheritance();
  test_tile_group();
  test_tile_decoder();
  test_intrabc_frame();
  test_skip_mode_header();
  test_real_stream();
  return testing::finish("test_frontend");
}
