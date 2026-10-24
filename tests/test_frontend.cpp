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
#include <cstring>
#include <string>
#include <vector>

#include "frontend/bitreader.hpp"
#include "frontend/coef_tables.hpp"
#include "frontend/headers.hpp"
#include "frontend/ivf.hpp"
#include "frontend/obu.hpp"
#include "frontend/tile_decoder.hpp"
#include "frontend/tiles.hpp"
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
  w.bit(false); /* frame_id_numbers_present */
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
};

std::vector<uint8_t> write_key_frame_header(const SequenceSpec& seq_spec,
                                            const KeyFrameSpec& spec) {
  BitWriter w;
  w.bit(false);            /* show_existing_frame */
  w.bits(0, 2);            /* frame_type = KEY */
  w.bit(spec.show);
  w.bit(false);            /* disable_cdf_update */
  w.bit(false);            /* allow_screen_content_tools */
  w.bit(false);            /* frame_size_override_flag */
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
  CHECK_EQ(header.size_from_reference, true);
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

/* Segmentation is part of the header, not a reason to stop: the flags and the
 * per segment feature values have to come back so later stages can use them. */
void test_segmentation() {
  std::printf("segmentation\n");
  SequenceSpec spec;
  const std::vector<uint8_t> seq_bytes = write_sequence_header(spec);
  SequenceHeader seq;
  std::string error;
  CHECK(parse_sequence_header(seq_bytes.data(), seq_bytes.size(), &seq, &error) ==
        ParseStatus::Ok);

  auto write_head = [](BitWriter* w) {
    w->bit(false);     /* show_existing_frame */
    w->bits(0, 2);     /* KEY */
    w->bit(true);      /* show_frame */
    w->bit(false);     /* disable_cdf_update */
    w->bit(false);     /* allow_screen_content_tools */
    w->bit(false);     /* frame_size_override */
    w->bit(false);     /* render size */
    w->bit(true);      /* refresh_frame_context */
    w->bit(true);      /* uniform tiles */
    w->bit(false);     /* stop: one column */
    w->bit(false);     /* stop: one row */
    w->bits(0, 8);     /* base_q_idx: lossless, so the filter bits disappear */
    w->bit(false);     /* delta_q_y_dc */
    w->bit(false);     /* delta_q_u_dc */
    w->bit(false);     /* delta_q_u_ac */
    w->bit(false);     /* using_qmatrix */
  };

  /* Segmentation on, map updated, no new feature data. */
  BitWriter plain;
  write_head(&plain);
  plain.bit(true);  /* segmentation_enabled */
  plain.bit(true);  /* update_map (primary_ref_frame is PRIMARY_REF_NONE) */
  plain.bit(false); /* temporal_update */
  plain.bit(false); /* update_data */
  plain.bit(false); /* reduced_tx_set */
  plain.align();
  const std::vector<uint8_t> plain_bytes = plain.bytes();
  FrameHeader header;
  CHECK(parse_frame_header(seq, plain_bytes.data(), plain_bytes.size(), &header, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(header.segmentation.enabled, true);
  CHECK_EQ(header.segmentation.update_map, true);
  CHECK_EQ(header.segmentation.update_data, false);
  CHECK_EQ(header.all_lossless, true);

  /* With feature data: segment 2 turns the quantiser down by 10. */
  BitWriter data;
  write_head(&data);
  data.bit(true);  /* segmentation_enabled */
  data.bit(true);  /* update_map */
  data.bit(false); /* temporal_update */
  data.bit(true);  /* update_data */
  for (int i = 0; i < kMaxSegments; ++i) {
    for (int j = 0; j < kSegFeatureCount; ++j) {
      const bool enabled = i == 2 && j == 0;
      data.bit(enabled);
      if (enabled) data.signed_bits(-10, 8);
    }
  }
  data.bit(false); /* reduced_tx_set */
  data.align();
  const std::vector<uint8_t> data_bytes = data.bytes();
  CHECK(parse_frame_header(seq, data_bytes.data(), data_bytes.size(), &header, &error) ==
        ParseStatus::Ok);
  CHECK_EQ(header.segmentation.update_data, true);
  CHECK_EQ(header.segmentation.feature_enabled[2][0], true);
  CHECK_EQ(header.segmentation.feature_data[2][0], -10);
  CHECK_EQ(header.segmentation.feature_enabled[1][0], false);
  CHECK_EQ(header.segmentation.feature_enabled[2][1], false);
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
          CHECK(h.size_from_reference);
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


/* The tile group is where the payload stops being plain bits. A single tile
 * that runs to the end of the payload is the common case; the group can also
 * carry a subset of the tiles, and the last tile of the group is never given a
 * size because it always takes the rest of the OBU. */
void test_tile_group() {
  std::printf("tile group\n");
  FrameHeader h;
  h.type = FrameType::Key;
  h.frame_width = 640;
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
  CHECK(partition_subsize(kPartitionHorizontalA, kBlock16x16) == kBlock8x8);

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
  /* The whole frame, as dav1d decodes it: 306 blocks, 1063 transform blocks
   * and 64301 coded coefficients. The tile's bytes are exactly consumed, which
   * is the end to end check that every symbol was read with the right table. */
  CHECK_EQ(result.blocks.size(), 306u);
  CHECK_EQ(result.transform_blocks, 1063);
  CHECK_EQ(result.coefficients, 64301u);
  CHECK_EQ(result.levels.size(), 64301u);
  CHECK_EQ(result.bytes_used, tile_size);

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
  test_segmentation();
  test_block_geometry();
  test_tile_group();
  test_tile_decoder();
  test_real_stream();
  return testing::finish("test_frontend");
}
