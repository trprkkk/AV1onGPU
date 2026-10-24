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
#include "frontend/headers.hpp"
#include "frontend/ivf.hpp"
#include "frontend/obu.hpp"
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

  /* Segmentation is understood but refused, not silently accepted. */
  CHECK(parse_frame_header(seq, bytes.data(), 8, &header, &error) ==
        ParseStatus::Invalid);

  /* An inter frame is reported as unsupported, not misparsed. */
  BitWriter inter;
  inter.bit(false); /* show_existing_frame */
  inter.bits(1, 2); /* INTER */
  inter.bit(true);  /* show_frame */
  inter.bit(false); /* error_resilient_mode */
  inter.bit(false); /* disable_cdf_update */
  inter.bit(false); /* allow_screen_content_tools */
  inter.bit(false); /* frame_size_override */
  inter.bits(0, 3); /* primary_ref_frame */
  inter.align();
  const std::vector<uint8_t> inter_bytes = inter.bytes();
  const ParseStatus status =
      parse_frame_header(seq, inter_bytes.data(), inter_bytes.size(), &header, &error);
  CHECK(status == ParseStatus::Unsupported);
  CHECK(header.type == FrameType::Inter);
  CHECK(!error.empty());
}

} /* namespace */

int main() {
  test_bit_reader();
  test_obu_framing();
  test_ivf();
  test_sequence_header();
  test_frame_header();
  return testing::finish("test_frontend");
}
