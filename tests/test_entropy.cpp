/*
 * AV1onGPU - entropy decoder tests
 *
 * The range coder is the one part of the decoder where a single wrong bit
 * produces plausible looking but wrong pixels, and where "close" is worthless:
 * a tile is either decoded exactly like the reference decoders or it is not
 * decoded at all. These tests therefore pin the output against golden vectors
 * produced by dav1d's `msac` (see tests/entropy_vectors.hpp) instead of testing
 * round trips against our own encoder, which would only prove we agree with
 * ourselves.
 *
 * Covered:
 *   - symbol decoding over alphabets from 2 to 16, with CDF adaptation
 *   - the adapted table itself, which is what keeps the following tiles in sync
 *   - disable_cdf_update, i.e. frames that freeze their probability tables
 *   - boolean and equiprobable reads, L(n) and ns(n)
 *   - end of tile data: overrun detection and the "keep going" contract
 *
 * The vectors live in a generated header so regenerating them (with a newer
 * dav1d, or more cases) cannot quietly change the expectations of this test.
 */
#include <cstdint>
#include <cstring>
#include <vector>

#include "data/tile_trace.hpp"
#include "entropy_vectors.hpp"
#include "frontend/cdf.hpp"
#include "frontend/headers.hpp"
#include "frontend/ivf.hpp"
#include "frontend/obu.hpp"
#include "frontend/tiles.hpp"
#include "frontend/range_decoder.hpp"
#include "test_common.hpp"

using av1gpu::frontend::RangeDecoder;

namespace {

/* Decodes one vector case and checks both the symbols and the adapted table.
 * `update_cdf` false is the disable_cdf_update path: the decoder must leave the
 * table alone, so only the first symbol can be compared. */
template <size_t S>
void check_symbols(const uint16_t (&cdf)[S], const uint8_t* payload, size_t payload_size,
                   const unsigned* symbols, int count, const uint16_t (&final_cdf)[S],
                   bool update_cdf) {
  const int N = static_cast<int>(S) - 1;
  uint16_t table[S];
  std::memcpy(table, cdf, sizeof(table));
  uint16_t before[S];
  std::memcpy(before, cdf, sizeof(table));

  RangeDecoder decoder;
  decoder.init(payload, payload_size, update_cdf);

  int mismatch = -1;
  unsigned got = 0;
  for (int i = 0; i < count; ++i) {
    const unsigned value = decoder.read_symbol(table, N);
    if (i == 0) CHECK_EQ(value, symbols[0]);
    if (decoder.failed()) {
      /* the payload ran out and the padding could not satisfy any interval */
      CHECK(decoder.overrun());
      mismatch = -2;
      break;
    }
    if (update_cdf && value != symbols[i] && mismatch < 0) {
      mismatch = i;
      got = value;
    }
  }
  if (mismatch >= 0) {
    std::printf("    symbol %d: decoded %u, dav1d decoded %u\n", mismatch, got,
                symbols[mismatch]);
  }
  CHECK_EQ(mismatch, -1);
  CHECK(!decoder.failed());

  if (update_cdf) {
    CHECK_EQ(std::memcmp(table, final_cdf, sizeof(table)), 0);
    /* the counter at the end saturates at 32 and the probabilities still
     * decrease towards the last symbol */
    CHECK_EQ(table[N], final_cdf[N]);
    CHECK_EQ(table[N - 1], 0u);
    for (int i = 1; i < N; ++i) CHECK(table[i] <= table[i - 1]);
  } else {
    CHECK_EQ(std::memcmp(table, before, sizeof(table)), 0);
  }
}

void test_symbols() {
  using namespace entropy_vectors;
  check_symbols(case0_cdf, case0_payload, sizeof(case0_payload), case0_symbols,
                case0_symbols_count, case0_final_cdf, true);
  check_symbols(case1_cdf, case1_payload, sizeof(case1_payload), case1_symbols,
                case1_symbols_count, case1_final_cdf, true);
  check_symbols(case2_cdf, case2_payload, sizeof(case2_payload), case2_symbols,
                case2_symbols_count, case2_final_cdf, true);
  check_symbols(case3_cdf, case3_payload, sizeof(case3_payload), case3_symbols,
                case3_symbols_count, case3_final_cdf, true);
  check_symbols(case4_cdf, case4_payload, sizeof(case4_payload), case4_symbols,
                case4_symbols_count, case4_final_cdf, true);
  check_symbols(case5_cdf, case5_payload, sizeof(case5_payload), case5_symbols,
                case5_symbols_count, case5_final_cdf, true);
}

/* disable_cdf_update: the tables must survive the decode untouched. */
void test_frozen_cdf() {
  using namespace entropy_vectors;
  uint16_t table[case2_n + 1];
  std::memcpy(table, case2_cdf, sizeof(table));
  RangeDecoder decoder;
  decoder.init(case2_payload, sizeof(case2_payload), false);
  CHECK_EQ(decoder.read_symbol(table, case2_n), case2_symbols[0]);
  CHECK_EQ(std::memcmp(table, case2_cdf, sizeof(table)), 0);
  /* decoding continues, just with the original probabilities */
  const unsigned second = decoder.read_symbol(table, case2_n);
  CHECK(second <= static_cast<unsigned>(case2_n));
  CHECK_EQ(std::memcmp(table, case2_cdf, sizeof(table)), 0);
}

void test_bits() {
  using namespace entropy_vectors;

  RangeDecoder decoder;
  decoder.init(bool_payload, sizeof(bool_payload), true);
  for (int i = 0; i < 64; ++i) CHECK_EQ(decoder.read_bool(bool_probabilities[i]), bool_bits[i]);
  CHECK(!decoder.overrun());

  RangeDecoder equi;
  equi.init(equi_payload, sizeof(equi_payload), true);
  for (int i = 0; i < 64; ++i) CHECK_EQ(equi.read_bit(), equi_bits[i]);
  CHECK(!equi.overrun());

  RangeDecoder literal;
  literal.init(literal_payload, sizeof(literal_payload), true);
  for (int i = 0; i < 8; ++i) CHECK_EQ(literal.read_literal(literal_widths[i]), literal_values[i]);

  RangeDecoder uniform;
  uniform.init(uniform_payload, sizeof(uniform_payload), true);
  for (int i = 0; i < 12; ++i) CHECK_EQ(uniform.read_uniform(uniform_sizes[i]), uniform_values[i]);
  CHECK(!uniform.overrun());
}

/* ns(1) is always zero and reads nothing: the decoder must not touch the
 * stream, which the following symbol proves. */
void test_uniform() {
  using namespace entropy_vectors;
  RangeDecoder a, b;
  a.init(case3_payload, sizeof(case3_payload), true);
  b.init(case3_payload, sizeof(case3_payload), true);
  CHECK_EQ(a.read_uniform(1), 0u);
  CHECK_EQ(a.read_uniform(0), 0u);
  uint16_t ta[case3_n + 1], tb[case3_n + 1];
  std::memcpy(ta, case3_cdf, sizeof(ta));
  std::memcpy(tb, case3_cdf, sizeof(tb));
  CHECK_EQ(a.read_symbol(ta, case3_n), b.read_symbol(tb, case3_n));
  CHECK_EQ(std::memcmp(ta, tb, sizeof(ta)), 0);
}

/* Past the end of the tile the decoder pads with one bits and keeps producing
 * values; callers detect that with overrun() at the end of the tile instead of
 * checking every symbol, so the state has to stay coherent. */
void test_overrun() {
  using namespace entropy_vectors;
  uint16_t table[case1_n + 1];
  std::memcpy(table, case1_cdf, sizeof(table));
  RangeDecoder decoder;
  decoder.init(case1_payload, sizeof(case1_payload), true);
  CHECK(!decoder.overrun());
  CHECK_EQ(decoder.read_symbol(table, case1_n), case1_symbols[0]);
  CHECK(!decoder.overrun());

  /* decoding the whole payload plus a good margin eventually runs out */
  for (int i = 0; i < 4000 && !decoder.overrun(); ++i) {
    std::memcpy(table, case1_cdf, sizeof(table));
    CHECK(decoder.read_symbol(table, case1_n) < static_cast<unsigned>(case1_n));
    CHECK(decoder.bytes_used() <= sizeof(case1_payload));
  }
  CHECK(decoder.overrun());
  CHECK(!decoder.failed());
  /* past the end it keeps decoding without crashing or leaving the alphabet */
  for (int i = 0; i < 64; ++i) {
    std::memcpy(table, case1_cdf, sizeof(table));
    CHECK(decoder.read_symbol(table, case1_n) < static_cast<unsigned>(case1_n));
    CHECK(decoder.read_bit() <= 1u);
  }
}

/* An empty tile is legal to construct (the caller reports the syntax error);
 * it must behave like an all-ones stream rather than read out of bounds. */
void test_empty() {
  /* a two symbol CDF needs three entries: the last one is the counter */
  uint16_t table[3] = {16384, 0, 0};
  RangeDecoder decoder;
  decoder.init(nullptr, 0, true);
  CHECK(decoder.overrun());
  CHECK_EQ(decoder.bytes_used(), 0u);
  CHECK(decoder.read_symbol(table, 2) < 2u);
  CHECK(decoder.read_literal(8) <= 0xffu);
  CHECK(!decoder.failed());
}

/* A boolean CDF is just a two symbol CDF in this decoder, which is how AV1
 * stores flag contexts (`intra_inter`, `skip`, ...) in the reference decoders. */
void test_boolean_cdf() {
  using namespace entropy_vectors;
  const uint16_t start[3] = {16384, 0, 0};
  uint16_t table[3];
  std::memcpy(table, start, sizeof(table));
  RangeDecoder decoder;
  decoder.init(case0_payload, sizeof(case0_payload), true);
  /* a balanced stream must keep the two probabilities inside (0, 32768): the
   * adaptation never lets a symbol become impossible */
  for (int i = 0; i < 32; ++i) {
    CHECK(decoder.read_symbol(table, 2) < 2u);
    CHECK(table[0] > 0u && table[0] < 32768u);
    CHECK_EQ(table[1], 0u);
  }
  CHECK_EQ(table[2], 32u);
}

/* The last symbol of a table has no stored probability of its own: its
 * interval is whatever range is left, so it is selected exactly when the
 * range coder stops above the last stored boundary. dav1d reaches it by
 * walking one entry past the probabilities and reading the adaptation
 * counter, which is small enough (< 64) that the entry contributes no range.
 * That is not an error path - real streams hit it for about one symbol in
 * seven - so it must not set failed(). */
void test_last_symbol() {
  /* zero bits in the window put the range coder below every boundary, so the
   * only interval left is the implied one for the last symbol */
  const uint8_t payload[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  uint16_t table[5] = {30000, 24000, 18000, 9000, 0};
  RangeDecoder decoder;
  decoder.init(payload, sizeof(payload), true);
  const unsigned value = decoder.read_symbol(table, 4);
  CHECK_EQ(value, 4u);
  CHECK(!decoder.failed());
  /* Every stored probability sits below the decoded symbol, so all of them are
   * adapted upwards; with four symbols the rate is 4 + 1 = 5. */
  CHECK(table[4] == 1);
  CHECK(table[0] == 30000u + ((32768u - 30000u) >> 5));
  CHECK(table[3] == 9000u + ((32768u - 9000u) >> 5));
}

/* A table whose counter entry is not a valid AV1 counter widens the last
 * interval instead of collapsing it, so the decoder would walk past the
 * array. That is a corrupt table (or a desynchronised stream) and must be
 * reported rather than read out of bounds. */
void test_corrupt_table() {
  const uint8_t payload[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  uint16_t table[4] = {32764, 32760, 0, 100};
  RangeDecoder decoder;
  decoder.init(payload, sizeof(payload), true);
  decoder.read_symbol(table, 3);
  CHECK(decoder.failed());
  /* the decoder stays usable and keeps answering within bounds */
  CHECK(decoder.read_bit() <= 1u);
  CHECK(!decoder.overrun());
  CHECK(decoder.bytes_used() <= sizeof(payload));
}


/* The vectors above are synthetic: an encoder written for the tests produced
 * the payload. This one is a real SVT-AV1 stream. dav1d decoded the tile and
 * logged every symbol, the table it used and the state it reached; replaying
 * that log through AV1onGPU's decoder must reproduce dav1d exactly, including
 * the window/range/counter state after the last event. Symbol values alone
 * would not catch an adaptation mismatch, and an adaptation mismatch shows up
 * as a wrong value on the next symbol of the same table. */
void test_real_tile_trace() {
  using namespace av1gpu::frontend;
  std::printf("real tile trace (SVT-AV1, dav1d)\n");
  const std::string path = std::string(AV1GPU_TEST_DATA_DIR) + "/svt_av1_320x240.ivf";
  std::vector<uint8_t> file;
  {
    std::FILE* handle = std::fopen(path.c_str(), "rb");
    CHECK(handle != nullptr);
    if (!handle) return;
    uint8_t chunk[16384];
    size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), handle)) > 0)
      file.insert(file.end(), chunk, chunk + got);
    std::fclose(handle);
  }

  IvfReader reader;
  std::string error;
  CHECK(reader.open(file, &error));
  IvfFrame frame;
  CHECK(reader.next(&frame, &error)); /* the key frame the trace is from */

  SequenceHeader seq;
  const uint8_t* tile_data = nullptr;
  size_t tile_size = 0;
  bool found = false;
  std::vector<Obu> obus;
  CHECK(split_obus(frame.payload.data(), frame.payload.size(), &obus, &error));
  for (const Obu& obu : obus) {
    if (obu.type == ObuType::SequenceHeader) {
      CHECK(parse_sequence_header(obu.data, obu.size, &seq, &error) == ParseStatus::Ok);
    } else if (obu.type == ObuType::Frame || obu.type == ObuType::FrameHeader) {
      FrameHeader h;
      CHECK(parse_frame_header(seq, obu.data, obu.size, &h, &error) == ParseStatus::Ok);
      TileGroup group;
      const ParseStatus status = parse_tile_group(seq, h, obu.data, obu.size, &group, &error);
      CHECK(status == ParseStatus::Ok);
      if (status != ParseStatus::Ok) {
        std::printf("  tile group: %s\n", error.c_str());
        continue;
      }
      CHECK_EQ(group.tiles.size(), 1u);
      if (group.tiles.size() == 1) {
        /* the whole frame payload is one tile, and it starts where the trace
         * says it does */
        CHECK_EQ(static_cast<size_t>(obu.data - frame.payload.data()) + group.tiles[0].offset,
                 tile_trace::kTileOffset);
        tile_data = obu.data + group.tiles[0].offset;
        tile_size = group.tiles[0].size;
        found = true;
      }
    }
  }
  CHECK(found);
  if (!found) return;

  RangeDecoder decoder;
  decoder.init(tile_data, tile_size, true);
  size_t next = 0;
  unsigned mismatches = 0;
  for (size_t i = 0; i < tile_trace::kEventCount; ++i) {
    const unsigned kind = tile_trace::kKind[i];
    unsigned value = 0;
    if (kind == 0) {
      const int count = tile_trace::kCount[i];
      if (i == 0) {
        /* the very first symbol of the tile is the 64x64 partition, decoded
         * out of our own default table rather than the recorded one */
        using namespace av1gpu::frontend;
        CdfContext defaults;
        init_default_cdf(&defaults, 0); /* partition rows do not depend on the quantiser */
        const unsigned first =
            decoder.read_symbol(defaults.partition.row(kBlockLevel64x64, 0), 9);
        CHECK_EQ(first, tile_trace::kValue[0]);
        next += static_cast<size_t>(count);
        continue;
      }
      uint16_t table[kMaxCdfSymbols] = {0};
      for (int j = 0; j < count; ++j) table[j] = tile_trace::kCdf[next + j];
      next += static_cast<size_t>(count);
      value = decoder.read_symbol(table, count);
    } else if (kind == 1) {
      value = decoder.read_bool(tile_trace::kCount[i]);
    } else {
      value = decoder.read_bit();
    }
    if (value != tile_trace::kValue[i] || decoder.failed()) {
      if (!mismatches) std::printf("  first mismatch at event %zu\n", i);
      ++mismatches;
      if (mismatches > 4) break;
    }
  }
  CHECK_EQ(mismatches, 0u);
  const RangeDecoder::State state = decoder.state();
  CHECK(state.window == tile_trace::kFinalDif);
  CHECK_EQ(state.range, tile_trace::kFinalRng);
  CHECK_EQ(state.count, tile_trace::kFinalCnt);
}


/* The defaults are normative numbers, so they can be checked against something
 * external: dav1d recorded the exact table it used for the first symbol of the
 * fixture's key frame, and that symbol is a partition of the 64x64 block at
 * (0,0) - the first thing every key frame decodes. If the 32768 - x
 * conversion, the row order or the counter placement were wrong, this fails
 * before any syntax exists to get confused by. */
void test_default_cdf() {
  using namespace av1gpu::frontend;
  CdfContext cdf;
  init_default_cdf(&cdf, 51);

  /* the 64x64 partition row dav1d used for the first symbol of the tile */
  const uint16_t* row = cdf.partition.row(kBlockLevel64x64, 0);
  for (int i = 0; i < tile_trace::kCount[0]; ++i) CHECK_EQ(row[i], tile_trace::kCdf[i]);
  CHECK_EQ(row[tile_trace::kCount[0]], 0u); /* the counter starts at zero */
  /* eight partition types are reachable from a 128x128 block, ten from the
   * levels below it, four from an 8x8 block */
  CHECK_EQ(PartitionCdf::symbols(kBlockLevel128x128), 8);
  CHECK_EQ(PartitionCdf::symbols(kBlockLevel64x64), 10);
  CHECK_EQ(PartitionCdf::symbols(kBlockLevel8x8), 4);

  /* skip: the 3 contexts are 32768 - {31671, 16515, 4576}; dav1d used 1097
   * for a 32x32 block with no coded neighbours, which is the second bool it
   * recorded */
  CHECK_EQ(cdf.skip.row(0)[0], 1097u);
  CHECK_EQ(cdf.skip.row(1)[0], 16253u);
  CHECK_EQ(cdf.skip.row(2)[0], 28192u);
  CHECK_EQ(cdf.skip.row(0)[1], 0u);

  /* every row is a valid inverse CDF: probabilities decrease, the counter
   * follows them, and nothing is left at 32768 (an interval of zero width) */
  for (int level = 0; level < kBlockLevelCount; ++level) {
    for (int context = 0; context < kPartitionContexts; ++context) {
      const uint16_t* r = cdf.partition.row(level, context);
      const int symbols = kPartitionSymbols[level];
      CHECK(r[0] < 32768u);
      for (int i = 1; i < symbols; ++i) CHECK(r[i] < r[i - 1]);
      CHECK_EQ(r[symbols], 0u);
    }
  }
}

/* One inverse CDF row: probabilities strictly decreasing, the adaptation
 * counter right after the last symbol, and nothing left in the padding that
 * follows it. */
void check_row(const uint16_t* row, int symbols, const char* what) {
  CHECK_EQ(row[symbols - 1], 0u);
  if (symbols > 1) CHECK(row[0] < 32768u);
  for (int i = 1; i < symbols - 1; ++i) {
    if (row[i] >= row[i - 1]) {
      std::printf("  %s: entry %d (%u) does not decrease from %u\n", what, i, row[i],
                  row[i - 1]);
      CHECK(false);
      return;
    }
  }
}

/* The tables an inter frame reads. Two independent checks:
 *
 *   - the numbers, against the table dav1d's `src/cdf.c` builds with
 *     CDF1/CDF2/... macros. Every row of every generated table was compared
 *     with dav1d when the generator was written: 245 of 258 rows appear there
 *     verbatim (the other 13 are the uniform wedge rows for the block sizes
 *     AV1 cannot wedge, which dav1d does not store). The rows below are the
 *     ones a mistake in the conversion, the transpose or the block size
 *     ordering would move, so they are pinned here as well.
 *
 *   - the shape: every table is a valid inverse CDF layout, which is what makes
 *     read_symbol safe on it. */
void test_inter_cdf() {
  using namespace av1gpu::frontend;
  CdfContext cdf;
  init_default_cdf(&cdf, 0); /* the inter tables do not depend on the quantiser */
  const InterCdf& in = cdf.inter;

  /* is_inter: dav1d's `.intra` is CDF1(806), CDF1(16662), CDF1(20186),
   * CDF1(26538) */
  CHECK_EQ(in.is_inter[0][0], 31962u);
  CHECK_EQ(in.is_inter[1][0], 16106u);
  CHECK_EQ(in.is_inter[2][0], 12582u);
  CHECK_EQ(in.is_inter[3][0], 6230u);

  /* skip_mode: CDF1(32621), CDF1(20708), CDF1(8127) */
  CHECK_EQ(in.skip_mode[0][0], 147u);
  CHECK_EQ(in.skip_mode[1][0], 12060u);
  CHECK_EQ(in.skip_mode[2][0], 24641u);

  /* compound or single: dav1d's `.comp`, CDF1(26828), CDF1(24035), ... */
  CHECK_EQ(in.comp_mode[0][0], 5940u);
  CHECK_EQ(in.comp_mode[4][0], 29867u);

  /* compound type: dav1d's `.comp_dir` */
  CHECK_EQ(in.comp_ref_type[0][0], 31570u);
  CHECK_EQ(in.comp_ref_type[2][0], 23602u);
  CHECK_EQ(in.comp_ref_type[4][0], 10293u);

  /* the reference trees: dav1d keeps these transposed (six steps by three
   * contexts), so a wrong axis swap shows up here */
  CHECK_EQ(in.single_ref[0][0][0], 27871u);
  CHECK_EQ(in.single_ref[2][5][0], 2464u);
  CHECK_EQ(in.comp_ref[0][0][0], 27822u);
  CHECK_EQ(in.comp_bwd_ref[1][1][0], 17593u);
  CHECK_EQ(in.uni_comp_ref[2][0][0], 994u);   /* 32768 - 31774 */
  CHECK_EQ(in.uni_comp_ref[2][2][0], 6058u);  /* 32768 - 26710 */
  CHECK_EQ(in.comp_ref_type[1][0], 30698u);

  /* compound mode and the two bits that choose the blend: dav1d's
   * `.comp_inter_mode`, `.jnt_comp` and `.mask_comp` */
  CHECK_EQ(in.compound_mode[0][0], 25008u);
  CHECK_EQ(in.compound_mode[0][6], 5877u);
  CHECK_EQ(in.compound_idx[0][0], 14524u);
  CHECK_EQ(in.comp_group_idx[0][0], 6161u);
  CHECK_EQ(in.comp_group_idx[5][0], 10094u);

  /* distance weighted or wedge: dav1d's `.wedge_comp`, which only covers the
   * block sizes that can be wedged - 8x8 is entry 3 of the specification's
   * table, and 9337 is 32768 - 23431 */
  CHECK_EQ(in.compound_type[kBlock8x8][0], 9337u);
  CHECK_EQ(in.compound_type[kBlock8x16][0], 19597u);
  CHECK_EQ(in.compound_type[kBlock4x4][0], 16384u); /* unreachable, but valid */
  /* the wedge index: dav1d's `.wedge_idx` starts at 8x8 too, CDF15(2438, ...) */
  CHECK_EQ(in.wedge_index[kBlock8x8][0], 30330u);
  CHECK_EQ(in.wedge_index[kBlock8x8][14], 1597u);

  /* inter/intra: dav1d's `.interintra` has four entries, the first of which is
   * never used, so its index 1 is the specification's index 0 */
  CHECK_EQ(in.inter_intra[0][0], 5881u);       /* 32768 - 26887 */
  CHECK_EQ(in.inter_intra_mode[0][0], 30893u); /* 32768 - 1875 */
  CHECK_EQ(in.inter_intra_mode[0][2], 5436u);
  CHECK_EQ(in.wedge_inter_intra[kBlock8x8][0], 12732u); /* 32768 - 20036 */

  /* motion mode: simple, obmc or warped, then whether obmc is used at all */
  CHECK_EQ(in.motion_mode[kBlock8x8][0], 25117u); /* 32768 - 7651 */
  CHECK_EQ(in.motion_mode[kBlock8x8][1], 8008u);  /* 32768 - 24760 */
  CHECK_EQ(in.use_obmc[kBlock8x8][0], 22331u);    /* 32768 - 10437 */
  CHECK_EQ(in.use_obmc[kBlock128x128][0], 130u);  /* 32768 - 32638 */

  /* the interpolation filter: dav1d's `.filter`, two directions by eight
   * filter contexts */
  CHECK_EQ(in.interp_filter[0][0], 833u); /* 32768 - 31935 */
  CHECK_EQ(in.interp_filter[0][1], 48u);
  CHECK_EQ(in.interp_filter[8][0], 858u); /* 32768 - 31910 */

  /* motion vectors: dav1d's `.mv.joint`, `.mv.comp.classes` and the fractional
   * parts */
  CHECK_EQ(in.mv_joint[0], 28672u);
  CHECK_EQ(in.mv_joint[2], 13440u);
  CHECK_EQ(in.mv_class[0][0], 4096u);
  CHECK_EQ(in.mv_class[0][9], 1u);
  CHECK_EQ(in.mv_class0_fr[0][0][0], 16384u);
  CHECK_EQ(in.mv_class0_fr[0][1][2], 8640u);
  CHECK_EQ(in.mv_fr[0][0], 24576u);
  CHECK_EQ(in.mv_bit[0][0], 15360u);
  CHECK_EQ(in.mv_bit[9][0], 2048u);

  /* new, zero or a predicted motion vector, and how far down the list */
  CHECK_EQ(in.new_mv[0][0], 8733u);
  CHECK_EQ(in.new_mv[5][0], 28092u);
  CHECK_EQ(in.zero_mv[0][0], 30593u);
  CHECK_EQ(in.ref_mv[3][0], 4146u);
  CHECK_EQ(in.drl_mode[0][0], 19664u);
  CHECK_EQ(in.drl_mode[2][0], 13823u);

  /* the transform types an inter block can use */
  CHECK_EQ(in.txtp_inter1[0][0], 28310u);
  CHECK_EQ(in.txtp_inter1[1][14], 2019u);
  CHECK_EQ(in.txtp_inter2[0], 31998u);
  CHECK_EQ(in.txtp_inter2[10], 2239u);
  CHECK_EQ(in.txtp_inter3[0][0], 16384u);
  CHECK_EQ(in.txtp_inter3[3][0], 32020u);

  /* every row of every table, then: a valid inverse CDF, counter zero */
  const auto rows = [](const uint16_t* flat, int count, int symbols, int stride,
                       const char* what) {
    for (int i = 0; i < count; ++i) check_row(flat + static_cast<size_t>(i) * stride, symbols, what);
  };
  rows(&in.is_inter[0][0], 4, 2, 2, "is_inter");
  rows(&in.skip_mode[0][0], 3, 2, 2, "skip_mode");
  rows(&in.comp_mode[0][0], 5, 2, 2, "comp_mode");
  rows(&in.comp_ref_type[0][0], 5, 2, 2, "comp_ref_type");
  rows(&in.uni_comp_ref[0][0][0], 9, 2, 2, "uni_comp_ref");
  rows(&in.comp_ref[0][0][0], 9, 2, 2, "comp_ref");
  rows(&in.comp_bwd_ref[0][0][0], 6, 2, 2, "comp_bwd_ref");
  rows(&in.single_ref[0][0][0], 18, 2, 2, "single_ref");
  rows(&in.compound_mode[0][0], 8, 8, 8, "compound_mode");
  rows(&in.compound_idx[0][0], 6, 2, 2, "compound_idx");
  rows(&in.comp_group_idx[0][0], 6, 2, 2, "comp_group_idx");
  rows(&in.compound_type[0][0], kBlockSizeCount, 2, 2, "compound_type");
  rows(&in.wedge_index[0][0], kBlockSizeCount, 16, 16, "wedge_index");
  rows(&in.inter_intra[0][0], 3, 2, 2, "inter_intra");
  rows(&in.inter_intra_mode[0][0], 3, 4, 4, "inter_intra_mode");
  rows(&in.wedge_inter_intra[0][0], kBlockSizeCount, 2, 2, "wedge_inter_intra");
  rows(&in.use_obmc[0][0], kBlockSizeCount, 2, 2, "use_obmc");
  rows(&in.motion_mode[0][0], kBlockSizeCount, 3, 3, "motion_mode");
  rows(&in.interp_filter[0][0], 16, 3, 3, "interp_filter");
  check_row(in.mv_joint, 4, "mv_joint");
  check_row(in.mv_sign, 2, "mv_sign");
  rows(&in.mv_class[0][0], 2, 11, 16, "mv_class");
  check_row(in.mv_class0_bit, 2, "mv_class0_bit");
  rows(&in.mv_class0_fr[0][0][0], 4, 4, 4, "mv_class0_fr");
  check_row(in.mv_class0_hp, 2, "mv_class0_hp");
  check_row(in.mv_hp, 2, "mv_hp");
  rows(&in.mv_bit[0][0], 10, 2, 2, "mv_bit");
  rows(&in.mv_fr[0][0], 2, 4, 4, "mv_fr");
  rows(&in.new_mv[0][0], 6, 2, 2, "new_mv");
  rows(&in.zero_mv[0][0], 2, 2, 2, "zero_mv");
  rows(&in.ref_mv[0][0], 6, 2, 2, "ref_mv");
  rows(&in.drl_mode[0][0], 3, 2, 2, "drl_mode");
  rows(&in.txtp_inter1[0][0], 2, 16, 16, "txtp_inter1");
  check_row(in.txtp_inter2, 12, "txtp_inter2");
  rows(&in.txtp_inter3[0][0], 4, 2, 2, "txtp_inter3");
}

} /* namespace */

int main() {
  std::printf("test_entropy: range decoder\n");
  test_symbols();
  test_frozen_cdf();
  test_bits();
  test_uniform();
  test_overrun();
  test_empty();
  test_boolean_cdf();
  test_last_symbol();
  test_corrupt_table();
  test_default_cdf();
  test_inter_cdf();
  test_real_tile_trace();
  return testing::finish("test_entropy");
}
