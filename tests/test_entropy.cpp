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

#include "entropy_vectors.hpp"
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
    if (value >= static_cast<unsigned>(N)) {
      /* the padding past the end of the payload finally ran out: the decoder
       * reports it instead of indexing outside the table */
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
  CHECK(second < static_cast<unsigned>(case2_n));
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

/* A stream that cannot satisfy any CDF interval must be reported, not silently
 * mapped to a symbol. The table below gives the last symbol a tiny interval, so
 * a payload of zero bytes (which the decoder reads as all ones) lands outside
 * every one of them. */
void test_no_symbol() {
  /* eight bytes so the seven byte lookahead does not pad yet */
  const uint8_t payload[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  uint16_t table[4] = {32764, 32760, 0, 0};
  RangeDecoder decoder;
  decoder.init(payload, sizeof(payload), true);
  const unsigned value = decoder.read_symbol(table, 3);
  CHECK_EQ(value, 3u);
  CHECK(decoder.failed());
  /* the decoder stays usable and keeps answering within bounds */
  CHECK(decoder.read_bit() <= 1u);
  CHECK(!decoder.overrun());
  CHECK(decoder.bytes_used() <= sizeof(payload));
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
  test_no_symbol();
  return testing::finish("test_entropy");
}
