#include "frontend/range_decoder.hpp"

namespace av1gpu {
namespace frontend {
namespace {

constexpr int kWindowBits = 64;
/* Position of the low probability bits: the reference decoders trade a little
 * precision (6 bits) for a cheaper multiply. Changing this changes the
 * bitstream, not just the speed. */
constexpr int kProbShift = 6;
constexpr unsigned kMinProb = 4;

/* Number of significant bits of `value` (value > 0). */
int bit_length(unsigned value) {
  int bits = 0;
  while (value) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

} /* namespace */

void RangeDecoder::init(const uint8_t* data, size_t size, bool update_cdf) {
  buf_ = data;
  buf_pos_ = data;
  buf_end_ = data + size;
  dif_ = 0;
  rng_ = 0x8000;
  cnt_ = -15;
  update_cdf_ = update_cdf;
  overrun_ = false;
  refill();
}

/* Pulls bytes into the window. When the tile data runs out the missing bytes
 * read as 0xff, which is what the specification prescribes. */
void RangeDecoder::refill() {
  int shift = kWindowBits - cnt_ - 24;
  uint64_t dif = dif_;
  while (true) {
    if (buf_pos_ >= buf_end_) {
      overrun_ = true;
      if (shift >= 0) dif |= ~(~static_cast<uint64_t>(0xff) << shift);
      break;
    }
    if (shift < 0) break;
    dif |= static_cast<uint64_t>(*buf_pos_++ ^ 0xff) << shift;
    shift -= 8;
  }
  dif_ = dif;
  cnt_ = kWindowBits - shift - 24;
}

/* Renormalises the range into [2^15, 2^16), pulling in bytes as it shifts. */
void RangeDecoder::normalize(uint64_t dif, unsigned rng) {
  const int shift = 15 - (bit_length(rng) - 1);
  dif_ = dif << shift;
  rng_ = rng << shift;
  cnt_ -= shift;
  if (cnt_ < shift) refill();
}

unsigned RangeDecoder::decode_bool_equi() {
  const unsigned r = rng_;
  const uint64_t dif = dif_;
  /* probability 1/2: the multiply degenerates into a shift */
  unsigned split = ((r >> 8) << 7) + kMinProb;
  const uint64_t threshold = static_cast<uint64_t>(split) << (kWindowBits - 16);
  const unsigned above = dif >= threshold;
  normalize(dif - (above ? threshold : 0), above ? split : r - split);
  return 1 - above;
}

unsigned RangeDecoder::decode_bool(unsigned probability) {
  const unsigned r = rng_;
  const uint64_t dif = dif_;
  unsigned split = ((r >> 8) * (probability >> kProbShift) >> (7 - kProbShift)) + kMinProb;
  const uint64_t threshold = static_cast<uint64_t>(split) << (kWindowBits - 16);
  const unsigned above = dif >= threshold;
  normalize(dif - (above ? threshold : 0), above ? split : r - split);
  return 1 - above;
}

unsigned RangeDecoder::decode_symbol(uint16_t* cdf, int nsymbols) {
  const unsigned c = static_cast<unsigned>(dif_ >> (kWindowBits - 16));
  const unsigned r = rng_ >> 8;
  unsigned u = 0;
  unsigned v = rng_;
  int symbol = -1;
  do {
    ++symbol;
    u = v;
    v = r * (cdf[symbol] >> kProbShift);
    v >>= 7 - kProbShift;
    v += kMinProb * static_cast<unsigned>(nsymbols - symbol);
  } while (symbol < nsymbols - 1 && c < v);

  normalize(dif_ - (static_cast<uint64_t>(v) << (kWindowBits - 16)), u - v);

  if (update_cdf_) {
    const unsigned count = cdf[nsymbols];
    const unsigned rate = 4 + (count >> 4) + (nsymbols > 2 ? 1u : 0u);
    int i = 0;
    for (; i < symbol; ++i) cdf[i] = static_cast<uint16_t>(cdf[i] + ((32768 - cdf[i]) >> rate));
    for (; i < nsymbols; ++i) cdf[i] = static_cast<uint16_t>(cdf[i] - (cdf[i] >> rate));
    cdf[nsymbols] = static_cast<uint16_t>(count + (count < 32 ? 1 : 0));
  }
  return static_cast<unsigned>(symbol);
}

uint32_t RangeDecoder::read_literal(int bits) {
  uint32_t value = 0;
  for (int i = 0; i < bits; ++i) value = (value << 1) | decode_bool_equi();
  return value;
}

uint32_t RangeDecoder::read_uniform(uint32_t n) {
  if (n <= 1) return 0;
  int width = 0;
  while ((1u << width) < n) ++width;
  const uint32_t m = (1u << width) - n;
  const uint32_t value = read_literal(width - 1);
  if (value < m) return value;
  return (value << 1) - m + decode_bool_equi();
}

} /* namespace frontend */
} /* namespace av1gpu */
