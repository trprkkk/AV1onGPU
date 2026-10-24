#include "frontend/range_decoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace av1gpu {
namespace frontend {
namespace {

constexpr int kWindowBits = 64;
/* Position of the low probability bits: the reference decoders trade a little
 * precision (6 bits) for a cheaper multiply. Changing this changes the
 * bitstream, not just the speed. */
constexpr int kProbShift = 6;
constexpr unsigned kMinProb = 4;

/* Symbol level tracing, off unless AV1GPU_TRACE_SYM is set.
 *
 * The format is the reference decoder's: one line per decoded symbol, with the
 * table it used and the decoder state afterwards, so a trace of this decoder
 * can be diffed against a trace of dav1d symbol by symbol. That is the only
 * practical way to find where two implementations of the same tile start to
 * disagree, because a desynchronised range coder produces plausible symbols
 * for thousands of blocks before anything looks wrong. */
FILE* trace_file() {
  static FILE* file = nullptr;
  static bool tried = false;
  if (!tried) {
    const char* name = std::getenv("AV1GPU_TRACE_SYM");
    if (name) file = std::strcmp(name, "-") == 0 ? stdout : std::fopen(name, "w");
    tried = true;
  }
  return file;
}

void trace_bool(const char* tag, unsigned bit, unsigned probability) {
  FILE* file = trace_file();
  if (!file) return;
  if (probability) {
    std::fprintf(file, "%s %u %u\n", tag, bit, probability);
  } else {
    std::fprintf(file, "%s %u\n", tag, bit);
  }
}

void trace_state(uint64_t dif, unsigned range, int count) {
  FILE* file = trace_file();
  if (!file) return;
  std::fprintf(file, "D %016llx %u %d\n", static_cast<unsigned long long>(dif), range, count);
}

void trace_symbol(unsigned value, const uint16_t* cdf, int nsymbols) {
  FILE* file = trace_file();
  if (!file) return;
  std::fprintf(file, "S %u %d [%u]", value, nsymbols, cdf[nsymbols]);
  for (int i = 0; i < nsymbols; ++i) std::fprintf(file, " %u", cdf[i]);
  std::fprintf(file, "\n");
}

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
  failed_ = false;
  refill();
  if (trace_file()) std::fprintf(trace_file(), "I\n");
}

/* Pulls bytes into the window. The bytes are complemented on the way in, and
 * when the tile data runs out the missing bits read as ones, which is what the
 * specification prescribes. */
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
  if (rng == 0 || failed_) {
    failed_ = true;
    return;
  }
  const int shift = 16 - bit_length(rng);
  const int old_cnt = cnt_;
  dif_ = dif << shift;
  rng_ = rng << shift;
  cnt_ = old_cnt - shift;
  /* The unsigned comparison skips the refill when the counter is already
   * saturated, exactly like the reference decoders. */
  if (static_cast<unsigned>(old_cnt) < static_cast<unsigned>(shift)) refill();
}

unsigned RangeDecoder::decode_bool_equi() {
  if (failed_) return 0;
  const unsigned r = rng_;
  const uint64_t dif = dif_;
  /* probability 1/2: the multiply degenerates into a shift */
  unsigned v = ((r >> 8) << 7) + kMinProb;
  const uint64_t threshold = static_cast<uint64_t>(v) << (kWindowBits - 16);
  const unsigned above = dif >= threshold;
  if (above) {
    v += r - 2 * v;
    normalize(dif - threshold, v);
    hash_symbol(0);
    trace_bool("E", 0, 0);
    trace_state(dif_, rng_, cnt_);
    return 0;
  }
  normalize(dif, v);
  hash_symbol(1);
  trace_bool("E", 1, 0);
  trace_state(dif_, rng_, cnt_);
  return 1;
}

unsigned RangeDecoder::decode_bool(unsigned probability) {
  if (failed_) return 0;
  const unsigned r = rng_;
  const uint64_t dif = dif_;
  unsigned v = ((r >> 8) * (probability >> kProbShift) >> (7 - kProbShift)) + kMinProb;
  const uint64_t threshold = static_cast<uint64_t>(v) << (kWindowBits - 16);
  const unsigned above = dif >= threshold;
  if (above) {
    v += r - 2 * v;
    normalize(dif - threshold, v);
    hash_symbol(0);
    trace_bool("B", 0, probability);
    trace_state(dif_, rng_, cnt_);
    return 0;
  }
  normalize(dif, v);
  hash_symbol(1);
  trace_bool("B", 1, probability);
  trace_state(dif_, rng_, cnt_);
  return 1;
}

unsigned RangeDecoder::decode_symbol(uint16_t* cdf, int nsymbols) {
  if (failed_) return 0;
  const unsigned c = static_cast<unsigned>(dif_ >> (kWindowBits - 16));
  const unsigned r = rng_ >> 8;
  /* the trace shows the table as it was before this symbol adapted it */
  uint16_t before[kMaxCdfSymbols + 1] = {0};
  const bool tracing = trace_file() != nullptr;
  if (tracing) {
    for (int i = 0; i <= nsymbols; ++i) before[i] = cdf[i];
  }
  unsigned u = 0;
  unsigned v = rng_;
  int symbol = 0;
  while (true) {
    u = v;
    /* Every interval keeps at least kMinProb per remaining symbol of range;
     * without the term a very skewed table can produce a zero width interval
     * and the range coder collapses. The term is zero for the last symbol. */
    v = r * (cdf[symbol] >> kProbShift);
    v >>= 7 - kProbShift;
    v += kMinProb * static_cast<unsigned>(nsymbols - symbol);
    if (c >= v) break;
    if (symbol == nsymbols) {
      /* The counter entry is not a probability: a valid one is at most 32, so
       * it contributes no range and always terminates the loop above. Getting
       * here means the table is not an AV1 CDF, and no symbol can be trusted. */
      if (std::getenv("AV1GPU_TRACE"))
        std::fprintf(stderr, "CDF-FAIL nsymbols=%d first=%u cdf=%p counter=%u\n", nsymbols,
                     cdf[0], static_cast<const void*>(cdf), cdf[nsymbols]);
      failed_ = true;
      return 0;
    }
    ++symbol;
  }

  normalize(dif_ - (static_cast<uint64_t>(v) << (kWindowBits - 16)), u - v);
  hash_symbol(static_cast<unsigned>(symbol));

  if (tracing) {
    trace_symbol(static_cast<unsigned>(symbol), before, nsymbols);
    trace_state(dif_, rng_, cnt_);
  }

  if (update_cdf_) {
    const unsigned count = cdf[nsymbols];
    const unsigned rate = 4 + (count >> 4) + (nsymbols > 2 ? 1u : 0u);
    int i = 0;
    for (; i < symbol; ++i) {
      cdf[i] = static_cast<uint16_t>(cdf[i] + ((32768 - cdf[i]) >> rate));
    }
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

/* ns(): FloorLog2(n) + 1 equiprobable bits, with the top of the range folded
 * back so every value below n is equally likely. */
uint32_t RangeDecoder::read_uniform(uint32_t n) {
  if (n <= 1) return 0;
  int width = 0;
  while ((1u << (width + 1)) <= n) ++width;
  ++width; /* FloorLog2(n) + 1 */
  const uint32_t m = (1u << width) - n;
  const uint32_t value = read_literal(width - 1);
  if (value < m) return value;
  return (value << 1) - m + decode_bool_equi();
}

} /* namespace frontend */
} /* namespace av1gpu */
