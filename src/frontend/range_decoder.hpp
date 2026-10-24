/*
 * AV1onGPU - AV1 arithmetic (boolean) decoder
 *
 * Everything in a tile payload is coded with AV1's range coder: the same
 * algorithm the specification calls the boolean decoder (section 8.2), the one
 * libaom ships as `od_ec_dec` and dav1d as `msac`. Unlike the literal bit reader
 * in bitreader.hpp, this one narrows a 16 bit range by the probability of each
 * symbol, so the encoder can spend fewer than one bit on likely symbols.
 *
 * CDFs
 *   A symbol alphabet is described by a 16 bit *inverse* cumulative
 *   distribution in Q15: cdf[i] is 32768 minus the cumulative frequency of the
 *   symbols before i, so cdf[] decreases and cdf[nsymbols] holds the adaptation
 *   counter (0..32), not a probability. That is exactly the layout the
 *   specification uses, so the default tables can be dropped in unchanged.
 *
 *   When `update_cdf` is enabled the decoder adapts the table after every
 *   symbol, which means encoder and decoder walk the same CDF evolution, and a
 *   corrupted stream diverges instead of silently decoding wrong pixels.
 *
 * The implementation follows the range scaling of the reference decoders
 * (`EC_PROB_SHIFT` 6, `EC_MIN_PROB` 4, a 64 bit window) so that it produces
 * bit-identical results; tests/test_range_coder.cpp pins it against symbols
 * produced by dav1d.
 */
#ifndef AV1GPU_FRONTEND_RANGE_DECODER_HPP
#define AV1GPU_FRONTEND_RANGE_DECODER_HPP

#include <cstddef>
#include <cstdint>

namespace av1gpu {
namespace frontend {

/* Q15 scale of the probability tables, and the largest symbol count a CDF may
 * describe (AV1 never codes more than 16 symbols at once). */
constexpr int kCdfScale = 32768;
constexpr int kMaxCdfSymbols = 16;

class RangeDecoder {
 public:
  RangeDecoder() = default;

  /* Resets the decoder over `size` bytes of tile data. `update_cdf` must match
   * the frame's disable_cdf_update flag. */
  void init(const uint8_t* data, size_t size, bool update_cdf = true);

  /* Decodes one equiprobable bit: `read_bit()` in the specification. */
  unsigned read_bit() { return decode_bool_equi(); }

  /* Decodes a bit whose probability of being 1 is `probability / 32768`. */
  unsigned read_bool(unsigned probability) { return decode_bool(probability); }

  /* Decodes a symbol from `cdf` (kMaxCdfSymbols entries plus the counter) and
   * adapts the table in place when updating is enabled. */
  unsigned read_symbol(uint16_t* cdf, int nsymbols) { return decode_symbol(cdf, nsymbols); }

  /* L(n): n equiprobable bits, most significant first. */
  uint32_t read_literal(int bits);

  /* ns(n): uniform value in [0, n), the AV1 "non symmetric" code. */
  uint32_t read_uniform(uint32_t n);

  /* True once the decoder has consumed the padding beyond the end of the tile
   * data. Decoding keeps returning zero bits, so callers check this at the end
   * of a tile instead of at every symbol. */
  bool overrun() const { return overrun_; }

  /* Bytes consumed so far, counting the lookahead window. */
  size_t bytes_used() const { return static_cast<size_t>(buf_pos_ - buf_); }

 private:
  unsigned decode_bool_equi();
  unsigned decode_bool(unsigned probability);
  unsigned decode_symbol(uint16_t* cdf, int nsymbols);
  void refill();
  void normalize(uint64_t dif, unsigned rng);

  const uint8_t* buf_ = nullptr;
  const uint8_t* buf_pos_ = nullptr;
  const uint8_t* buf_end_ = nullptr;
  uint64_t dif_ = 0;
  unsigned rng_ = 0;
  int cnt_ = 0;
  bool update_cdf_ = true;
  bool overrun_ = false;
};

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_RANGE_DECODER_HPP */
