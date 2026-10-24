/*
 * AV1onGPU - AV1 arithmetic (boolean) decoder
 *
 * Everything in a tile payload is coded with AV1's range coder: the algorithm
 * the specification calls the boolean decoder (section 8.2), the one libaom
 * ships as `od_ec_dec` and dav1d as `msac`. Unlike the literal bit reader in
 * bitreader.hpp, this one narrows a 16 bit range by the probability of each
 * symbol, so the encoder can spend fewer than one bit on likely symbols.
 *
 * CDFs
 *   A symbol alphabet is described by a 16 bit *inverse* cumulative
 *   distribution in Q15: cdf[i] is 32768 minus the cumulative frequency of the
 *   symbols before i, so cdf[] decreases towards 0 at the last symbol, and
 *   cdf[nsymbols] holds the adaptation counter (0..32), not a probability. That
 *   is the layout the specification uses, so the default tables can be dropped
 *   in unchanged.
 *
 *   When `update_cdf` is enabled the decoder adapts the table after every
 *   symbol, so encoder and decoder walk the same CDF evolution and a corrupted
 *   stream diverges instead of silently decoding wrong pixels.
 *
 * The arithmetic is written to be bit identical to dav1d's `msac` and libaom's
 * `od_ec`: EC_PROB_SHIFT is 6, EC_MIN_PROB is 4, the window is 64 bit and the
 * range is renormalised into [32768, 65536). tests/test_entropy.cpp pins it
 * against symbols decoded by dav1d itself.
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

  /* Decodes one equiprobable bit. */
  unsigned read_bit() { return decode_bool_equi(); }

  /* Decodes a bit whose probability of being 1 is `probability / 32768`. */
  unsigned read_bool(unsigned probability) { return decode_bool(probability); }

  /* Decodes a symbol from `cdf`, which must hold `nsymbols` probabilities plus
   * the adaptation counter (nsymbols + 1 entries in total), and adapts the
   * table in place when updating is enabled. A result equal to nsymbols
   * means the stream or the table is corrupt: the range coder could satisfy no
   * symbol at all; in that case failed() becomes true and no table lookup can
   * be trusted afterwards. */
  unsigned read_symbol(uint16_t* cdf, int nsymbols) { return decode_symbol(cdf, nsymbols); }

  /* L(n): n equiprobable bits, most significant first. */
  uint32_t read_literal(int bits);

  /* ns(n): a uniform value in [0, n), the AV1 non symmetric code. */
  uint32_t read_uniform(uint32_t n);

  /* True once the decoder has reached the end of the tile data and is filling
   * its window with padding (missing bytes read as ones, which is what the
   * specification prescribes). The window looks seven bytes ahead, so this
   * becomes true slightly before the last coded symbol and is a "this tile is
   * exhausted" signal - a truncated tile sets it too - rather than per symbol
   * error detection. Two consequences for callers: a tile shorter than seven
   * bytes reports it immediately, and the check that actually catches a
   * desynchronised tile is the trailing bit position at the end of the tile,
   * not this flag. */
  bool overrun() const { return overrun_; }

  /* True if the stream is broken badly enough that decoding cannot continue: a
   * symbol fell outside every CDF interval, or a range collapsed to zero. */
  bool failed() const { return failed_; }

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
  bool failed_ = false;
};

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_RANGE_DECODER_HPP */
