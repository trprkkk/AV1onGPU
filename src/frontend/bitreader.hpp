/*
 * AV1onGPU - literal bit reader
 *
 * AV1 codes everything outside the tile data with plain, uncompressed bits:
 * the sequence header and the frame header are read with `f(n)` (n bits, most
 * significant first), `su(n)` (two's complement style signed) and `ns(n)`
 * (uniform, non power of two). Only the tile payload needs an arithmetic
 * decoder, so the front-end starts here.
 *
 * Reads past the end of the buffer do not throw: they return zero and set
 * `overrun()`, which the callers check once at the end. That keeps the parsing
 * code free of error plumbing at every field and still makes truncated input a
 * definite failure.
 */
#ifndef AV1GPU_FRONTEND_BITREADER_HPP
#define AV1GPU_FRONTEND_BITREADER_HPP

#include <cstddef>
#include <cstdint>

namespace av1gpu {
namespace frontend {

class BitReader {
 public:
  BitReader() = default;
  BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  void reset(const uint8_t* data, size_t size) {
    data_ = data;
    size_ = size;
    bits_ = 0;
    overrun_ = false;
  }

  /* f(1) */
  bool read_bit() {
    if (bits_ >= size_ * 8) {
      overrun_ = true;
      return false;
    }
    const size_t byte = bits_ >> 3;
    const int shift = 7 - static_cast<int>(bits_ & 7);
    ++bits_;
    return ((data_[byte] >> shift) & 1) != 0;
  }

  /* f(n): n literal bits, most significant first. */
  uint32_t read(int n) {
    if (n <= 0) return 0;
    if (n > 32) {
      overrun_ = true;
      return 0;
    }
    uint32_t value = 0;
    for (int i = 0; i < n; ++i) value = (value << 1) | (read_bit() ? 1u : 0u);
    return value;
  }

  /* su(n): f(n) reinterpreted as a signed value in [-2^(n-1), 2^(n-1)-1]. */
  int32_t read_signed(int n) {
    if (n <= 0 || n > 32) {
      overrun_ = true;
      return 0;
    }
    const uint32_t value = read(n);
    if (n == 32) return static_cast<int32_t>(value);
    const uint32_t sign = 1u << (n - 1);
    return (value & sign) ? static_cast<int32_t>(value) - static_cast<int32_t>(sign << 1)
                          : static_cast<int32_t>(value);
  }

  /* ns(n): uniform value in [0, n), n >= 1. */
  uint32_t read_uniform(uint32_t n) {
    if (n <= 1) return 0;
    int w = 0;
    while ((1u << w) < n) ++w; /* w = ceil(log2(n)) */
    const uint32_t m = (1u << w) - n;
    const uint32_t value = read(w - 1);
    if (value < m) return value;
    return (value << 1) - m + (read_bit() ? 1u : 0u);
  }

  /* uvlc(): exponential golomb with a unary prefix, as used by AV1 for
   * variable length unsigned values. Returns 0xffffffff on an invalid code. */
  uint32_t read_uvlc() {
    int leading_zeros = 0;
    while (!read_bit()) {
      if (overrun_ || ++leading_zeros >= 32) return 0xffffffffu;
    }
    if (leading_zeros >= 32) return 0xffffffffu;
    return read(leading_zeros) + (1u << leading_zeros) - 1u;
  }

  /* leb128, as used for OBU sizes and the metadata OBUs. */
  uint64_t read_uleb128() {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      const uint64_t byte = read(8);
      value |= (byte & 0x7f) << (7 * i);
      if (!(byte & 0x80)) return value;
    }
    overrun_ = true; /* more than eight bytes is not a valid leb128 */
    return 0;
  }

  void skip(size_t count) {
    if (bits_ + count > size_ * 8) {
      overrun_ = true;
      bits_ = size_ * 8;
      return;
    }
    bits_ += count;
  }

  /* Advances to the next byte boundary and returns the bits that were skipped. */
  uint32_t align_to_byte() {
    const uint32_t padding = static_cast<uint32_t>((8 - (bits_ & 7)) & 7);
    if (padding) skip(padding);
    return padding;
  }

  bool overrun() const { return overrun_; }
  size_t bits_read() const { return bits_; }
  size_t bytes_read() const { return (bits_ + 7) / 8; }
  size_t bits_left() const { return bits_ < size_ * 8 ? size_ * 8 - bits_ : 0; }
  size_t size() const { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  size_t bits_ = 0;
  bool overrun_ = false;
};

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_BITREADER_HPP */
