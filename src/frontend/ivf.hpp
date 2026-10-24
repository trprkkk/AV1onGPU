/*
 * AV1onGPU - IVF container demuxer
 *
 * IVF is not a standard, it is the little container the AV1 test tools use: a
 * 32 byte file header ("DKIF", version, header length, fourcc, size, timebase,
 * frame count) followed by frames, each with a 12 byte header (size, timestamp)
 * and a payload that is one temporal unit of OBUs.
 *
 * The whole file is loaded into memory. That is fine for test clips and keeps
 * the reader free of I/O state; a player that streams would replace this with a
 * small parser over its own reader.
 */
#ifndef AV1GPU_FRONTEND_IVF_HPP
#define AV1GPU_FRONTEND_IVF_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av1gpu {
namespace frontend {

struct IvfHeader {
  int version = 0;
  int header_size = 0;
  char fourcc[5] = {0, 0, 0, 0, 0};
  int width = 0;
  int height = 0;
  uint32_t timebase_num = 0;
  uint32_t timebase_den = 0;
  uint32_t frame_count = 0;
};

struct IvfFrame {
  std::vector<uint8_t> payload;
  uint64_t timestamp = 0;
};

class IvfReader {
 public:
  /* Loads and validates the header. Returns false with a message on error. */
  bool open(const std::string& path, std::string* error);
  bool open(const std::vector<uint8_t>& file, std::string* error);

  const IvfHeader& header() const { return header_; }
  const std::string& error() const { return error_; }

  /* Reads the next frame. Returns false at the end of the stream or on error;
   * `at_end()` tells the two apart. */
  bool next(IvfFrame* frame, std::string* error);
  bool at_end() const { return pos_ >= file_.size(); }
  size_t frames_read() const { return frames_read_; }

 private:
  std::vector<uint8_t> file_;
  size_t pos_ = 0;
  size_t frames_read_ = 0;
  IvfHeader header_;
  std::string error_;
};

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_IVF_HPP */
