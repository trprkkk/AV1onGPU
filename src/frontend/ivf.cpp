#include "frontend/ivf.hpp"

#include <cstdio>
#include <cstring>

namespace av1gpu {
namespace frontend {
namespace {

uint32_t read_u32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

uint64_t read_u64(const uint8_t* data) {
  return static_cast<uint64_t>(read_u32(data)) |
         (static_cast<uint64_t>(read_u32(data + 4)) << 32);
}

constexpr size_t kFileHeaderSize = 32;
constexpr size_t kFrameHeaderSize = 12;

} /* namespace */

bool IvfReader::open(const std::string& path, std::string* error) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  std::vector<uint8_t> data;
  uint8_t chunk[16384];
  size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
    data.insert(data.end(), chunk, chunk + got);
  const bool failed = std::ferror(file) != 0;
  std::fclose(file);
  if (failed) {
    if (error) *error = "cannot read " + path;
    return false;
  }
  return open(data, error);
}

bool IvfReader::open(const std::vector<uint8_t>& file, std::string* error) {
  file_ = file;
  pos_ = kFileHeaderSize;
  frames_read_ = 0;
  error_.clear();
  header_ = IvfHeader();

  if (file_.size() < kFileHeaderSize) {
    if (error) *error = "file is too small to be an IVF stream";
    return false;
  }
  if (std::memcmp(file_.data(), "DKIF", 4) != 0) {
    if (error) *error = "missing IVF signature (expected \"DKIF\")";
    return false;
  }
  header_.version = file_[4] | (file_[5] << 8);
  header_.header_size = file_[6] | (file_[7] << 8);
  std::memcpy(header_.fourcc, file_.data() + 8, 4);
  header_.fourcc[4] = '\0';
  header_.width = static_cast<int>(file_[12] | (file_[13] << 8));
  header_.height = static_cast<int>(file_[14] | (file_[15] << 8));
  header_.timebase_num = read_u32(file_.data() + 16);
  header_.timebase_den = read_u32(file_.data() + 20);
  header_.frame_count = read_u32(file_.data() + 24);

  if (header_.version != 0) {
    if (error) *error = "unsupported IVF version " + std::to_string(header_.version);
    return false;
  }
  if (header_.header_size != kFileHeaderSize) {
    if (error) {
      *error = "unsupported IVF header size " + std::to_string(header_.header_size);
    }
    return false;
  }
  if (std::strcmp(header_.fourcc, "AV01") != 0) {
    if (error) {
      *error = std::string("IVF stream is ") + header_.fourcc + ", not AV01";
    }
    return false;
  }
  return true;
}

bool IvfReader::next(IvfFrame* frame, std::string* error) {
  if (pos_ >= file_.size()) return false; /* clean end of stream */
  if (file_.size() - pos_ < kFrameHeaderSize) {
    if (error) *error = "truncated IVF frame header";
    return false;
  }
  const uint32_t size = read_u32(file_.data() + pos_);
  const uint64_t timestamp = read_u64(file_.data() + pos_ + 4);
  pos_ += kFrameHeaderSize;
  if (size > file_.size() - pos_) {
    if (error) {
      *error = "IVF frame of " + std::to_string(size) + " bytes runs past the end";
    }
    return false;
  }
  frame->payload.assign(file_.begin() + static_cast<ptrdiff_t>(pos_),
                        file_.begin() + static_cast<ptrdiff_t>(pos_ + size));
  frame->timestamp = timestamp;
  pos_ += size;
  ++frames_read_;
  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */
