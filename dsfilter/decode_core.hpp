/*
 * AV1onGPU - DirectShow decode core
 *
 * The part of the DirectShow filter that is worth testing without DirectShow:
 * it takes the byte stream a demuxer pushes (one media sample at a time, which
 * is one temporal unit for every splitter this tree has seen) and produces the
 * shown frames as NV12 byte buffers. The filter's pins wrap this class.
 *
 * The per OBU state machine is the one tools/av1gpu_decode.cpp walks: sequence
 * header into a SequenceHeader, frame OBU through the tile decoder and
 * plan_from_tile into a FrameSyntax, decode_reference, then the refresh book
 * keeping that makes the next inter frame's references resolvable. The core
 * adds byte level buffering with an incremental OBU parser, so a media sample
 * may end in the middle of an OBU.
 */
#ifndef AV1GPU_DSFILTER_DECODE_CORE_HPP
#define AV1GPU_DSFILTER_DECODE_CORE_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "frontend/obu.hpp"

namespace av1gpu {
namespace dsfilter {

/* Size of a frame's NV12 buffer for a given stride, or 0 when the geometry
 * cannot produce one (a non 4:2:0 subsampling, or a nonsensical pitch). */
size_t nv12_size(int width, int height, size_t pitch);

class StreamDecoder {
 public:
  /* One shown frame, already packed. `pitch` is the caller's row stride in
   * bytes; rows are that wide, padding included. */
  using Sink = std::function<bool(const std::vector<uint8_t>& nv12, int width,
                                  int height, std::string* error)>;

  explicit StreamDecoder(const BackendOptions& options);
  ~StreamDecoder();

  StreamDecoder(const StreamDecoder&) = delete;
  StreamDecoder& operator=(const StreamDecoder&) = delete;

  /* Creates the backend decoder. Returns false and fills `error` when no
   * usable device is present (a missing GPU with fallback disabled included). */
  bool init(std::string* error);

  const DeviceInfo& device() const;

  /* Feed one media sample. Complete OBUs are decoded as they arrive; an OBU
   * that ends inside this sample waits for the next push. Every shown frame
   * goes to `sink` in coding order; `pitch` is its row stride. Returns false
   * with `error` filled for a stream this decoder cannot follow; the sample
   * is consumed either way. */
  bool push(const uint8_t* data, size_t size, size_t pitch, const Sink& sink,
            std::string* error);

  /* Drops the buffered bytes and every reference. Call on a discontinuity. */
  void reset_stream();

  size_t shown_frames() const { return shown_; }
  size_t decoded_frames() const { return decoded_; }
  size_t dropped_frames() const { return dropped_; }
  const std::string& notice() const { return notice_; }

 private:
  bool decode_obu(frontend::ObuType type, const uint8_t* payload, size_t size,
                  size_t pitch, const Sink& sink, std::string* error);
  bool emit(const Frame& picture, size_t pitch, const Sink& sink, std::string* error);

  BackendOptions options_;
  std::unique_ptr<Decoder> decoder_;
  void* state_; /* StreamState, defined in the .cpp */

  std::vector<uint8_t> pending_; /* OBU bytes that wait for their tail */
  bool have_seq_ = false;
  std::string notice_; /* a one shot message worth surfacing once */

  size_t shown_ = 0;
  size_t decoded_ = 0;
  size_t dropped_ = 0;
};

} /* namespace dsfilter */
} /* namespace av1gpu */

#endif /* AV1GPU_DSFILTER_DECODE_CORE_HPP */
