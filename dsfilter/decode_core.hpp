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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "frontend/obu.hpp"

namespace av1gpu {
namespace dsfilter {

/* Size of a frame's NV12 buffer for a given stride, or 0 when the geometry
 * cannot produce one (a non 4:2:0 subsampling, or a nonsensical pitch). */
size_t nv12_size(int width, int height, size_t pitch);

/* One shown frame, already packed. `pitch` is the caller's row stride in
 * bytes; rows are that wide, padding included. */
using Sink = std::function<bool(const std::vector<uint8_t>& nv12, int width,
                                int height, std::string* error)>;

/* One frame the worker thread still owes a decode and (maybe) a show. The
 * main thread does everything that fixes what later frames read - tile decode,
 * reference bookkeeping, planning - and then hands the finished plan over, so
 * the GPU work of frame N overlaps the CPU prep of frame N+1. */
struct PipeJob {
  FrameSyntax syntax;      /* the frame syntax the plan was built from */
  /* Opaque pre-built frame plan (the CPU-heavy scheduler), when the caller
   * planned on its own thread so it overlaps the previous job's kernels. */
  void* frame_plan = nullptr;
  int slot = -1;           /* reserved reference slot, or -1 to drop the copy */
  int refresh_flags = 0;   /* AV1 slots this frame hands its picture to */
  bool show_frame = false; /* push the picture through the sink */
  bool show_existing = false; /* re-show an already decoded picture instead */
  int existing_idx = -1;   /* AV1 slot show_existing_frame names */
  size_t pitch = 0;
  Sink sink;
  /* Backend slots the pool dropped at this frame's plan time. Releasing them
   * is deferred to after this job's decode: a queued earlier job may still be
   * about to write its picture into one of them. */
  std::vector<int> release_slots;
};

class StreamDecoder {
 public:
  using Sink = dsfilter::Sink;

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

  /* Waits for the worker to drain every queued frame. Call at end of stream so
   * the last frames are shown before EOS is delivered downstream. */
  void finish();

  /* Drops queued frames and waits for the worker to go idle. Call on flush;
   * after this the decoder accepts a fresh stream. */
  void abort_pipe();

  size_t shown_frames() const { return shown_.load(std::memory_order_relaxed); }
  size_t decoded_frames() const { return decoded_.load(std::memory_order_relaxed); }
  size_t dropped_frames() const { return dropped_.load(std::memory_order_relaxed); }
  std::string notice() const;

 private:
  bool decode_obu(frontend::ObuType type, const uint8_t* payload, size_t size,
                  size_t pitch, const Sink& sink, std::string* error);
  bool emit(const Frame& picture, size_t pitch, const Sink& sink, std::string* error);

  void queue_job(PipeJob job, std::string* error);
  void worker_loop();
  void run_job(PipeJob* job);
  void pipe_fail(const std::string& message);

  BackendOptions options_;
  std::unique_ptr<Decoder> decoder_;
  void* state_; /* StreamState, defined in the .cpp */

  std::vector<uint8_t> pending_; /* OBU bytes that wait for their tail */
  bool have_seq_ = false;

  /* The frame pipeline. `pipe_picture_`/`pipe_valid_` are the decoded-picture
   * half of the reference pool (for show_existing_frame); both the main thread
   * (synchronous fallback) and the worker write them under `pipe_mutex_`. */
  std::thread worker_;
  mutable std::mutex pipe_mutex_;
  std::condition_variable pipe_cv_;    /* queue space or drain */
  std::condition_variable worker_cv_;  /* wake the worker */
  std::deque<PipeJob> jobs_;
  /* Back pressure: a bounded queue keeps the decoder a few frames behind the
   * demuxer instead of planning the whole file into memory. Six frames of
   * slack is what lets the pipeline run ahead during simple scenes and hide a
   * keyframe's cost at a scene cut: with three, every GPU or scheduler spike
   * reached the renderer as a visible hitch. */
  static constexpr size_t kPipeDepth = 3;
  bool pipelined_ = false;   /* frames go through the queue */
  bool preplan_ = false;     /* the decoder can pre-build frame plans */
  bool fallback_used_ = false; /* a failed reservation switched us to sync */
  bool quitting_ = false;    /* worker must exit */
  bool busy_ = false;        /* the worker is inside a job */
  bool pipe_failed_ = false; /* a queued job failed; the stream is broken */
  Frame pipe_picture_[8];
  bool pipe_valid_[8] = {false, false, false, false, false,
                         false, false, false};

  std::string notice_; /* a one shot message worth surfacing once */

  std::atomic<size_t> shown_{0};
  std::atomic<size_t> decoded_{0};
  std::atomic<size_t> dropped_{0};
};

} /* namespace dsfilter */
} /* namespace av1gpu */

#endif /* AV1GPU_DSFILTER_DECODE_CORE_HPP */
