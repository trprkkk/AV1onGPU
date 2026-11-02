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
  /* Frames that were neither shown nor stored in the reference pool, and so
   * were not decoded at all. */
  size_t skipped_invisible_frames() const {
    return skipped_invisible_.load(std::memory_order_relaxed);
  }
  std::string notice() const;

 private:
  bool decode_obu(frontend::ObuType type, const uint8_t* payload, size_t size,
                  size_t pitch, const Sink& sink, std::string* error);
  bool emit(const Frame& picture, size_t pitch, const Sink& sink, std::string* error);

  void queue_job(PipeJob job, std::string* error, uint64_t generation = 0);
  /* Starts the pipeline's threads if they are not running yet. */
  void ensure_threads();
  void planner_loop();
  void frontend_loop();
  void worker_loop();
  void output_loop();
  /* Hands a decoded picture to the output stage: the worker starts the next
   * frame's kernels while this one is packed into NV12 and shown. */
  void queue_output(Frame picture, size_t pitch, const Sink& sink);
  /* Hands a reserved backend slot back to the pool and counts it out. */
  void release_reserved(int slot);
  /* Hands a frame's frontend work (the per block syntax) to the frontend
   * thread. The task runs in order, so the frames reach the planner in the
   * order they were parsed. */
  void submit_frontend_task(std::function<void()> task, int slot);
  void run_job(PipeJob* job);
  void pipe_fail(const std::string& message);

  BackendOptions options_;
  std::unique_ptr<Decoder> decoder_;
  void* state_; /* StreamState, defined in the .cpp */

  /* push() runs on the demuxer's streaming thread; reset_stream() on whatever
   * thread delivers the seek (EndFlush/NewSegment/Stop). They share pending_,
   * the StreamState and the decoder's slot bookkeeping, so a dedicated mutex
   * serializes them: without it a seek racing a Receive corrupts the heap
   * (seen as 0xC0000374 in the player). Lock order is always stream_mutex_
   * before pipe_mutex_, never the other way round. */
  std::mutex stream_mutex_;
  std::vector<uint8_t> pending_; /* OBU bytes that wait for their tail */
  bool have_seq_ = false;

  /* The frame pipeline is three stages: the demuxer's thread parses and plans
   * the syntax (jobs_), a planner thread turns the syntax into the CPU-heavy
   * frame plan (ready_jobs_), and the worker runs the kernels and emits. The
   * planner stage is what overlaps frame N's plan build with frame N-1's GPU
   * work; without it the worker paid the two back to back. `pipe_picture_`/
   * `pipe_valid_` are the decoded-picture half of the reference pool (for
   * show_existing_frame); both the main thread (synchronous fallback) and the
   * worker write them under `pipe_mutex_`. */
  std::thread planner_;
  std::thread worker_;
  /* The frontend stage: the demuxer parses a frame (the tiles are decoded and
   * the reference pool is updated - that part is a chain, because a frame's
   * motion vectors are what the next frame predicts from), and then hands the
   * frame over. Building the per block syntax - the frontend plan - depends on
   * nothing but the frame itself, and it is the demuxer's single biggest cost
   * (13 ms a frame), so it runs here: the demuxer parses the next frame while
   * this one is being planned, and the planner and the GPU pick it up when it
   * is ready. The queue is FIFO, so the frames stay in order. */
  std::thread frontend_;
  mutable std::mutex front_mutex_;
  std::condition_variable front_cv_;    /* a new task arrived */
  std::condition_variable front_idle_;  /* the frontend thread ran dry */
  std::deque<std::function<void()>> front_tasks_;
  /* The backend slot each queued frame reserved, so a flush can hand the
   * slots of the frames it drops back to the pool. */
  std::deque<int> front_slots_;
  bool front_busy_ = false;   /* the frontend thread is inside a task */
  bool front_quit_ = false;   /* the frontend thread must exit */
  /* The output stage: packing a picture into NV12 and handing it to the
   * filter costs 2.8 ms a frame at 1080p, and it is pure host work that does
   * not need the GPU - the worker can start the next frame's kernels while it
   * runs. The queue is FIFO and the thread is single, so the pictures are
   * shown in coding order. */
  struct OutputJob {
    Frame picture;
    size_t pitch = 0;
    Sink sink;
  };
  std::thread output_;
  std::deque<OutputJob> out_jobs_;
  bool output_busy_ = false;  /* the output thread is inside a job */
  bool output_quit_ = false;  /* the output thread must exit */
  /* Bumped by abort_pipe(): a task that was queued before the flush belongs
   * to the old stream and must not reach the worker. */
  uint64_t generation_ = 0;
  mutable std::mutex pipe_mutex_;
  std::condition_variable pipe_cv_;     /* queue space or drain */
  std::condition_variable planner_cv_;  /* wake the planner */
  std::condition_variable worker_cv_;   /* wake the worker */
  std::condition_variable space_cv_;    /* a reserved slot came back */
  std::deque<PipeJob> jobs_;       /* parsed, waiting for the planner */
  std::deque<PipeJob> ready_jobs_; /* planned, waiting for the worker */
  /* Back pressure: a bounded number of in-flight jobs (parsed + planned) keeps
   * the decoder a few frames behind the demuxer instead of planning the whole
   * file into memory. The bound has to cover both queues (see queue_job), and
   * it also has to stay under kReferenceSlots - 8: each in-flight kept frame
   * holds one reserved backend slot on top of the eight the reference pool can
   * name, and running out is what triggers the reserve-fail drain. */
  static constexpr size_t kPipeDepth = 12;
  /* The backend's slot pool is the hard limit on how far the decoder may run
   * ahead of the GPU: every frame that is parsed but not decoded yet holds a
   * reserved slot, and the reference pool holds eight more. The pipeline has
   * three queues in front of the worker (the frontend queue, the planner's and
   * the worker's), so the number of reserved slots has to be counted
   * explicitly - bounding the queues separately let all three fill up at once
   * and the pool ran dry, which deadlocked the pipeline (the worker waited for
   * a slot, the demuxer waited for the queue, the queue waited for the
   * worker). */
  static constexpr size_t kMaxReservedSlots = 12;
  size_t reserved_slots_ = 0;  /* slots reserved and not handed back yet */
  bool pipelined_ = false;   /* frames go through the queue */
  bool plan_thread_ = true;  /* the planner thread builds the plans */
  bool skip_invisible_ = true; /* frames nobody shows or references are skipped */
  bool preplan_ = false;     /* the decoder can pre-build frame plans */
  bool fallback_used_ = false; /* a failed reservation switched us to sync */
  bool quitting_ = false;    /* threads must exit once the queues are empty */
  bool busy_ = false;        /* the worker is inside a job */
  bool planner_busy_ = false; /* the planner is inside a job */
  bool pipe_failed_ = false; /* a queued job failed; the stream is broken */
  Frame pipe_picture_[8];
  bool pipe_valid_[8] = {false, false, false, false, false,
                         false, false, false};

  std::string notice_; /* a one shot message worth surfacing once */

  std::atomic<size_t> shown_{0};
  std::atomic<size_t> decoded_{0};
  std::atomic<size_t> dropped_{0};
  std::atomic<size_t> skipped_invisible_{0};
};

} /* namespace dsfilter */
} /* namespace av1gpu */

#endif /* AV1GPU_DSFILTER_DECODE_CORE_HPP */
