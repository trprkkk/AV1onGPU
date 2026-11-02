/*
 * AV1onGPU - DirectShow decode core
 *
 * See decode_core.hpp. The decode state is the exact per OBU walk of
 * tools/av1gpu_decode.cpp: the same pool book keeping that keeps the backend's
 * reference slots alive only as long as an AV1 slot names them, and the same
 * RefPool updates that let the next frame resolve its references.
 */
#include "decode_core.hpp"
#include <chrono>
#include <cstdio>
#include <cstdarg>
#define NOMINMAX
#include <windows.h>

void debug_log(const char* fmt, ...);

inline bool ds_debug_enabled() {
  /* cached: this is called on the per-frame hot path, and walking the
   * environment block per call showed up in the pipeline timings */
  static const bool enabled = [] {
    char flag[8] = {0};
    const DWORD n = GetEnvironmentVariableA("AV1GPU_DS_DEBUG", flag, sizeof(flag));
    return n > 0 && n < sizeof(flag) && flag[0] != '0';
  }();
  return enabled;
}

void ds_debug_log(const char* fmt, ...) {
  static const bool enabled = [] {
    char flag[8] = {0};
    const DWORD n = GetEnvironmentVariableA("AV1GPU_DS_DEBUG", flag, sizeof(flag));
    return n > 0 && n < sizeof(flag) && flag[0] != '0';
  }();
  if (!enabled) return;
  char line[512];
  va_list args;
  va_start(args, fmt);
  _vsnprintf_s(line, sizeof(line), _TRUNCATE, fmt, args);
  va_end(args);
  line[sizeof(line) - 1] = 0;
  char path[MAX_PATH];
  if (GetTempPathA(MAX_PATH, path)) {
    strncat_s(path, "av1gpu_ds.log", _TRUNCATE);
    if (FILE* f = std::fopen(path, "a")) {
      SYSTEMTIME st;
      GetLocalTime(&st);
      std::fprintf(f, "%02u:%02u:%02u.%03u pid %lu thr %lu  %s\n", st.wHour,
                   st.wMinute, st.wSecond, st.wMilliseconds, GetCurrentProcessId(),
                   GetCurrentThreadId(), line);
      std::fclose(f);
    }
  }
}


#include <cstring>

#include "core/backend.hpp"
#include "frontend/dequant.hpp"
#include "frontend/headers.hpp"
#include "frontend/reconstruct.hpp"
#include "frontend/tile_decoder.hpp"
#include "frontend/tiles.hpp"

namespace av1gpu {
namespace dsfilter {

namespace {

/* Everything the per OBU state machine carries from one OBU to the next. */
struct StreamState {
  frontend::SequenceHeader seq;
  frontend::RefPool refs;

  /* The decoder's reference pool, exactly as av1gpu_decode keeps it: each AV1
   * slot names a backend pool slot. The decoded pictures themselves live in
   * the StreamDecoder's pipe arrays, because the worker fills them after the
   * main thread has already moved on to the next frame's plan. A stream may
   * refresh several AV1 slots with one picture, so a backend slot is released
   * only when no AV1 slot names it any more. */
  int pool_slot[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

  void clear() {
    refs = frontend::RefPool();
    for (int i = 0; i < 8; ++i) {
      pool_slot[i] = -1;
    }
  }
};

/* What the incremental OBU header parse says about the buffer's front. */
enum class ObuPeek { NeedMore, Ok, Invalid };

/* Parses the fixed part of the next OBU at `pos` without consuming payload.
 * Mirrors src/frontend/obu.cpp's parse_obu_header, but reports "the buffer
 * ends inside this OBU" as NeedMore instead of an error, which is what a
 * stream fed one media sample at a time needs. */
ObuPeek peek_obu(const uint8_t* data, size_t size, size_t* pos,
                 frontend::ObuType* type, bool* has_size, size_t* payload_size) {
  if (*pos >= size) return ObuPeek::NeedMore;
  const uint8_t byte = data[(*pos)++];
  if (byte & 0x80) return ObuPeek::Invalid; /* forbidden bit: not an AV1 stream */
  if (!frontend::obu_type_from_int((byte >> 3) & 0x0f, type)) {
    return ObuPeek::Invalid; /* reserved type */
  }
  const bool has_extension = (byte >> 2) & 1;
  *has_size = (byte >> 1) & 1;
  if (byte & 1) return ObuPeek::Invalid; /* reserved bit */
  if (has_extension) {
    if (*pos >= size) return ObuPeek::NeedMore;
    const uint8_t extension = data[(*pos)++];
    if (extension & 0x07) return ObuPeek::Invalid; /* reserved extension bits */
  }
  if (!*has_size) {
    /* low overhead bitstream: the OBU runs to the end of the buffer, which
     * only the last OBU of a unit does */
    *payload_size = size - *pos;
    return ObuPeek::Ok;
  }
  uint64_t declared = 0;
  for (int i = 0; i < 8; ++i) {
    if (*pos >= size) return ObuPeek::NeedMore; /* leb128 still arriving */
    const uint8_t leb = data[(*pos)++];
    declared |= static_cast<uint64_t>(leb & 0x7f) << (7 * i);
    if (!(leb & 0x80)) {
      if (declared > (1ull << 32)) return ObuPeek::Invalid;
      *payload_size = static_cast<size_t>(declared);
      return *pos + *payload_size > size ? ObuPeek::NeedMore : ObuPeek::Ok;
    }
  }
  return ObuPeek::Invalid; /* leb128 longer than eight bytes */
}

/* Releases a backend slot unless some AV1 slot still names it. */
void release_slot(StreamState* st, Decoder* decoder, int slot) {
  if (slot < 0) return;
  for (int i = 0; i < 8; ++i)
    if (st->pool_slot[i] == slot) return;
  decoder->release_reference(slot);
}

} /* namespace */

size_t nv12_size(int width, int height, size_t pitch) {
  if (width <= 0 || height <= 0 || pitch % 2 != 0 ||
      pitch < 2u * static_cast<size_t>((width + 1) / 2)) {
    return 0;
  }
  return pitch * static_cast<size_t>(height) * 3 / 2;
}

StreamDecoder::StreamDecoder(const BackendOptions& options) : options_(options) {
  state_ = new StreamState();
}

StreamDecoder::~StreamDecoder() {
  {
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    quitting_ = true;
  }
  {
    std::lock_guard<std::mutex> lk(front_mutex_);
    front_quit_ = true;
    front_tasks_.clear();
    front_slots_.clear();
  }
  pipe_cv_.notify_all();
  planner_cv_.notify_all();
  worker_cv_.notify_all();
  front_cv_.notify_all();
  if (frontend_.joinable()) frontend_.join();
  if (planner_.joinable()) planner_.join();
  if (worker_.joinable()) worker_.join();
  delete static_cast<StreamState*>(state_);
}

std::string StreamDecoder::notice() const {
  std::lock_guard<std::mutex> lk(pipe_mutex_);
  return notice_;
}

void StreamDecoder::pipe_fail(const std::string& message) {
  ds_debug_log("pipeline job failed: %s", message.c_str());
  std::lock_guard<std::mutex> lk(pipe_mutex_);
  if (!pipe_failed_) {
    pipe_failed_ = true;
    if (notice_.empty()) notice_ = message;
  }
}

void StreamDecoder::ensure_threads() {
  /* The pipeline's threads are started from whichever thread hands the first
   * frame over, and that is now two threads (the demuxer for a
   * show_existing_frame, the frontend stage for a real frame). Starting a
   * thread twice is a fatal error, so the check and the start happen under
   * the pipe lock. */
  std::lock_guard<std::mutex> lk(pipe_mutex_);
  if (!planner_.joinable()) planner_ = std::thread(&StreamDecoder::planner_loop, this);
  if (!worker_.joinable()) worker_ = std::thread(&StreamDecoder::worker_loop, this);
  if (!frontend_.joinable()) frontend_ = std::thread(&StreamDecoder::frontend_loop, this);
}

void StreamDecoder::release_reserved(int slot) {
  if (slot < 0) return;
  decoder_->release_reference(slot);
  {
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    if (reserved_slots_) --reserved_slots_;
  }
  space_cv_.notify_all();
}

void StreamDecoder::queue_job(PipeJob job, std::string* error, uint64_t generation) {
  ensure_threads();
  std::unique_lock<std::mutex> lk(pipe_mutex_);
  /* A frame that was queued before a flush belongs to the old stream: the
   * flush dropped it, so it must not be decoded now. */
  if (generation != 0 && generation != generation_) {
    const int slot = job.slot;
    decoder_->discard_frame_plan(job.frame_plan);
    job.frame_plan = nullptr;
    lk.unlock();
    release_reserved(slot);
    if (error && error->empty()) *error = "the frame was dropped by a flush";
    return;
  }
  /* back pressure: a bounded queue keeps the decoder a few frames behind the
   * demuxer instead of planning the whole file into memory */
  /* Back pressure counts both queues: the planner moves a job from jobs_ to
   * ready_jobs_ without running it, so bounding jobs_ alone would let the
   * demuxer queue far more reserved slots than the backend pool has and force
   * the reserve-fail drain (a visible hitch) once 8 + kPipeDepth is passed. */
  pipe_cv_.wait(lk, [&] {
    return jobs_.size() + ready_jobs_.size() < kPipeDepth || pipe_failed_ || quitting_;
  });
  if (pipe_failed_ || quitting_) {
    /* the job will never run: hand its reservation straight back */
    const int slot = job.slot;
    decoder_->discard_frame_plan(job.frame_plan);
    lk.unlock();
    release_reserved(slot);
    if (error && error->empty()) *error = notice_;
    return;
  }
  jobs_.push_back(std::move(job));
  planner_cv_.notify_one();
}

void StreamDecoder::frontend_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lk(front_mutex_);
      front_cv_.wait(lk, [&] { return front_quit_ || !front_tasks_.empty(); });
      if (front_tasks_.empty()) {
        if (front_quit_) return;
        continue;
      }
      task = std::move(front_tasks_.front());
      front_tasks_.pop_front();
      front_slots_.pop_front();
      front_busy_ = true;
      /* The queue has room again: wake a producer that is waiting for it. */
      front_cv_.notify_all();
    }
    task();
    {
      std::lock_guard<std::mutex> lk(front_mutex_);
      front_busy_ = false;
    }
    front_idle_.notify_all();
  }
}

void StreamDecoder::submit_frontend_task(std::function<void()> task, int slot) {
  ensure_threads();
  {
    std::unique_lock<std::mutex> lk(front_mutex_);
    /* The queue is bounded like the pipe itself: a frame waiting here still
     * holds a reserved reference slot, and the tile data of a queued frame is
     * not small. */
    front_cv_.wait(lk, [&] { return front_tasks_.size() < kPipeDepth || front_quit_; });
    if (front_quit_) return;
    front_tasks_.push_back(std::move(task));
    front_slots_.push_back(slot);
  }
  front_cv_.notify_one();
}

void StreamDecoder::planner_loop() {
  std::unique_lock<std::mutex> lk(pipe_mutex_);
  LARGE_INTEGER p0{}, p1{}, pf{};
  QueryPerformanceFrequency(&pf);
  for (;;) {
    planner_cv_.wait(lk, [&] {
      return (quitting_ && jobs_.empty()) || pipe_failed_ || !jobs_.empty();
    });
    if (jobs_.empty()) return; /* quitting_ or failed */
    PipeJob job = std::move(jobs_.front());
    jobs_.pop_front();
    const bool dead = pipe_failed_ || quitting_; /* read under the lock */
    pipe_cv_.notify_all(); /* queue space for queue_job() */
    planner_busy_ = true;
    lk.unlock();
    bool ok = true;
    std::string err;
    if (!dead && !job.show_existing && job.slot >= 0 && !job.frame_plan && plan_thread_) {
      if (ds_debug_enabled()) QueryPerformanceCounter(&p0);
      ok = decoder_->plan_frame(job.syntax, &job.frame_plan, &err);
      if (ds_debug_enabled()) {
        QueryPerformanceCounter(&p1);
        const double ms = double(p1.QuadPart - p0.QuadPart) * 1000.0 / double(pf.QuadPart);
        static double acc = 0, worst = 0;
        static long n = 0;
        if (ms > 60.0) ds_debug_log("SLOW planner plan: %.1f ms", ms);
        acc += ms;
        if (ms > worst) worst = ms;
        if (++n % 50 == 0) {
          ds_debug_log("perf planner plan: avg %.1f, worst %.1f ms over %ld", acc / n, worst, n);
          acc = worst = 0;
          n = 0;
        }
      }
      if (!ok) pipe_fail(err.empty() ? "planner: plan_frame failed" : err);
    }
    lk.lock();
    planner_busy_ = false;
    if (ok && !pipe_failed_) {
      ready_jobs_.push_back(std::move(job));
      worker_cv_.notify_one();
    } else {
      /* dropped (failed plan, or the pipe is broken): hand the reservation
       * back (decoder calls are made without the lock, like queue_job()'s
       * failure path) */
      const int slot = job.slot;
      decoder_->discard_frame_plan(job.frame_plan);
      job.frame_plan = nullptr;
      lk.unlock();
      release_reserved(slot);
      lk.lock();
    }
    pipe_cv_.notify_all(); /* drain waiters (finish) */
  }
}

void StreamDecoder::worker_loop() {
  std::unique_lock<std::mutex> lk(pipe_mutex_);
  LARGE_INTEGER i0{}, iff{};
  QueryPerformanceFrequency(&iff);
  for (;;) {
    if (ds_debug_enabled()) QueryPerformanceCounter(&i0);
    worker_cv_.wait(lk,
                    [&] { return (quitting_ && ready_jobs_.empty()) || !ready_jobs_.empty(); });
    if (ds_debug_enabled() && !ready_jobs_.empty()) {
      LARGE_INTEGER i1;
      QueryPerformanceCounter(&i1);
      const double idle = double(i1.QuadPart - i0.QuadPart) * 1000.0 / double(iff.QuadPart);
      if (idle > 50.0) ds_debug_log("SLOW worker idle: %.1f ms", idle);
    }
    if (ready_jobs_.empty()) return; /* quitting_ */
    PipeJob job = std::move(ready_jobs_.front());
    ready_jobs_.pop_front();
    pipe_cv_.notify_all(); /* in-flight space for queue_job() */
    busy_ = true;
    lk.unlock();
    run_job(&job);
    lk.lock();
    busy_ = false;
    pipe_cv_.notify_all();
  }
}

void StreamDecoder::run_job(PipeJob* job) {
  /* AV1GPU_DS_DEBUG=1 also reports how long each queued job took on the
   * worker: a hitch in playback shows up here as a spike long before the
   * average moves. */
  LARGE_INTEGER j0{}, j1{}, jf{};
  const bool jtimed = ds_debug_enabled();
  if (jtimed) {
    QueryPerformanceFrequency(&jf);
    QueryPerformanceCounter(&j0);
  }
  if (job->show_existing) {
    Frame picture;
    bool valid = false;
    {
      std::lock_guard<std::mutex> lk(pipe_mutex_);
      valid = job->existing_idx >= 0 && job->existing_idx < 8 &&
              pipe_valid_[job->existing_idx];
      if (valid) picture = pipe_picture_[job->existing_idx];
    }
    if (!valid) {
      /* the frames this re-show depends on were dropped; skip it rather than
         break a stream that is otherwise fine */
      ds_debug_log("show_existing_frame names a slot that was never decoded");
      return;
    }
    std::string err;
    if (!emit(picture, job->pitch, job->sink, &err)) pipe_fail(err);
    return;
  }

  Frame picture;
  std::string err;
  LARGE_INTEGER b0{}, b1{};
  if (jtimed) QueryPerformanceCounter(&b0);
  if (job->frame_plan && job->slot < 0) {
    /* a frame nobody keeps has no reserved slot to decode into: drop the plan
     * and take the ordinary synchronous path */
    decoder_->discard_frame_plan(job->frame_plan);
    job->frame_plan = nullptr;
  }
  if (job->frame_plan) {
    /* pre-planned on the demuxer's thread: submit it and read the picture
     * back here, on the worker */
    const int slot =
        decoder_->decode_reference_planned(job->frame_plan, job->slot, &picture, &err);
    job->frame_plan = nullptr; /* consumed either way */
    if (slot < 0) {
      pipe_fail(err.empty() ? "queued frame decode failed" : err);
      return;
    }
  } else if (job->slot >= 0) {
    const int slot = decoder_->decode_reference_reserved(job->syntax, job->slot, &picture, &err);
    if (slot < 0) {
      pipe_fail(err.empty() ? "queued frame decode failed" : err);
      return;
    }
  } else {
    if (!decoder_->decode(job->syntax, &picture, &err)) {
      pipe_fail(err.empty() ? "queued frame decode failed" : err);
      return;
    }
  }
  decoded_.fetch_add(1, std::memory_order_relaxed);

  if (jtimed) {
    QueryPerformanceCounter(&b1);
    static double acc_b = 0, worst_b = 0;
    static long nb = 0;
    const double ms = double(b1.QuadPart - b0.QuadPart) * 1000.0 / double(jf.QuadPart);
    if (ms > 50.0) ds_debug_log("SLOW worker decode: %.1f ms", ms);
    acc_b += ms;
    if (ms > worst_b) worst_b = ms;
    if (++nb % 50 == 0) {
      ds_debug_log("perf worker decode: avg %.1f, worst %.1f ms over %ld",
                   acc_b / nb, worst_b, nb);
      acc_b = worst_b = 0;
      nb = 0;
    }
  }

  if (job->show_frame && !emit(picture, job->pitch, job->sink, &err)) {
    pipe_fail(err);
    return;
  }

  if (job->slot >= 0) {
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    for (int i = 0; i < 8; ++i) {
      if (!(job->refresh_flags & (1 << i))) continue;
      pipe_picture_[i] = picture;
      pipe_valid_[i] = true;
    }
  }

  /* every job queued before this one has finished, so slots dropped at plan
   * time are safe to hand back now */
  for (int slot : job->release_slots) release_reserved(slot);
  if (jtimed) {
    QueryPerformanceCounter(&j1);
    static double acc = 0, worst = 0;
    static long n = 0;
    const double ms = double(j1.QuadPart - j0.QuadPart) * 1000.0 / double(jf.QuadPart);
    if (ms > 50.0) ds_debug_log("SLOW worker job: %.1f ms", ms);
    acc += ms;
    if (ms > worst) worst = ms;
    if (++n % 50 == 0) {
      ds_debug_log("perf worker job: avg %.1f, worst %.1f ms over %ld", acc / n, worst, n);
      acc = worst = 0;
      n = 0;
    }
  }
}

void StreamDecoder::finish() {
  if (!worker_.joinable()) return;
  /* The frontend stage comes first in the chain: a frame that is still being
   * planned has not reached the pipe yet. */
  {
    std::unique_lock<std::mutex> lk(front_mutex_);
    front_idle_.wait(lk, [&] { return front_tasks_.empty() && !front_busy_; });
  }
  std::unique_lock<std::mutex> lk(pipe_mutex_);
  pipe_cv_.wait(lk, [&] {
    return jobs_.empty() && ready_jobs_.empty() && !busy_ && !planner_busy_;
  });
}

void StreamDecoder::abort_pipe() {
  /* The frontend stage first: a frame that is being planned right now would
   * otherwise reach the worker after the flush. */
  std::vector<int> front_orphans;
  {
    std::lock_guard<std::mutex> lk(front_mutex_);
    for (int slot : front_slots_)
      if (slot >= 0) front_orphans.push_back(slot);
    front_tasks_.clear();
    front_slots_.clear();
  }
  {
    std::unique_lock<std::mutex> lk(front_mutex_);
    front_idle_.wait(lk, [&] { return !front_busy_; });
  }
  if (!worker_.joinable()) {
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    for (int i = 0; i < 8; ++i) pipe_valid_[i] = false;
    pipe_failed_ = false;
    for (int slot : front_orphans) release_reserved(slot);
    return;
  }
  std::vector<int> orphan_slots;
  {
    std::unique_lock<std::mutex> lk(pipe_mutex_);
    /* Everything queued so far belongs to the old stream. */
    ++generation_;
    for (auto& job : jobs_) {
      if (job.slot >= 0) orphan_slots.push_back(job.slot);
      decoder_->discard_frame_plan(job.frame_plan);
      job.frame_plan = nullptr;
    }
    jobs_.clear();
    /* the planner may be mid-plan on a job it already popped; wait for it,
     * then drop what it queued */
    pipe_cv_.wait(lk, [&] { return !busy_ && !planner_busy_; });
    for (auto& job : ready_jobs_) {
      if (job.slot >= 0) orphan_slots.push_back(job.slot);
      decoder_->discard_frame_plan(job.frame_plan);
      job.frame_plan = nullptr;
    }
    ready_jobs_.clear();
    for (int i = 0; i < 8; ++i) pipe_valid_[i] = false;
    pipe_failed_ = false;
  }
  pipe_cv_.notify_all();
  planner_cv_.notify_all();
  worker_cv_.notify_all();
  /* slots reserved for frames that will now never be decoded */
  for (int slot : orphan_slots) release_reserved(slot);
  for (int slot : front_orphans) release_reserved(slot);
}

void StreamDecoder::reset_stream() {
  /* Serialized against push(): this runs on the seeking thread while the
   * streaming thread may be inside push(). abort_pipe() waits for the worker
   * under pipe_mutex_; push() can be blocked in queue_job()'s back pressure
   * at the same time, but it holds no lock the worker needs, so no deadlock. */
  std::lock_guard<std::mutex> stream_lock(stream_mutex_);
  abort_pipe();
  pending_.clear();
  static_cast<StreamState*>(state_)->clear();
  if (decoder_) decoder_->reset();
  {
    /* The backend's pool was reset with it, so every slot it held is gone. */
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    reserved_slots_ = 0;
  }
  space_cv_.notify_all();
}

bool StreamDecoder::init(std::string* error) {
  decoder_ = Decoder::create(options_, error);
  if (!decoder_) return false;
  /* the pipeline needs a backend that can name a decode's slot before it
   * runs; without that every frame stays on the synchronous path */
  const int probe = decoder_->reserve_reference();
  if (probe >= 0) {
    pipelined_ = true;
    decoder_->release_reference(probe);
  }
  /* Building the plan on the demuxer's thread sounds like free parallelism,
   * but in the filter that thread is the one parsing and planning the frame,
   * and moving the scheduler there made it the bottleneck: 25.7 fps with
   * pre-planning against 28.0 without. The pipeline therefore has a dedicated
   * planner thread between the demuxer's and the worker's (see planner_loop);
   * AV1GPU_DS_NOPLANNER=1 turns it off and the worker plans as it decodes. */
  plan_thread_ = pipelined_ && std::getenv("AV1GPU_DS_NOPLANNER") == nullptr;
  preplan_ = pipelined_ && decoder_->supports_preplanning() &&
             std::getenv("AV1GPU_DS_PREPLAN") != nullptr;
  /* An invisible frame that refreshes no reference slot cannot affect the
   * decoded output, so it does not have to be decoded at all (see
   * decode_obu). AV1GPU_SKIP_INVISIBLE=0 turns the skip off. */
  skip_invisible_ = true;
  if (const char* e = std::getenv("AV1GPU_SKIP_INVISIBLE")) skip_invisible_ = e[0] != '0';
  return true;
}

const DeviceInfo& StreamDecoder::device() const { return decoder_->device(); }

bool StreamDecoder::push(const uint8_t* data, size_t size, size_t pitch,
                         const Sink& sink, std::string* error) {
  /* The whole sample is processed under stream_mutex_: a concurrent
   * reset_stream() (a seek from another thread) must not clear pending_ or
   * reset the decoder while this thread is mid-OBU in them. queue_job()
   * takes pipe_mutex_ inside, which is fine: the worker never takes
   * stream_mutex_, so the lock order stream -> pipe cannot deadlock. */
  LARGE_INTEGER d0{}, d1{}, df{};
  const bool dtimed = ds_debug_enabled();
  if (dtimed) {
    QueryPerformanceFrequency(&df);
    QueryPerformanceCounter(&d0);
  }
  std::unique_lock<std::mutex> stream_lock(stream_mutex_);
  if (dtimed) {
    QueryPerformanceCounter(&d1);
    const double lms = double(d1.QuadPart - d0.QuadPart) * 1000.0 / double(df.QuadPart);
    if (lms > 60.0) ds_debug_log("SLOW stream_lock: %.1f ms", lms);
    QueryPerformanceCounter(&d0);
  }
  /* A stream whose OBUs never complete would grow the buffer without bound;
   * 64 MiB is far past any real temporal unit. */
  constexpr size_t kPendingLimit = 64u << 20;
  LARGE_INTEGER q0{}, q1{};
  if (dtimed) QueryPerformanceCounter(&q0);
  {
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    if (pipe_failed_) {
      if (error) *error = notice_;
      return false;
    }
  }
  if (dtimed) QueryPerformanceCounter(&q0);
  if (pending_.size() + size > kPendingLimit) {
    *error = "the stream's OBU framing is broken (no complete OBU in 64 MiB)";
    pending_.clear();
    return false;
  }
  pending_.insert(pending_.end(), data, data + size);
  if (dtimed) QueryPerformanceCounter(&q0);

  size_t pos = 0;
  bool ok = true;
  while (pos < pending_.size()) {
    const size_t start = pos; /* the OBU's first byte, for the NeedMore case */
    frontend::ObuType type = frontend::ObuType::Reserved0;
    bool has_size = false;
    size_t payload_size = 0;
    const ObuPeek peek =
        peek_obu(pending_.data(), pending_.size(), &pos, &type, &has_size, &payload_size);
    if (peek == ObuPeek::NeedMore) {
      pos = start; /* the bytes from `start` on wait for the next sample */
      break;
    }
    if (peek == ObuPeek::Invalid) {
      if (error) *error = "not an AV1 elementary stream (bad OBU header)";
      ok = false;
      break;
    }
    ok = decode_obu(type, pending_.data() + pos, payload_size, pitch, sink, error);
    pos += payload_size;
    if (!ok) break;
  }
  if (dtimed) QueryPerformanceCounter(&q0);
  /* Consume everything decoded (and, on an error, the OBU that failed); keep
   * an incomplete tail for the next push. */
  pending_.erase(pending_.begin(),
                 pending_.begin() + static_cast<std::ptrdiff_t>(pos));
  if (dtimed) {
    QueryPerformanceCounter(&d1);
    const double ms = double(d1.QuadPart - d0.QuadPart) * 1000.0 / double(df.QuadPart);
    if (ms > 60.0) ds_debug_log("SLOW push body: %.1f ms (sample %zu bytes)", ms, size);
  }
  return ok;
}

bool StreamDecoder::emit(const Frame& picture, size_t pitch, const Sink& sink,
                         std::string* error) {
  if (picture.ss_x != 1 || picture.ss_y != 1) {
    /* NV12 cannot carry 4:4:4 or 4:2:2; the output negotiates NV12, so a frame
     * like this is dropped rather than shown with wrong chroma. */
    {
      std::lock_guard<std::mutex> lk(pipe_mutex_);
      if (notice_.empty())
        notice_ = "stream is not 4:2:0; the NV12 output drops those frames";
    }
    ++dropped_;
    return true;
  }
  std::vector<uint8_t> nv12;
  const auto pc0 = std::chrono::steady_clock::now();
  if (!frame_to_nv12(picture, pitch, &nv12)) {
    *error = "NV12 pack failed (geometry " + std::to_string(picture.width) + "x" +
             std::to_string(picture.height) + ", pitch " + std::to_string(pitch) + ")";
    return false;
  }
  const auto pc1 = std::chrono::steady_clock::now();
  static struct {
    double total = 0, worst = 0; long n = 0;
    void add(double ms) {
      total += ms; ++n; if (ms > worst) worst = ms;
      if (n % 50 == 0)
        ds_debug_log("perf stage nv12_pack: avg %.1f ms, worst %.1f ms over %ld",
                  total / n, worst, n);
    }
  } st_nv12;
  st_nv12.add(std::chrono::duration<double, std::milli>(pc1 - pc0).count());
  ++shown_;
  return sink(nv12, picture.width, picture.height, error);
}

bool StreamDecoder::decode_obu(frontend::ObuType type, const uint8_t* data, size_t size,
                               size_t pitch, const Sink& sink, std::string* error) {
  StreamState* st = static_cast<StreamState*>(state_);

  if (type == frontend::ObuType::SequenceHeader) {
    if (frontend::parse_sequence_header(data, size, &st->seq, error) ==
        frontend::ParseStatus::Invalid) {
      return false;
    }
    have_seq_ = true;
    return true;
  }
  if (type == frontend::ObuType::TileList) {
    /* large scale tile: a frame assembled from independently stored tiles; the
     * parser does not read it yet */
    {
      std::lock_guard<std::mutex> lk(pipe_mutex_);
      if (notice_.empty()) notice_ = "OBU_TILE_LIST is not supported";
    }
    ++dropped_;
    return true;
  }
  if (type != frontend::ObuType::FrameHeader && type != frontend::ObuType::Frame &&
      type != frontend::ObuType::RedundantFrameHeader) {
    return true; /* temporal delimiter, metadata, padding */
  }
  if (!have_seq_) {
    *error = "frame before any sequence header";
    return false;
  }

  frontend::FrameHeader frame;
  LARGE_INTEGER h0{}, h1{}, hf{};
  double hdr_ms = 0, tg_ms = 0, res_ms = 0, destroy_ms = 0;
  double tiles_ms = 0, deq_ms = 0, plan_ms = 0, queue_ms = 0;
  if (ds_debug_enabled()) {
    QueryPerformanceFrequency(&hf);
    QueryPerformanceCounter(&h0);
  }
  const frontend::ParseStatus parsed =
      frontend::parse_frame_header(st->seq, data, size, &frame, error, &st->refs);
  if (ds_debug_enabled()) {
    QueryPerformanceCounter(&h1);
    hdr_ms = double(h1.QuadPart - h0.QuadPart) * 1000.0 / double(hf.QuadPart);
    if (hdr_ms > 20.0) ds_debug_log("SLOW stage frame_header: %.1f ms", hdr_ms);
  }
  if (parsed == frontend::ParseStatus::Invalid) return false;
  if (parsed == frontend::ParseStatus::Unsupported) {
    frontend::refresh_ref_pool(frame, &st->refs);
    ++dropped_;
    return true;
  }

  if (frame.show_existing_frame) {
    const int which = frame.existing_frame_idx;
    if (which < 0 || which >= 8) {
      *error = "show_existing_frame names an out of range slot";
      return false;
    }
    /* The validity check runs on the worker: earlier queued frames may still
     * owe the picture this frame re-shows. */
    PipeJob job;
    job.show_existing = true;
    job.show_frame = true;
    job.existing_idx = which;
    job.pitch = pitch;
    job.sink = sink;
    /* It goes through the frontend queue like every other frame: the frames
     * before it are still being planned there, and a re-show that overtakes
     * them would name a slot whose picture is not decoded yet. */
    uint64_t generation = 0;
    {
      std::lock_guard<std::mutex> lk(pipe_mutex_);
      generation = generation_;
    }
    submit_frontend_task(
        [this, job = std::move(job), generation]() mutable {
          std::string err;
          queue_job(std::move(job), &err, generation);
        },
        -1);
    return true;
  }

  /* A frame that is neither shown nor stored in the reference pool cannot
   * affect the decoded output: nothing displays it, and no later frame can
   * reference it - show_existing_frame only re-shows a slot, and a frame that
   * refreshes no slot is in none. Such frames are the invisible "scratch"
   * frames of a temporal layered stream, and they are the largest units in
   * it: decoding them was the demuxer thread's single biggest cost and the
   * source of the visible hitches (a 50 kB invisible frame took the demuxer
   * 150-220 ms, and the shown frame behind it waited).
   *
   * Nothing else in the decoder carries state across a frame: the reference
   * pool is written by refresh_frame_flags only, the CDFs are per frame, and
   * the segmentation, grain and motion vector grids live in the slots. */
  if (skip_invisible_ && !frame.show_frame && frame.refresh_frame_flags == 0) {
    skipped_invisible_.fetch_add(1, std::memory_order_relaxed);
    if (ds_debug_enabled())
      ds_debug_log("skip invisible non-reference %s frame (%d bytes)",
                   frontend::frame_type_name(frame.type), static_cast<int>(size));
    return true;
  }

  frontend::TileGroup group;
  if (ds_debug_enabled()) QueryPerformanceCounter(&h0);
  if (frontend::parse_tile_group(st->seq, frame, data, size, &group, error) !=
      frontend::ParseStatus::Ok) {
    return false;
  }
  if (ds_debug_enabled()) {
    QueryPerformanceCounter(&h1);
    tg_ms = double(h1.QuadPart - h0.QuadPart) * 1000.0 / double(hf.QuadPart);
    if (tg_ms > 20.0) ds_debug_log("SLOW stage tile_group: %.1f ms", tg_ms);
  }
  if (!frontend::tile_group_is_complete(frame, group)) {
    *error = "the tile group covers " + std::to_string(group.tiles.size()) +
             " of the frame's " + std::to_string(frame.tiles.cols) + "x" +
             std::to_string(frame.tiles.rows) + " tiles";
    ds_debug_log("tile group incomplete: %s", error->c_str());
    return false;
  }

  frontend::TileDecodeResult decoded;
  /* stage timing (release instrumentation, av1gpu_ds.log "perf:") */
  struct PerfClock {
    LARGE_INTEGER freq;
    PerfClock() { QueryPerformanceFrequency(&freq); }
    double ms(LARGE_INTEGER a, LARGE_INTEGER b) const {
      return double(b.QuadPart - a.QuadPart) / double(freq.QuadPart) * 1000.0;
    }
  };
  static const PerfClock perf;
  struct Stage {
    const char* name; double total = 0.0; double worst = 0.0; long n = 0;
    void add(double ms) {
      total += ms; ++n; if (ms > worst) worst = ms;
      /* a one-off stall is what a viewer sees as a hitch; log it on the spot
       * instead of waiting for the 50 frame summary */
      if (ms > 60.0) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "SLOW stage %s: %.1f ms", name, ms);
        ds_debug_log("%s", msg);
      }
      if (n % 50 == 0)
        ds_debug_log("perf stage %s: avg %.1f ms, worst %.1f ms over %ld",
                  name, total / n, worst, n);
    }
  };
  static Stage st_tiles{"tiles"}, st_deq{"dequantise"}, st_plan{"plan"},
               st_gpu{"gpu_decode"}, st_nv12{"nv12_pack"}, st_preplan{"preplan"},
               st_queue{"queue_wait"}, st_refresh{"refpool"}, st_hdr{"frame_header"},
               st_tg{"tile_group"}, st_res{"reserve"};
  LARGE_INTEGER pc0, pc1;

  QueryPerformanceCounter(&pc0);
  const frontend::ParseStatus tile_status = frontend::decode_tiles(
      st->seq, frame, &st->refs, group.tiles.data(), group.tiles.size(), data, size,
      &decoded, error);
  QueryPerformanceCounter(&pc1);
  tiles_ms = perf.ms(pc0, pc1);
  st_tiles.add(tiles_ms);
  if (tile_status != frontend::ParseStatus::Ok) {
    if (error->empty()) *error = decoded.stop_reason;
    return false;
  }

  frontend::DequantisedTile quant;
  QueryPerformanceCounter(&pc0);
  frontend::dequantise_tile(frame, st->seq.bit_depth, decoded, &quant);
  QueryPerformanceCounter(&pc1);
  deq_ms = perf.ms(pc0, pc1);
  st_deq.add(deq_ms);

  frontend::ReferenceInfo reference_info;
  for (int i = 0; i < 7; ++i) {
    const int which = frame.ref_frame_idx[i];
    if (which < 0 || which >= 8 || !st->refs.slots[which].valid) continue;
    reference_info.slot[i] = st->pool_slot[which];
    reference_info.width[i] = st->refs.slots[which].width;
    reference_info.height[i] = st->refs.slots[which].height;
    reference_info.upscaled_width[i] = st->refs.slots[which].upscaled_width > 0
                                           ? st->refs.slots[which].upscaled_width
                                           : st->refs.slots[which].width;
  }

  /* The reference pool update happens *before* the decode is queued: a
   * pipelined next frame plans against it while this frame's kernels still
   * run. It must come after the loop above, because a frame is allowed to
   * reference a slot it also refreshes, and the refresh would then overwrite
   * the data the loop still needs. */
  QueryPerformanceCounter(&pc0);
  frontend::refresh_ref_pool(frame, &st->refs, decoded.ref_mvs, decoded.cdf,
                             decoded.seg_map);
  QueryPerformanceCounter(&pc1);
  st_refresh.add(perf.ms(pc0, pc1));

  FrameSyntax syntax;

  /* Free the slots this frame refreshes but does not predict from before it is
   * decoded; the frame is allowed to refresh a slot it reads, so those stay.
   * On the pipelined path the release is deferred to this frame's job, because
   * a queued earlier job may still be about to write into the slot.
   *
   * A backend slot may be named by several AV1 slots (one picture refreshing
   * many), so a freed slot is only handed back when this frame was its last
   * namer - the slot that still names it releases it in its own turn. */
  std::vector<int> deferred_release;
  auto defer_release = [&](int slot) {
    if (slot < 0) return;
    for (int k = 0; k < 8; ++k)
      if (st->pool_slot[k] == slot) return; /* still named: its owner releases it */
    for (int held : deferred_release)
      if (held == slot) return; /* already queued for release */
    deferred_release.push_back(slot);
  };
  for (int i = 0; i < 8; ++i) {
    if (!(frame.refresh_frame_flags & (1 << i))) continue;
    bool referenced = false;
    for (int j = 0; j < 7; ++j) {
      if (frame.ref_frame_idx[j] == i) {
        referenced = true;
        break;
      }
    }
    if (referenced) continue;
    const int freed = st->pool_slot[i];
    st->pool_slot[i] = -1;
    if (pipelined_) {
      defer_release(freed);
    } else {
      release_slot(st, decoder_.get(), freed);
    }
  }

  /* Decide up front which backend slot the picture will be decoded into, so
   * the pool names below are final before the decode is queued: later frames
   * plan against these names while this frame's kernels still run. */
  bool kept = false;
  for (int i = 0; i < 8; ++i)
    if (frame.refresh_frame_flags & (1 << i)) kept = true;

  bool pipelined = pipelined_;
  int reserved = -1;
  if (ds_debug_enabled()) QueryPerformanceCounter(&h0);
  if (pipelined && kept) {
    /* Wait for a reserved slot to come back before asking for one: the
     * pipeline has three queues in front of the worker, and every frame that
     * waits in one of them holds a slot. Without this the pool ran dry and
     * the pipeline deadlocked (the worker waited for a slot to decode into,
     * the demuxer waited for queue space, the queue waited for the worker). */
    {
      std::unique_lock<std::mutex> lk(pipe_mutex_);
      space_cv_.wait(lk, [&] {
        return reserved_slots_ < kMaxReservedSlots || pipe_failed_ || quitting_ ||
               fallback_used_;
      });
      if (pipe_failed_ || quitting_) {
        *error = notice_.empty() ? "the decoder is shutting down" : notice_;
        return false;
      }
    }
    reserved = decoder_->reserve_reference();
    if (reserved >= 0) {
      std::lock_guard<std::mutex> lk(pipe_mutex_);
      ++reserved_slots_;
    }
    if (reserved < 0) {
      /* every slot may be marked because queued frames still hold the ones
       * the pool dropped; drain the queue and try again before giving up */
      {
        LARGE_INTEGER f0, f1, ff;
        QueryPerformanceFrequency(&ff);
        QueryPerformanceCounter(&f0);
        finish();
        QueryPerformanceCounter(&f1);
        ds_debug_log("SLOW reserve-fail drain: %.1f ms",
                     double(f1.QuadPart - f0.QuadPart) * 1000.0 / double(ff.QuadPart));
      }
      reserved = decoder_->reserve_reference();
    }
    if (reserved < 0) {
      if (!fallback_used_) {
        /* first failure: drain what is queued and finish this frame on the
         * calling thread; every later frame stays synchronous too */
        fallback_used_ = true;
        pipelined_ = false;
        pipelined = false;
        finish();
        /* slots dropped earlier in this frame's plan: no job will carry them */
        for (int slot : deferred_release) release_reserved(slot);
        deferred_release.clear();
        ds_debug_log("reference reservation failed; falling back to the "
                     "synchronous decode path");
      } else {
        *error = "reference pool exhausted";
        return false;
      }
    }
  }
  if (ds_debug_enabled()) {
    QueryPerformanceCounter(&h1);
    res_ms = double(h1.QuadPart - h0.QuadPart) * 1000.0 / double(hf.QuadPart);
    if (res_ms > 20.0) ds_debug_log("SLOW stage reserve: %.1f ms", res_ms);
  }

  if (pipelined) {
    for (int i = 0; i < 8; ++i) {
      if (!(frame.refresh_frame_flags & (1 << i))) continue;
      const int previous = st->pool_slot[i];
      st->pool_slot[i] = reserved;
      if (previous != reserved) defer_release(previous);
    }

    /* The frontend plan - the per block syntax the engine reads - is the
     * demuxer's single biggest cost (13 ms a frame) and it depends on nothing
     * but this frame, so it runs on the frontend thread: the demuxer parses
     * the next frame while this one is being planned, and the planner and the
     * GPU pick the frame up when it is ready. The frames keep their order,
     * because the frontend queue is FIFO. */
    PipeJob job;
    job.slot = reserved;
    job.refresh_flags = frame.refresh_frame_flags;
    job.show_frame = frame.show_frame;
    job.pitch = pitch;
    job.sink = sink;
    job.release_slots = std::move(deferred_release);

    uint64_t generation = 0;
    {
      std::lock_guard<std::mutex> lk(pipe_mutex_);
      generation = generation_;
    }
    frontend::SequenceHeader seq_copy = st->seq;
    frontend::FrameHeader frame_copy = frame;
    frontend::ReferenceInfo refs_copy = reference_info;
    frontend::TileDecodeResult decoded_move = std::move(decoded);
    frontend::DequantisedTile quant_move = std::move(quant);
    const int job_slot = job.slot;
    submit_frontend_task([this, seq_copy = std::move(seq_copy),
                          frame_copy = std::move(frame_copy), refs_copy = std::move(refs_copy),
                          decoded_move = std::move(decoded_move),
                          quant_move = std::move(quant_move), job = std::move(job),
                          generation]() mutable {
      FrameSyntax syntax;
      std::string err;
      LARGE_INTEGER f0{}, f1{}, ff{};
      QueryPerformanceFrequency(&ff);
      QueryPerformanceCounter(&f0);
      if (!frontend::plan_from_tile(seq_copy, frame_copy, decoded_move, std::move(quant_move),
                                    &syntax, &err, &refs_copy)) {
        pipe_fail(err.empty() ? "frontend: the frame could not be planned" : err);
        return;
      }
      QueryPerformanceCounter(&f1);
      st_plan.add(double(f1.QuadPart - f0.QuadPart) * 1000.0 / double(ff.QuadPart));
      if (preplan_) {
        QueryPerformanceCounter(&f0);
        const bool planned = decoder_->plan_frame(syntax, &job.frame_plan, &err);
        QueryPerformanceCounter(&f1);
        st_preplan.add(double(f1.QuadPart - f0.QuadPart) * 1000.0 / double(ff.QuadPart));
        if (!planned) {
          pipe_fail(err.empty() ? "frontend: the frame plan could not be built" : err);
          return;
        }
      }
      job.syntax = std::move(syntax);
      queue_job(std::move(job), &err, generation);
    }, job_slot);
    return true;
  }

  /* Synchronous path: the plan is built here, on the calling thread. */
  QueryPerformanceCounter(&pc0);
  if (!frontend::plan_from_tile(st->seq, frame, decoded, std::move(quant), &syntax, error,
                                &reference_info)) {
    return false;
  }
  QueryPerformanceCounter(&pc1);
  plan_ms = perf.ms(pc0, pc1);
  st_plan.add(plan_ms);

  /* Synchronous path: decode on the calling thread, then do the book keeping
   * that the pipelined path does at plan time. */
  Frame picture;
  QueryPerformanceCounter(&pc0);
  const int slot = decoder_->decode_reference(syntax, &picture, error);
  QueryPerformanceCounter(&pc1);
  st_gpu.add(perf.ms(pc0, pc1));
  if (slot < 0) return false;
  {
    const DecodeStats& s = decoder_->stats();
    static double acc[4] = {0, 0, 0, 0};
    static long n = 0;
    acc[0] += s.reconstruct_ms; acc[1] += s.filter_ms; acc[2] += s.readback_ms;
    acc[3] += s.total_ms; ++n;
    if (n % 50 == 0)
      ds_debug_log("perf gpu split: reconstruct avg %.1f, filter avg %.1f, "
                   "readback avg %.1f, total avg %.1f ms over %ld",
                   acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n, n);
  }
  ++decoded_;

  if (frame.show_frame) {
    if (!emit(picture, pitch, sink, error)) return false;
  }

  /* Hand the picture to the AV1 slots this frame refreshes. */
  for (int i = 0; i < 8; ++i) {
    if (!(frame.refresh_frame_flags & (1 << i))) continue;
    const int previous = st->pool_slot[i];
    st->pool_slot[i] = slot;
    release_slot(st, decoder_.get(), previous);
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    pipe_picture_[i] = picture;
    pipe_valid_[i] = true;
  }
  if (!kept) release_slot(st, decoder_.get(), slot);

  return true;
}

} /* namespace dsfilter */
} /* namespace av1gpu */
