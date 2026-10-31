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
  char flag[8] = {0};
  const DWORD n = GetEnvironmentVariableA("AV1GPU_DS_DEBUG", flag, sizeof(flag));
  return n > 0 && n < sizeof(flag) && flag[0] != '0';
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
  worker_cv_.notify_all();
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

void StreamDecoder::queue_job(PipeJob job, std::string* error) {
  if (!worker_.joinable()) worker_ = std::thread(&StreamDecoder::worker_loop, this);
  std::unique_lock<std::mutex> lk(pipe_mutex_);
  /* back pressure: a bounded queue keeps the decoder a few frames behind the
   * demuxer instead of planning the whole file into memory */
  pipe_cv_.wait(lk, [&] { return jobs_.size() < kPipeDepth || pipe_failed_ || quitting_; });
  if (pipe_failed_ || quitting_) {
    /* the job will never run: hand its reservation straight back */
    const int slot = job.slot;
    decoder_->discard_frame_plan(job.frame_plan);
    lk.unlock();
    if (slot >= 0) decoder_->release_reference(slot);
    if (error && error->empty()) *error = notice_;
    return;
  }
  jobs_.push_back(std::move(job));
  worker_cv_.notify_one();
}

void StreamDecoder::worker_loop() {
  std::unique_lock<std::mutex> lk(pipe_mutex_);
  for (;;) {
    worker_cv_.wait(lk, [&] { return (quitting_ && jobs_.empty()) || !jobs_.empty(); });
    if (jobs_.empty()) return; /* quitting_ */
    PipeJob job = std::move(jobs_.front());
    jobs_.pop_front();
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
  for (int slot : job->release_slots) decoder_->release_reference(slot);
  if (jtimed) {
    QueryPerformanceCounter(&j1);
    static double acc = 0, worst = 0;
    static long n = 0;
    const double ms = double(j1.QuadPart - j0.QuadPart) * 1000.0 / double(jf.QuadPart);
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
  std::unique_lock<std::mutex> lk(pipe_mutex_);
  pipe_cv_.wait(lk, [&] { return jobs_.empty() && !busy_; });
}

void StreamDecoder::abort_pipe() {
  if (!worker_.joinable()) {
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    for (int i = 0; i < 8; ++i) pipe_valid_[i] = false;
    pipe_failed_ = false;
    return;
  }
  std::vector<int> orphan_slots;
  {
    std::unique_lock<std::mutex> lk(pipe_mutex_);
    for (auto& job : jobs_) {
      if (job.slot >= 0) orphan_slots.push_back(job.slot);
      decoder_->discard_frame_plan(job.frame_plan);
      job.frame_plan = nullptr;
    }
    jobs_.clear();
    pipe_cv_.wait(lk, [&] { return !busy_; });
    for (int i = 0; i < 8; ++i) pipe_valid_[i] = false;
    pipe_failed_ = false;
  }
  /* slots reserved for frames that will now never be decoded */
  for (int slot : orphan_slots) decoder_->release_reference(slot);
}

void StreamDecoder::reset_stream() {
  abort_pipe();
  pending_.clear();
  static_cast<StreamState*>(state_)->clear();
  if (decoder_) decoder_->reset();
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
  preplan_ = pipelined_ && decoder_->supports_preplanning();
  return true;
}

const DeviceInfo& StreamDecoder::device() const { return decoder_->device(); }

bool StreamDecoder::push(const uint8_t* data, size_t size, size_t pitch,
                         const Sink& sink, std::string* error) {
  /* A stream whose OBUs never complete would grow the buffer without bound;
   * 64 MiB is far past any real temporal unit. */
  constexpr size_t kPendingLimit = 64u << 20;
  {
    std::lock_guard<std::mutex> lk(pipe_mutex_);
    if (pipe_failed_) {
      if (error) *error = notice_;
      return false;
    }
  }
  if (pending_.size() + size > kPendingLimit) {
    *error = "the stream's OBU framing is broken (no complete OBU in 64 MiB)";
    pending_.clear();
    return false;
  }
  pending_.insert(pending_.end(), data, data + size);

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
  /* Consume everything decoded (and, on an error, the OBU that failed); keep
   * an incomplete tail for the next push. */
  pending_.erase(pending_.begin(),
                 pending_.begin() + static_cast<std::ptrdiff_t>(pos));
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
  const frontend::ParseStatus parsed =
      frontend::parse_frame_header(st->seq, data, size, &frame, error, &st->refs);
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
    queue_job(std::move(job), error);
    return true;
  }

  frontend::TileGroup group;
  if (frontend::parse_tile_group(st->seq, frame, data, size, &group, error) !=
      frontend::ParseStatus::Ok) {
    return false;
  }
  if (!frontend::tile_group_is_complete(frame, group)) {
    *error = "the tile group covers " + std::to_string(group.tiles.size()) +
             " of the frame's " + std::to_string(frame.tiles.cols) + "x" +
             std::to_string(frame.tiles.rows) + " tiles";
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
      if (n % 50 == 0)
        ds_debug_log("perf stage %s: avg %.1f ms, worst %.1f ms over %ld",
                  name, total / n, worst, n);
    }
  };
  static Stage st_tiles{"tiles"}, st_deq{"dequantise"}, st_plan{"plan"},
               st_gpu{"gpu_decode"}, st_nv12{"nv12_pack"};
  LARGE_INTEGER pc0, pc1;

  QueryPerformanceCounter(&pc0);
  const frontend::ParseStatus tile_status = frontend::decode_tiles(
      st->seq, frame, &st->refs, group.tiles.data(), group.tiles.size(), data, size,
      &decoded, error);
  QueryPerformanceCounter(&pc1);
  st_tiles.add(perf.ms(pc0, pc1));
  if (tile_status != frontend::ParseStatus::Ok) {
    if (error->empty()) *error = decoded.stop_reason;
    return false;
  }

  frontend::DequantisedTile quant;
  QueryPerformanceCounter(&pc0);
  frontend::dequantise_tile(frame, st->seq.bit_depth, decoded, &quant);
  QueryPerformanceCounter(&pc1);
  st_deq.add(perf.ms(pc0, pc1));

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
  frontend::refresh_ref_pool(frame, &st->refs, decoded.ref_mvs, decoded.cdf,
                             decoded.seg_map);

  FrameSyntax syntax;
  QueryPerformanceCounter(&pc0);
  if (!frontend::plan_from_tile(st->seq, frame, decoded, std::move(quant), &syntax, error,
                                &reference_info)) {
    return false;
  }
  QueryPerformanceCounter(&pc1);
  st_plan.add(perf.ms(pc0, pc1));

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
  if (pipelined && kept) {
    reserved = decoder_->reserve_reference();
    if (reserved < 0) {
      /* every slot may be marked because queued frames still hold the ones
       * the pool dropped; drain the queue and try again before giving up */
      finish();
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
        for (int slot : deferred_release) decoder_->release_reference(slot);
        deferred_release.clear();
        ds_debug_log("reference reservation failed; falling back to the "
                     "synchronous decode path");
      } else {
        *error = "reference pool exhausted";
        return false;
      }
    }
  }

  if (pipelined) {
    for (int i = 0; i < 8; ++i) {
      if (!(frame.refresh_frame_flags & (1 << i))) continue;
      const int previous = st->pool_slot[i];
      st->pool_slot[i] = reserved;
      if (previous != reserved) defer_release(previous);
    }

    /* Build the frame plan *here*, on the demuxer's thread, so it overlaps the
     * kernels of the job still running on the worker. The worker then only has
     * to submit the plan. */
    PipeJob job;
    if (preplan_) {
      if (!decoder_->plan_frame(syntax, &job.frame_plan, error)) return false;
    }
    job.syntax = std::move(syntax);
    job.slot = reserved;
    job.refresh_flags = frame.refresh_frame_flags;
    job.show_frame = frame.show_frame;
    job.pitch = pitch;
    job.sink = sink;
    job.release_slots = std::move(deferred_release);
    queue_job(std::move(job), error);
    return true;
  }

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
