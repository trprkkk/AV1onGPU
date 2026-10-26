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
   * slot names a backend pool slot plus the picture, for show_existing_frame.
   * A stream may refresh several AV1 slots with one picture, so a backend slot
   * is released only when no AV1 slot names it any more. */
  int pool_slot[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  Frame pool_picture[8];
  bool pool_picture_valid[8] = {false, false, false, false, false,
                                false, false, false};

  void clear() {
    refs = frontend::RefPool();
    for (int i = 0; i < 8; ++i) {
      pool_slot[i] = -1;
      pool_picture[i] = Frame();
      pool_picture_valid[i] = false;
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

StreamDecoder::~StreamDecoder() { delete static_cast<StreamState*>(state_); }

bool StreamDecoder::init(std::string* error) {
  decoder_ = Decoder::create(options_, error);
  return decoder_ != nullptr;
}

const DeviceInfo& StreamDecoder::device() const { return decoder_->device(); }

void StreamDecoder::reset_stream() {
  pending_.clear();
  static_cast<StreamState*>(state_)->clear();
  if (decoder_) decoder_->reset();
}

bool StreamDecoder::push(const uint8_t* data, size_t size, size_t pitch,
                         const Sink& sink, std::string* error) {
  /* A stream whose OBUs never complete would grow the buffer without bound;
   * 64 MiB is far past any real temporal unit. */
  constexpr size_t kPendingLimit = 64u << 20;
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
    if (notice_.empty()) {
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
    if (notice_.empty()) notice_ = "OBU_TILE_LIST is not supported";
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
    if (which < 0 || which >= 8 || !st->pool_picture_valid[which]) {
      *error = "show_existing_frame names a slot that was never decoded";
      return false;
    }
    return emit(st->pool_picture[which], pitch, sink, error);
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

  FrameSyntax syntax;
  QueryPerformanceCounter(&pc0);
  if (!frontend::plan_from_tile(st->seq, frame, decoded, quant, &syntax, error,
                                &reference_info)) {
    return false;
  }
  QueryPerformanceCounter(&pc1);
  st_plan.add(perf.ms(pc0, pc1));

  /* Free the slots this frame refreshes but does not predict from before it is
   * decoded; the frame is allowed to refresh a slot it reads, so those stay. */
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
    st->pool_picture_valid[i] = false;
    release_slot(st, decoder_.get(), freed);
  }

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
  bool kept = false;
  for (int i = 0; i < 8; ++i) {
    if (!(frame.refresh_frame_flags & (1 << i))) continue;
    const int previous = st->pool_slot[i];
    st->pool_slot[i] = -1;
    release_slot(st, decoder_.get(), previous);
    st->pool_slot[i] = slot;
    st->pool_picture[i] = picture;
    st->pool_picture_valid[i] = true;
    kept = true;
  }
  if (!kept) release_slot(st, decoder_.get(), slot);

  frontend::refresh_ref_pool(frame, &st->refs, decoded.ref_mvs, decoded.cdf,
                             decoded.seg_map);
  return true;
}

} /* namespace dsfilter */
} /* namespace av1gpu */
