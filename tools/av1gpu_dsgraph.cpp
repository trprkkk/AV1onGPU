/*
 * av1gpu_dsgraph - exercise the DirectShow filter's real COM surface
 *
 * A test harness for av1gpu_dshow.dll that avoids two things at once: it does
 * not need a splitter, and it does not need a registered DLL. It
 *
 *   1. LoadLibrary's the DLL the way a player does and gets the class factory
 *      straight from DllGetClassObject;
 *   2. connects a minimal source pin into the filter's input pin (the way a
 *      splitter's output pin connects), then the filter's output pin into a
 *      minimal sink that plays the renderer role: ReceiveConnection,
 *      IMemInputPin, an allocator with the commit/decommit lifecycle. The
 *      input is connected first because the output geometry comes from the
 *      input media type - the same order IGraphBuilder::Render builds;
 *   3. feeds the input pin the temporal units of an .ivf file through
 *      Receive, one IMediaSample per unit;
 *   4. counts the NV12 frames the filter delivers, and checks the first
 *      frame's size against the connected media type, so a decode failure
 *      shows up as a wrong count instead of a silent pass;
 *   5. checks the small COM details players rely on: QueryAccept per
 *      subtype, the output's media types, FindPin, EndOfStream, a flush plus
 *      NewSegment round trip (the seek path), and a Stop/Pause cycle (the
 *      allocator decommit/commit path).
 *
 * The sink and source are deliberately tiny: NullRenderer would also work,
 * but it wants a graph and a clock to not block, which makes the test slower
 * and flakier for no extra coverage.
 */
#include <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <dshow.h>
/* VIDEOINFOHEADER2 is not in dshow.h's default include set */
#include <dvdmedia.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

/* {B7E0E1C2-5A3D-4F6B-9C48-2A1D7E5F3B60} - keep in step with dsfilter */
const GUID kDecoderClsid = {
    0xb7e0e1c2, 0x5a3d, 0x4f6b, {0x9c, 0x48, 0x2a, 0x1d, 0x7e, 0x5f, 0x3b, 0x60}};
/* 'AV01' */
const GUID kSubtypeAv1 = {
    0x31305641, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
/* 'H264', for the QueryAccept rejection check */
const GUID kSubtypeH264 = {
    0x34363248, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

[[noreturn]] void die(const std::string& what, HRESULT hr) {
  std::printf("av1gpu_dsgraph: %s failed: 0x%08lx\n", what.c_str(),
              static_cast<unsigned long>(hr));
  std::exit(1);
}

[[noreturn]] void die(const std::string& what) {
  std::printf("av1gpu_dsgraph: %s\n", what.c_str());
  std::exit(1);
}

void delete_media_type(AM_MEDIA_TYPE* mt) {
  if (!mt) return;
  if (mt->cbFormat) CoTaskMemFree(mt->pbFormat);
  CoTaskMemFree(mt);
}

LONG interlocked(LONG* p, LONG delta) { return InterlockedExchangeAdd(p, delta) + delta; }

/* ---------------------------------------------------------------- IVF read */

struct Unit {
  std::vector<uint8_t> data;
};

void read_ivf(const std::string& path, std::vector<Unit>* units) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) die("cannot open " + path);
  uint8_t header[32];
  if (std::fread(header, 1, sizeof(header), f) != sizeof(header) ||
      std::memcmp(header, "DKIF", 4) != 0) {
    std::fclose(f);
    die(path + " is not an IVF file");
  }
  for (;;) {
    uint8_t fh[12];
    if (std::fread(fh, 1, sizeof(fh), f) != sizeof(fh)) break;
    const size_t size = static_cast<size_t>(fh[0]) | (static_cast<size_t>(fh[1]) << 8) |
                        (static_cast<size_t>(fh[2]) << 16) |
                        (static_cast<size_t>(fh[3]) << 24);
    if (size == 0) break;
    Unit unit;
    unit.data.resize(size);
    if (std::fread(unit.data.data(), 1, size, f) != size) {
      std::fclose(f);
      die(path + " ends inside a temporal unit");
    }
    units->push_back(std::move(unit));
  }
  std::fclose(f);
}

/* --------------------------------------------------------- helper objects */

AM_MEDIA_TYPE* clone_type(const AM_MEDIA_TYPE& src) {
  AM_MEDIA_TYPE* mt = static_cast<AM_MEDIA_TYPE*>(CoTaskMemAlloc(sizeof(AM_MEDIA_TYPE)));
  if (!mt) return nullptr;
  *mt = src;
  mt->pUnk = nullptr;
  mt->pbFormat = nullptr;
  mt->cbFormat = 0;
  if (src.cbFormat && src.pbFormat) {
    mt->pbFormat = static_cast<BYTE*>(CoTaskMemAlloc(src.cbFormat));
    if (!mt->pbFormat) {
      CoTaskMemFree(mt);
      return nullptr;
    }
    CopyMemory(mt->pbFormat, src.pbFormat, src.cbFormat);
    mt->cbFormat = src.cbFormat;
  }
  return mt;
}

/* A media sample the source hands to Receive; the decoder filter only reads
 * from it (pointer, length, times, discontinuity flag). */
class FakeSample : public IMediaSample {
 public:
  FakeSample(const uint8_t* data, long size, REFERENCE_TIME start, REFERENCE_TIME stop,
             bool discontinuity)
      : data_(data), size_(size), start_(start), stop_(stop), disc_(discontinuity) {}

  ULONG __stdcall AddRef() override { return interlocked(&refs_, 1); }
  ULONG __stdcall Release() override {
    const LONG r = interlocked(&refs_, -1);
    if (r == 0) delete this;
    return static_cast<ULONG>(r);
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IMediaSample) {
      *out = static_cast<IMediaSample*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }

  HRESULT __stdcall GetPointer(BYTE** buffer) override {
    if (!buffer) return E_POINTER;
    *buffer = const_cast<BYTE*>(data_); /* the filter reads; the interface is not const */
    return S_OK;
  }
  long __stdcall GetSize() override { return size_; }
  HRESULT __stdcall GetTime(REFERENCE_TIME* start, REFERENCE_TIME* stop) override {
    if (!start || !stop) return E_POINTER;
    *start = start_;
    *stop = stop_;
    return S_OK;
  }
  HRESULT __stdcall SetTime(REFERENCE_TIME*, REFERENCE_TIME*) override { return E_NOTIMPL; }
  HRESULT __stdcall GetMediaTime(LONGLONG*, LONGLONG*) override { return E_NOTIMPL; }
  HRESULT __stdcall SetMediaTime(LONGLONG*, LONGLONG*) override { return E_NOTIMPL; }
  HRESULT __stdcall IsSyncPoint() override { return S_OK; }
  HRESULT __stdcall SetSyncPoint(BOOL) override { return E_NOTIMPL; }
  HRESULT __stdcall IsPreroll() override { return S_FALSE; }
  HRESULT __stdcall SetPreroll(BOOL) override { return E_NOTIMPL; }
  long __stdcall GetActualDataLength() override { return size_; }
  HRESULT __stdcall SetActualDataLength(long) override { return E_NOTIMPL; }
  HRESULT __stdcall GetMediaType(AM_MEDIA_TYPE**) override { return S_FALSE; }
  HRESULT __stdcall SetMediaType(AM_MEDIA_TYPE*) override { return E_NOTIMPL; }
  HRESULT __stdcall IsDiscontinuity() override { return disc_ ? S_OK : S_FALSE; }
  HRESULT __stdcall SetDiscontinuity(BOOL) override { return E_NOTIMPL; }

 private:
  LONG refs_ = 1;
  const uint8_t* data_;
  long size_;
  REFERENCE_TIME start_, stop_;
  bool disc_;
};

/* The minimal allocator a renderer's input pin would hand out: enough bytes
 * for one frame, one sample per GetBuffer, and the commit rule. */
class SinkAllocator;

class SinkSample : public IMediaSample {
 public:
  explicit SinkSample(size_t bytes) : buffer_(bytes) {}

  ULONG __stdcall AddRef() override { return interlocked(&refs_, 1); }
  ULONG __stdcall Release() override {
    const LONG r = interlocked(&refs_, -1);
    if (r == 0) delete this;
    return static_cast<ULONG>(r);
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IMediaSample) {
      *out = static_cast<IMediaSample*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }

  HRESULT __stdcall GetPointer(BYTE** buffer) override {
    if (!buffer) return E_POINTER;
    *buffer = buffer_.data();
    return S_OK;
  }  long __stdcall GetSize() override { return static_cast<long>(buffer_.size()); }
  HRESULT __stdcall GetTime(REFERENCE_TIME* start, REFERENCE_TIME* stop) override {
    if (no_time_) return VFW_E_SAMPLE_TIME_NOT_SET;
    *start = start_;
    *stop = stop_;
    return S_OK;
  }
  HRESULT __stdcall SetTime(REFERENCE_TIME* start, REFERENCE_TIME* stop) override {
    if (!start || !stop) {
      no_time_ = true;
      return S_OK;
    }
    start_ = *start;
    stop_ = *stop;
    no_time_ = false;
    return S_OK;
  }
  HRESULT __stdcall GetMediaTime(LONGLONG*, LONGLONG*) override { return E_NOTIMPL; }
  HRESULT __stdcall SetMediaTime(LONGLONG*, LONGLONG*) override { return S_OK; }
  HRESULT __stdcall IsSyncPoint() override { return sync_ ? S_OK : S_FALSE; }
  HRESULT __stdcall SetSyncPoint(BOOL is) override {
    sync_ = is == TRUE;
    return S_OK;
  }
  HRESULT __stdcall IsPreroll() override { return preroll_ ? S_OK : S_FALSE; }
  HRESULT __stdcall SetPreroll(BOOL is) override {
    preroll_ = is == TRUE;
    return S_OK;
  }
  long __stdcall GetActualDataLength() override { return actual_; }
  HRESULT __stdcall SetActualDataLength(long n) override {
    if (n < 0 || static_cast<size_t>(n) > buffer_.size()) return VFW_E_BUFFER_OVERFLOW;
    actual_ = n;
    return S_OK;
  }
  HRESULT __stdcall GetMediaType(AM_MEDIA_TYPE**) override { return S_FALSE; }
  HRESULT __stdcall SetMediaType(AM_MEDIA_TYPE*) override { return E_NOTIMPL; }
  HRESULT __stdcall IsDiscontinuity() override { return disc_ ? S_OK : S_FALSE; }
  HRESULT __stdcall SetDiscontinuity(BOOL is) override {
    disc_ = is == TRUE;
    return S_OK;
  }

 private:
  LONG refs_ = 1;
  std::vector<BYTE> buffer_;
  long actual_ = 0;
  REFERENCE_TIME start_ = 0, stop_ = 0;
  bool no_time_ = true, sync_ = false, preroll_ = false, disc_ = false;
};

class SinkAllocator : public IMemAllocator {
 public:
  ULONG __stdcall AddRef() override { return interlocked(&refs_, 1); }
  ULONG __stdcall Release() override {
    const LONG r = interlocked(&refs_, -1);
    if (r == 0) delete this;
    return static_cast<ULONG>(r);
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IMemAllocator) {
      *out = static_cast<IMemAllocator*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }

  HRESULT __stdcall SetProperties(ALLOCATOR_PROPERTIES* want,
                                  ALLOCATOR_PROPERTIES* actual) override {
    if (!want || !actual) return E_POINTER;
    if (want->cBuffers <= 0 || want->cbBuffer <= 0) {
      return E_INVALIDARG;
    }
    props_ = *want;
    *actual = props_;
    return S_OK;
  }
  HRESULT __stdcall GetProperties(ALLOCATOR_PROPERTIES* actual) override {
    if (!actual) return E_POINTER;
    *actual = props_;
    return S_OK;
  }
  HRESULT __stdcall Commit() override {
    committed_ = TRUE;
    return S_OK;
  }
  HRESULT __stdcall Decommit() override {
    committed_ = FALSE;
    return S_OK;
  }
  HRESULT __stdcall GetBuffer(IMediaSample** sample, REFERENCE_TIME*, REFERENCE_TIME*,
                              DWORD) override {
    if (!sample) return E_POINTER;
    if (!committed_) return VFW_E_NOT_COMMITTED;
    *sample = new (std::nothrow) SinkSample(static_cast<size_t>(props_.cbBuffer));
    return *sample ? S_OK : E_OUTOFMEMORY;
  }
  HRESULT __stdcall ReleaseBuffer(IMediaSample*) override { return S_OK; }

 private:
  LONG refs_ = 1;
  ALLOCATOR_PROPERTIES props_{};
  BOOL committed_ = FALSE;
};

/* Counts what the decoder delivers, and keeps the first frame for a size
 * check. Plays the role of a renderer's input pin. */
class SinkPin : public IPin, public IMemInputPin {
 public:
  size_t frames() const { return frames_; }
  bool connected() const { return peer_ != nullptr; }
  bool got_eos() const { return eos_; }
  bool notified() const { return notified_; }
  const std::vector<BYTE>& first_frame() const { return first_frame_; }

  ULONG __stdcall AddRef() override { return interlocked(&refs_, 1); }
  ULONG __stdcall Release() override {
    const LONG r = interlocked(&refs_, -1);
    if (r == 0) delete this;
    return static_cast<ULONG>(r);
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IPin) {
      *out = static_cast<IPin*>(this);
    } else if (riid == IID_IMemInputPin) {
      *out = static_cast<IMemInputPin*>(this);
    } else {
      *out = nullptr;
      return E_NOINTERFACE;
    }
    reinterpret_cast<IUnknown*>(*out)->AddRef();
    return S_OK;
  }

  HRESULT __stdcall Connect(IPin*, const AM_MEDIA_TYPE*) override {
    return E_UNEXPECTED;
  }
  HRESULT __stdcall ReceiveConnection(IPin* peer, const AM_MEDIA_TYPE* mt) override {
    if (!peer || !mt) return E_POINTER;
    peer_ = peer;
    return S_OK;
  }
  HRESULT __stdcall Disconnect() override {
    peer_ = nullptr;
    return S_OK;
  }
  HRESULT __stdcall ConnectedTo(IPin** peer) override {
    if (!peer) return E_POINTER;
    *peer = peer_;
    if (peer_) peer_->AddRef();
    return peer_ ? S_OK : VFW_E_NOT_CONNECTED;
  }
  HRESULT __stdcall ConnectionMediaType(AM_MEDIA_TYPE* mt) override {
    if (!mt) return E_POINTER;
    ZeroMemory(mt, sizeof(*mt));
    return VFW_E_NOT_CONNECTED; /* the harness reads the type from the filter */
  }
  HRESULT __stdcall QueryPinInfo(PIN_INFO* info) override {
    if (!info) return E_POINTER;
    info->pFilter = nullptr;
    info->dir = PINDIR_INPUT;
    wcscpy_s(info->achName, L"Sink");
    return S_OK;
  }
  HRESULT __stdcall QueryDirection(PIN_DIRECTION* dir) override {
    if (!dir) return E_POINTER;
    *dir = PINDIR_INPUT;
    return S_OK;
  }
  HRESULT __stdcall QueryId(LPWSTR* id) override {
    if (!id) return E_POINTER;
    *id = static_cast<LPWSTR>(CoTaskMemAlloc(8 * sizeof(wchar_t)));
    if (!*id) return E_OUTOFMEMORY;
    wcscpy_s(*id, 8, L"Sink");
    return S_OK;
  }
  HRESULT __stdcall QueryAccept(const AM_MEDIA_TYPE*) override { return S_OK; }
  HRESULT __stdcall EnumMediaTypes(IEnumMediaTypes**) override { return E_NOTIMPL; }
  HRESULT __stdcall QueryInternalConnections(IPin**, ULONG*) override {
    return E_NOTIMPL;
  }
  HRESULT __stdcall EndOfStream() override {
    eos_ = true;
    return S_OK;
  }
  HRESULT __stdcall BeginFlush() override { return S_OK; }
  HRESULT __stdcall EndFlush() override { return S_OK; }
  HRESULT __stdcall NewSegment(REFERENCE_TIME, REFERENCE_TIME, double) override {
    return S_OK;
  }

  /* IMemInputPin */
  HRESULT __stdcall GetAllocator(IMemAllocator** allocator) override {
    if (!allocator) return E_POINTER;
    if (!allocator_) allocator_ = new (std::nothrow) SinkAllocator();
    if (!allocator_) return E_OUTOFMEMORY;
    *allocator = allocator_;
    allocator_->AddRef();
    return S_OK;
  }
  HRESULT __stdcall NotifyAllocator(IMemAllocator* allocator, BOOL read_only) override {
    if (!allocator) return E_POINTER;
    (void)read_only;
    notified_ = true;
    return S_OK;
  }
  HRESULT __stdcall GetAllocatorRequirements(ALLOCATOR_PROPERTIES*) override {
    return E_NOTIMPL;
  }
  HRESULT __stdcall Receive(IMediaSample* sample) override {
    if (!sample) return E_POINTER;
    /* the protocol the filter has to keep: times that were set, a data
     * length that fits the sample */
    REFERENCE_TIME start = 0, stop = 0;
    if (FAILED(sample->GetTime(&start, &stop))) return VFW_E_SAMPLE_TIME_NOT_SET;
    if (stop <= start) return VFW_E_SAMPLE_TIME_NOT_SET;
    BYTE* data = nullptr;
    if (FAILED(sample->GetPointer(&data))) return E_FAIL;
    if (sample->GetActualDataLength() <= 0) return E_FAIL;
    if (first_frame_.empty()) {
      const long n = sample->GetActualDataLength();
      first_frame_.assign(data, data + n);
    }
    ++frames_;
    return S_OK;
  }
  HRESULT __stdcall ReceiveMultiple(IMediaSample** samples, long count,
                                    long* processed) override {
    if (!samples || !processed) return E_POINTER;
    for (long i = 0; i < count; ++i) {
      const HRESULT hr = Receive(samples[i]);
      if (FAILED(hr)) return hr;
    }
    *processed = count;
    return S_OK;
  }
  HRESULT __stdcall ReceiveCanBlock() override { return S_OK; }

 private:
  LONG refs_ = 1;
  IPin* peer_ = nullptr;
  IMemAllocator* allocator_ = nullptr;
  size_t frames_ = 0;
  std::vector<BYTE> first_frame_;
  bool eos_ = false, notified_ = false;
};

/* A splitter's output pin: accepts the connection, and that is all the
 * decoder's input pin needs from its peer. */
class SourcePin : public IPin {
 public:
  bool connected() const { return peer_ != nullptr; }

  ULONG __stdcall AddRef() override { return interlocked(&refs_, 1); }
  ULONG __stdcall Release() override {
    const LONG r = interlocked(&refs_, -1);
    if (r == 0) delete this;
    return static_cast<ULONG>(r);
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IPin) {
      *out = static_cast<IPin*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }

  HRESULT __stdcall Connect(IPin*, const AM_MEDIA_TYPE*) override {
    return E_UNEXPECTED;
  }
  HRESULT __stdcall ReceiveConnection(IPin* peer, const AM_MEDIA_TYPE* mt) override {
    if (!peer || !mt) return E_POINTER;
    peer_ = peer;
    return S_OK;
  }
  HRESULT __stdcall Disconnect() override {
    peer_ = nullptr;
    return S_OK;
  }
  HRESULT __stdcall ConnectedTo(IPin** peer) override {
    if (!peer) return E_POINTER;
    *peer = peer_;
    if (peer_) peer_->AddRef();
    return peer_ ? S_OK : VFW_E_NOT_CONNECTED;
  }
  HRESULT __stdcall ConnectionMediaType(AM_MEDIA_TYPE* mt) override {
    if (!mt) return E_POINTER;
    ZeroMemory(mt, sizeof(*mt));
    return VFW_E_NOT_CONNECTED;
  }
  HRESULT __stdcall QueryPinInfo(PIN_INFO* info) override {
    if (!info) return E_POINTER;
    info->pFilter = nullptr;
    info->dir = PINDIR_OUTPUT;
    wcscpy_s(info->achName, L"AV1 Source");
    return S_OK;
  }
  HRESULT __stdcall QueryDirection(PIN_DIRECTION* dir) override {
    if (!dir) return E_POINTER;
    *dir = PINDIR_OUTPUT;
    return S_OK;
  }
  HRESULT __stdcall QueryId(LPWSTR* id) override {
    if (!id) return E_POINTER;
    *id = static_cast<LPWSTR>(CoTaskMemAlloc(8 * sizeof(wchar_t)));
    if (!*id) return E_OUTOFMEMORY;
    wcscpy_s(*id, 8, L"Source");
    return S_OK;
  }
  HRESULT __stdcall QueryAccept(const AM_MEDIA_TYPE*) override { return S_OK; }
  HRESULT __stdcall EnumMediaTypes(IEnumMediaTypes**) override { return E_NOTIMPL; }
  HRESULT __stdcall QueryInternalConnections(IPin**, ULONG*) override {
    return E_NOTIMPL;
  }
  HRESULT __stdcall EndOfStream() override { return S_OK; }
  HRESULT __stdcall BeginFlush() override { return S_OK; }
  HRESULT __stdcall EndFlush() override { return S_OK; }
  HRESULT __stdcall NewSegment(REFERENCE_TIME, REFERENCE_TIME, double) override {
    return S_OK;
  }

 private:
  LONG refs_ = 1;
  IPin* peer_ = nullptr;
};

/* Builds the AV1 input media type: VideoInfo2, one temporal unit per sample,
 * dimensions the caller knows from the fixture. */
AM_MEDIA_TYPE* make_av1_type(int width, int height) {
  VIDEOINFOHEADER2* v = static_cast<VIDEOINFOHEADER2*>(CoTaskMemAlloc(sizeof(*v)));
  if (!v) return nullptr;
  ZeroMemory(v, sizeof(*v));
  v->rcSource.right = width;
  v->rcSource.bottom = height;
  v->rcTarget = v->rcSource;
  v->AvgTimePerFrame = 333667;
  v->dwPictAspectRatioX = static_cast<DWORD>(width);
  v->dwPictAspectRatioY = static_cast<DWORD>(height);
  v->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  v->bmiHeader.biWidth = width;
  v->bmiHeader.biHeight = height;
  v->bmiHeader.biPlanes = 1;
  v->bmiHeader.biCompression = '1V4A'; /* 'AV01' little endian */
  AM_MEDIA_TYPE* mt = static_cast<AM_MEDIA_TYPE*>(CoTaskMemAlloc(sizeof(*mt)));
  if (!mt) {
    CoTaskMemFree(v);
    return nullptr;
  }
  ZeroMemory(mt, sizeof(*mt));
  mt->majortype = MEDIATYPE_Video;
  mt->subtype = kSubtypeAv1;
  mt->bFixedSizeSamples = FALSE;
  mt->bTemporalCompression = TRUE;
  mt->lSampleSize = 1;
  mt->formattype = FORMAT_VideoInfo2;
  mt->cbFormat = sizeof(VIDEOINFOHEADER2);
  mt->pbFormat = reinterpret_cast<BYTE*>(v);
  return mt;
}

/* ------------------------------------------------------------- fixtures */

struct KnownStream {
  const char* name;
  int width, height, frames;
};

const KnownStream kKnownStreams[] = {
    {"lw_t2", 320, 240, 30},
    {"lr", 320, 240, 8},
    {"tiles", 256, 160, 4},
    {"grain", 320, 240, 30},
    {"superres", 320, 240, 30},
    {"odd_size", 130, 100, 6},
    {"lossless", 160, 120, 4},
};

} /* namespace */

int main(int argc, char** argv) {
  std::string dll_path, stream_path;
  int expect = -1; /* -1: just require at least one frame */
  int width = 0, height = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--dll") {
      dll_path = next();
    } else if (arg == "--stream") {
      stream_path = next();
    } else if (arg == "--expect") {
      expect = std::atoi(next().c_str());
    } else if (arg == "--size") {
      const std::string s = next();
      const size_t x = s.find('x');
      if (x == std::string::npos) die("--size wants WxH");
      width = std::atoi(s.substr(0, x).c_str());
      height = std::atoi(s.substr(x + 1).c_str());
    } else {
      die("unknown option " + arg);
    }
  }
  if (dll_path.empty() || stream_path.empty()) {
    std::printf("usage: av1gpu_dsgraph --dll av1gpu_dshow.dll --stream x.ivf "
                "[--expect N] [--size WxH]\n");
    return 2;
  }
  const std::string stem = stream_path.substr(stream_path.find_last_of("/\\") + 1);
  if (width == 0) {
    bool known = false;
    for (const KnownStream& s : kKnownStreams) {
      if (stem.find(s.name) == 0) {
        width = s.width;
        height = s.height;
        if (expect < 0) expect = s.frames;
        known = true;
        break;
      }
    }
    if (!known) die("pass --size WxH for " + stream_path);
  }

  HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  if (FAILED(hr)) die("CoInitializeEx", hr);

  /* 1. the DLL, the way a player loads an unregistered filter */
  HMODULE module = LoadLibraryA(dll_path.c_str());
  if (!module) {
    die("LoadLibrary " + dll_path + " failed: " + std::to_string(GetLastError()));
  }
  const auto get_class_object = reinterpret_cast<HRESULT(__stdcall*)(REFCLSID, REFIID,
                                                                     void**)>(
      GetProcAddress(module, "DllGetClassObject"));
  if (!get_class_object) die("the DLL does not export DllGetClassObject");
  /* a class id the DLL does not know has to be refused */
  {
    const GUID kUnknown = {0x12345678, 0x1111, 0x2222, {0x33, 0x44, 0x55, 0x66, 0x77, 0x88}};
    IClassFactory* refused = nullptr;
    if (get_class_object(kUnknown, IID_IClassFactory, reinterpret_cast<void**>(&refused)) !=
        CLASS_E_CLASSNOTAVAILABLE) {
      die("DllGetClassObject accepted an unknown CLSID");
    }
  }

  IClassFactory* factory = nullptr;
  hr = get_class_object(kDecoderClsid, IID_IClassFactory, reinterpret_cast<void**>(&factory));
  if (FAILED(hr)) die("DllGetClassObject", hr);
  IBaseFilter* filter = nullptr;
  hr = factory->CreateInstance(nullptr, IID_IBaseFilter, reinterpret_cast<void**>(&filter));
  factory->Release();
  if (FAILED(hr)) die("CreateInstance(IBaseFilter)", hr);

  CLSID clsid{};
  hr = filter->GetClassID(&clsid);
  if (FAILED(hr) || clsid != kDecoderClsid) die("GetClassID mismatch");
  FILTER_INFO info{};
  hr = filter->QueryFilterInfo(&info);
  if (FAILED(hr) || info.pGraph != nullptr ||
      wcscmp(info.achName, L"AV1onGPU AV1 Decoder") != 0) {
    if (SUCCEEDED(hr) && info.pGraph) info.pGraph->Release();
    die("QueryFilterInfo: unexpected name or graph");
  }

  /* 2. the pins */
  IPin* in_pin = nullptr;
  IPin* out_pin = nullptr;
  {
    IEnumPins* e = nullptr;
    hr = filter->EnumPins(&e);
    if (FAILED(hr)) die("EnumPins", hr);
    IPin* pin = nullptr;
    while (e->Next(1, &pin, nullptr) == S_OK) {
      PIN_DIRECTION dir;
      pin->QueryDirection(&dir);
      if (dir == PINDIR_INPUT && !in_pin) {
        in_pin = pin;
      } else if (dir == PINDIR_OUTPUT && !out_pin) {
        out_pin = pin;
      } else {
        pin->Release();
      }
    }
    e->Release();
  }
  if (!in_pin || !out_pin) die("the filter does not have one input and one output pin");
  /* FindPin answers by id, and misses with the right error */
  {
    IPin* by_id = nullptr;
    hr = filter->FindPin(L"In", &by_id);
    if (FAILED(hr)) die("FindPin(In)", hr);
    by_id->Release();
    hr = filter->FindPin(L"Out", &by_id);
    if (FAILED(hr)) die("FindPin(Out)", hr);
    by_id->Release();
    hr = filter->FindPin(L"Nope", &by_id);
    if (hr != VFW_E_NOT_FOUND) die("FindPin(Nope) should be VFW_E_NOT_FOUND");
  }

  /* 3. connect. The input goes first: the output geometry comes from the
   * input media type, and that is the order IGraphBuilder::Render uses for a
   * decoder. */
  SourcePin source;
  AM_MEDIA_TYPE* av1_type = make_av1_type(width, height);
  if (!av1_type) die("out of memory for the input media type");
  /* QueryAccept: yes for AV1, no for anything else */
  if (in_pin->QueryAccept(av1_type) != S_OK) die("QueryAccept(AV01) should be S_OK");
  {
    AM_MEDIA_TYPE wrong = *av1_type;
    wrong.subtype = kSubtypeH264;
    if (in_pin->QueryAccept(&wrong) != S_FALSE) {
      die("QueryAccept(H264) should be S_FALSE");
    }
    AM_MEDIA_TYPE no_dims = *av1_type;
    no_dims.formattype = GUID_NULL;
    no_dims.pbFormat = nullptr;
    no_dims.cbFormat = 0;
    if (in_pin->QueryAccept(&no_dims) != S_FALSE) {
      die("QueryAccept without dimensions should be S_FALSE");
    }
  }
  /* the filter's input pin receives the connection from the source, the way
   * the graph calls it when the splitter's output pin connects */
  hr = in_pin->ReceiveConnection(&source, av1_type);
  if (FAILED(hr)) die("input pin ReceiveConnection", hr);
  delete_media_type(av1_type);
  {
    IPin* peer = nullptr;
    hr = in_pin->ConnectedTo(&peer);
    if (FAILED(hr) || peer != static_cast<IPin*>(&source)) die("input ConnectedTo");
    if (peer) peer->Release();
  }

  /* with the input connected, the output knows its geometry and names NV12 */
  {
    IEnumMediaTypes* e = nullptr;
    hr = out_pin->EnumMediaTypes(&e);
    if (FAILED(hr)) die("output EnumMediaTypes", hr);
    AM_MEDIA_TYPE* mt = nullptr;
    ULONG fetched = 0;
    if (e->Next(1, &mt, &fetched) != S_OK || fetched != 1 || !mt) {
      die("the output pin proposes no media type");
    }
    if (mt->majortype != MEDIATYPE_Video || mt->subtype != MEDIASUBTYPE_NV12 ||
        mt->formattype != FORMAT_VideoInfo2) {
      die("the output pin's proposed type is not VideoInfo2/NV12");
    }
    const VIDEOINFOHEADER2* v = reinterpret_cast<const VIDEOINFOHEADER2*>(mt->pbFormat);
    if (v->bmiHeader.biWidth < width ||
        (v->bmiHeader.biHeight != height && -v->bmiHeader.biHeight != height)) {
      die("the proposed type's geometry does not match the stream");
    }
    delete_media_type(mt);
    e->Release();
  }

  SinkPin sink;
  hr = out_pin->Connect(&sink, nullptr);
  if (FAILED(hr)) die("output pin Connect(sink)", hr);
  if (!sink.notified()) die("the output pin never called NotifyAllocator");
  {
    IPin* peer = nullptr;
    hr = out_pin->ConnectedTo(&peer);
    if (FAILED(hr) || peer != static_cast<IPin*>(&sink)) die("output ConnectedTo");
    if (peer) peer->Release();
    AM_MEDIA_TYPE mt{};
    hr = out_pin->ConnectionMediaType(&mt);
    if (FAILED(hr) || mt.subtype != MEDIASUBTYPE_NV12) die("output ConnectionMediaType");
    if (mt.pbFormat) CoTaskMemFree(mt.pbFormat);
  }

  /* 4. active, then feed every temporal unit */
  hr = filter->Pause();
  if (FAILED(hr)) die("Pause", hr);
  {
    FILTER_STATE state = State_Stopped;
    hr = filter->GetState(0, &state);
    if (FAILED(hr) || state != State_Paused) die("GetState after Pause");
  }

  IMemInputPin* input = nullptr;
  hr = in_pin->QueryInterface(IID_IMemInputPin, reinterpret_cast<void**>(&input));
  if (FAILED(hr)) die("input pin is not an IMemInputPin", hr);

  std::vector<Unit> units;
  read_ivf(stream_path, &units);
  if (units.empty()) die("the stream has no temporal units");

  REFERENCE_TIME now = 0;
  for (size_t i = 0; i < units.size(); ++i) {
    const REFERENCE_TIME start = now;
    now += 333667;
    FakeSample sample(units[i].data.data(), static_cast<long>(units[i].data.size()),
                      start, now, i == 0);
    hr = input->Receive(&sample);
    if (FAILED(hr)) {
      input->Release();
      die("Receive failed on temporal unit " + std::to_string(i), hr);
    }
  }

  /* 5. what came out */
  const size_t got = sink.frames();
  std::printf("av1gpu_dsgraph: %s: %zu temporal unit(s) in, %zu frame(s) out", stem.c_str(),
              units.size(), got);
  if (expect >= 0) std::printf(", expected %d", expect);
  std::printf("\n");
  if (got == 0) die("the filter delivered no frames");
  if (expect >= 0 && static_cast<int>(got) != expect) {
    input->Release();
    die("delivered " + std::to_string(got) + " frames, expected " + std::to_string(expect));
  }

  /* the first delivered frame fills the connected NV12 geometry */
  {
    AM_MEDIA_TYPE mt{};
    hr = out_pin->ConnectionMediaType(&mt);
    if (FAILED(hr)) die("output ConnectionMediaType", hr);
    if (mt.formattype == FORMAT_VideoInfo2 && mt.cbFormat >= sizeof(VIDEOINFOHEADER2)) {
      const VIDEOINFOHEADER2* v = reinterpret_cast<const VIDEOINFOHEADER2*>(mt.pbFormat);
      const size_t want = static_cast<size_t>(v->bmiHeader.biWidth) *
                          (v->bmiHeader.biHeight < 0 ? -v->bmiHeader.biHeight
                                                     : v->bmiHeader.biHeight) * 3 / 2;
      if (sink.first_frame().size() < want) {
        die("the first frame is " + std::to_string(sink.first_frame().size()) +
            " bytes, want " + std::to_string(want));
      }
    }
    if (mt.pbFormat) CoTaskMemFree(mt.pbFormat);
  }

  /* end of stream travels downstream */
  hr = in_pin->EndOfStream();
  if (FAILED(hr)) die("EndOfStream", hr);
  if (!sink.got_eos()) die("EndOfStream never reached the sink");

  /* a flush plus NewSegment is the seek path: the decoder starts over, and
   * the first units decode again from the key frame */
  {
    in_pin->BeginFlush();
    in_pin->EndFlush();
    in_pin->NewSegment(0, 0, 1.0);
    const size_t before = sink.frames();
    now = 0;
    for (size_t i = 0; i < units.size() && i < 3; ++i) {
      const REFERENCE_TIME start = now;
      now += 333667;
      FakeSample sample(units[i].data.data(), static_cast<long>(units[i].data.size()),
                        start, now, i == 0);
      hr = input->Receive(&sample);
      if (FAILED(hr)) {
        input->Release();
        die("Receive after flush failed on unit " + std::to_string(i), hr);
      }
    }
    const size_t after = sink.frames() - before;
    std::printf("  after flush+NewSegment: %zu more frame(s) from 3 unit(s)\n", after);
    if (after == 0) die("no frames after a flush and re-feed");
  }

  /* 6. a Stop/Pause cycle: Stop decommits the allocator, Pause commits it
   * again, and the decoder still decodes */
  hr = filter->Stop();
  if (FAILED(hr)) die("Stop", hr);
  hr = filter->Pause();
  if (FAILED(hr)) die("re-Pause", hr);
  {
    FakeSample sample(units[0].data.data(), static_cast<long>(units[0].data.size()), 0,
                      333667, false);
    hr = input->Receive(&sample);
    if (FAILED(hr)) die("Receive after a Stop/Pause cycle failed", hr);
  }

  input->Release();
  in_pin->Release();
  out_pin->Release();
  filter->Release();
  CoUninitialize();
  std::printf("av1gpu_dsgraph: all checks passed\n");
  return 0;
}
