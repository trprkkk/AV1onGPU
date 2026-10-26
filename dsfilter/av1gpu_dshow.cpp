/*
 * AV1onGPU - DirectShow AV1 decoder filter
 *
 * av1gpu_dshow.dll registers one in process filter: an input pin that accepts
 * the AV1 elementary stream a splitter delivers (MEDIASUBTYPE_AV1 / 'av01',
 * one temporal unit per media sample) and an output pin that delivers NV12.
 * The decode runs on the CUDA or OpenCL backend of the av1gpu library, so a
 * GPU without AV1 fixed function hardware decodes the stream while the CPU
 * only parses and submits.
 *
 * Players: MPC-BE, MPC-HC and PotPlayer take the filter as an external filter
 * (add by DLL path, or by name after regsvr32), and every DirectShow graph
 * editor can insert it by CLSID. The COM surface is hand written: no
 * DirectShow base classes, so the only dependencies are the Windows SDK
 * headers and strmiids.lib for the standard IIDs.
 *
 * Deliberate v1 limitations:
 *  - the output is NV12 only, so a stream that is not 4:2:0 drops its frames
 *    (a message is logged once);
 *  - the output dimensions come from the input pin's media type. A splitter
 *    that reports no dimensions cannot be connected;
 *  - everything decodes synchronously inside Receive, one frame at a time,
 *    which keeps the code small and the latency zero.
 *
 * Debug output goes to OutputDebugString when the AV1GPU_DS_DEBUG environment
 * variable is set (visible in DbgView or a debugger).
 */
#include <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <dshow.h>
/* VIDEOINFOHEADER2 is not in dshow.h's default include set */
#include <dvdmedia.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "decode_core.hpp"

/* {B7E0E1C2-5A3D-4F6B-9C48-2A1D7E5F3B60} */
extern "C" const GUID CLSID_AV1GpuDecoder = {
    0xb7e0e1c2, 0x5a3d, 0x4f6b, {0x9c, 0x48, 0x2a, 0x1d, 0x7e, 0x5f, 0x3b, 0x60}};

/* 'AV01' and 'av01' as FOURCC media subtypes: {fourcc-0000-0010-8000-00AA00389B71} */
static const GUID kSubtypeAv1Upper = {
    0x31305641, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
static const GUID kSubtypeAv1Lower = {
    0x31307661, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

static const wchar_t kFilterNameW[] = L"AV1onGPU AV1 Decoder";
static const char kFilterNameA[] = "AV1onGPU AV1 Decoder";
static const char kClsidPath[] = "CLSID\\{B7E0E1C2-5A3D-4F6B-9C48-2A1D7E5F3B60}";
#ifndef MERIT_NORMAL
#define MERIT_NORMAL 0x00600000
#endif

using av1gpu::dsfilter::StreamDecoder;
using av1gpu::dsfilter::nv12_size;
using av1gpu::BackendKind;
using av1gpu::BackendOptions;
using av1gpu::DeviceInfo;

namespace {

/* ---------------------------------------------------------------- logging */

void debug_log(const char* fmt, ...) {
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
  OutputDebugStringA("av1gpu: ");
  OutputDebugStringA(line);
  OutputDebugStringA("\n");
  /* also append to a file: a crash in the host kills DbgView's live view, but
   * everything logged up to the crash is on disk */
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

/* ----------------------------------------------------------- media types */

AM_MEDIA_TYPE* make_media_type(const GUID* major, const GUID* subtype,
                               BOOL fixed_sizes, const GUID* format_type,
                               ULONG format_size, const void* format) {
  AM_MEDIA_TYPE* mt = static_cast<AM_MEDIA_TYPE*>(CoTaskMemAlloc(sizeof(AM_MEDIA_TYPE)));
  if (!mt) return nullptr;
  ZeroMemory(mt, sizeof(*mt));
  mt->majortype = *major;
  mt->subtype = *subtype;
  mt->bFixedSizeSamples = fixed_sizes;
  mt->bTemporalCompression = TRUE; /* an AV1 sample does not decode on its own */
  mt->lSampleSize = 1;
  mt->formattype = *format_type;
  if (format_size > 0) {
    mt->cbFormat = format_size;
    mt->pbFormat = static_cast<BYTE*>(CoTaskMemAlloc(format_size));
    if (!mt->pbFormat) {
      CoTaskMemFree(mt);
      return nullptr;
    }
    CopyMemory(mt->pbFormat, format, format_size);
  }
  return mt;
}

void free_media_type(AM_MEDIA_TYPE* mt) {
  if (!mt) return;
  if (mt->cbFormat && mt->pbFormat) CoTaskMemFree(mt->pbFormat);
  CoTaskMemFree(mt);
}

/* Releases the format block of a media type that lives inline - a pin's
 * member copy - and clears it. The struct itself is not a heap allocation,
 * so it must never go through CoTaskMemFree. */
void clear_media_type(AM_MEDIA_TYPE* mt) {
  if (!mt) return;
  if (mt->cbFormat && mt->pbFormat) CoTaskMemFree(mt->pbFormat);
  ZeroMemory(mt, sizeof(*mt));
}

AM_MEDIA_TYPE* copy_media_type(const AM_MEDIA_TYPE* src) {
  if (!src) return nullptr;
  AM_MEDIA_TYPE* mt = static_cast<AM_MEDIA_TYPE*>(CoTaskMemAlloc(sizeof(AM_MEDIA_TYPE)));
  if (!mt) return nullptr;
  *mt = *src;
  mt->pUnk = nullptr;
  mt->pbFormat = nullptr;
  mt->cbFormat = 0;
  if (src->cbFormat && src->pbFormat) {
    mt->pbFormat = static_cast<BYTE*>(CoTaskMemAlloc(src->cbFormat));
    if (!mt->pbFormat) {
      CoTaskMemFree(mt);
      return nullptr;
    }
    CopyMemory(mt->pbFormat, src->pbFormat, src->cbFormat);
    mt->cbFormat = src->cbFormat;
  }
  return mt;
}

/* Reads the picture dimensions out of an input media type; false when the
 * format block carries none (FORMAT_None, GUID_NULL, or a partial type). */
bool input_dimensions(const AM_MEDIA_TYPE* mt, int* width, int* height,
                      unsigned long long* avg_time_per_frame) {
  *width = 0;
  *height = 0;
  *avg_time_per_frame = 0;
  if (!mt || !mt->pbFormat) return false;
  if (mt->formattype == FORMAT_VideoInfo2 && mt->cbFormat >= sizeof(VIDEOINFOHEADER2)) {
    const VIDEOINFOHEADER2* v = reinterpret_cast<const VIDEOINFOHEADER2*>(mt->pbFormat);
    *width = v->bmiHeader.biWidth;
    *height = v->bmiHeader.biHeight < 0 ? -v->bmiHeader.biHeight : v->bmiHeader.biHeight;
    *avg_time_per_frame = v->AvgTimePerFrame;
    return *width > 0 && *height > 0;
  }
  if (mt->formattype == FORMAT_VideoInfo && mt->cbFormat >= sizeof(VIDEOINFOHEADER)) {
    const VIDEOINFOHEADER* v = reinterpret_cast<const VIDEOINFOHEADER*>(mt->pbFormat);
    *width = v->bmiHeader.biWidth;
    *height = v->bmiHeader.biHeight < 0 ? -v->bmiHeader.biHeight : v->bmiHeader.biHeight;
    *avg_time_per_frame = v->AvgTimePerFrame;
    return *width > 0 && *height > 0;
  }
  return false;
}

/* The format types the input accepts; FORMAT_VideoInfo2, FORMAT_VideoInfo and
 * FORMAT_None all appear in the wild. */
bool format_acceptable(const GUID& format_type) {
  return format_type == FORMAT_VideoInfo2 || format_type == FORMAT_VideoInfo ||
         format_type == FORMAT_None || format_type == GUID_NULL;
}

/* Fills a VIDEOINFOHEADER2 for NV12 at `width` x `height` with `stride` wide
 * rows. Positive biHeight: NV12 is top down in DirectShow. */
void fill_nv12_format(BYTE* format, int width, int height, int stride,
                      unsigned long long avg_time_per_frame) {
  VIDEOINFOHEADER2* v = reinterpret_cast<VIDEOINFOHEADER2*>(format);
  ZeroMemory(v, sizeof(VIDEOINFOHEADER2));
  v->rcSource.right = width;
  v->rcSource.bottom = height;
  v->rcTarget = v->rcSource;
  v->AvgTimePerFrame = avg_time_per_frame ? avg_time_per_frame : 333667;
  v->dwPictAspectRatioX = static_cast<DWORD>(width);
  v->dwPictAspectRatioY = static_cast<DWORD>(height);
  BITMAPINFOHEADER* bi = &v->bmiHeader;
  bi->biSize = sizeof(BITMAPINFOHEADER);
  bi->biWidth = stride;
  bi->biHeight = height;
  bi->biPlanes = 1;
  bi->biBitCount = 12;
  bi->biCompression = '2VMN'; /* 'NV12' little endian */
  bi->biSizeImage = static_cast<DWORD>(stride) * height * 3 / 2;
}

/* Rewrites the same payload as a FORMAT_VideoInfo header for peers that ask
 * for one. */
void videoinfo2_to_videoinfo(const VIDEOINFOHEADER2* from, VIDEOINFOHEADER* to) {
  ZeroMemory(to, sizeof(VIDEOINFOHEADER));
  to->rcSource = from->rcSource;
  to->rcTarget = from->rcTarget;
  to->dwBitRate = from->dwBitRate;
  to->AvgTimePerFrame = from->AvgTimePerFrame;
  to->bmiHeader = from->bmiHeader;
}

/* Reads the output geometry back out of a connected media type, which may be
 * VideoInfo2 or VideoInfo depending on what the peer accepted. */
void read_output_format(const AM_MEDIA_TYPE& mt, int* width, int* height, int* stride,
                        unsigned long long* avg_time_per_frame) {
  *width = *height = *stride = 0;
  *avg_time_per_frame = 0;
  if (mt.formattype == FORMAT_VideoInfo2 && mt.cbFormat >= sizeof(VIDEOINFOHEADER2)) {
    const VIDEOINFOHEADER2* v = reinterpret_cast<const VIDEOINFOHEADER2*>(mt.pbFormat);
    *width = v->bmiHeader.biWidth;
    *height = v->bmiHeader.biHeight < 0 ? -v->bmiHeader.biHeight : v->bmiHeader.biHeight;
    *stride = v->bmiHeader.biWidth;
    *avg_time_per_frame = v->AvgTimePerFrame;
  } else if (mt.formattype == FORMAT_VideoInfo && mt.cbFormat >= sizeof(VIDEOINFOHEADER)) {
    const VIDEOINFOHEADER* v = reinterpret_cast<const VIDEOINFOHEADER*>(mt.pbFormat);
    *width = v->bmiHeader.biWidth;
    *height = v->bmiHeader.biHeight < 0 ? -v->bmiHeader.biHeight : v->bmiHeader.biHeight;
    *stride = v->bmiHeader.biWidth;
    *avg_time_per_frame = v->AvgTimePerFrame;
  }
}

int nv12_stride_for(int width) { return (width + 31) & ~31; }

size_t nv12_frame_bytes(int stride, int height) {
  return static_cast<size_t>(stride) * height * 3 / 2;
}

/* ------------------------------------------------------------ lock guard */

class CritSec {
 public:
  CritSec() { InitializeCriticalSection(&cs_); }
  ~CritSec() { DeleteCriticalSection(&cs_); }
  void enter() { EnterCriticalSection(&cs_); }
  void leave() { LeaveCriticalSection(&cs_); }

 private:
  CRITICAL_SECTION cs_;
};

class Hold {
 public:
  explicit Hold(CritSec& cs) : cs_(cs) { cs_.enter(); }
  ~Hold() { cs_.leave(); }
  Hold(const Hold&) = delete;
  Hold& operator=(const Hold&) = delete;

 private:
  CritSec& cs_;
};

/* ---------------------------------------------------------- enumerators */

class EnumMediaTypesImpl : public IEnumMediaTypes {
 public:
  /* `owner` is the pin; `make` builds the type for one index. */
  typedef HRESULT (*MakeFn)(void* owner, int index, AM_MEDIA_TYPE** out);
  EnumMediaTypesImpl(void* owner, MakeFn make, int count)
      : owner_(owner), make_(make), count_(count) {
    debug_log("EnumMediaTypesImpl %p created", static_cast<void*>(this));
  }

  ULONG __stdcall AddRef() override { return InterlockedIncrement(&refs_); }
  ULONG __stdcall Release() override {
    const ULONG r = InterlockedDecrement(&refs_);
    if (r == 0) delete this;
    return r;
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IEnumMediaTypes) {
      *out = static_cast<IEnumMediaTypes*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  HRESULT __stdcall Next(ULONG count, AM_MEDIA_TYPE** out, ULONG* fetched) override {
    if (!out || (!fetched && count != 1)) return E_POINTER;
    ULONG got = 0;
    while (got < count && at_ < count_) {
      AM_MEDIA_TYPE* mt = nullptr;
      if (FAILED(make_(owner_, static_cast<int>(at_), &mt))) break;
      out[got++] = mt;
      ++at_;
    }
    if (fetched) *fetched = got;
    return got == count ? S_OK : S_FALSE;
  }
  HRESULT __stdcall Skip(ULONG count) override {
    at_ = static_cast<ULONG>(std::min<unsigned long long>(at_ + count, count_));
    return at_ < count_ ? S_OK : S_FALSE;
  }
  HRESULT __stdcall Reset() override {
    at_ = 0;
    return S_OK;
  }
  HRESULT __stdcall Clone(IEnumMediaTypes** out) override {
    if (!out) return E_POINTER;
    EnumMediaTypesImpl* e = new (std::nothrow) EnumMediaTypesImpl(owner_, make_, count_);
    if (!e) return E_OUTOFMEMORY;
    e->at_ = at_;
    *out = e;
    return S_OK;
  }

 private:
  ULONG refs_ = 1;
  ULONG at_ = 0;
  void* owner_;
  MakeFn make_;
  int count_;
};

class EnumPinsImpl : public IEnumPins {
 public:
  EnumPinsImpl(IPin* const* pins, int count) : count_(count) {
    debug_log("EnumPinsImpl %p created", static_cast<void*>(this));
    for (int i = 0; i < count; ++i) {
      pins_[i] = pins[i];
      pins_[i]->AddRef();
    }
  }
  ~EnumPinsImpl() {
    for (int i = 0; i < count_; ++i) pins_[i]->Release();
  }
  ULONG __stdcall AddRef() override { return InterlockedIncrement(&refs_); }
  ULONG __stdcall Release() override {
    const ULONG r = InterlockedDecrement(&refs_);
    if (r == 0) delete this;
    return r;
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IEnumPins) {
      *out = static_cast<IEnumPins*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  HRESULT __stdcall Next(ULONG count, IPin** out, ULONG* fetched) override {
    if (!out || (!fetched && count != 1)) return E_POINTER;
    ULONG got = 0;
    while (got < count && at_ < count_) {
      out[got] = pins_[at_++];
      out[got]->AddRef();
      ++got;
    }
    if (fetched) *fetched = got;
    return got == count ? S_OK : S_FALSE;
  }
  HRESULT __stdcall Skip(ULONG count) override {
    at_ = static_cast<ULONG>(std::min<unsigned long long>(at_ + count, count_));
    return at_ < count_ ? S_OK : S_FALSE;
  }
  HRESULT __stdcall Reset() override {
    at_ = 0;
    return S_OK;
  }
  HRESULT __stdcall Clone(IEnumPins** out) override {
    if (!out) return E_POINTER;
    EnumPinsImpl* e = new (std::nothrow) EnumPinsImpl(pins_, count_);
    if (!e) return E_OUTOFMEMORY;
    e->at_ = at_;
    *out = e;
    return S_OK;
  }

 private:
  ULONG refs_ = 1;
  ULONG at_ = 0;
  int count_;
  IPin* pins_[2];
};

/* ------------------------------------------------------------- forward */

class Av1GpuFilter;
class OutputPin;

/* -------------------------------------------------------------- input pin */

class InputPin : public IPin, public IMemInputPin {
 public:
  InputPin(Av1GpuFilter* filter, const wchar_t* name) : filter_(filter), name_(name) {}

  /* IUnknown: the pins are aggregate parts of the filter. */
  ULONG __stdcall AddRef() override;
  ULONG __stdcall Release() override;
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

  /* IPin */
  HRESULT __stdcall Connect(IPin*, const AM_MEDIA_TYPE*) override {
    return E_UNEXPECTED; /* only an output pin drives a connection */
  }
  HRESULT __stdcall ReceiveConnection(IPin* peer, const AM_MEDIA_TYPE* mt) override;
  HRESULT __stdcall Disconnect() override;
  HRESULT __stdcall ConnectedTo(IPin** peer) override;
  HRESULT __stdcall ConnectionMediaType(AM_MEDIA_TYPE* mt) override;
  HRESULT __stdcall QueryPinInfo(PIN_INFO* info) override;
  HRESULT __stdcall QueryDirection(PIN_DIRECTION* dir) override {
    debug_log("input QueryDirection");
    if (!dir) return E_POINTER;
    *dir = PINDIR_INPUT;
    return S_OK;
  }
  HRESULT __stdcall QueryId(LPWSTR* id) override;
  HRESULT __stdcall QueryAccept(const AM_MEDIA_TYPE* mt) override;
  HRESULT __stdcall EnumMediaTypes(IEnumMediaTypes** out) override;
  HRESULT __stdcall QueryInternalConnections(IPin**, ULONG*) override {
    return E_NOTIMPL;
  }
  HRESULT __stdcall EndOfStream() override;
  HRESULT __stdcall BeginFlush() override;
  HRESULT __stdcall EndFlush() override;
  HRESULT __stdcall NewSegment(REFERENCE_TIME start, REFERENCE_TIME stop,
                               double rate) override;

  /* IMemInputPin */
  HRESULT __stdcall GetAllocator(IMemAllocator** allocator) override {
    (void)allocator;
    return E_NOTIMPL; /* the source's allocator is fine: the bytes are copied */
  }
  HRESULT __stdcall NotifyAllocator(IMemAllocator*, BOOL) override { return S_OK; }
  HRESULT __stdcall GetAllocatorRequirements(ALLOCATOR_PROPERTIES*) override {
    return E_NOTIMPL;
  }
  HRESULT __stdcall Receive(IMediaSample* sample) override;
  HRESULT __stdcall ReceiveMultiple(IMediaSample** samples, long count,
                                    long* processed) override {
    if (!samples || !processed) return E_POINTER;
    HRESULT hr = S_OK;
    for (long i = 0; i < count; ++i) {
      hr = Receive(samples[i]);
      if (FAILED(hr)) break;
    }
    *processed = count;
    return hr;
  }
  HRESULT __stdcall ReceiveCanBlock() override {
    return S_OK; /* Receive decodes and delivers inside the call */
  }

  /* filter internals */
  bool connected() const { return peer_ != nullptr; }
  bool flushing() const { return flushing_; }
  const AM_MEDIA_TYPE& media_type() const { return media_type_; }
  OutputPin* output_pin();

 private:
  Av1GpuFilter* filter_;
  const wchar_t* name_;
  IPin* peer_ = nullptr;
  AM_MEDIA_TYPE media_type_ = {};
  bool flushing_ = false;
};

/* ------------------------------------------------------------- output pin */

class OutputPin : public IPin {
 public:
  OutputPin(Av1GpuFilter* filter, const wchar_t* name) : filter_(filter), name_(name) {}

  ULONG __stdcall AddRef() override;
  ULONG __stdcall Release() override;
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IPin) {
      *out = static_cast<IPin*>(this);
    } else {
      *out = nullptr;
      return E_NOINTERFACE;
    }
    reinterpret_cast<IUnknown*>(*out)->AddRef();
    return S_OK;
  }

  /* IPin: the output side drives the connection. */
  HRESULT __stdcall Connect(IPin* peer, const AM_MEDIA_TYPE* mt) override;
  HRESULT __stdcall ReceiveConnection(IPin*, const AM_MEDIA_TYPE*) override {
    return E_UNEXPECTED;
  }
  HRESULT __stdcall Disconnect() override;
  HRESULT __stdcall ConnectedTo(IPin** peer) override;
  HRESULT __stdcall ConnectionMediaType(AM_MEDIA_TYPE* mt) override;
  HRESULT __stdcall QueryPinInfo(PIN_INFO* info) override;
  HRESULT __stdcall QueryDirection(PIN_DIRECTION* dir) override {
    if (!dir) return E_POINTER;
    *dir = PINDIR_OUTPUT;
    return S_OK;
  }
  HRESULT __stdcall QueryId(LPWSTR* id) override;
  HRESULT __stdcall QueryAccept(const AM_MEDIA_TYPE* mt) override {
    if (!mt || mt->majortype != MEDIATYPE_Video || mt->subtype != MEDIASUBTYPE_NV12) {
      return S_FALSE;
    }
    return S_OK;
  }
  HRESULT __stdcall EnumMediaTypes(IEnumMediaTypes** out) override;
  HRESULT __stdcall QueryInternalConnections(IPin**, ULONG*) override {
    return E_NOTIMPL;
  }
  HRESULT __stdcall EndOfStream() override { return E_UNEXPECTED; }
  HRESULT __stdcall BeginFlush() override { return E_UNEXPECTED; }
  HRESULT __stdcall EndFlush() override { return E_UNEXPECTED; }
  HRESULT __stdcall NewSegment(REFERENCE_TIME, REFERENCE_TIME, double) override {
    return S_OK;
  }

  /* streaming */
  bool connected() const { return peer_ != nullptr && allocator_ != nullptr; }
  int width() const { return width_; }
  int height() const { return height_; }
  int stride() const { return stride_; }
  IPin* peer_handle() const { return peer_; }

  /* the allocator lifecycle: the output pin commits whatever allocator the
   * connection uses when the filter goes active, and decommits on stop (this
   * is CBaseOutputPin::Active/Inactive's job in the base classes) */
  void on_activate() {
    if (allocator_) allocator_->Commit();
  }
  void on_deactivate() {
    if (allocator_) allocator_->Decommit();
  }

  HRESULT deliver(const std::vector<uint8_t>& nv12, int width, int height,
                  REFERENCE_TIME start, REFERENCE_TIME stop, bool discontinuity);
  HRESULT deliver_end_of_stream();
  HRESULT deliver_flush(bool begin);

 private:
  HRESULT propose_type(int index, AM_MEDIA_TYPE** out);
  HRESULT decide_allocator(IPin* peer);

  Av1GpuFilter* filter_;
  const wchar_t* name_;
  IPin* peer_ = nullptr;
  IMemInputPin* sink_ = nullptr;
  IMemAllocator* allocator_ = nullptr;
  AM_MEDIA_TYPE media_type_ = {};
  int width_ = 0, height_ = 0, stride_ = 0;
  unsigned long long frame_time_ = 0;
};

/* ---------------------------------------------------------------- filter */

class Av1GpuFilter : public IBaseFilter {
 public:
  Av1GpuFilter() : input_(this, L"In"), output_(this, L"Out") {
    pins_[0] = &input_;
    pins_[1] = &output_;
    wcsncpy_s(name_, kFilterNameW, _TRUNCATE);
  }

  /* IUnknown */
  ULONG __stdcall AddRef() override { return InterlockedIncrement(&refs_); }
  ULONG __stdcall Release() override {
    const ULONG r = InterlockedDecrement(&refs_);
    if (r == 0) delete this;
    return r;
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown) {
      *out = static_cast<IUnknown*>(this);
    } else if (riid == IID_IPersist) {
      *out = static_cast<IPersist*>(this);
    } else if (riid == IID_IMediaFilter || riid == IID_IBaseFilter) {
      *out = static_cast<IBaseFilter*>(this);
    } else {
      *out = nullptr;
      return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
  }

  /* IPersist */
  HRESULT __stdcall GetClassID(CLSID* id) override {
    if (!id) return E_POINTER;
    *id = CLSID_AV1GpuDecoder;
    return S_OK;
  }

  /* IMediaFilter */
  HRESULT __stdcall Stop() override {
    Hold hold(lock_);
    state_ = State_Stopped;
    output_.on_deactivate();
    return S_OK;
  }
  HRESULT __stdcall Pause() override {
    Hold hold(lock_);
    if (state_ == State_Stopped) state_ = State_Paused;
    output_.on_activate();
    return S_OK;
  }
  HRESULT __stdcall Run(REFERENCE_TIME) override {
    Hold hold(lock_);
    if (state_ != State_Stopped) state_ = State_Running;
    return S_OK;
  }
  HRESULT __stdcall GetState(DWORD ms_timeout, FILTER_STATE* state) override {
    (void)ms_timeout;
    if (!state) return E_POINTER;
    Hold hold(lock_);
    *state = state_;
    return S_OK;
  }
  HRESULT __stdcall SetSyncSource(IReferenceClock* clock) override {
    Hold hold(lock_);
    if (clock_) clock_->Release();
    clock_ = clock;
    if (clock_) clock_->AddRef();
    return S_OK;
  }
  HRESULT __stdcall GetSyncSource(IReferenceClock** clock) override {
    if (!clock) return E_POINTER;
    Hold hold(lock_);
    *clock = clock_;
    if (clock_) clock_->AddRef();
    return S_OK;
  }

  /* IBaseFilter */
  HRESULT __stdcall EnumPins(IEnumPins** out) override {
    if (!out) return E_POINTER;
    Hold hold(lock_);
    *out = new (std::nothrow) EnumPinsImpl(pins_, 2);
    return *out ? S_OK : E_OUTOFMEMORY;
  }
  HRESULT __stdcall FindPin(LPCWSTR id, IPin** out) override {
    if (!id || !out) return E_POINTER;
    for (int i = 0; i < 2; ++i) {
      LPWSTR pin_id = nullptr;
      if (SUCCEEDED(pins_[i]->QueryId(&pin_id))) {
        const bool match = wcscmp(pin_id, id) == 0;
        CoTaskMemFree(pin_id);
        if (match) {
          *out = pins_[i];
          (*out)->AddRef();
          return S_OK;
        }
      }
    }
    return VFW_E_NOT_FOUND;
  }
  HRESULT __stdcall QueryFilterInfo(FILTER_INFO* info) override {
    if (!info) return E_POINTER;
    Hold hold(lock_);
    CopyMemory(info->achName, name_, sizeof(name_));
    info->pGraph = graph_;
    if (graph_) graph_->AddRef();
    return S_OK;
  }
  HRESULT __stdcall JoinFilterGraph(IFilterGraph* graph, LPCWSTR name) override {
    Hold hold(lock_);
    graph_ = graph;
    if (name) {
      wcsncpy_s(name_, name, _TRUNCATE);
    } else {
      name_[0] = 0;
    }
    return S_OK;
  }
  HRESULT __stdcall QueryVendorInfo(LPWSTR* vendor) override {
    if (!vendor) return E_POINTER;
    *vendor = static_cast<LPWSTR>(CoTaskMemAlloc(sizeof(kFilterNameW)));
    if (!*vendor) return E_OUTOFMEMORY;
    CopyMemory(*vendor, kFilterNameW, sizeof(kFilterNameW));
    return S_OK;
  }

  /* internals shared with the pins */
  CritSec& lock() { return lock_; }
  InputPin* in() { return &input_; }
  OutputPin* out() { return &output_; }
  FILTER_STATE state() {
    Hold hold(lock_);
    return state_;
  }

 private:
  /* the pins reach the filter's private state (state_, core_) directly */
  friend class InputPin;
  friend class OutputPin;

  /* Lazily created on the first sample: creating a CUDA context inside a
   * connect call would slow every graph build down and break the ones that
   * never run. */
  bool ensure_core(std::string* error);
  StreamDecoder* core() {
    Hold hold(lock_);
    return core_.get();
  }

 private:
  ULONG refs_ = 1;
  CritSec lock_;
  FILTER_STATE state_ = State_Stopped;
  IReferenceClock* clock_ = nullptr;
  IFilterGraph* graph_ = nullptr;
  wchar_t name_[128] = {0};

  InputPin input_;
  OutputPin output_;
  IPin* pins_[2];

  std::unique_ptr<StreamDecoder> core_;
  std::string core_error_;
};

/* IUnknown of the pins delegates to the filter that owns them */
ULONG InputPin::AddRef() { return filter_->AddRef(); }
ULONG InputPin::Release() { return filter_->Release(); }
ULONG OutputPin::AddRef() { return filter_->AddRef(); }
ULONG OutputPin::Release() { return filter_->Release(); }

OutputPin* InputPin::output_pin() { return filter_->out(); }

/* The internal-connection mapping helper is no longer needed; the real fix
 * was initializing out-parameters on the failure paths. */

/* ------------------------------------------------------------ input pin */

HRESULT InputPin::ReceiveConnection(IPin* peer, const AM_MEDIA_TYPE* mt) {
  if (!peer || !mt) return E_POINTER;
  debug_log("input ReceiveConnection: peer=%p format=%08lx cbFormat=%lu",
            static_cast<void*>(peer), mt->formattype.Data1,
            static_cast<unsigned long>(mt->cbFormat));
  Hold hold(filter_->lock());
  if (peer_) return VFW_E_ALREADY_CONNECTED;
  if (filter_->state_ != State_Stopped) return VFW_E_NOT_STOPPED;

  PIN_DIRECTION dir;
  if (FAILED(peer->QueryDirection(&dir)) || dir != PINDIR_OUTPUT) {
    return VFW_E_CANNOT_CONNECT;
  }
  if (mt->majortype != MEDIATYPE_Video ||
      (mt->subtype != kSubtypeAv1Upper && mt->subtype != kSubtypeAv1Lower)) {
    debug_log("input rejected: subtype is not AV1");
    return VFW_E_TYPE_NOT_ACCEPTED;
  }
  if (!format_acceptable(mt->formattype)) {
    debug_log("input rejected: format type is not one we read");
    return VFW_E_TYPE_NOT_ACCEPTED;
  }
  int width = 0, height = 0;
  unsigned long long frame_time = 0;
  if (!input_dimensions(mt, &width, &height, &frame_time)) {
    debug_log("input rejected: the format block carries no dimensions");
    return VFW_E_TYPE_NOT_ACCEPTED;
  }

  peer_ = peer;
  peer_->AddRef();
  clear_media_type(&media_type_);
  AM_MEDIA_TYPE* copy = copy_media_type(mt);
  if (!copy) {
    peer_->Release();
    peer_ = nullptr;
    return E_OUTOFMEMORY;
  }
  media_type_ = *copy;
  CoTaskMemFree(copy);
  debug_log("input connected: %dx%d, %llu fps", width, height,
            frame_time ? 10000000ull / frame_time : 0ull);
  return S_OK;
}

HRESULT InputPin::Disconnect() {
  debug_log("input Disconnect");
  Hold hold(filter_->lock());
  if (!peer_) return S_FALSE;
  if (filter_->state_ != State_Stopped) return VFW_E_NOT_STOPPED;
  peer_->Release();
  peer_ = nullptr;
  clear_media_type(&media_type_);
  return S_OK;
}

HRESULT InputPin::ConnectedTo(IPin** peer) {
  debug_log("input ConnectedTo");
  if (!peer) return E_POINTER;
  Hold hold(filter_->lock());
  /* out parameters must be initialized even on failure: quartz reads the
   * peer unconditionally after the call */
  *peer = nullptr;
  if (!peer_) return VFW_E_NOT_CONNECTED;
  *peer = peer_;
  peer_->AddRef();
  return S_OK;
}

HRESULT InputPin::ConnectionMediaType(AM_MEDIA_TYPE* mt) {
  debug_log("input ConnectionMediaType");
  if (!mt) return E_POINTER;
  ZeroMemory(mt, sizeof(*mt));
  Hold hold(filter_->lock());
  if (!peer_) return VFW_E_NOT_CONNECTED;
  AM_MEDIA_TYPE* copy = copy_media_type(&media_type_);
  if (!copy) return E_OUTOFMEMORY;
  *mt = *copy;
  CoTaskMemFree(copy);
  return S_OK;
}

HRESULT InputPin::QueryPinInfo(PIN_INFO* info) {
  debug_log("input %p QueryPinInfo", static_cast<void*>(this));
  if (!info) return E_POINTER;
  info->pFilter = static_cast<IBaseFilter*>(filter_);
  filter_->AddRef();
  info->dir = PINDIR_INPUT;
  wcsncpy_s(info->achName, name_, _TRUNCATE);
  return S_OK;
}

HRESULT InputPin::QueryId(LPWSTR* id) {
  debug_log("input QueryId");
  if (!id) return E_POINTER;
  const size_t bytes = (wcslen(name_) + 1) * sizeof(wchar_t);
  LPWSTR copy = static_cast<LPWSTR>(CoTaskMemAlloc(bytes));
  if (!copy) return E_OUTOFMEMORY;
  CopyMemory(copy, name_, bytes);
  *id = copy;
  return S_OK;
}

HRESULT InputPin::QueryAccept(const AM_MEDIA_TYPE* mt) {
  if (!mt || mt->majortype != MEDIATYPE_Video ||
      (mt->subtype != kSubtypeAv1Upper && mt->subtype != kSubtypeAv1Lower)) {
    debug_log("input QueryAccept: rejected (subtype %08lx-%04lx)",
              mt ? mt->subtype.Data1 : 0, mt ? mt->subtype.Data2 : 0);
    return S_FALSE;
  }
  if (!format_acceptable(mt->formattype)) {
    debug_log("input QueryAccept: rejected (format type)");
    return S_FALSE;
  }
  int width = 0, height = 0;
  unsigned long long frame_time = 0;
  if (!input_dimensions(mt, &width, &height, &frame_time)) {
    debug_log("input QueryAccept: rejected (no dimensions)");
    return S_FALSE;
  }
  debug_log("input QueryAccept: accepted %dx%d", width, height);
  return S_OK;
}

HRESULT InputPin::EnumMediaTypes(IEnumMediaTypes** out) {
  debug_log("input EnumMediaTypes");
  if (!out) return E_POINTER;
  class Make {
   public:
    static HRESULT call(void*, int index, AM_MEDIA_TYPE** result) {
      if (index != 0) return S_FALSE;
      /* A partial type: majortype and subtype only, no format block, which is
       * what an input pin's enumeration honestly knows before connecting. */
      AM_MEDIA_TYPE* mt = make_media_type(&MEDIATYPE_Video, &kSubtypeAv1Upper, FALSE,
                                          &FORMAT_None, 0, nullptr);
      if (!mt) return E_OUTOFMEMORY;
      *result = mt;
      return S_OK;
    }
  };
  *out = new (std::nothrow) EnumMediaTypesImpl(nullptr, &Make::call, 1);
  return *out ? S_OK : E_OUTOFMEMORY;
}

HRESULT InputPin::EndOfStream() {
  /* let the worker show the frames it still has queued before EOS */
  if (StreamDecoder* core = filter_->core()) core->finish();
  return output_pin()->deliver_end_of_stream();
}

HRESULT InputPin::BeginFlush() {
  debug_log("BeginFlush");
  {
    Hold hold(filter_->lock());
    flushing_ = true;
  }
  IPin* peer = output_pin()->peer_handle();
  return peer ? peer->BeginFlush() : S_OK;
}

HRESULT InputPin::EndFlush() {
  debug_log("EndFlush");
  IPin* peer = output_pin()->peer_handle();
  HRESULT hr = peer ? peer->EndFlush() : S_OK;
  {
    Hold hold(filter_->lock());
    flushing_ = false;
    if (StreamDecoder* core = filter_->core()) core->reset_stream();
  }
  return hr;
}

HRESULT InputPin::NewSegment(REFERENCE_TIME start, REFERENCE_TIME stop, double rate) {
  (void)stop;
  (void)start;
  (void)rate;
  Hold hold(filter_->lock());
  if (StreamDecoder* core = filter_->core()) {
    /* a seek: references from before the seek are wrong; the sequence header
     * arrives again with the next key frame */
    core->reset_stream();
  }
  return S_OK;
}

HRESULT InputPin::Receive(IMediaSample* sample) {
  if (!sample) return E_POINTER;
  {
    Hold hold(filter_->lock());
    if (flushing_) return S_FALSE;
    if (!peer_) return VFW_E_NOT_CONNECTED;
    if (filter_->state_ == State_Stopped) return VFW_E_NOT_RUNNING;
  }
  debug_log("Receive: sample %ld bytes", sample->GetActualDataLength());

  std::string error;
  if (!filter_->ensure_core(&error)) {
    debug_log("decoder init failed: %s", error.c_str());
    return E_FAIL;
  }

  BYTE* data = nullptr;
  const long size = sample->GetActualDataLength();
  if (FAILED(sample->GetPointer(&data)) || size <= 0 || !data) return S_OK;

  REFERENCE_TIME start = 0, stop = 0;
  const bool have_time = SUCCEEDED(sample->GetTime(&start, &stop));

  OutputPin* out = filter_->out();
  const int stride = out->stride();
  const int width = out->width();
  const int height = out->height();
  const size_t pitch = static_cast<size_t>(stride);
  if (!out->connected() || nv12_size(width, height, pitch) == 0) {
    return VFW_E_NOT_CONNECTED;
  }

  bool discontinuity = sample->IsDiscontinuity() == S_OK;
  const REFERENCE_TIME frame_dur = have_time && stop > start ? stop - start : 0;
  REFERENCE_TIME frame_start = start;

  StreamDecoder* core = filter_->core();
  /* The pipeline's worker delivers frames after Receive returned, so the
   * timing state must outlive this call: shared, value captured. */
  struct Timing {
    REFERENCE_TIME dur = 333667;
    REFERENCE_TIME start = 0;
    bool discontinuity = false;
  };
  auto timing = std::make_shared<Timing>();
  timing->dur = frame_dur > 0 ? frame_dur : 333667;
  timing->start = frame_start;
  timing->discontinuity = discontinuity;
  StreamDecoder::Sink sink = [out, timing](const std::vector<uint8_t>& nv12, int w, int h,
                                           std::string* sink_error) -> bool {
    (void)sink_error;
    const HRESULT hr = out->deliver(nv12, w, h, timing->start,
                                    timing->start + timing->dur,
                                    timing->discontinuity);
    timing->discontinuity = false; /* only the first frame after a seek carries it */
    timing->start += timing->dur;
    if (FAILED(hr)) {
      debug_log("downstream Receive failed: 0x%08lx", static_cast<unsigned long>(hr));
    }
    /* A renderer that refuses one frame must not stop the decode. */
    return hr != E_POINTER;
  };

  if (true) { /* perf instrumentation: measure decode time per sample */
    static LARGE_INTEGER freq = [] {
      LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f;
    }();
    static double worst = 0.0, total = 0.0;
    static long count = 0;
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    const bool ok = core->push(data, static_cast<size_t>(size), pitch, sink, &error);
    QueryPerformanceCounter(&t1);
    const double ms = double(t1.QuadPart - t0.QuadPart) / double(freq.QuadPart) * 1000.0;
    ++count; total += ms;
    if (ms > worst) worst = ms;
    if (count % 50 == 0) {
      debug_log("perf: last %.1f ms, avg %.1f ms, worst %.1f ms over %ld samples",
                ms, total / count, worst, count);
    }
    if (!ok) {
      debug_log("decode failed: %s", error.c_str());
      /* A broken frame must not stop the graph: keep going. A stream that never
       * decodes anything reports through the absence of frames. */
      return S_OK;
    }
    return S_OK;
  }
}

/* ------------------------------------------------------------ output pin */

/* Output media types, in enumeration order: 0 = VideoInfo2, 1 = VideoInfo.
 * Both describe NV12 at the geometry the input media type named. */
HRESULT OutputPin::propose_type(int index, AM_MEDIA_TYPE** out) {
  int width = 0, height = 0;
  unsigned long long frame_time = 0;
  {
    Hold hold(filter_->lock());
    if (!input_dimensions(&filter_->in()->media_type(), &width, &height, &frame_time)) {
      return VFW_E_NO_ACCEPTABLE_TYPES;
    }
  }
  const int stride = nv12_stride_for(width);
  std::vector<BYTE> format(sizeof(VIDEOINFOHEADER2));
  fill_nv12_format(format.data(), width, height, stride, frame_time);
  const GUID* format_type = index == 0 ? &FORMAT_VideoInfo2 : &FORMAT_VideoInfo;
  AM_MEDIA_TYPE* mt = make_media_type(&MEDIATYPE_Video, &MEDIASUBTYPE_NV12, TRUE,
                                      format_type, sizeof(VIDEOINFOHEADER2),
                                      format.data());
  if (!mt) return E_OUTOFMEMORY;
  mt->lSampleSize = nv12_frame_bytes(stride, height);
  *out = mt;
  return S_OK;
}

HRESULT OutputPin::Connect(IPin* peer, const AM_MEDIA_TYPE* mt) {
  if (!peer) return E_POINTER;
  debug_log("output Connect: peer=%p mt=%p", static_cast<void*>(peer),
            static_cast<const void*>(mt));
  Hold hold(filter_->lock());
  if (peer_) return VFW_E_ALREADY_CONNECTED;

  PIN_DIRECTION dir;
  if (FAILED(peer->QueryDirection(&dir)) || dir != PINDIR_INPUT) {
    return VFW_E_CANNOT_CONNECT;
  }

  /* Which media type to propose: the caller's (completed from ours), or the
   * first of ours the peer takes. The peer answers through ReceiveConnection,
   * which is what makes it the authority. */
  AM_MEDIA_TYPE* full = nullptr;
  auto complete_from_input = [&]() -> HRESULT {
    int width = 0, height = 0;
    unsigned long long frame_time = 0;
    if (!input_dimensions(&filter_->in()->media_type(), &width, &height, &frame_time)) {
      return VFW_E_NO_ACCEPTABLE_TYPES;
    }
    const int stride = nv12_stride_for(width);
    std::vector<BYTE> format(sizeof(VIDEOINFOHEADER2));
    fill_nv12_format(format.data(), width, height, stride, frame_time);
    full = make_media_type(&MEDIATYPE_Video, &MEDIASUBTYPE_NV12, TRUE,
                           &FORMAT_VideoInfo2, sizeof(VIDEOINFOHEADER2), format.data());
    if (!full) return E_OUTOFMEMORY;
    full->lSampleSize = nv12_frame_bytes(stride, height);
    return S_OK;
  };

  HRESULT hr = E_FAIL;
  if (mt && (mt->majortype != GUID_NULL || mt->subtype != GUID_NULL ||
             mt->formattype != GUID_NULL)) {
    if (mt->majortype != GUID_NULL && mt->majortype != MEDIATYPE_Video) {
      return VFW_E_TYPE_NOT_ACCEPTED;
    }
    if (mt->subtype != GUID_NULL && mt->subtype != MEDIASUBTYPE_NV12) {
      return VFW_E_TYPE_NOT_ACCEPTED;
    }
    if (mt->formattype != GUID_NULL && !format_acceptable(mt->formattype)) {
      return VFW_E_TYPE_NOT_ACCEPTED;
    }
    hr = complete_from_input();
    if (SUCCEEDED(hr) && mt->formattype == FORMAT_VideoInfo &&
        full->formattype == FORMAT_VideoInfo2) {
      /* the peer asked for VideoInfo specifically */
      VIDEOINFOHEADER v;
      videoinfo2_to_videoinfo(
          reinterpret_cast<const VIDEOINFOHEADER2*>(full->pbFormat), &v);
      free_media_type(full);
      full = make_media_type(&MEDIATYPE_Video, &MEDIASUBTYPE_NV12, TRUE,
                             &FORMAT_VideoInfo, sizeof(VIDEOINFOHEADER), &v);
      if (!full) return E_OUTOFMEMORY;
    }
  } else {
    for (int i = 0; i < 2 && FAILED(hr); ++i) hr = propose_type(i, &full);
  }
  if (FAILED(hr)) return hr;

  hr = peer->ReceiveConnection(this, full);
  if (FAILED(hr)) {
    free_media_type(full);
    return hr;
  }

  media_type_ = *full;
  /* take ownership of the format block; the wrapper itself is heap allocated */
  full->pbFormat = nullptr;
  full->cbFormat = 0;
  free_media_type(full);
  read_output_format(media_type_, &width_, &height_, &stride_, &frame_time_);
  if (width_ <= 0 || height_ <= 0) {
    peer->Disconnect(); /* the peer took a type it cannot describe */
    clear_media_type(&media_type_);
    return VFW_E_TYPE_NOT_ACCEPTED;
  }

  peer_ = peer;
  peer_->AddRef();
  hr = decide_allocator(peer);
  if (FAILED(hr)) {
    peer_->Release();
    peer_ = nullptr;
    peer->Disconnect(); /* undo the ReceiveConnection the peer accepted */
    clear_media_type(&media_type_);
    width_ = height_ = stride_ = 0;
    return hr;
  }
  debug_log("output connected: %dx%d stride %d, %llu bytes/frame", width_, height_,
            stride_, static_cast<unsigned long long>(nv12_frame_bytes(stride_, height_)));
  return S_OK;
}

HRESULT OutputPin::decide_allocator(IPin* peer) {
  IMemInputPin* sink = nullptr;
  HRESULT hr = peer->QueryInterface(IID_IMemInputPin, reinterpret_cast<void**>(&sink));
  if (FAILED(hr)) return VFW_E_NO_ALLOCATOR;

  const size_t frame_bytes = nv12_frame_bytes(stride_, height_);
  ALLOCATOR_PROPERTIES want{};
  want.cBuffers = 3;
  want.cbBuffer = static_cast<long>(frame_bytes);
  want.cbAlign = 1;
  want.cbPrefix = 0;

  IMemAllocator* allocator = nullptr;
  hr = sink->GetAllocator(&allocator);
  if (FAILED(hr)) {
    sink->Release();
    return VFW_E_NO_ALLOCATOR;
  }
  ALLOCATOR_PROPERTIES actual{};
  hr = allocator->SetProperties(&want, &actual);
  if (SUCCEEDED(hr) &&
      (static_cast<size_t>(actual.cbBuffer) < frame_bytes || actual.cBuffers < 2)) {
    hr = E_FAIL; /* the peer cannot fit a frame */
  }
  if (SUCCEEDED(hr)) hr = sink->NotifyAllocator(allocator, FALSE);
  if (FAILED(hr)) {
    allocator->Release();
    sink->Release();
    return hr;
  }
  allocator_ = allocator;
  sink_ = sink;
  return S_OK;
}

HRESULT OutputPin::Disconnect() {
  Hold hold(filter_->lock());
  if (!peer_) return S_FALSE;
  if (filter_->state_ != State_Stopped) return VFW_E_NOT_STOPPED;
  peer_->Release();
  peer_ = nullptr;
  if (sink_) sink_->Release();
  sink_ = nullptr;
  if (allocator_) allocator_->Release();
  allocator_ = nullptr;
  clear_media_type(&media_type_);
  width_ = height_ = stride_ = 0;
  return S_OK;
}

HRESULT OutputPin::ConnectedTo(IPin** peer) {
  if (!peer) return E_POINTER;
  Hold hold(filter_->lock());
  *peer = nullptr;
  if (!peer_) return VFW_E_NOT_CONNECTED;
  *peer = peer_;
  peer_->AddRef();
  return S_OK;
}

HRESULT OutputPin::ConnectionMediaType(AM_MEDIA_TYPE* mt) {
  if (!mt) return E_POINTER;
  ZeroMemory(mt, sizeof(*mt));
  Hold hold(filter_->lock());
  if (!peer_) return VFW_E_NOT_CONNECTED;
  AM_MEDIA_TYPE* copy = copy_media_type(&media_type_);
  if (!copy) return E_OUTOFMEMORY;
  *mt = *copy;
  CoTaskMemFree(copy);
  return S_OK;
}

HRESULT OutputPin::QueryPinInfo(PIN_INFO* info) {
  if (!info) return E_POINTER;
  info->pFilter = static_cast<IBaseFilter*>(filter_);
  filter_->AddRef();
  info->dir = PINDIR_OUTPUT;
  wcsncpy_s(info->achName, name_, _TRUNCATE);
  return S_OK;
}

HRESULT OutputPin::QueryId(LPWSTR* id) {
  if (!id) return E_POINTER;
  const size_t bytes = (wcslen(name_) + 1) * sizeof(wchar_t);
  LPWSTR copy = static_cast<LPWSTR>(CoTaskMemAlloc(bytes));
  if (!copy) return E_OUTOFMEMORY;
  CopyMemory(copy, name_, bytes);
  *id = copy;
  return S_OK;
}

HRESULT OutputPin::EnumMediaTypes(IEnumMediaTypes** out) {
  if (!out) return E_POINTER;
  class Make {
   public:
    static HRESULT call(void* owner, int index, AM_MEDIA_TYPE** result) {
      return static_cast<OutputPin*>(owner)->propose_type(index, result);
    }
  };
  *out = new (std::nothrow) EnumMediaTypesImpl(this, &Make::call, 2);
  return *out ? S_OK : E_OUTOFMEMORY;
}

HRESULT OutputPin::deliver(const std::vector<uint8_t>& nv12, int width, int height,
                           REFERENCE_TIME start, REFERENCE_TIME stop,
                           bool discontinuity) {
  if (!connected()) return VFW_E_NOT_CONNECTED;
  const size_t want = nv12_frame_bytes(stride_, height_);
  if (nv12.size() < want || width != width_ || height != height_) {
    debug_log("frame geometry %dx%d does not match the output type %dx%d; dropped",
              width, height, width_, height_);
    return S_FALSE; /* skip the frame, keep playing */
  }
  IMediaSample* sample = nullptr;
  HRESULT hr = allocator_->GetBuffer(&sample, nullptr, nullptr, 0);
  if (FAILED(hr)) return hr;
  BYTE* buffer = nullptr;
  hr = sample->GetPointer(&buffer);
  if (SUCCEEDED(hr)) {
    CopyMemory(buffer, nv12.data(), want);
    sample->SetActualDataLength(static_cast<long>(want));
    sample->SetTime(&start, &stop);
    sample->SetMediaTime(nullptr, nullptr);
    sample->SetSyncPoint(TRUE);
    sample->SetDiscontinuity(discontinuity ? TRUE : FALSE);
    sample->SetPreroll(FALSE);
    hr = sink_->Receive(sample);
  }
  sample->Release();
  return hr;
}

HRESULT OutputPin::deliver_end_of_stream() {
  IPin* peer = peer_handle();
  return peer ? peer->EndOfStream() : S_FALSE;
}

HRESULT OutputPin::deliver_flush(bool begin) {
  IPin* peer = peer_handle();
  return peer ? (begin ? peer->BeginFlush() : peer->EndFlush()) : S_FALSE;
}

/* ---------------------------------------------------------------- filter */

bool Av1GpuFilter::ensure_core(std::string* error) {
  Hold hold(lock_);
  if (core_) return true;
  if (!core_error_.empty()) {
    if (error) *error = core_error_;
    return false;
  }
  BackendOptions options; /* Auto: CUDA, then OpenCL, then the CPU reference */
  core_.reset(new (std::nothrow) StreamDecoder(options));
  if (!core_) {
    core_error_ = "out of memory";
    if (error) *error = core_error_;
    return false;
  }
  if (!core_->init(&core_error_)) {
    core_.reset();
    debug_log("backend init failed: %s", core_error_.c_str());
    if (error) *error = core_error_;
    return false;
  }
  const DeviceInfo& device = core_->device();
  debug_log("backend: %s (%s)", device.name.c_str(),
            device.kind == BackendKind::CUDA
                ? "cuda"
                : (device.kind == BackendKind::OpenCL ? "opencl" : "cpu"));
  return true;
}

/* --------------------------------------------------------- class factory */

class Factory : public IClassFactory {
 public:
  ULONG __stdcall AddRef() override { return InterlockedIncrement(&refs_); }
  ULONG __stdcall Release() override {
    const ULONG r = InterlockedDecrement(&refs_);
    if (r == 0) delete this;
    return r;
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IClassFactory) {
      *out = static_cast<IClassFactory*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  HRESULT __stdcall CreateInstance(IUnknown* outer, REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (outer) return CLASS_E_NOAGGREGATION;
    Av1GpuFilter* filter = new (std::nothrow) Av1GpuFilter();
    if (!filter) return E_OUTOFMEMORY;
    const HRESULT hr = filter->QueryInterface(riid, out);
    filter->Release();
    return hr;
  }
  HRESULT __stdcall LockServer(BOOL) override { return S_OK; }

 private:
  ULONG refs_ = 1;
};

} /* namespace */

/* ------------------------------------------------------- module plumbing */

static volatile LONG g_module_refs = 0;
static HMODULE g_module = nullptr;

extern "C" BOOL __stdcall DllMain(HMODULE module, DWORD reason, void*) {
  if (reason == DLL_PROCESS_ATTACH) {
    g_module = module;
    DisableThreadLibraryCalls(module);
  }
  return TRUE;
}

extern "C" HRESULT __stdcall DllGetClassObject(REFCLSID clsid, REFIID riid, void** out) {
  if (!out) return E_POINTER;
  if (clsid != CLSID_AV1GpuDecoder) return CLASS_E_CLASSNOTAVAILABLE;
  InterlockedIncrement(&g_module_refs);
  Factory* factory = new (std::nothrow) Factory();
  if (!factory) return E_OUTOFMEMORY;
  const HRESULT hr = factory->QueryInterface(riid, out);
  factory->Release();
  InterlockedDecrement(&g_module_refs);
  return hr;
}

extern "C" HRESULT __stdcall DllCanUnloadNow() {
  return g_module_refs == 0 ? S_OK : S_FALSE;
}

/* --------------------------------------------------------- registration */

/* Writes a registry value under HKCR, falling back to HKCU's class view when
 * HKLM is denied, so the filter works for a user without admin rights too
 * (players read both). */
static LONG reg_write_string(const char* path, const char* value_name,
                             const char* data) {
  HKEY key = nullptr;
  LONG rc = RegCreateKeyExA(HKEY_CLASSES_ROOT, path, 0, nullptr,
                            REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, nullptr, &key,
                            nullptr);
  if (rc == ERROR_ACCESS_DENIED) {
    char user_path[600];
    _snprintf_s(user_path, sizeof(user_path), _TRUNCATE, "Software\\Classes\\%s", path);
    rc = RegCreateKeyExA(HKEY_CURRENT_USER, user_path, 0, nullptr,
                         REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, nullptr, &key,
                         nullptr);
  }
  if (rc != ERROR_SUCCESS) return rc;
  rc = RegSetValueExA(key, value_name, 0, REG_SZ,
                      reinterpret_cast<const BYTE*>(data),
                      static_cast<DWORD>(strlen(data)) + 1);
  RegCloseKey(key);
  return rc;
}

static LONG reg_delete_tree(const char* path) {
  LONG rc = RegDeleteTreeA(HKEY_CLASSES_ROOT, path);
  if (rc == ERROR_ACCESS_DENIED) {
    char user_path[600];
    _snprintf_s(user_path, sizeof(user_path), _TRUNCATE, "Software\\Classes\\%s", path);
    rc = RegDeleteTreeA(HKEY_CURRENT_USER, user_path);
  }
  return rc;
}

static HRESULT register_com_class(bool unregister) {
  if (unregister) {
    reg_delete_tree(kClsidPath);
    return S_OK;
  }
  char path[512];
  char dll[MAX_PATH + 2];
  const DWORD n = GetModuleFileNameA(g_module, dll, MAX_PATH);
  if (n == 0) return E_FAIL;
  dll[MAX_PATH] = 0;

  _snprintf_s(path, sizeof(path), _TRUNCATE, "%s\\InprocServer32", kClsidPath);
  LONG rc = reg_write_string(path, nullptr, dll);
  if (rc != ERROR_SUCCESS) return HRESULT_FROM_WIN32(rc);
  reg_write_string(path, "ThreadingModel", "Both");
  reg_write_string(kClsidPath, nullptr, kFilterNameA);
  return S_OK;
}

static HRESULT register_with_filter_mapper(bool unregister) {
  IFilterMapper2* mapper = nullptr;
  HRESULT hr = CoCreateInstance(CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER,
                                IID_IFilterMapper2, reinterpret_cast<void**>(&mapper));
  if (FAILED(hr)) return S_OK; /* players still load the DLL by path without it */

  if (unregister) {
    hr = mapper->UnregisterFilter(&CLSID_LegacyAmFilterCategory, nullptr,
                                  CLSID_AV1GpuDecoder);
    mapper->Release();
    return hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) ? S_OK : hr;
  }

  const REGPINTYPES in_types[] = {
      {&MEDIATYPE_Video, &kSubtypeAv1Upper},
      {&MEDIATYPE_Video, &kSubtypeAv1Lower},
  };
  const REGPINTYPES out_types[] = {
      {&MEDIATYPE_Video, &MEDIASUBTYPE_NV12},
  };
  /* version 2 pin descriptors (REGFILTER2::dwVersion == 2) */
  REGFILTERPINS2 pin_descriptors[2];
  ZeroMemory(pin_descriptors, sizeof(pin_descriptors));
  pin_descriptors[0].cInstances = 0;
  pin_descriptors[0].nMediaTypes = 2;
  pin_descriptors[0].lpMediaType = in_types;
  pin_descriptors[0].nMediums = 0;
  pin_descriptors[0].lpMedium = nullptr;
  pin_descriptors[0].clsPinCategory = nullptr;
  pin_descriptors[1].cInstances = 0;
  pin_descriptors[1].nMediaTypes = 1;
  pin_descriptors[1].lpMediaType = out_types;
  pin_descriptors[1].nMediums = 0;
  pin_descriptors[1].lpMedium = nullptr;
  pin_descriptors[1].clsPinCategory = nullptr;

  REGFILTER2 rf{};
  rf.dwVersion = 2;
  rf.dwMerit = MERIT_NORMAL;
  rf.cPins2 = 2;
  rf.rgPins2 = pin_descriptors;

  hr = mapper->RegisterFilter(CLSID_AV1GpuDecoder, kFilterNameW, nullptr,
                              &CLSID_LegacyAmFilterCategory, nullptr, &rf);
  mapper->Release();
  return hr;
}

extern "C" HRESULT __stdcall DllRegisterServer() {
  HRESULT hr = register_com_class(false);
  if (SUCCEEDED(hr)) hr = register_with_filter_mapper(false);
  return hr;
}

extern "C" HRESULT __stdcall DllUnregisterServer() {
  register_with_filter_mapper(true);
  register_com_class(true);
  return S_OK;
}
