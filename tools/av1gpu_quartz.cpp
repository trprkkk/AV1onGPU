/*
 * av1gpu_quartz - reproduce a host's graph build with the real quartz.dll
 *
 * MPC-HC crashed inside quartz's ConnectDirect without ever calling into the
 * filter's pin methods. This harness builds the same three filter graph with
 * the same quartz graph manager the player uses - a minimal AV1 source, the
 * av1gpu decoder (loaded by DLL path), and the real Enhanced Video Renderer -
 * and connects it with IGraphBuilder::ConnectDirect, so whatever quartz does
 * to foreign filters during a connect happens here too.
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

namespace {

/* {B7E0E1C2-5A3D-4F6B-9C48-2A1D7E5F3B60} - keep in step with dsfilter */
const GUID kDecoderClsid = {
    0xb7e0e1c2, 0x5a3d, 0x4f6b, {0x9c, 0x48, 0x2a, 0x1d, 0x7e, 0x5f, 0x3b, 0x60}};
/* 'AV01' */
const GUID kSubtypeAv1 = {
    0x31305641, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

[[noreturn]] void die(const char* what, HRESULT hr) {
  std::printf("av1gpu_quartz: %s failed: 0x%08lx\n", what,
              static_cast<unsigned long>(hr));
  std::exit(1);
}

LONG interlocked(LONG* p, LONG delta) { return InterlockedExchangeAdd(p, delta) + delta; }

/* A media sample the source hands to Receive: read-only for the decoder. */
class FakeSample : public IMediaSample {
 public:
  FakeSample(const uint8_t* data, long size, REFERENCE_TIME start, REFERENCE_TIME stop)
      : data_(data), size_(size), start_(start), stop_(stop) {}
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
    *buffer = const_cast<BYTE*>(data_);
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
  HRESULT __stdcall IsDiscontinuity() override { return S_FALSE; }
  HRESULT __stdcall SetDiscontinuity(BOOL) override { return E_NOTIMPL; }

 private:
  LONG refs_ = 1;
  const uint8_t* data_;
  long size_;
  REFERENCE_TIME start_, stop_;
};

class SourceFilter;

/* A proper IEnumPins over the source filter's single output pin. */
class SourceEnumPins : public IEnumPins {
 public:
  explicit SourceEnumPins(IPin* pin) : pin_(pin) { pin_->AddRef(); }
  ~SourceEnumPins() { pin_->Release(); }
  ULONG __stdcall AddRef() override { return interlocked(&refs_, 1); }
  ULONG __stdcall Release() override {
    const LONG r = interlocked(&refs_, -1);
    if (r == 0) delete this;
    return static_cast<ULONG>(r);
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
    if (at_ == 0 && got < count) {
      out[got++] = pin_;
      pin_->AddRef();
      at_ = 1;
    }
    if (fetched) *fetched = got;
    return got == count ? S_OK : S_FALSE;
  }
  HRESULT __stdcall Skip(ULONG count) override {
    at_ = std::min<unsigned long>(at_ + count, 1);
    return at_ < 1 ? S_OK : S_FALSE;
  }
  HRESULT __stdcall Reset() override {
    at_ = 0;
    return S_OK;
  }
  HRESULT __stdcall Clone(IEnumPins** out) override {
    if (!out) return E_POINTER;
    SourceEnumPins* e = new SourceEnumPins(pin_);
    e->at_ = at_;
    *out = e;
    return S_OK;
  }

 private:
  LONG refs_ = 1;
  ULONG at_ = 0;
  IPin* pin_;
};

/* A splitter's output pin in the smallest form that quartz accepts: it hands
 * out one temporal unit per Receive, from a buffer the tool fills. */
class SourceFilter;

AM_MEDIA_TYPE* copy_mt(const AM_MEDIA_TYPE* src);
void free_mt_shim(AM_MEDIA_TYPE* mt);

class SourceOutPin : public IPin, public IMemInputPin /* not used; id only */ {
 public:
  explicit SourceOutPin(SourceFilter* filter) : filter_(filter) {}

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

  /* IPin */
  HRESULT __stdcall Connect(IPin* peer, const AM_MEDIA_TYPE* mt) override {
    std::printf("  trace: source pin Connect\n");
    if (!peer) return E_POINTER;
    if (peer_) return VFW_E_ALREADY_CONNECTED;
    AM_MEDIA_TYPE* cand = nullptr;
    if (mt) {
      if (peer->QueryAccept(mt) != S_OK) return VFW_E_TYPE_NOT_ACCEPTED;
      cand = copy_mt(mt);
    } else {
      IEnumMediaTypes* e = nullptr;
      HRESULT hr = peer->EnumMediaTypes(&e);
      if (FAILED(hr)) return hr;
      AM_MEDIA_TYPE* t = nullptr;
      while (e->Next(1, &t, nullptr) == S_OK) {
        if (peer->QueryAccept(t) == S_OK) {
          cand = t;
          break;
        }
        free_mt_shim(t);
        t = nullptr;
      }
      e->Release();
      if (!cand) return VFW_E_NO_ACCEPTABLE_TYPES;
    }
    const HRESULT hr = peer->ReceiveConnection(this, cand);
    free_mt_shim(cand);
    if (SUCCEEDED(hr)) {
      peer_ = peer;
      peer_->AddRef();
    }
    return hr;
  }
  HRESULT __stdcall ReceiveConnection(IPin* peer, const AM_MEDIA_TYPE* mt) override {
    if (!peer || !mt) return E_POINTER;
    std::printf("  trace: source pin ReceiveConnection\n");
    peer_ = peer;
    return S_OK;
  }
  HRESULT __stdcall Disconnect() override {
    std::printf("  trace: source pin Disconnect\n");
    peer_ = nullptr;
    return S_OK;
  }
  HRESULT __stdcall ConnectedTo(IPin** peer) override {
    std::printf("  trace: source pin ConnectedTo\n");
    if (!peer) return E_POINTER;
    *peer = peer_;
    if (peer_) peer_->AddRef();
    return peer_ ? S_OK : VFW_E_NOT_CONNECTED;
  }
  HRESULT __stdcall ConnectionMediaType(AM_MEDIA_TYPE* mt) override {
    std::printf("  trace: source pin ConnectionMediaType\n");
    if (!mt) return E_POINTER;
    ZeroMemory(mt, sizeof(*mt));
    return VFW_E_NOT_CONNECTED;
  }
  HRESULT __stdcall QueryPinInfo(PIN_INFO* info) override; /* defined below */
  HRESULT __stdcall QueryDirection(PIN_DIRECTION* dir) override {
    std::printf("  trace: source pin QueryDirection\n");
    if (!dir) return E_POINTER;
    *dir = PINDIR_OUTPUT;
    return S_OK;
  }
  HRESULT __stdcall QueryId(LPWSTR* id) override {
    std::printf("  trace: source pin QueryId\n");
    if (!id) return E_POINTER;
    *id = static_cast<LPWSTR>(CoTaskMemAlloc(16));
    if (!*id) return E_OUTOFMEMORY;
    wcscpy_s(*id, 16, L"AV1 out");
    return S_OK;
  }
  HRESULT __stdcall QueryAccept(const AM_MEDIA_TYPE*) override {
    std::printf("  trace: source pin QueryAccept\n");
    return S_OK;
  }
  HRESULT __stdcall EnumMediaTypes(IEnumMediaTypes**) override {
    std::printf("  trace: source pin EnumMediaTypes\n");
    return E_NOTIMPL;
  }
  HRESULT __stdcall QueryInternalConnections(IPin**, ULONG*) override {
    std::printf("  trace: source pin QueryInternalConnections\n");
    return E_NOTIMPL;
  }
  HRESULT __stdcall EndOfStream() override { return S_OK; }
  HRESULT __stdcall BeginFlush() override { return S_OK; }
  HRESULT __stdcall EndFlush() override { return S_OK; }
  HRESULT __stdcall NewSegment(REFERENCE_TIME, REFERENCE_TIME, double) override {
    return S_OK;
  }

  /* IMemInputPin (never used on an output pin) */
  HRESULT __stdcall GetAllocator(IMemAllocator**) override { return E_NOTIMPL; }
  HRESULT __stdcall NotifyAllocator(IMemAllocator*, BOOL) override { return S_OK; }
  HRESULT __stdcall GetAllocatorRequirements(ALLOCATOR_PROPERTIES*) override {
    return E_NOTIMPL;
  }
  HRESULT __stdcall Receive(IMediaSample*) override { return E_UNEXPECTED; }
  HRESULT __stdcall ReceiveMultiple(IMediaSample**, long, long*) override {
    return E_UNEXPECTED;
  }
  HRESULT __stdcall ReceiveCanBlock() override { return S_OK; }

 private:
  LONG refs_ = 1;
  SourceFilter* filter_;
  IPin* peer_ = nullptr;
};

/* The owning filter; quartz asks for it through QueryPinInfo. */
class SourceFilter : public IBaseFilter {
 public:
  SourceFilter() : pin_(this) { name_[0] = 0; }
  SourceOutPin* pin() { return &pin_; }

  ULONG __stdcall AddRef() override { return interlocked(&refs_, 1); }
  ULONG __stdcall Release() override {
    const LONG r = interlocked(&refs_, -1);
    if (r == 0) delete this;
    return static_cast<ULONG>(r);
  }
  HRESULT __stdcall QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_IPersist) {
      *out = static_cast<IPersist*>(this);
    } else if (riid == IID_IMediaFilter || riid == IID_IBaseFilter) {
      *out = static_cast<IBaseFilter*>(this);
    } else {
      *out = nullptr;
      return E_NOINTERFACE;
    }
    reinterpret_cast<IUnknown*>(*out)->AddRef();
    return S_OK;
  }
  HRESULT __stdcall GetClassID(CLSID*) override { return E_NOTIMPL; }
  HRESULT __stdcall Stop() override {
    std::printf("  trace: source filter Stop\n");
    return S_OK;
  }
  HRESULT __stdcall Pause() override {
    std::printf("  trace: source filter Pause\n");
    return S_OK;
  }
  HRESULT __stdcall Run(REFERENCE_TIME) override {
    std::printf("  trace: source filter Run\n");
    return S_OK;
  }
  HRESULT __stdcall GetState(DWORD, FILTER_STATE* state) override {
    std::printf("  trace: source filter GetState\n");
    if (!state) return E_POINTER;
    *state = State_Paused;
    return S_OK;
  }
  HRESULT __stdcall SetSyncSource(IReferenceClock*) override {
    std::printf("  trace: source filter SetSyncSource\n");
    return S_OK;
  }
  HRESULT __stdcall GetSyncSource(IReferenceClock** clock) override {
    std::printf("  trace: source filter GetSyncSource\n");
    if (!clock) return E_POINTER;
    *clock = nullptr;
    return S_OK;
  }
  HRESULT __stdcall QueryFilterInfo(FILTER_INFO* info) override {
    std::printf("  trace: source filter QueryFilterInfo\n");
    if (!info) return E_POINTER;
    wcscpy_s(info->achName, L"av1gpu quartz test source");
    info->pGraph = graph_;
    if (graph_) graph_->AddRef();
    return S_OK;
  }
  HRESULT __stdcall JoinFilterGraph(IFilterGraph* graph, LPCWSTR) override {
    std::printf("  trace: source filter JoinFilterGraph\n");
    graph_ = graph;
    return S_OK;
  }
  HRESULT __stdcall QueryVendorInfo(LPWSTR*) override { return E_NOTIMPL; }
  HRESULT __stdcall EnumPins(IEnumPins** out) override {
    std::printf("  trace: source filter EnumPins\n");
    if (!out) return E_POINTER;
    *out = new SourceEnumPins(&pin_);
    return S_OK;
  }
  HRESULT __stdcall FindPin(LPCWSTR, IPin**) override { return VFW_E_NOT_FOUND; }

 private:
  LONG refs_ = 1;
  SourceOutPin pin_;
  IFilterGraph* graph_ = nullptr;
  wchar_t name_[128];
};

/* quartz walks to the owning filter during ConnectDirect */
HRESULT SourceOutPin::QueryPinInfo(PIN_INFO* info) {
  std::printf("  trace: source pin QueryPinInfo\n");
  if (!info) return E_POINTER;
  info->pFilter = filter_;
  filter_->AddRef();
  info->dir = PINDIR_OUTPUT;
  wcscpy_s(info->achName, L"AV1 out");
  return S_OK;
}

AM_MEDIA_TYPE* copy_mt(const AM_MEDIA_TYPE* src) {
  if (!src) return nullptr;
  AM_MEDIA_TYPE* mt = static_cast<AM_MEDIA_TYPE*>(CoTaskMemAlloc(sizeof(*mt)));
  if (!mt) return nullptr;
  *mt = *src;
  mt->pbFormat = nullptr;
  if (src->cbFormat && src->pbFormat) {
    mt->pbFormat = static_cast<BYTE*>(CoTaskMemAlloc(src->cbFormat));
    if (!mt->pbFormat) {
      CoTaskMemFree(mt);
      return nullptr;
    }
    CopyMemory(mt->pbFormat, src->pbFormat, src->cbFormat);
  }
  return mt;
}

void free_mt_shim(AM_MEDIA_TYPE* mt) {
  if (!mt) return;
  if (mt->cbFormat && mt->pbFormat) CoTaskMemFree(mt->pbFormat);
  CoTaskMemFree(mt);
}

AM_MEDIA_TYPE* make_av1_type(int width, int height) {  VIDEOINFOHEADER2* v = static_cast<VIDEOINFOHEADER2*>(CoTaskMemAlloc(sizeof(*v)));
  if (!v) return nullptr;
  ZeroMemory(v, sizeof(*v));
  v->rcSource.right = width;
  v->rcSource.bottom = height;
  v->rcTarget = v->rcSource;
  v->AvgTimePerFrame = 333667;
  v->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  v->bmiHeader.biWidth = width;
  v->bmiHeader.biHeight = height;
  v->bmiHeader.biPlanes = 1;
  v->bmiHeader.biCompression = '1V4A';
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

} /* namespace */

namespace {
/* No debugger on this machine: catch the access violation ourselves and dump
 * the faulting instruction so the NULL vtable slot can be identified. */
bool g_in_handler = false;

LONG WINAPI av_handler(EXCEPTION_POINTERS* ep) {
  if (g_in_handler) return EXCEPTION_CONTINUE_SEARCH;
  if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
    return EXCEPTION_CONTINUE_SEARCH; /* e.g. OutputDebugString's 0x40010006 */
  g_in_handler = true;
  const CONTEXT* c = ep->ContextRecord;
  std::printf("\n!!! EXCEPTION 0x%08lx at rip=%016llx\n",
              static_cast<unsigned long>(ep->ExceptionRecord->ExceptionCode),
              static_cast<unsigned long long>(c->Rip));
  std::printf("  rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n",
              static_cast<unsigned long long>(c->Rax),
              static_cast<unsigned long long>(c->Rbx),
              static_cast<unsigned long long>(c->Rcx),
              static_cast<unsigned long long>(c->Rdx));
  std::printf("  r8=%016llx  r9=%016llx  rsp=%016llx rbp=%016llx\n",
              static_cast<unsigned long long>(c->R8),
              static_cast<unsigned long long>(c->R9),
              static_cast<unsigned long long>(c->Rsp),
              static_cast<unsigned long long>(c->Rbp));
  /* dump the 16 instruction bytes at rip */
  const unsigned char* rip = reinterpret_cast<const unsigned char*>(c->Rip);
  std::printf("  bytes at rip:");
  __try {
    for (int i = 0; i < 16; ++i) std::printf(" %02x", rip[i]);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    std::printf(" <unreadable>");
  }
  std::printf("\n");
  /* what object is rcx pointing at? dump its first qwords */
  unsigned long long objdump[8] = {0};
  bool rcx_readable = false;
  __try {
    const unsigned long long* obj =
        reinterpret_cast<const unsigned long long*>(c->Rcx);
    for (int i = 0; i < 8; ++i) objdump[i] = obj[i];
    rcx_readable = true;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
  }
  if (rcx_readable) {
    std::printf("  [rcx+0x00..0x38]:");
    for (int i = 0; i < 8; ++i) std::printf(" %016llx", objdump[i]);
    std::printf("\n");
    /* context before rcx: is rcx inside a copied pin-pointer array? */
    __try {
      const unsigned long long* back =
          reinterpret_cast<const unsigned long long*>(c->Rcx) - 8;
      std::printf("  [rcx-0x40..-0x08]:");
      for (int i = 0; i < 8; ++i)
        std::printf(" %016llx", static_cast<unsigned long long>(back[i]));
      std::printf("\n");
    } __except (EXCEPTION_EXECUTE_HANDLER) {
      std::printf("  [rcx-...] unreadable\n");
    }
  } else {
    std::printf("  [rcx] unreadable\n");
  }
  const unsigned long long* sp =
      reinterpret_cast<const unsigned long long*>(c->Rsp);
  /* bytes just before the return address = the call instruction itself */
  __try {
    const unsigned char* call_site =
        reinterpret_cast<const unsigned char*>(sp[0]) - 16;
    std::printf("  bytes before return addr:");
    for (int i = 0; i < 20; ++i) std::printf(" %02x", call_site[i]);
    std::printf("\n");
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    std::printf("  return addr unreadable\n");
  }
  /* which module holds [rcx+0x18]? */
  if (rcx_readable && objdump[3] > 0x10000) {
    HMODULE owner = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(objdump[3]), &owner) &&
        owner) {
      char name[MAX_PATH] = {0};
      GetModuleFileNameA(owner, name, MAX_PATH);
      std::printf("  [rcx+0x18] module: %s+0x%llx\n", name, objdump[3] -
                  reinterpret_cast<unsigned long long>(owner));
    }
  }
  /* walk the stack: resolve each plausible return address to module+offset */
  __try {
    for (int i = 0; i < 220; ++i) {
      const unsigned long long v = sp[i];
      if (v < 0x10000) continue;
      HMODULE mod = nullptr;
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCSTR>(v), &mod) && mod) {
        char name[MAX_PATH] = {0};
        GetModuleFileNameA(mod, name, MAX_PATH);
        const char* base_name = name;
        for (const char* p = name; *p; ++p)
          if (*p == '\\' || *p == '/') base_name = p + 1;
        std::printf("  [rsp+%4d] %016llx  %s+0x%llx\n", i * 8, v, base_name,
                    v - reinterpret_cast<unsigned long long>(mod));
      }
    }
  } __except (EXCEPTION_EXECUTE_HANDLER) {
  }
  /* dump the vtable of the object at rcx: entry [13] (offset 0x68) is the
   * NULL that got called through quartz's jmp rax thunk */
  if (rcx_readable && objdump[0] > 0x10000) {
    __try {
      const unsigned long long* vtbl =
          reinterpret_cast<const unsigned long long*>(objdump[0]);
      std::printf("  vtable at %016llx:\n", objdump[0]);
      for (int i = 0; i < 16; ++i) {
        const unsigned long long fn = vtbl[i];
        HMODULE mod = nullptr;
        char where[64] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(fn), &mod) && mod) {
          char name[MAX_PATH] = {0};
          GetModuleFileNameA(mod, name, MAX_PATH);
          const char* base_name = name;
          for (const char* p = name; *p; ++p)
            if (*p == '\\' || *p == '/') base_name = p + 1;
          std::snprintf(where, sizeof(where), "%s+0x%llx", base_name,
                        fn - reinterpret_cast<unsigned long long>(mod));
        }
        std::printf("    [%2d] %016llx  %s%s\n", i, fn, where,
                    i == 13 ? "   <== called NULL" : "");
      }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
      std::printf("  vtable unreadable\n");
    }
  }
  __try {
    const unsigned long long ret = sp[0];
    const unsigned long long slot =
        ret + 0x0fe2a4 + 4; /* call [rip+0x0fe2a4]; next insn = ret */
    HMODULE qmod = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(ret), &qmod) && qmod) {
      const unsigned char* dos =
          reinterpret_cast<const unsigned char*>(qmod);
      const unsigned long long e_lfanew = static_cast<unsigned long long>(
          *reinterpret_cast<const unsigned long*>(dos + 0x3c));
      const unsigned char* nt = dos + e_lfanew;
      const unsigned char* opt = nt + 0x18;
      const unsigned long long img_base =
          *reinterpret_cast<const unsigned long long*>(opt + 0x18);      const unsigned long long iat_rva =
          *reinterpret_cast<const unsigned long*>(opt + 0xD0); /* DataDir[12] */
      const unsigned long long iat_size =
          *reinterpret_cast<const unsigned long*>(opt + 0xD4);
      const unsigned long long slot_rva = slot - img_base;
      std::printf("  called slot rva=0x%llx (iat rva=0x%llx size=0x%llx)\n",
                  slot_rva, iat_rva, iat_size);
      /* walk import descriptors */
      const unsigned long long impdir_rva =
          *reinterpret_cast<const unsigned long*>(opt + 0x78); /* DataDir[1] */
      if (impdir_rva) {
        struct ImpDesc {
          unsigned long long OriginalFirstThunk;
          unsigned long long TimeDateStamp;
          unsigned long long ForwarderChain;
          unsigned long long Name;
          unsigned long long FirstThunk;
        };
        const ImpDesc* desc =
            reinterpret_cast<const ImpDesc*>(dos + impdir_rva);
        for (; desc->FirstThunk; ++desc) {
          const unsigned long long ft_start = desc->FirstThunk;
          const unsigned long long oft_start = desc->OriginalFirstThunk
                                                   ? desc->OriginalFirstThunk
                                                   : desc->FirstThunk;
          const char* dll_name = reinterpret_cast<const char*>(dos + desc->Name);
          int n = 0;
          for (;; ++n) {
            const unsigned long long ft_rva =
                ft_start + static_cast<unsigned long long>(n) * 8;
            if (slot_rva >= ft_rva && slot_rva < ft_rva + 8) {
              const unsigned long long oft =
                  *reinterpret_cast<const unsigned long long*>(dos + oft_start +
                                                               n * 8);
              if (oft & 0x8000000000000000ULL) {
                std::printf("  import: %s ordinal #%llu\n", dll_name,
                            oft & 0xFFFFULL);
              } else {
                const unsigned short hint =
                    *reinterpret_cast<const unsigned short*>(dos + oft);
                std::printf("  import: %s !%s\n", dll_name,
                            reinterpret_cast<const char*>(dos + oft + 2));
              }
              break;
            }
            if (*reinterpret_cast<const unsigned long long*>(dos + ft_rva) == 0)
              break;
          }
        }
      }
    }
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    std::printf("  import lookup failed\n");
  }
  std::printf("\n");
  std::fflush(stdout);
  return EXCEPTION_CONTINUE_SEARCH;
}
} /* namespace */

int main(int argc, char** argv) {
  AddVectoredExceptionHandler(1, &av_handler);
  setvbuf(stdout, nullptr, _IONBF, 0); /* crashes must not swallow the trace */
  std::string dll_path = "build/bin/Release/av1gpu_dshow.dll";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--dll") && i + 1 < argc) dll_path = argv[++i];
  }
  HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  if (FAILED(hr)) die("CoInitializeEx", hr);

  /* the real quartz graph manager, the object MPC-HC's CFGManager talks to */
  IGraphBuilder* graph = nullptr;
  hr = CoCreateInstance(CLSID_FilterGraph, nullptr, CLSCTX_INPROC_SERVER,
                        IID_IGraphBuilder, reinterpret_cast<void**>(&graph));
  if (FAILED(hr)) die("CoCreateInstance(CLSID_FilterGraph)", hr);

  SourceFilter* source = new SourceFilter();
  hr = graph->AddFilter(source, L"Source");
  if (FAILED(hr)) die("AddFilter(source)", hr);
  std::printf("  source added to the graph\n");

  HMODULE module = LoadLibraryA(dll_path.c_str());
  if (!module) {
    die("LoadLibrary (see the path above)", HRESULT_FROM_WIN32(GetLastError()));
  }
  const auto get_class_object = reinterpret_cast<HRESULT(__stdcall*)(REFCLSID, REFIID,
                                                                     void**)>(
      GetProcAddress(module, "DllGetClassObject"));
  if (!get_class_object) die("no DllGetClassObject", E_FAIL);
  IClassFactory* factory = nullptr;
  hr = get_class_object(kDecoderClsid, IID_IClassFactory,
                        reinterpret_cast<void**>(&factory));
  if (FAILED(hr)) die("DllGetClassObject", hr);
  IBaseFilter* decoder = nullptr;
  hr = factory->CreateInstance(nullptr, IID_IBaseFilter,
                               reinterpret_cast<void**>(&decoder));
  factory->Release();
  if (FAILED(hr)) die("CreateInstance", hr);
  hr = graph->AddFilter(decoder, L"AV1onGPU AV1 Decoder");
  if (FAILED(hr)) die("AddFilter(decoder)", hr);
  std::printf("  decoder added to the graph\n");

  IBaseFilter* evr = nullptr;
  hr = CoCreateInstance(CLSID_EnhancedVideoRenderer, nullptr, CLSCTX_INPROC_SERVER,
                        IID_IBaseFilter, reinterpret_cast<void**>(&evr));
  if (FAILED(hr)) {
    std::printf("av1gpu_quartz: no EVR on this machine (0x%08lx), trying the"
                " video renderer\n",
                static_cast<unsigned long>(hr));
    hr = CoCreateInstance(CLSID_VideoRendererDefault, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IBaseFilter, reinterpret_cast<void**>(&evr));
  }
  if (SUCCEEDED(hr)) {
    hr = graph->AddFilter(evr, L"Renderer");
    if (FAILED(hr)) die("AddFilter(renderer)", hr);
    std::printf("  renderer added to the graph\n");
  }

  /* 1. source -> decoder through quartz, the way the player does it */
  std::printf("av1gpu_quartz: connecting source -> decoder\n");
  IPin* out_pin = source->pin();
  IPin* in_pin = nullptr;
  {
    IEnumPins* e = nullptr;
    hr = decoder->EnumPins(&e);
    if (FAILED(hr)) die("decoder EnumPins", hr);
    while (e->Next(1, &in_pin, nullptr) == S_OK) {
      PIN_DIRECTION dir;
      in_pin->QueryDirection(&dir);
      if (dir == PINDIR_INPUT) break;
      in_pin->Release();
      in_pin = nullptr;
    }
    e->Release();
  }
  if (!in_pin) die("decoder has no input pin", E_FAIL);
  bool free_mt = false;
  bool no_decoder = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--free-mt")) free_mt = true;
    if (!std::strcmp(argv[i], "--no-decoder")) no_decoder = true;
  }
  IPin* sink_pin = in_pin;
  if (no_decoder) {
    std::printf("av1gpu_quartz: connecting source -> renderer (no decoder)\n");
    sink_pin = nullptr;
    IEnumPins* e = nullptr;
    hr = evr->EnumPins(&e);
    if (FAILED(hr)) die("renderer EnumPins", hr);
    while (e->Next(1, &sink_pin, nullptr) == S_OK) {
      PIN_DIRECTION dir;
      sink_pin->QueryDirection(&dir);
      if (dir == PINDIR_INPUT) break;
      sink_pin->Release();
      sink_pin = nullptr;
    }
    e->Release();
    if (!sink_pin) die("renderer has no input pin", E_FAIL);
  }
  AM_MEDIA_TYPE* av1 = free_mt ? nullptr : make_av1_type(1920, 1080);
  hr = graph->ConnectDirect(out_pin, sink_pin, av1);
  if (FAILED(hr)) {
    std::printf("  ConnectDirect(source->decoder) refused: 0x%08lx\n",
                static_cast<unsigned long>(hr));
  } else {
    std::printf("  connected source -> decoder\n");
  }
  if (av1) {
    if (av1->pbFormat) CoTaskMemFree(av1->pbFormat);
    CoTaskMemFree(av1);
  }
  in_pin->Release();

  /* 2. decoder -> renderer, media type free (let the pins agree) */
  if (SUCCEEDED(graph->AddFilter(evr, L"Renderer")) || evr) {
    IPin* dec_out = nullptr;
    {
      IEnumPins* e = nullptr;
      hr = decoder->EnumPins(&e);
      if (FAILED(hr)) die("decoder EnumPins", hr);
      while (e->Next(1, &dec_out, nullptr) == S_OK) {
        PIN_DIRECTION dir;
        dec_out->QueryDirection(&dir);
        if (dir == PINDIR_OUTPUT) break;
        dec_out->Release();
        dec_out = nullptr;
      }
      e->Release();
    }
    if (dec_out) {
      IPin* evr_in = nullptr;
      {
        IEnumPins* e = nullptr;
        hr = evr->EnumPins(&e);
        if (FAILED(hr)) die("renderer EnumPins", hr);
        while (e->Next(1, &evr_in, nullptr) == S_OK) {
          PIN_DIRECTION dir;
          evr_in->QueryDirection(&dir);
          if (dir == PINDIR_INPUT) break;
          evr_in->Release();
          evr_in = nullptr;
        }
        e->Release();
      }
      if (evr_in) {
        std::printf("av1gpu_quartz: connecting decoder -> renderer\n");
        hr = graph->ConnectDirect(dec_out, evr_in, nullptr);
        if (FAILED(hr)) {
          std::printf("  ConnectDirect(decoder->renderer) refused: 0x%08lx\n",
                      static_cast<unsigned long>(hr));
        } else {
          std::printf("  connected decoder -> renderer\n");
        }
        evr_in->Release();
      }
      dec_out->Release();
    }
  }

  std::printf("av1gpu_quartz: graph built, no crash\n");
  if (evr) evr->Release();
  decoder->Release();
  source->Release();
  graph->Release();
  CoUninitialize();
  return 0;
}

