/*
 * AV1onGPU - minimal CUDA driver + NVRTC loader
 *
 * Like the OpenCL side, everything is resolved with dlopen so that the library
 * builds and runs on machines without the CUDA toolkit: if libcuda or libnvrtc
 * is missing, the CUDA backend simply reports itself as unavailable instead of
 * preventing the program from starting.
 */
#ifndef AV1GPU_CUDA_CUDA_MIN_HPP
#define AV1GPU_CUDA_CUDA_MIN_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace av1gpu {
namespace cuda {

#if defined(_WIN32)
/* the loader code reads the same on every OS: what dlsym is on POSIX is
 * GetProcAddress on Windows (dlopen is LoadLibraryA at the call sites) */
inline void* dlsym(void* handle, const char* name) {
  return reinterpret_cast<void*>(::GetProcAddress(static_cast<::HMODULE>(handle), name));
}
#endif

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUevent = void*;
using CUdeviceptr = unsigned long long;
using CUgraph = void*;
using CUgraphNode = void*;
using CUgraphExec = void*;

/* CUDA 11.8 header layout (which the 13.x driver accepts via the unversioned
 * cuGraphAddKernelNode symbol): no hStream field. The CUDA 12 layout inserts
 * hStream before kernelParams, which shifts the fields and is rejected by
 * this driver with rc=700/1. */
struct CUDA_KERNEL_NODE_PARAMS {
  CUfunction func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  void** kernelParams;
  void** extra;
};

constexpr CUresult CUDA_SUCCESS = 0;

/* attribute ids we use */
constexpr int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
constexpr int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

struct Driver {
  CUresult (*Init)(unsigned);
  CUresult (*DeviceGetCount)(int*);
  CUresult (*DeviceGet)(CUdevice*, int);
  CUresult (*DeviceGetName)(char*, int, CUdevice);
  CUresult (*DeviceGetAttribute)(int*, int, CUdevice);
  CUresult (*DeviceTotalMem)(size_t*, CUdevice);
  CUresult (*CtxCreate)(CUcontext*, unsigned, CUdevice);
  CUresult (*CtxDestroy)(CUcontext);
  CUresult (*CtxSynchronize)(void);
  CUresult (*CtxSetCurrent)(CUcontext); /* workers must adopt the context */
  CUresult (*CtxGetCurrent)(CUcontext*);
  CUresult (*DriverGetVersion)(int*);
  /* events, for GPU side timing */
  CUresult (*EventCreate)(void**, unsigned);
  CUresult (*EventRecord)(void*, CUstream);
  CUresult (*EventSynchronize)(void*);
  CUresult (*EventElapsedTime)(float*, void*, void*);
  CUresult (*EventDestroy)(void*);
  CUresult (*MemAlloc)(CUdeviceptr*, size_t);
  CUresult (*MemFree)(CUdeviceptr);
  CUresult (*MemcpyHtoD)(CUdeviceptr, const void*, size_t);
  CUresult (*MemcpyDtoH)(void*, CUdeviceptr, size_t);
  /* the asynchronous copy, for a read back that runs beside the decode it
   * follows; may be null on a driver that does not export it, in which case
   * the caller falls back to the synchronous one */
  CUresult (*MemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
  CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
  CUresult (*MemAllocHost)(void**, size_t); /* page-locked host memory */
  CUresult (*MemFreeHost)(void*);
  CUresult (*ModuleLoadDataEx)(CUmodule*, const void*, unsigned, int*, void**);
  CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*);
  CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, CUstream, void**, void**);
  /* cuLaunchKernelEx (CUDA 11.8+), may be null. CUlaunchConfig layout:
   * gridDimX/Y/Z, blockDimX/Y/Z, sharedMemBytes, hStream, attrs, numAttrs */
  CUresult (*LaunchKernelEx)(const void* config, CUfunction,
                             void** kernelParams, void** extra);
  /* graphs: one submission for the whole frame instead of a kernel launch per
   * wave, which is what a keyframe's several hundred waves cost on WDDM.
   * CUDA 12 API: parameters ride in a struct. */
  CUresult (*GraphCreate)(CUgraph*, unsigned);
  CUresult (*GraphAddKernelNode)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t,
                                 const CUDA_KERNEL_NODE_PARAMS*);  CUresult (*GraphAddDependencies)(CUgraph, const CUgraphNode*, const CUgraphNode*, size_t);
  /* cuGraphInstantiate unversioned symbol = v2: 5 args (errorNode, logBuf, logBufSize) */
  CUresult (*GraphInstantiate)(CUgraphExec*, CUgraph, CUgraphNode*, char*, size_t);
  CUresult (*GraphLaunch)(CUgraphExec, CUstream);
  CUresult (*GraphExecDestroy)(CUgraphExec);
  CUresult (*GraphDestroy)(CUgraph);
  /* Stream for graph kernel nodes - WDDM drivers may reject nullptr hStream */
  CUresult (*StreamCreate)(CUstream*, unsigned);
  CUresult (*StreamDestroy)(CUstream);
  CUresult (*StreamWaitEvent)(CUstream, CUevent, unsigned);
  CUresult (*StreamSynchronize)(CUstream);
  /* stream capture: records the kernel launches of a frame into a CUDA graph
   * without touching the versioned CUDA_KERNEL_NODE_PARAMS ABI. Optional. */
  CUresult (*StreamBeginCapture)(CUstream, unsigned);
  CUresult (*StreamEndCapture)(CUstream, CUgraph*);
};

struct Nvrtc {
  int (*CreateProgram)(void**, const char*, const char*, int, const char* const*,
                       const char* const*);
  int (*CompileProgram)(void*, int, const char* const*);
  int (*GetProgramLogSize)(void*, size_t*);
  int (*GetProgramLog)(void*, char*);
  int (*GetPTXSize)(void*, size_t*);
  int (*GetPTX)(void*, char*);
  /* CUDA 11 and up can hand out the assembled cubin directly, which spares the
   * driver's PTX JIT - several seconds on a large kernel set. Optional. */
  int (*GetCUBINSize)(void*, size_t*);
  int (*GetCUBIN)(void*, char*);
  int (*DestroyProgram)(void**);
};

bool load_driver(Driver* driver, void** library, std::string* error);
bool load_nvrtc(Nvrtc* nvrtc, void** library, std::string* error);
void close_library(void* library);

} /* namespace cuda */
} /* namespace av1gpu */

#endif /* AV1GPU_CUDA_CUDA_MIN_HPP */
