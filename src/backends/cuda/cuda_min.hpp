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
using CUdeviceptr = unsigned long long;

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
  CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
  CUresult (*MemAllocHost)(void**, size_t); /* page-locked host memory */
  CUresult (*MemFreeHost)(void*);
  CUresult (*ModuleLoadDataEx)(CUmodule*, const void*, unsigned, int*, void**);
  CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*);
  CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, CUstream, void**, void**);
};

struct Nvrtc {
  int (*CreateProgram)(void**, const char*, const char*, int, const char* const*,
                       const char* const*);
  int (*CompileProgram)(void*, int, const char* const*);
  int (*GetProgramLogSize)(void*, size_t*);
  int (*GetProgramLog)(void*, char*);
  int (*GetPTXSize)(void*, size_t*);
  int (*GetPTX)(void*, char*);
  int (*DestroyProgram)(void**);
};

bool load_driver(Driver* driver, void** library, std::string* error);
bool load_nvrtc(Nvrtc* nvrtc, void** library, std::string* error);
void close_library(void* library);

} /* namespace cuda */
} /* namespace av1gpu */

#endif /* AV1GPU_CUDA_CUDA_MIN_HPP */
