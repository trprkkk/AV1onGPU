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

namespace av1gpu {
namespace cuda {

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
  CUresult (*MemAlloc)(CUdeviceptr*, size_t);
  CUresult (*MemFree)(CUdeviceptr);
  CUresult (*MemcpyHtoD)(CUdeviceptr, const void*, size_t);
  CUresult (*MemcpyDtoH)(void*, CUdeviceptr, size_t);
  CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
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
