/*
 * AV1onGPU - minimal OpenCL loader
 *
 * The OpenCL entry points are resolved at run time with dlopen/dlsym instead of
 * being linked against libOpenCL. That has three practical benefits:
 *
 *   - the project builds on machines without an OpenCL SDK or headers,
 *   - a missing or broken ICD loader degrades to "backend unavailable" instead
 *     of failing the whole binary at load time (a real problem on headless
 *     Linux servers and inside containers),
 *   - the same binary can pick between several installed runtimes.
 *
 * Only the small subset of OpenCL 2.0 that the backend actually uses is
 * declared here.
 */
#ifndef AV1GPU_OPENCL_CL_MIN_HPP
#define AV1GPU_OPENCL_CL_MIN_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace av1gpu {
namespace opencl {

using cl_int = int;
using cl_uint = unsigned int;
using cl_ulong = unsigned long long;
using cl_bitfield = unsigned long long;
using cl_bool = cl_uint;

struct _cl_platform_id;
struct _cl_device_id;
struct _cl_context;
struct _cl_command_queue;
struct _cl_mem;
struct _cl_program;
struct _cl_kernel;
struct _cl_event;

using cl_platform_id = _cl_platform_id*;
using cl_device_id = _cl_device_id*;
using cl_context = _cl_context*;
using cl_command_queue = _cl_command_queue*;
using cl_mem = _cl_mem*;
using cl_program = _cl_program*;
using cl_kernel = _cl_kernel*;
using cl_event = _cl_event*;
/* an integer type in the real headers */
using cl_context_properties = intptr_t;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_int CL_INVALID_WORK_GROUP_SIZE = -54;
constexpr cl_int CL_TRUE = 1;
constexpr cl_bitfield CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;
constexpr cl_bitfield CL_MEM_READ_WRITE = 1 << 0;
constexpr cl_bitfield CL_MEM_WRITE_ONLY = 1 << 1;
constexpr cl_bitfield CL_MEM_READ_ONLY = 1 << 2;
constexpr cl_bitfield CL_MEM_COPY_HOST_PTR = 1 << 5;

constexpr cl_uint CL_PLATFORM_NAME = 0x0902;
constexpr cl_uint CL_PLATFORM_VENDOR = 0x0903;
constexpr cl_uint CL_PLATFORM_VERSION = 0x0901;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_VENDOR = 0x102C;
constexpr cl_uint CL_DEVICE_VERSION = 0x102F;
constexpr cl_uint CL_DEVICE_DRIVER_VERSION = 0x102D;
constexpr cl_uint CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_uint CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
constexpr cl_uint CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
constexpr cl_uint CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_uint CL_DEVICE_TYPE_CPU = 1 << 1;
constexpr cl_uint CL_DEVICE_TYPE_GPU = 1 << 2;

/* The API surface, resolved at run time. */
struct Api {
  cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
  cl_int (*GetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
  cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
  cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
  cl_context (*CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                              void*, void*, cl_int*);
  cl_int (*ReleaseContext)(cl_context);
  cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
  cl_int (*ReleaseCommandQueue)(cl_command_queue);
  cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*,
                                        cl_int*);
  cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                         void (*)(cl_program, void*), void*);
  cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
  cl_int (*ReleaseProgram)(cl_program);
  cl_kernel (*CreateKernel)(cl_program, const char*, cl_int*);
  cl_int (*ReleaseKernel)(cl_kernel);
  cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
  cl_mem (*CreateBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*);
  cl_int (*ReleaseMemObject)(cl_mem);
  cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                               const void*, cl_uint, const cl_event*, cl_event*);
  cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*,
                              cl_uint, const cl_event*, cl_event*);
  cl_int (*EnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t,
                              cl_uint, const cl_event*, cl_event*);
  cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                                 const size_t*, const size_t*, cl_uint, const cl_event*,
                                 cl_event*);
  cl_int (*Finish)(cl_command_queue);
  cl_int (*Flush)(cl_command_queue);
};

/* Loads libOpenCL and resolves every entry point. Returns false and fills
 * `error` when no usable loader is installed. */
bool load(Api* api, void** library, std::string* error);

} /* namespace opencl */
} /* namespace av1gpu */

#endif /* AV1GPU_OPENCL_CL_MIN_HPP */
