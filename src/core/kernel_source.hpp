/*
 * AV1onGPU - embedded kernel sources
 *
 * The OpenCL and CUDA sources are generated at build time by cmake/embed.cmake
 * (the shared header is inlined, because NVRTC has no include path that can be
 * relied upon) and compiled into the library. That way the library works from
 * any working directory and needs no installed kernel files.
 */
#ifndef AV1GPU_CORE_KERNEL_SOURCE_HPP
#define AV1GPU_CORE_KERNEL_SOURCE_HPP

#include <string>

namespace av1gpu {
namespace internal {
namespace kernel_source {

const std::string& opencl();
const std::string& cuda();

} /* namespace kernel_source */
} /* namespace internal */
} /* namespace av1gpu */

#endif /* AV1GPU_CORE_KERNEL_SOURCE_HPP */
