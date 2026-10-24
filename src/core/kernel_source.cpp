/*
 * AV1onGPU - embedded kernel sources
 *
 * The real definitions are generated at build time by cmake/embed.cmake, one
 * translation unit per language (generated/opencl_source.cpp and
 * generated/cuda_source.cpp). This file only provides a safe fallback so that a
 * build with both GPU backends disabled still links.
 */
#include "core/kernel_source.hpp"

namespace av1gpu {
namespace internal {
namespace kernel_source {

#if !defined(AV1GPU_HAVE_OPENCL) && !defined(AV1GPU_HAVE_CUDA)
const std::string& const_empty() {
  static const std::string empty;
  return empty;
}
#endif

const std::string& opencl() {
#if defined(AV1GPU_HAVE_OPENCL)
  extern const std::string& opencl_generated();
  return opencl_generated();
#else
  static const std::string empty;
  return empty;
#endif
}

const std::string& cuda() {
#if defined(AV1GPU_HAVE_CUDA)
  extern const std::string& cuda_generated();
  return cuda_generated();
#else
  static const std::string empty;
  return empty;
#endif
}

} /* namespace kernel_source */
} /* namespace internal */
} /* namespace av1gpu */
