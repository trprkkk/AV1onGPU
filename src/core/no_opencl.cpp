/*
 * AV1onGPU - OpenCL backend disabled at build time
 */
#include <memory>
#include <string>

#include "core/backend.hpp"

namespace av1gpu {
namespace internal {

std::unique_ptr<Backend> create_opencl_backend(const BackendOptions&, std::string* error) {
  if (error) *error = "this build has no OpenCL backend (AV1GPU_ENABLE_OPENCL=OFF)";
  return nullptr;
}

std::vector<DeviceInfo> list_opencl_devices() { return std::vector<DeviceInfo>(); }

} /* namespace internal */
} /* namespace av1gpu */
