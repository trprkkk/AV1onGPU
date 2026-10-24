/*
 * AV1onGPU - CUDA backend disabled at build time
 */
#include <memory>
#include <string>

#include "core/backend.hpp"

namespace av1gpu {
namespace internal {

std::unique_ptr<Backend> create_cuda_backend(const BackendOptions&, std::string* error) {
  if (error) *error = "this build has no CUDA backend (AV1GPU_ENABLE_CUDA=OFF)";
  return nullptr;
}

std::vector<DeviceInfo> list_cuda_devices() { return std::vector<DeviceInfo>(); }

} /* namespace internal */
} /* namespace av1gpu */
