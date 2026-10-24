/*
 * av1gpu_info - show the build information and every device AV1onGPU can use.
 */
#include <cstdio>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"

namespace {

std::string human_bytes(uint64_t bytes) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f %s", value, units[unit]);
  return buf;
}

} /* namespace */

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::printf("%s\n", av1gpu::build_info().c_str());

  std::printf("devices:\n");
  const std::vector<av1gpu::DeviceInfo> devices = av1gpu::enumerate_devices();
  int gpu_index = 0;
  for (const av1gpu::DeviceInfo& d : devices) {
    const char* label = av1gpu::to_string(d.kind);
    if (d.kind == av1gpu::BackendKind::Cpu) {
      std::printf("  [cpu] %-28s %s (%s)\n", d.name.c_str(), d.version.c_str(),
                  d.detail.c_str());
      continue;
    }
    std::printf("  [%s %d] %-24s %s\n", label, gpu_index++, d.name.c_str(),
                d.version.c_str());
    if (!d.detail.empty()) std::printf("        %s\n", d.detail.c_str());
    if (d.global_mem)
      std::printf("        %s of memory, %d compute units\n", human_bytes(d.global_mem).c_str(),
                  d.compute_units);
  }

  std::printf("\ntry: av1gpu_bench --backend auto\n");
  return 0;
}
