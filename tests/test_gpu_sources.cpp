/*
 * AV1onGPU - the GPU launch ABI, checked without a GPU
 *
 * The two device backends are thin: they pack jobs into a buffer, build an
 * argument list by hand and launch one of the twelve kernels. Nothing in that path is
 * type checked by the compiler - cuLaunchKernel takes `void**` and OpenCL takes
 * `clSetKernelArg(void*, size_t, size_t, const void*)` - so a missing argument,
 * a swapped pair or an argument set twice is a silent read of whatever was on
 * the stack. On hardware it shows up as wrong pixels somewhere in the frame, or
 * as a device fault a few frames later, and it cannot be found on a machine
 * without a GPU at all.
 *
 * This test makes those mistakes fail on any machine, in milliseconds:
 *
 *   * the embedded CUDA and OpenCL sources must declare the same kernels with
 *     the same parameter types in the same order, so a new argument added to
 *     one language cannot be forgotten in the other;
 *   * neither source may use a construct of the other language, because the
 *     embedded text is compiled by NVRTC and by the OpenCL driver as is;
 *   * every launch site in the backends must pass exactly as many arguments as
 *     the kernel it names declares, and the OpenCL indices must be exactly
 *     0..n-1 (the indices are written by hand);
 *   * every kernel a backend launches must exist in that backend's source.
 *
 * It reads the backend sources from the source tree, which CMake passes as
 * AV1GPU_SOURCE_DIR; the device sources come from the library itself, so what
 * is checked is what is embedded in the binary under test.
 */
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/kernel_source.hpp"
#include "test_common.hpp"

namespace {

const char* kSourceDir =
#if defined(AV1GPU_SOURCE_DIR)
    AV1GPU_SOURCE_DIR;
#else
    "";
#endif

std::string read_file(const std::string& path, bool* ok) {
  std::ifstream in(path);
  if (!in) {
    *ok = false;
    return std::string();
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *ok = true;
  return ss.str();
}

bool starts_with(const std::string& s, const char* prefix) {
  return s.compare(0, std::strlen(prefix), prefix) == 0;
}

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n'))
    --e;
  return s.substr(b, e - b);
}

/* A parameter type as both languages spell it: the storage and const
 * qualifiers are dropped, because `__global const int *x` and `const int *x`
 * are the same argument to the driver, and the name is dropped, because the two
 * files are allowed to name it differently. What is left is what decides whether
 * the bytes of the argument are read the way the kernel expects them. */
std::string parameter_type(const std::string& raw) {
  std::string t = trim(raw);
  static const char* const kQualifiers[] = {"__global", "__global__", "__constant", "__read_only",
                                            "__restrict", "__attribute__((aligned(16)))", "const",
                                            "volatile", "restrict", "struct"};
  bool changed = true;
  while (changed) {
    changed = false;
    for (const char* q : kQualifiers) {
      const size_t len = std::strlen(q);
      if (t.size() > len && t.compare(0, len, q) == 0 && (t[len] == ' ' || t[len] == '*')) {
        t = trim(t.substr(len));
        changed = true;
      }
    }
  }
  /* drop the trailing identifier, keeping the pointer stars in front of it */
  size_t end = t.size();
  while (end > 0 && (std::isalnum(static_cast<unsigned char>(t[end - 1])) || t[end - 1] == '_'))
    --end;
  if (end < t.size() && end > 0 &&
      (t[end - 1] == ' ' || t[end - 1] == '*' || t[end - 1] == '&'))
    t = t.substr(0, end);
  t = trim(t);
  /* one space around the stars, so `int *` and `int*` compare equal */
  std::string out;
  for (size_t i = 0; i < t.size(); ++i) {
    if (t[i] == '*') {
      while (!out.empty() && out.back() == ' ') out.pop_back();
      out += " *";
      while (i + 1 < t.size() && t[i + 1] == ' ') ++i;
    } else {
      out += t[i];
    }
  }
  return trim(out);
}

struct Kernel {
  std::string name;
  std::vector<std::string> params;
};

std::vector<std::string> split_parameters(const std::string& text) {
  std::vector<std::string> out;
  std::string current;
  int depth = 0;
  for (char c : text) {
    if (c == '(' || c == '[') ++depth;
    if (c == ')' || c == ']') --depth;
    if (c == ',' && depth == 0) {
      out.push_back(current);
      current.clear();
      continue;
    }
    current += c;
  }
  if (!trim(current).empty()) out.push_back(current);
  return out;
}

/* Finds every `av1g_k_<name>( ... )` declaration in a device source. The name
 * is always introduced by __kernel or __global__, which also keeps a mention in
 * a comment from being read as a declaration. */
std::map<std::string, Kernel> parse_kernels(const std::string& source) {
  std::map<std::string, Kernel> out;
  size_t pos = 0;
  while ((pos = source.find("av1g_k_", pos)) != std::string::npos) {
    const size_t name_start = pos;
    size_t name_end = pos;
    while (name_end < source.size() &&
           (std::isalnum(static_cast<unsigned char>(source[name_end])) ||
            source[name_end] == '_'))
      ++name_end;
    const size_t open = source.find('(', name_end);
    if (open == std::string::npos) break;
    /* between the name and the parenthesis there may only be whitespace */
    if (!trim(source.substr(name_end, open - name_end)).empty()) {
      pos = name_end;
      continue;
    }
    /* the declaration has to be preceded by the language's kernel keyword */
    const std::string before = source.substr(name_start > 48 ? name_start - 48 : 0,
                                             name_start > 48 ? 48 : name_start);
    if (before.find("__global__") == std::string::npos &&
        before.find("__kernel") == std::string::npos) {
      pos = name_end;
      continue;
    }
    int depth = 0;
    size_t close = open;
    for (; close < source.size(); ++close) {
      if (source[close] == '(') ++depth;
      if (source[close] == ')') {
        --depth;
        if (depth == 0) break;
      }
    }
    if (close >= source.size()) break;
    Kernel kernel;
    kernel.name = source.substr(name_start, name_end - name_start);
    for (const std::string& raw : split_parameters(source.substr(open + 1, close - open - 1)))
      kernel.params.push_back(parameter_type(raw));
    out[kernel.name] = kernel;
    pos = close;
  }
  return out;
}

std::string join(const std::vector<std::string>& parts) {
  std::string out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) out += ", ";
    out += parts[i];
  }
  return out;
}

/* The parameter list, as a comparable string. */
std::string signature(const Kernel& kernel) {
  return kernel.name + "(" + join(kernel.params) + ")";
}

void check_languages_agree(const std::string& opencl, const std::string& cuda) {
  const std::map<std::string, Kernel> a = parse_kernels(opencl);
  const std::map<std::string, Kernel> b = parse_kernels(cuda);
  std::printf("  opencl declares %zu kernel(s), cuda %zu\n", a.size(), b.size());
  CHECK_EQ(a.size(), b.size());
  for (const auto& entry : a) {
    const auto it = b.find(entry.first);
    ::testing::report(it != b.end(), "kernel declared in both languages", __FILE__, __LINE__,
                      entry.first);
    if (it == b.end()) continue;
    if (it->second.params != entry.second.params) {
      std::printf("  %s differs:\n    opencl: %s\n    cuda:   %s\n", entry.first.c_str(),
                  signature(entry.second).c_str(), signature(it->second).c_str());
    }
    CHECK(it->second.params == entry.second.params);
  }
}

void check_no_foreign_constructs(const std::string& opencl, const std::string& cuda) {
  /* NVRTC does not know __kernel or get_global_id, and an OpenCL driver does not
   * know __global__ or blockIdx: either one is a build failure at run time, on
   * the user's machine and nowhere else. */
  /* __constant and __kernel are deliberately absent from this list: the shared
   * header defines AV1G_TBL as `__constant` inside the branch OpenCL compiles
   * and as `__device__ __constant__` inside the one CUDA does, so the token
   * appears in both files and means nothing on its own. */
  static const char* const kOpenClOnly[] = {"get_global_id", "get_local_id", "get_group_id",
                                            "get_local_size", "__read_only", "__write_only"};
  for (const char* token : kOpenClOnly)
    CHECK(cuda.find(token) == std::string::npos);
  static const char* const kCudaOnly[] = {"__global__", "blockIdx", "threadIdx", "__shared__"};
  for (const char* token : kCudaOnly)
    CHECK(opencl.find(token) == std::string::npos);
  /* the shared header is inlined and removed, never included twice */
  CHECK(opencl.find("#include \"av1gpu_kernels.h\"") == std::string::npos);
  CHECK(cuda.find("#include \"av1gpu_kernels.h\"") == std::string::npos);
  CHECK(opencl.find("av1g_intra_block") != std::string::npos);
  CHECK(cuda.find("av1g_intra_block") != std::string::npos);
}

/* Walk back from a launch line over the argument packing and count the
 * arguments. Both backends pack the arguments immediately before the launch,
 * one per line, which is what makes this readable and checkable. */
int cuda_argument_count(const std::vector<std::string>& lines, size_t launch_line) {
  int count = 0;
  for (size_t i = launch_line; i-- > 0;) {
    const std::string line = trim(lines[i]);
    if (starts_with(line, "add_mem(a,") || starts_with(line, "add_int(a,")) {
      ++count;
      continue;
    }
    if (starts_with(line, "Launcher a;")) return count;
    if (line.empty() || starts_with(line, "{")) continue;
    return -1; /* not the shape this test knows how to read */
  }
  return -1;
}

void check_cuda_launches(const std::string& backend, const std::string& device_source) {
  const std::map<std::string, Kernel> kernels = parse_kernels(device_source);
  std::vector<std::string> lines;
  {
    std::istringstream in(backend);
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
  }
  int launches = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    const size_t at = lines[i].find("launch(\"av1g_k_");
    if (at == std::string::npos) continue;
    const size_t name_start = at + std::strlen("launch(\"");
    const size_t name_end = lines[i].find('"', name_start);
    if (name_end == std::string::npos) continue;
    const std::string name = lines[i].substr(name_start, name_end - name_start);
    ++launches;
    const auto it = kernels.find(name);
    ::testing::report(it != kernels.end(), "launched kernel exists in the CUDA source", __FILE__,
                      __LINE__, name);
    if (it == kernels.end()) continue;
    const int count = cuda_argument_count(lines, i);
    if (count < 0) {
      std::printf("  could not read the arguments of %s\n", name.c_str());
      CHECK(count >= 0);
      continue;
    }
    if (count != static_cast<int>(it->second.params.size())) {
      std::printf("  %s: %d argument(s) passed, %zu declared (%s)\n", name.c_str(), count,
                  it->second.params.size(), signature(it->second).c_str());
    }
    CHECK_EQ(count, static_cast<int>(it->second.params.size()));
  }
  CHECK(launches > 0);
}

void check_opencl_launches(const std::string& backend, const std::string& device_source) {
  const std::map<std::string, Kernel> kernels = parse_kernels(device_source);
  std::vector<std::string> lines;
  {
    std::istringstream in(backend);
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
  }
  int launches = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    const size_t at = lines[i].find("kernel(\"av1g_k_");
    if (at == std::string::npos) continue;
    /* cl_kernel k = kernel("name"); -> k */
    const std::string decl = trim(lines[i]);
    const size_t var_start = decl.find("cl_kernel ");
    if (var_start == std::string::npos) continue;
    const size_t var_end = decl.find_first_of(" =", var_start + 10);
    const std::string var = decl.substr(var_start + 10, var_end - var_start - 10);
    const size_t name_start = at + std::strlen("kernel(\"");
    const size_t name_end = lines[i].find('"', name_start);
    if (name_end == std::string::npos) continue;
    const std::string name = lines[i].substr(name_start, name_end - name_start);
    ++launches;
    const auto it = kernels.find(name);
    ::testing::report(it != kernels.end(), "launched kernel exists in the OpenCL source", __FILE__,
                      __LINE__, name);
    if (it == kernels.end()) continue;
    /* the indices are written by hand: they must be exactly 0..n-1. The block
     * between the declaration and the enqueue may also contain local variables
     * (the inter launch hoists the reference pool), so only set_* lines count
     * and the scan stops at the next kernel. */
    std::set<int> indices;
    for (size_t j = i + 1; j < lines.size(); ++j) {
      if (lines[j].find("kernel(\"av1g_k_") != std::string::npos) break;
      const std::string line = trim(lines[j]);
      const std::string mem = "set_mem(" + var + ",";
      const std::string num = "set_int(" + var + ",";
      const bool is_mem = starts_with(line, mem.c_str());
      const bool is_int = starts_with(line, num.c_str());
      if (!is_mem && !is_int) continue;
      const size_t idx_start = (is_mem ? mem.size() : num.size());
      const int index = std::atoi(line.c_str() + idx_start);
      if (!indices.insert(index).second) {
        std::printf("  %s: argument index %d is set twice\n", name.c_str(), index);
        CHECK(false);
      }
    }
    const size_t expected = it->second.params.size();
    bool contiguous = indices.size() == expected;
    for (size_t k = 0; contiguous && k < expected; ++k)
      contiguous = indices.count(static_cast<int>(k)) == 1;
    if (!contiguous) {
      std::printf("  %s: %zu argument index(es) set, %zu declared (%s)\n", name.c_str(),
                  indices.size(), expected, signature(it->second).c_str());
    }
    CHECK(contiguous);
  }
  CHECK(launches > 0);
}

/* Every offset in a job that points into the reference pool has to be moved by
 * the slot's base before the launch, and both backends have to do it to the same
 * fields: a field rebased in one backend and not the other reads another
 * reference frame (or another plane) and produces plausible pictures, which is
 * the worst kind of difference to debug. Collecting the `job.<field> +=` lines
 * of each backend catches a new offset that was only wired up in one of them. */
std::set<std::string> rebased_fields(const std::string& backend) {
  std::set<std::string> out;
  static const char* const kMarkers[] = {"job.ref0_off += ",  "job.ref1_off += ",
                                         "job.luma_ref0_off += ", "job.luma_ref1_off += ",
                                         "job.obmc[k].ref_off += "};
  for (const char* marker : kMarkers)
    if (backend.find(marker) != std::string::npos) out.insert(marker);
  return out;
}

void check_reference_rebase(const std::string& opencl_backend, const std::string& cuda_backend) {
  const std::set<std::string> a = rebased_fields(opencl_backend);
  const std::set<std::string> b = rebased_fields(cuda_backend);
  std::printf("  %zu reference offset(s) rebased in both backends\n", a.size());
  CHECK(!a.empty());
  for (const std::string& field : a) {
    if (b.count(field) == 0) std::printf("  %smissing from the CUDA backend\n", field.c_str());
    CHECK(b.count(field) == 1);
  }
  for (const std::string& field : b) {
    if (a.count(field) == 0) std::printf("  %smissing from the OpenCL backend\n", field.c_str());
    CHECK(a.count(field) == 1);
  }
  /* the difference weighted mask is built from luma, so its offsets are the
   * ones that are easy to forget */
  CHECK(opencl_backend.find("AV1G_MASK_DIFFWTD") != std::string::npos);
  CHECK(cuda_backend.find("AV1G_MASK_DIFFWTD") != std::string::npos);
}

} /* namespace */

int main() {
  std::printf("GPU launch ABI (no device needed)\n");
  const std::string opencl = av1gpu::internal::kernel_source::opencl();
  const std::string cuda = av1gpu::internal::kernel_source::cuda();
  if (opencl.empty() || cuda.empty()) {
    std::printf("  a backend is disabled in this build, nothing to check\n");
    return ::testing::finish("test_gpu_sources");
  }

  check_languages_agree(opencl, cuda);
  check_no_foreign_constructs(opencl, cuda);

  if (!kSourceDir[0]) {
    std::printf("  AV1GPU_SOURCE_DIR is not defined, skipping the launch sites\n");
    return ::testing::finish("test_gpu_sources");
  }
  bool ok = false;
  const std::string cuda_backend =
      read_file(std::string(kSourceDir) + "/src/backends/cuda/cuda_backend.cpp", &ok);
  CHECK(ok);
  const std::string opencl_backend =
      read_file(std::string(kSourceDir) + "/src/backends/opencl/opencl_backend.cpp", &ok);
  CHECK(ok);
  if (!cuda_backend.empty()) check_cuda_launches(cuda_backend, cuda);
  if (!opencl_backend.empty()) check_opencl_launches(opencl_backend, opencl);
  if (!cuda_backend.empty() && !opencl_backend.empty())
    check_reference_rebase(opencl_backend, cuda_backend);

  return ::testing::finish("test_gpu_sources");
}
