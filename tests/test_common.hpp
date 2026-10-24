/*
 * AV1onGPU - tiny test harness
 *
 * No external dependency on purpose: the tests must build on a bare machine with
 * nothing but a compiler, because that is the environment the GPU backends are
 * meant to work in.
 */
#ifndef AV1GPU_TEST_COMMON_HPP
#define AV1GPU_TEST_COMMON_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace testing {

inline int& failures() {
  static int count = 0;
  return count;
}

inline int& checks() {
  static int count = 0;
  return count;
}

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& extra) {
  ++checks();
  if (ok) return;
  ++failures();
  std::printf("  FAIL %s:%d  %s%s%s\n", file, line, expr, extra.empty() ? "" : "  ",
              extra.c_str());
}

template <typename A, typename B>
void check_eq(const A& a, const B& b, const char* expr_a, const char* expr_b,
              const char* file, int line) {
  const bool ok = a == b;
  std::string extra;
  if (!ok) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "(%s = %lld, %s = %lld)", expr_a,
                  static_cast<long long>(a), expr_b, static_cast<long long>(b));
    extra = buf;
  }
  report(ok, "equality", file, line, extra);
}

inline int finish(const char* name) {
  std::printf("%s: %d checks, %d failure(s)\n", name, checks(), failures());
  return failures() == 0 ? 0 : 1;
}

} /* namespace testing */

#define CHECK(cond) ::testing::report((cond), #cond, __FILE__, __LINE__, "")
#define CHECK_EQ(a, b) ::testing::check_eq((a), (b), #a, #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol)                                                       \
  ::testing::report(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <=    \
                        static_cast<double>(tol),                                   \
                    #a " ~= " #b, __FILE__, __LINE__, "")

#endif /* AV1GPU_TEST_COMMON_HPP */
