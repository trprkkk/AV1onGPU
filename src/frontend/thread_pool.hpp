/*
 * AV1onGPU - a small persistent worker pool
 *
 * The host-side stages of a frame (dequantise, the planner's job building,
 * the NV12 pack) split their loops over a few threads. Spawning that many
 * std::thread objects per frame costs more than the work on a six core part
 * with SMT - thread creation is tens of microseconds each, and twenty of them
 * a frame is a scheduling storm that shows up as playback hitches - so the
 * workers are created once and reused.
 *
 * The pool is a singleton: the demuxer's thread and the pipeline's worker may
 * both want it at the same time, and one queue serves both.
 */
#ifndef AV1GPU_FRONTEND_THREAD_POOL_HPP
#define AV1GPU_FRONTEND_THREAD_POOL_HPP

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace av1gpu {
namespace frontend {

class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }

  /* Runs fn(chunk_index, begin, end) over [0, n) split into contiguous chunks,
   * one per worker, and blocks until every chunk is done. Contiguous chunks
   * keep each chunk's output in place, so a merge that walks the chunks in
   * order sees exactly the sequential order. Ranges too small to be worth it
   * run on the calling thread. fn must not throw and must not call run(). */
  template <typename F>
  void run(size_t n, unsigned max_workers, F&& fn) {
    const unsigned hw = std::thread::hardware_concurrency();
    unsigned k = n >= 256 && hw > 1 ? std::min<unsigned>(max_workers, hw) : 1;
    const unsigned pool_n = static_cast<unsigned>(workers_.size());
    if (pool_n > 0 && k > pool_n) k = pool_n;
    if (k <= 1) {
      fn(0, 0, n);
      return;
    }
    const size_t chunk = (n + k - 1) / k;
    /* Per call: the tasks below read both, and they cannot start before the
     * push loop finishes, because popping needs the mutex it holds. */
    unsigned pushed = k;
    std::atomic<int> done{0};
    {
      std::lock_guard<std::mutex> lk(m_);
      for (unsigned t = 0; t < k; ++t) {
        const size_t b = static_cast<size_t>(t) * chunk;
        const size_t e = std::min(b + chunk, n);
        tasks_.emplace_back([this, &fn, &pushed, &done, t, b, e] {
          if (b < e) fn(t, b, e);
          if (done.fetch_add(1, std::memory_order_acq_rel) + 1 == static_cast<int>(pushed)) {
            std::lock_guard<std::mutex> lk(m_);
            cv_done_.notify_all();
          }
        });
      }
      cv_work_.notify_all();
    }
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return done.load(std::memory_order_acquire) >= static_cast<int>(pushed); });
  }

 private:
  ThreadPool() {
    const unsigned hw = std::thread::hardware_concurrency();
    /* Leave the pipeline threads (demuxer, planner, worker) cores of their
     * own: the pool is shared by all of them, and on a six core part six pool
     * workers plus three pipeline threads oversubscribe the machine and slow
     * the GPU launch path down. AV1GPU_POOL_N overrides the count. */
    unsigned n = hw > 2 ? std::min<unsigned>(hw - 2, 6) : (hw > 1 ? 1u : 0u);
    if (const char* e = std::getenv("AV1GPU_POOL_N")) {
      const long v = std::atol(e);
      if (v >= 0 && v <= 16) n = static_cast<unsigned>(v);
    }
    workers_.reserve(n);
    for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_work_.notify_all();
    for (std::thread& t : workers_) t.join();
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void loop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_work_.wait(lk, [&] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  std::deque<std::function<void()>> tasks_;
  bool stop_ = false;
};

/* Runs fn(chunk_index, begin, end) over [0, n) in contiguous chunks on the
 * shared pool, joining before returning. Small ranges stay on the calling
 * thread. */
template <typename F>
void parallel_chunks(size_t n, unsigned max_workers, F&& fn) {
  ThreadPool::instance().run(n, max_workers,
                             [&fn](size_t t, size_t b, size_t e) { fn(t, b, e); });
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_THREAD_POOL_HPP */
