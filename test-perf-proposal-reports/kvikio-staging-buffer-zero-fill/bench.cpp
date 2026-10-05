// Microbenchmark: cost of value-initializing (zero-filling) a staging buffer per open,
// versus allocating it uninitialized. Single-threaded, CPU only.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <sys/resource.h>

static inline void escape(void* p) { asm volatile("" : : "g"(p) : "memory"); }

static long minflt()
{
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_minflt;
}

struct result {
  double median_ms;
  double min_ms;
  double faults_per_iter;
};

// payload: bytes copied into the staging buffer after it is allocated (what the stream stages).
template <typename F>
result run(F&& alloc_and_use, int iters)
{
  std::vector<double> t;
  t.reserve(iters);
  long f0 = minflt();
  for (int i = 0; i < iters; ++i) {
    auto a = std::chrono::steady_clock::now();
    alloc_and_use();
    auto b = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
  }
  long f1 = minflt();
  std::sort(t.begin(), t.end());
  return {t[t.size() / 2], t.front(), double(f1 - f0) / iters};
}

int main(int argc, char** argv)
{
  const int iters = argc > 1 ? std::atoi(argv[1]) : 40;
  std::vector<char> src(size_t{32} << 20, 'x');  // source of staged bytes

  for (size_t cap : {size_t{32} << 20, size_t{1} << 20}) {
    for (size_t payload : {size_t{4096}, size_t{1} << 20, cap}) {
      if (payload > cap) continue;
      auto vec = [&] {
        std::vector<char> buf(cap);  // value-initialized (current code)
        std::memcpy(buf.data(), src.data(), payload);
        escape(buf.data());
      };
      auto uninit = [&] {
        auto buf = std::make_unique_for_overwrite<char[]>(cap);  // proposed
        std::memcpy(buf.get(), src.data(), payload);
        escape(buf.get());
      };
      // warm-up
      vec();
      uninit();
      auto rv = run(vec, iters);
      auto ru = run(uninit, iters);
      std::printf(
        "cap=%6zu KiB staged=%6zu KiB | vector<char>(n): median %7.3f ms (min %7.3f) %7.0f faults | "
        "make_unique_for_overwrite: median %7.3f ms (min %7.3f) %7.0f faults\n",
        cap >> 10, payload >> 10, rv.median_ms, rv.min_ms, rv.faults_per_iter, ru.median_ms,
        ru.min_ms, ru.faults_per_iter);
    }
  }
  return 0;
}
