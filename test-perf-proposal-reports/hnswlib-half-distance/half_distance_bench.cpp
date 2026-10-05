// CPU-only microbenchmark and exactness check for the half distance kernels in change.patch.
// Not part of the patch. Build against a tree with the patch applied (cpp/src) and the fetched,
// cuVS-patched hnswlib:
//   g++ -march=nocona -mtune=haswell -O3 -std=gnu++20 -Wall -Wno-unused-function
//     -I<patched cuvs>/cpp/src -I<build>/_deps/hnswlib-src
//     -I$CONDA_PREFIX/targets/x86_64-linux/include
//     half_distance_bench.cpp -o half_distance_bench && taskset -c 0 ./half_distance_bench
// It compares all 65536 fp16 bit patterns against __half2float, checks that the generic and F16C
// kernels agree bit for bit, and times hnswlib's L2Sqr<half, float> / InnerProductDistance<half,
// float> against the new kernels.
#include <cuda_fp16.h>
#include <hnswlib/hnswlib.h>
#include "neighbors/detail/hnsw_half_distance.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace cuvs::neighbors::hnsw::detail;

template <typename F>
double bench(F f, const std::vector<half>& data, size_t n, size_t dim, float& sink) {
  auto t0 = std::chrono::steady_clock::now();
  size_t iters = 0; float s = 0;
  do {
    for (size_t i = 0; i + 1 < n; i++) { s += f(data.data() + i * dim, data.data() + (i + 1) * dim, &dim); iters++; }
  } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.2);
  sink += s;
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main() {
  // exhaustive conversion check
  size_t bad = 0;
  for (uint32_t h = 0; h < 65536; h++) {
    __half_raw r; r.x = (unsigned short)h;
    float ref = __half2float(half(r));
    float got = half_bits_to_float((uint16_t)h);
    uint32_t a, b; memcpy(&a, &ref, 4); memcpy(&b, &got, 4);
    if (std::isnan(ref) ? !std::isnan(got) : a != b) { if (bad < 5) printf("mismatch %04x ref %g got %g\n", h, ref, got); bad++; }
  }
  printf("conversion mismatches: %zu\n", bad);
  printf("cpu_supports_f16c: %d\n", (int)cpu_supports_f16c());

  {
    // path equality on many dims and distributions (incl. negatives and subnormals)
    std::mt19937 r2(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    size_t diff = 0, total = 0;
    for (int dist = 0; dist < 3; dist++) {
      for (size_t dim = 1; dim <= 70; dim++) {
        std::vector<half> v(64 * dim);
        for (auto& x : v) {
          float f = dist == 0 ? nd(r2) : dist == 1 ? nd(r2) * 1e-5f : nd(r2) * 100.f;
          x = __float2half(f);
        }
        for (size_t i = 0; i + 1 < 64; i++) {
          for (int ip = 0; ip < 2; ip++) {
            float g = ip ? half_distance_generic<true>(&v[i * dim], &v[(i + 1) * dim], &dim) : half_distance_generic<false>(&v[i * dim], &v[(i + 1) * dim], &dim);
            float f = ip ? half_distance_f16c<true>(&v[i * dim], &v[(i + 1) * dim], &dim) : half_distance_f16c<false>(&v[i * dim], &v[(i + 1) * dim], &dim);
            total++; if (memcmp(&g, &f, 4)) { if (diff < 5) printf("path diff dim %zu ip %d: %a vs %a\n", dim, ip, g, f); diff++; }
          }
        }
      }
    }
    printf("generic vs f16c bitwise differences: %zu / %zu\n", diff, total);
  }
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> u(0.1f, 2.0f);
  for (size_t dim : {5, 64, 128, 250, 1000}) {
    size_t n = 512;
    std::vector<half> data(n * dim);
    for (auto& x : data) x = __float2half(u(rng));
    // compare accuracy vs double reference
    double err_old = 0, err_new = 0, err_ip_old = 0, err_ip_new = 0; size_t mismatch_paths = 0;
    for (size_t i = 0; i + 1 < n; i++) {
      const half* a = data.data() + i * dim; const half* b = a + dim;
      double ref = 0, ref_ip = 0;
      for (size_t j = 0; j < dim; j++) { double d = (double)__half2float(a[j]) - (double)__half2float(b[j]); ref += d * d; ref_ip += (double)__half2float(a[j]) * (double)__half2float(b[j]); }
      ref_ip = 1.0 - ref_ip;
      float o = hnswlib::L2Sqr<half, float>(a, b, &dim);
      float g = half_distance_generic<false>(a, b, &dim);
      float f = half_distance_f16c<false>(a, b, &dim);
      float oi = hnswlib::InnerProductDistance<half, float>(a, b, &dim);
      float gi = half_distance_generic<true>(a, b, &dim);
      float fi = half_distance_f16c<true>(a, b, &dim);
      if (memcmp(&g, &f, 4) || memcmp(&gi, &fi, 4)) mismatch_paths++;
      err_old = std::max(err_old, std::fabs(o - ref) / ref); err_new = std::max(err_new, std::fabs(g - ref) / ref);
      err_ip_old = std::max(err_ip_old, std::fabs(oi - ref_ip) / std::fabs(ref_ip)); err_ip_new = std::max(err_ip_new, std::fabs(gi - ref_ip) / std::fabs(ref_ip));
    }
    float sink = 0;
    double t_old = bench([](const void* a, const void* b, const void* q) { return hnswlib::L2Sqr<half, float>(a, b, q); }, data, n, dim, sink);
    double t_gen = bench(half_distance_generic<false>, data, n, dim, sink);
    double t_f16 = bench(half_distance_f16c<false>, data, n, dim, sink);
    double ti_old = bench([](const void* a, const void* b, const void* q) { return hnswlib::InnerProductDistance<half, float>(a, b, q); }, data, n, dim, sink);
    double ti_f16 = bench(half_distance_f16c<true>, data, n, dim, sink);
    printf("dim %4zu: L2 ns/call old %8.1f generic %7.1f f16c %6.1f | IP old %8.1f f16c %6.1f | max rel err L2 old %.2e new %.2e, IP old %.2e new %.2e | generic!=f16c %zu (sink %g)\n",
           dim, t_old, t_gen, t_f16, ti_old, ti_f16, err_old, err_new, err_ip_old, err_ip_new, mismatch_paths, (double)sink);
  }
}
