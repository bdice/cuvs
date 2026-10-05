// Single-threaded per-row cost of the GnndGraph host loops (replicas of nn_descent.cuh code).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
using DistData_t = float;
struct ID { int id_{std::numeric_limits<int>::max()};
  bool is_new() const { return id_ >= 0; } int& id_with_flag() { return id_; }
  int id() const { return is_new() ? id_ : -id_ - 1; } void mark_old() { if (id_ >= 0) id_ = -id_ - 1; } };
struct Bloom { size_t nrow, sets, nh; std::vector<bool> b; static constexpr int bits = 512;
  Bloom(size_t n, size_t s, size_t h) : nrow(n), sets(s), nh(h), b(n * bits * s) {}
  uint32_t h0(uint32_t v){v*=1103515245;v+=12345;v^=v<<13;v^=v>>17;v^=v<<5;return v;}
  uint32_t h1(uint32_t v){v*=1664525;v+=1013904223;v^=v<<13;v^=v>>17;v^=v<<5;return v;}
  void add(size_t l, int key){uint32_t h=h0(key);size_t g=l*bits*sets+key%sets*bits;b[g+h%bits]=1;for(size_t i=1;i<nh;i++){h=h+h1(key);b[g+h%bits]=1;}}
  bool check(size_t l,int key){bool p=true;uint32_t h=h0(key);size_t g=l*bits*sets+key%sets*bits;p&=b[g+h%bits];if(!p)return false;for(size_t i=1;i<nh;i++){h=h+h1(key);p&=b[g+h%bits];if(!p)return false;}return true;}};
int insert_to_ordered_list(ID* list, DistData_t* dl, int width, ID nid, DistData_t dist) {
  if (dist > dl[width - 1]) return width;
  int idx = width; bool found = false;
  for (int i = 0; i < width; i++) { if (list[i].id() == nid.id()) return width;
    if (!found && dl[i] > dist) { idx = i; found = true; } }
  if (idx == width) return idx;
  memmove(list + idx + 1, list + idx, sizeof(*list) * (width - idx - 1));
  memmove(dl + idx + 1, dl + idx, sizeof(*dl) * (width - idx - 1));
  list[idx] = nid; dl[idx] = dist; return idx; }
int main(int argc, char** argv) {
  size_t nrow = argc > 1 ? atol(argv[1]) : 4000, node_degree = argc > 2 ? atol(argv[2]) : 96, internal = argc > 3 ? atol(argv[3]) : 192;
  const int seg = 32, num_samples = 32, width = 32; int num_segments = node_degree / seg;
  std::mt19937 rng(1); std::uniform_int_distribution<int> rid(0, nrow - 1); std::uniform_real_distribution<float> rd(0, 1);
  std::vector<ID> g(nrow * node_degree); std::vector<float> d(nrow * node_degree);
  std::vector<int> g_old(nrow * num_samples), g_new(nrow * num_samples); std::vector<std::pair<int,int>> ls_old(nrow), ls_new(nrow);
  std::vector<ID> cand(nrow * width); std::vector<float> cd(nrow * width);
  Bloom bloom(nrow, internal / seg, 3);
  auto reset_graph = [&](float lo, float hi) { for (size_t i = 0; i < nrow; i++) for (int s = 0; s < num_segments; s++) {
      std::vector<float> v(seg); for (auto& x : v) x = lo + (hi - lo) * rd(rng); std::sort(v.begin(), v.end());
      for (int j = 0; j < seg; j++) { g[i*node_degree+s*seg+j].id_with_flag() = rid(rng); d[i*node_degree+s*seg+j] = v[j]; } } };
  auto make_cand = [&](float lo, float hi) { for (size_t i = 0; i < nrow; i++) { std::vector<float> v(width);
      for (auto& x : v) x = lo + (hi - lo) * rd(rng); std::sort(v.begin(), v.end());
      for (int j = 0; j < width; j++) { cand[i*width+j].id_with_flag() = rid(rng); cd[i*width+j] = v[j]; } } };
  std::atomic<int64_t> counter{0};
  auto update_graph = [&]() { for (size_t i = 0; i < nrow; i++) for (int j = 0; j < width; j++) {
      auto nid = cand[i*width+j]; auto nd = cd[i*width+j]; if (nd == std::numeric_limits<float>::max()) break;
      if ((size_t)nid.id() == i) continue; int si = nid.id() % num_segments;
      int pos = insert_to_ordered_list(&g[i*node_degree+si*seg], &d[i*node_degree+si*seg], seg, nid, nd);
      if (i % 100 == 0 && pos != seg) counter++; } };
  auto sample_graph = [&]() { std::fill(g_old.begin(), g_old.end(), std::numeric_limits<int>::max());
    for (size_t i = 0; i < nrow; i++) { ls_old[i] = {0,0}; ls_new[i] = {0,0}; auto* list = &g[i*node_degree]; int* lo = &g_old[i*num_samples];
      for (int j = 0; j < seg; j++) { for (int k = 0; k < num_segments; k++) { auto nb = list[k*seg+j]; if ((size_t)nb.id() >= nrow) continue;
          if (!nb.is_new()) { if (ls_old[i].first < num_samples) lo[ls_old[i].first++] = nb.id(); }
          if (ls_old[i].first == num_samples && ls_new[i].first == num_samples) break; }
        if (ls_old[i].first == num_samples && ls_new[i].first == num_samples) break; } } };
  auto sample_graph_new = [&]() { std::fill(g_new.begin(), g_new.end(), std::numeric_limits<int>::max());
    for (size_t i = 0; i < nrow; i++) { int* ln = &g_new[i*num_samples]; ls_new[i] = {0,0};
      for (int j = 0; j < width; j++) { int id = cand[i*width+j].id(); if ((size_t)id >= nrow) break;
        if (bloom.check(i, id)) continue; bloom.add(i, id); cand[i*width+j].mark_old(); ln[ls_new[i].first++] = id;
        if (ls_new[i].first == num_samples) break; } } };
  auto time = [&](const char* name, auto&& setup, auto&& fn) { double best = 1e30;
    for (int r = 0; r < 7; r++) { setup(); auto t0 = std::chrono::steady_clock::now(); fn(); auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count()); }
    printf("%-34s nrow=%zu deg=%zu: %8.1f us total, %.3f us/row\n", name, nrow, node_degree, best, best / nrow); };
  time("update_graph (all inserted)", [&]{ reset_graph(0.5f, 1.0f); make_cand(0.0f, 0.5f); }, update_graph);
  time("update_graph (half inserted)", [&]{ reset_graph(0.0f, 1.0f); make_cand(0.0f, 1.0f); }, update_graph);
  time("update_graph (all rejected)", [&]{ reset_graph(0.0f, 0.5f); make_cand(0.6f, 1.0f); }, update_graph);
  time("sample_graph(false)", [&]{ reset_graph(0.0f, 1.0f); for (auto& x : g) if (rd(rng) < 0.5f) x.mark_old(); }, sample_graph);
  time("sample_graph_new", [&]{ make_cand(0.0f, 1.0f); }, sample_graph_new);
  printf("checksum %lld\n", (long long)counter.load());
}
