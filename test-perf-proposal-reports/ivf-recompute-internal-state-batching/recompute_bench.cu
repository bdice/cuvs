// Times ivf_flat::helpers::recompute_internal_state and a one-row ivf_flat::extend (which calls it)
// on IVF-Flat indexes with many lists.
#include <cuvs/neighbors/ivf_flat.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

template <typename F>
static std::pair<double, double> time_ms(const raft::resources& res, int reps, F&& f)
{
  std::vector<double> t;
  for (int i = 0; i < reps; i++) {
    raft::resource::sync_stream(res);
    auto t0 = std::chrono::steady_clock::now();
    f();
    raft::resource::sync_stream(res);
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return {t[t.size() / 2], t.front()};
}

int main()
{
  raft::resources res;
  raft::random::RngState rng(42);
  constexpr int64_t dim = 8;
  std::printf("n_lists\tnonempty_lists\trecompute_median_ms\trecompute_min_ms\textend1_median_ms\textend1_min_ms\n");
  for (uint32_t n_lists : {1024u, 4096u, 16384u, 65536u}) {
    int64_t n_rows = int64_t(n_lists) * 16;
    auto data      = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
    raft::random::uniform(res, rng, data.data_handle(), n_rows * dim, -1.0f, 1.0f);
    cuvs::neighbors::ivf_flat::index_params params;
    params.n_lists                  = n_lists;
    params.kmeans_n_iters           = 2;
    params.kmeans_trainset_fraction = 1.0;
    params.add_data_on_build        = true;
    auto index = cuvs::neighbors::ivf_flat::build(res, params, raft::make_const_mdspan(data.view()));
    raft::resource::sync_stream(res);
    uint32_t nonempty = 0;
    for (auto& l : index.lists()) { nonempty += l != nullptr; }
    auto recompute = [&]() { cuvs::neighbors::ivf_flat::helpers::recompute_internal_state(res, &index); };
    time_ms(res, 3, recompute);  // warm-up
    auto [rc_med, rc_min] = time_ms(res, 21, recompute);
    auto one_row = raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), 1, dim);
    auto one_idx = raft::make_device_vector<int64_t, int64_t>(res, 1);
    int64_t next_id = n_rows;
    raft::copy(one_idx.data_handle(), &next_id, 1, raft::resource::get_cuda_stream(res));
    auto extend = [&]() {
      cuvs::neighbors::ivf_flat::extend(
        res, one_row, std::make_optional(raft::make_const_mdspan(one_idx.view())), &index);
    };
    time_ms(res, 2, extend);  // warm-up
    auto [ex_med, ex_min] = time_ms(res, 11, extend);
    std::printf("%u\t%u\t%.3f\t%.3f\t%.3f\t%.3f\n", n_lists, nonempty, rc_med, rc_min, ex_med, ex_min);
    std::fflush(stdout);
  }
  return 0;
}
