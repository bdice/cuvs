// Byte-identity check for IVF list (de)serialization across two libcuvs builds.
//   ivf_io_check write <flat|pq|sq> <case> <out>   build an index and serialize it
//   ivf_io_check roundtrip <flat|pq|sq> <in> <out> deserialize and serialize again
// case: 0 = 5000 rows, dim 33, 64 lists; 1 = train on 5000 rows, extend with 50 rows (empty lists);
//       2 = 40000 rows, dim 128, 2 lists (lists above 4 MiB).
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace cuvs::neighbors;

int main(int argc, char** argv)
{
  if (argc != 5) { std::fprintf(stderr, "usage\n"); return 2; }
  std::string mode = argv[1], algo = argv[2], arg = argv[3], out = argv[4];
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  if (mode == "write") {
    int c        = std::stoi(arg);
    int64_t n    = c == 2 ? 40000 : 5000;
    int64_t dim  = c == 2 ? 128 : 33;
    int64_t next = c == 1 ? 50 : 0;
    uint32_t nl  = c == 2 ? 2 : 64;
    std::mt19937 gen(42);
    std::normal_distribution<float> d;
    std::vector<float> h(n * dim);
    for (auto& v : h) v = d(gen);
    auto data = raft::make_device_matrix<float, int64_t>(res, n, dim);
    raft::copy(data.data_handle(), h.data(), h.size(), stream);
    auto dv  = raft::make_const_mdspan(data.view());
    auto sub = raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), next, dim);
    if (algo == "flat") {
      ivf_flat::index_params p;
      p.n_lists            = nl;
      p.add_data_on_build  = next == 0;
      auto idx             = ivf_flat::build(res, p, dv);
      if (next) { idx = ivf_flat::extend(res, sub, std::nullopt, idx); }
      ivf_flat::serialize(res, out, idx);
    } else if (algo == "pq") {
      ivf_pq::index_params p;
      p.n_lists           = nl;
      p.pq_dim            = c == 2 ? 32 : 11;
      p.add_data_on_build = next == 0;
      auto idx            = ivf_pq::build(res, p, dv);
      if (next) { idx = ivf_pq::extend(res, sub, std::nullopt, idx); }
      ivf_pq::serialize(res, out, idx);
    } else {
      ivf_sq::index_params p;
      p.n_lists           = nl;
      p.add_data_on_build = next == 0;
      auto idx            = ivf_sq::build(res, p, dv);
      if (next) { ivf_sq::extend(res, sub, std::nullopt, &idx); }
      ivf_sq::serialize(res, out, idx);
    }
  } else {
    if (algo == "flat") {
      ivf_flat::index<float, int64_t> idx(res);
      ivf_flat::deserialize(res, arg, &idx);
      ivf_flat::serialize(res, out, idx);
    } else if (algo == "pq") {
      ivf_pq::index<int64_t> idx(res);
      ivf_pq::deserialize(res, arg, &idx);
      std::fprintf(stderr, "deserialized: n_lists=%u size=%lld\n", idx.n_lists(), (long long)idx.size());
      if (mode == "read") { return 0; }
      ivf_pq::serialize(res, out, idx);
    } else {
      ivf_sq::index<uint8_t> idx(res);
      ivf_sq::deserialize(res, arg, &idx);
      ivf_sq::serialize(res, out, idx);
    }
  }
  raft::resource::sync_stream(res);
  return 0;
}
