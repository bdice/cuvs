// Host-only check: list_record_layouts produces the same bytes as both legacy serialize_list paths,
// and the legacy parsers accept the batched bytes. No GPU work is done.
#include <neighbors/ivf_list.cuh>

#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>

#include <cstdio>
#include <random>
#include <sstream>
#include <string>

template <typename ListT>
int check(const raft::resources& res, const typename ListT::spec_type& spec, const char* name)
{
  using value_type = typename ListT::value_type;
  using index_type = typename ListT::index_type;
  using size_type  = typename ListT::size_type;
  cuvs::neighbors::ivf::detail::list_record_layouts<ListT> layouts(res, spec);
  std::mt19937 rng(42);
  int failures = 0;
  std::string all_legacy, all_batched;
  std::vector<size_type> sizes_used;
  for (size_type size : {0u, 1u, 7u, 31u, 32u, 33u, 100u, 1000u, 12345u, 0u, 32u}) {
    sizes_used.push_back(size);
    std::ostringstream legacy_host;  // serialize_list, std::ostream branch
    std::ostringstream legacy_kvikio;  // serialize_list, kvikio branch (headers + write_device)
    raft::serialize_scalar(res, legacy_host, size);
    raft::serialize_scalar(res, legacy_kvikio, size);
    std::string data_bytes, idx_bytes;
    if (size > 0) {
      auto ext  = spec.make_list_extents(size);
      auto data = raft::make_host_mdarray<value_type, size_type, raft::row_major>(ext);
      auto inds = raft::make_host_mdarray<index_type, size_type, raft::row_major>(
        raft::make_extents<size_type>(size));
      auto* d = reinterpret_cast<unsigned char*>(data.data_handle());
      for (size_t i = 0; i < data.size() * sizeof(value_type); i++) { d[i] = rng() & 0xff; }
      for (size_t i = 0; i < inds.size(); i++) { inds.data_handle()[i] = index_type(rng()); }
      raft::serialize_mdspan(res, legacy_host, data.view());
      raft::serialize_mdspan(res, legacy_host, inds.view());
      data_bytes.assign(reinterpret_cast<const char*>(data.data_handle()),
                        data.size() * sizeof(value_type));
      idx_bytes.assign(reinterpret_cast<const char*>(inds.data_handle()),
                       inds.size() * sizeof(index_type));
      std::vector<size_t> shape;
      for (size_t i = 0; i < ext.rank(); i++) { shape.push_back(ext.extent(i)); }
      raft::numpy_serializer::write_header(
        legacy_kvikio, cuvs::util::detail::get_numpy_header<const value_type>(shape, false));
      legacy_kvikio.write(data_bytes.data(), data_bytes.size());
      raft::numpy_serializer::write_header(
        legacy_kvikio,
        cuvs::util::detail::get_numpy_header<const index_type>({size_t(size)}, false));
      legacy_kvikio.write(idx_bytes.data(), idx_bytes.size());
    }
    const auto& l = layouts(size);
    std::string batched =
      l.head + data_bytes + l.mid + idx_bytes;
    bool ok = batched == legacy_host.str() && batched == legacy_kvikio.str() &&
              l.data_bytes == data_bytes.size() && l.indices_bytes == idx_bytes.size() &&
              l.record_bytes() == batched.size() && l.staged() == (l.payload_bytes() <= (4u << 20));
    if (!ok) {
      failures++;
      std::printf("%s size %u: MISMATCH (batched %zu, host %zu, kvikio %zu)\n",
                  name, unsigned(size), batched.size(), legacy_host.str().size(),
                  legacy_kvikio.str().size());
    }
    all_legacy += legacy_host.str();
    all_batched += batched;
  }
  // The legacy host-path parser accepts the concatenated batched records.
  std::istringstream is(all_batched);
  for (auto expected : sizes_used) {
    auto size = raft::deserialize_scalar<size_type>(res, is);
    if (size != expected) { failures++; }
    if (size == 0) { continue; }
    auto data = raft::make_host_mdarray<value_type, size_type, raft::row_major>(
      spec.make_list_extents(size));
    auto inds = raft::make_host_mdarray<index_type, size_type, raft::row_major>(
      raft::make_extents<size_type>(size));
    raft::deserialize_mdspan(res, is, data.view());
    raft::deserialize_mdspan(res, is, inds.view());
  }
  if (is.peek() != std::char_traits<char>::eof()) { failures++; }
  if (all_legacy != all_batched) { failures++; }
  std::printf("%-28s %s (%zu bytes, %zu sizes)\n", name, failures ? "FAIL" : "ok",
              all_batched.size(), sizes_used.size());
  return failures;
}

int main()
{
  raft::resources res;
  int f = 0;
  using namespace cuvs::neighbors;
  f += check<ivf_flat::list_data<float, int64_t>>(res, {16, true}, "ivf_flat<float> dim16");
  f += check<ivf_flat::list_data<half, int64_t>>(res, {3, true}, "ivf_flat<half> dim3");
  f += check<ivf_flat::list_data<int8_t, int64_t>>(res, {2048, false}, "ivf_flat<int8> dim2048");
  f += check<ivf_flat::list_data<uint8_t, int64_t>>(res, {4096, true}, "ivf_flat<uint8> dim4096");
  f += check<ivf_sq::list_data<uint8_t, int64_t>>(res, {13, true}, "ivf_sq<uint8> dim13");
  f += check<ivf_pq::list_data_flat<int64_t>>(res, {8, 64, true}, "ivf_pq flat 8b x64");
  f += check<ivf_pq::list_data_interleaved<int64_t>>(res, {5, 17, true}, "ivf_pq interleaved 5b x17");
  f += check<ivf_pq::list_data_interleaved<int64_t>>(res, {8, 3072, true}, "ivf_pq interleaved 8b x3072");
  std::printf("%s\n", f ? "FAILED" : "ALL OK");
  return f != 0;
}
