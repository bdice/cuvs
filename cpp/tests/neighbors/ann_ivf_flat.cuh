/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"

#include <cuda/stream>
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/linalg/normalize.cuh>
#include <raft/stats/mean.cuh>

#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/add.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/util/fast_int_div.cuh>
#include <raft/util/pow2_utils.cuh>
#include <rmm/cuda_stream_pool.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

namespace cuvs::neighbors::ivf_flat {

struct test_ivf_sample_filter {
  static constexpr unsigned offset = 300;
};

template <typename Pred>
RAFT_KERNEL count_if_kernel(uint32_t n, Pred pred, uint32_t* count)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // All threads of the block must take part, including those past the end.
  auto n_true = __syncthreads_count(i < n && pred(i));
  if (threadIdx.x == 0 && n_true > 0) { atomicAdd(count, static_cast<uint32_t>(n_true)); }
}

/**
 * Adds to `*count` (in device memory) the number of `i < n` for which `pred(i)` holds. It does not
 * synchronize the stream, so many counts can be read back with a single copy.
 */
template <typename Pred>
void count_if_async(const raft::resources& res, uint32_t n, Pred pred, uint32_t* count)
{
  if (n == 0) { return; }
  constexpr uint32_t kBlockSize = 256;
  count_if_kernel<<<raft::ceildiv(n, kBlockSize),
                    kBlockSize,
                    0,
                    raft::resource::get_cuda_stream(res).get()>>>(n, pred, count);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Whether the element `i` of the data of an interleaved list of `list_size` rows of `dim`
 * components holds a component of a row, rather than padding.
 */
struct interleaved_list_mask {
  uint32_t dim;
  uint32_t list_size;
  raft::util::FastIntDiv<int32_t> chunk_size;  // veclen

  __device__ auto operator()(uint32_t i) const -> bool
  {
    using interleaved_group   = raft::Pow2<kIndexGroupSize>;
    uint32_t max_group_offset = interleaved_group::roundDown(list_size);
    if (i < max_group_offset * dim) { return true; }
    uint32_t surplus    = (i - max_group_offset * dim);
    uint32_t ingroup_id = interleaved_group::mod(static_cast<int32_t>(surplus) / chunk_size);
    return ingroup_id < (list_size - max_group_offset);
  }
};

template <typename IdxT>
struct AnnIvfFlatInputs {
  IdxT num_queries;
  IdxT num_db_vecs;
  IdxT dim;
  IdxT k;
  IdxT nprobe;
  IdxT nlist;
  cuvs::distance::DistanceType metric;
  bool adaptive_centers;
  bool host_dataset = false;
  // The kernel_copy_overlapping option is only applicable when host dataset is enabled.
  bool kernel_copy_overlapping = false;
  // By default, testPacker and testFilter reuse the index trained by testIVFFlat. If set, they
  // build their own indexes instead (kmeans_trainset_fraction = 1 and add_data_on_build = true,
  // respectively), and the case does not use the index cache of the fixture.
  bool independent_builds = false;
};

template <typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const AnnIvfFlatInputs<IdxT>& p)
{
  os << "{ " << p.num_queries << ", " << p.num_db_vecs << ", " << p.dim << ", " << p.k << ", "
     << p.nprobe << ", " << p.nlist << ", "
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << ", " << p.adaptive_centers << "," << p.host_dataset << "," << p.kernel_copy_overlapping
     << "," << p.independent_builds << '}';
  return os;
}

template <typename T, typename DataT, typename IdxT>
class AnnIVFFlatTest : public ::testing::TestWithParam<AnnIvfFlatInputs<IdxT>> {
 public:
  AnnIVFFlatTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnIvfFlatInputs<IdxT>>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void testIVFFlat()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivfflat(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<T> distances_ivfflat(queries_size);
    std::vector<T> distances_naive(queries_size);

    if (ps.kernel_copy_overlapping) {
      size_t n_streams = 1;
      raft::resource::set_cuda_stream_pool(handle_,
                                           std::make_shared<rmm::cuda_stream_pool>(n_streams));
    }

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      cuvs::neighbors::naive_knn<T, DataT, IdxT>(handle_,
                                                 distances_naive_dev.data(),
                                                 indices_naive_dev.data(),
                                                 search_queries.data(),
                                                 database.data(),
                                                 ps.num_queries,
                                                 ps.num_db_vecs,
                                                 ps.dim,
                                                 ps.k,
                                                 ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      // unless something is really wrong with clustering, this could serve as a lower bound on
      // recall
      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);

      rmm::device_uvector<T> distances_ivfflat_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_ivfflat_dev(queries_size, stream_);

      {
        cuvs::neighbors::ivf_flat::search_params search_params;
        search_params.n_probes = ps.nprobe;

        // A case with the build_key of a cached case searches the cached indexes; these passed all
        // the checks that do not depend on the search parameters.
        if (!cached_) { ASSERT_NO_FATAL_FAILURE(buildIndexes()); }
        const auto& index_loaded = cached_ ? cached_->loaded : *loaded_;

        auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
          search_queries.data(), ps.num_queries, ps.dim);
        auto indices_out_view = raft::make_device_matrix_view<IdxT, IdxT>(
          indices_ivfflat_dev.data(), ps.num_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<T, IdxT>(
          distances_ivfflat_dev.data(), ps.num_queries, ps.k);
        cuvs::neighbors::ivf_flat::search(handle_,
                                          search_params,
                                          index_loaded,
                                          search_queries_view,
                                          indices_out_view,
                                          dists_out_view);

        raft::update_host(
          distances_ivfflat.data(), distances_ivfflat_dev.data(), queries_size, stream_);
        raft::update_host(
          indices_ivfflat.data(), indices_ivfflat_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);

        // Keep `loaded_` for the cache only if it can be cached (see TearDown()).
        if (loaded_ && !cacheable(device_bytes(*loaded_))) { loaded_.reset(); }
      }
      float eps = std::is_same_v<DataT, half> ? 0.005 : 0.001;
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivfflat,
                                  distances_naive,
                                  distances_ivfflat,
                                  ps.num_queries,
                                  ps.k,
                                  eps,
                                  min_recall));
    }
  }

  /**
   * Trains the index of the case (`trained_`), adds the database to a copy of it with two calls to
   * `extend`, round-trips that through a file (`loaded_`) and checks the result. Skipped if the
   * case reuses the indexes of an earlier case with the same build_key.
   */
  void buildIndexes()
  {
    cuvs::neighbors::ivf_flat::index_params index_params;
    index_params.n_lists          = ps.nlist;
    index_params.metric           = ps.metric;
    index_params.adaptive_centers = ps.adaptive_centers;

    index_params.add_data_on_build        = false;
    index_params.kmeans_trainset_fraction = 0.5;
    index_params.metric_arg               = 0;

    auto& idx = trained_.emplace(handle_, index_params, ps.dim);
    cuvs::neighbors::ivf_flat::index<DataT, IdxT> index_2(handle_, index_params, ps.dim);

    if (!ps.host_dataset) {
      auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
        (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
      idx                 = cuvs::neighbors::ivf_flat::build(handle_, index_params, database_view);
      auto vector_indices = raft::make_device_vector<IdxT, IdxT>(handle_, ps.num_db_vecs);
      raft::linalg::map_offset(handle_, vector_indices.view(), raft::identity_op{});
      raft::resource::sync_stream(handle_);

      IdxT half_of_data = ps.num_db_vecs / 2;

      auto half_of_data_view = raft::make_device_matrix_view<const DataT, IdxT>(
        (const DataT*)database.data(), half_of_data, ps.dim);

      const std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
      index_2 = cuvs::neighbors::ivf_flat::extend(handle_, half_of_data_view, no_opt, idx);

      auto new_half_of_data_view = raft::make_device_matrix_view<const DataT, IdxT>(
        database.data() + half_of_data * ps.dim, IdxT(ps.num_db_vecs) - half_of_data, ps.dim);

      auto new_half_of_data_indices_view = raft::make_device_vector_view<const IdxT, IdxT>(
        vector_indices.data_handle() + half_of_data, IdxT(ps.num_db_vecs) - half_of_data);

      cuvs::neighbors::ivf_flat::extend(
        handle_,
        new_half_of_data_view,
        std::make_optional<raft::device_vector_view<const IdxT, IdxT>>(
          new_half_of_data_indices_view),
        &index_2);
    } else {
      auto host_database = raft::make_host_matrix<DataT, IdxT>(ps.num_db_vecs, ps.dim);
      raft::copy(host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
      idx = ivf_flat::build(handle_, index_params, raft::make_const_mdspan(host_database.view()));

      auto vector_indices = raft::make_host_vector<IdxT>(handle_, ps.num_db_vecs);
      std::iota(vector_indices.data_handle(), vector_indices.data_handle() + ps.num_db_vecs, 0);

      IdxT half_of_data = ps.num_db_vecs / 2;

      auto half_of_data_view = raft::make_host_matrix_view<const DataT, IdxT>(
        (const DataT*)host_database.data_handle(), half_of_data, ps.dim);

      const std::optional<raft::host_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
      index_2 = ivf_flat::extend(handle_, half_of_data_view, no_opt, idx);

      auto new_half_of_data_view = raft::make_host_matrix_view<const DataT, IdxT>(
        host_database.data_handle() + half_of_data * ps.dim,
        IdxT(ps.num_db_vecs) - half_of_data,
        ps.dim);
      auto new_half_of_data_indices_view = raft::make_host_vector_view<const IdxT, IdxT>(
        vector_indices.data_handle() + half_of_data, IdxT(ps.num_db_vecs) - half_of_data);
      ivf_flat::extend(
        handle_,
        new_half_of_data_view,
        std::make_optional<raft::host_vector_view<const IdxT, IdxT>>(new_half_of_data_indices_view),
        &index_2);
    }

    tmp_index_file index_file;
    cuvs::neighbors::ivf_flat::serialize(handle_, index_file.filename, index_2);
    auto& index_loaded = loaded_.emplace(handle_);
    cuvs::neighbors::ivf_flat::deserialize(handle_, index_file.filename, &index_loaded);
    ASSERT_EQ(index_2.size(), index_loaded.size());

    // Test the centroid invariants
    if (index_2.adaptive_centers()) {
      // The centers must be up-to-date with the corresponding data. The mean of each list is
      // computed on the device, and all of them are compared with the centers at once.
      std::vector<uint32_t> list_sizes(index_2.n_lists());
      std::vector<IdxT*> list_indices(index_2.n_lists());
      raft::copy(list_sizes.data(), index_2.list_sizes().data_handle(), index_2.n_lists(), stream_);
      raft::copy(
        list_indices.data(), index_2.inds_ptrs().data_handle(), index_2.n_lists(), stream_);
      raft::resource::sync_stream(handle_);
      uint32_t max_list_size = *std::max_element(list_sizes.begin(), list_sizes.end());
      rmm::device_uvector<float> cluster_data(size_t{max_list_size} * ps.dim, stream_);
      rmm::device_uvector<float> centroids(size_t{index_2.n_lists()} * ps.dim, stream_);
      for (uint32_t l = 0; l < index_2.n_lists(); l++) {
        if (list_sizes[l] == 0) continue;
        cuvs::spatial::knn::detail::utils::copy_selected<float>((IdxT)list_sizes[l],
                                                                (IdxT)ps.dim,
                                                                database.data(),
                                                                list_indices[l],
                                                                (IdxT)ps.dim,
                                                                cluster_data.data(),
                                                                (IdxT)ps.dim,
                                                                stream_);
        raft::stats::mean<true, float, uint32_t>(centroids.data() + ps.dim * l,
                                                 cluster_data.data(),
                                                 ps.dim,
                                                 list_sizes[l],
                                                 false,
                                                 stream_.get());
      }
      std::vector<float> centers_host(centroids.size());
      std::vector<float> centroids_host(centroids.size());
      raft::copy(
        centers_host.data(), index_2.centers().data_handle(), centers_host.size(), stream_);
      raft::copy(centroids_host.data(), centroids.data(), centroids_host.size(), stream_);
      raft::resource::sync_stream(handle_);
      cuvs::CompareApprox<float> eq_compare(0.001);
      for (uint32_t l = 0; l < index_2.n_lists(); l++) {
        if (list_sizes[l] == 0) continue;
        for (IdxT j = 0; j < ps.dim; j++) {
          float expected = centers_host[ps.dim * l + j];
          float actual   = centroids_host[ps.dim * l + j];
          ASSERT_TRUE(eq_compare(expected, actual))
            << "actual=" << actual << " != expected=" << expected << " @" << j << " (list " << l
            << ")";
        }
      }
    } else {
      // The centers must be immutable
      ASSERT_TRUE(cuvs::devArrMatch(index_2.centers().data_handle(),
                                    idx.centers().data_handle(),
                                    index_2.centers().size(),
                                    cuvs::Compare<float>(),
                                    stream_.get()));
    }
  }

  void testPacker()
  {
    // The packer checks depend only on the build_key; they passed for the cached indexes.
    if (cached_) { return; }

    // The reference: all database rows added to an empty, trained index with `extend`.
    std::optional<index<DataT, IdxT>> own_extend_index;
    if (ps.independent_builds) {
      ivf_flat::index_params index_params;
      index_params.n_lists          = ps.nlist;
      index_params.metric           = ps.metric;
      index_params.adaptive_centers = false;

      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;

      auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
        (const DataT*)database.data(), ps.num_db_vecs, ps.dim);

      auto trained_index = ivf_flat::build(handle_, index_params, database_view);

      const std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
      own_extend_index.emplace(ivf_flat::extend(handle_, database_view, no_opt, trained_index));
    }
    const auto& extend_index = own_extend_index ? *own_extend_index : full_index();

    auto list_sizes = raft::make_host_vector<uint32_t>(extend_index.n_lists());
    raft::update_host(list_sizes.data_handle(),
                      extend_index.list_sizes().data_handle(),
                      extend_index.n_lists(),
                      stream_);
    raft::resource::sync_stream(handle_);

    // An empty index of the same shape to pack the flat codes into.
    index<DataT, IdxT> idx(
      handle_, extend_index.metric(), extend_index.n_lists(), false, false, extend_index.dim());
    ivf_flat::helpers::reset_index(handle_, &idx);

    auto& lists = idx.lists();

    // conservative memory allocation for codepacking
    auto list_device_spec = list_spec<uint32_t, DataT, IdxT>{idx.dim(), false};

    for (uint32_t label = 0; label < idx.n_lists(); label++) {
      uint32_t list_size = list_sizes.data_handle()[label];

      ivf::resize_list(handle_, lists[label], list_device_spec, list_size, 0);
    }

    ivf_flat::helpers::recompute_internal_state(handle_, &idx);

    using interleaved_group = raft::Pow2<kIndexGroupSize>;

    // The checks of all lists are counted on the device and read back at once after the loop, so
    // that the loop does not synchronize the stream. Per list: the number of elements of the
    // interleaved data that hold a component of a row (`kMasked`), and the numbers of mismatches of
    // the packed data (`kPackMismatches`) and of the unpacked rows (`kUnpackMismatches`).
    enum : uint32_t { kMasked = 0, kPackMismatches, kUnpackMismatches, kNumCounts };
    auto counts = raft::make_device_matrix<uint32_t, uint32_t>(handle_, idx.n_lists(), kNumCounts);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(counts.data_handle(), 0, counts.size() * sizeof(uint32_t), stream_.get()));

    uint32_t max_list_size =
      *std::max_element(list_sizes.data_handle(), list_sizes.data_handle() + list_sizes.extent(0));
    auto flat_codes = raft::make_device_matrix<DataT, uint32_t>(handle_, max_list_size, idx.dim());
    auto unpacked_flat_codes =
      raft::make_device_matrix<DataT, uint32_t>(handle_, max_list_size, idx.dim());

    for (uint32_t label = 0; label < idx.n_lists(); label++) {
      uint32_t list_size = list_sizes.data_handle()[label];

      if (list_size > 0) {
        uint32_t padded_list_size = interleaved_group::roundUp(list_size);
        uint32_t n_elems          = padded_list_size * idx.dim();
        auto& list_data           = lists[label]->data;
        auto& list_inds           = extend_index.lists()[label]->indices;
        auto* list_counts         = counts.data_handle() + label * kNumCounts;

        // fetch the flat codes
        auto flat_codes_view = raft::make_device_matrix_view<DataT, uint32_t>(
          flat_codes.data_handle(), list_size, idx.dim());

        raft::matrix::gather(
          handle_,
          raft::make_device_matrix_view<const DataT, uint32_t>(
            (const DataT*)database.data(), static_cast<uint32_t>(ps.num_db_vecs), idx.dim()),
          raft::make_device_vector_view<const IdxT, uint32_t>((const IdxT*)list_inds.data_handle(),
                                                              list_size),
          flat_codes_view);

        helpers::codepacker::pack(
          handle_, make_const_mdspan(flat_codes_view), idx.veclen(), 0, list_data.view());

        interleaved_list_mask mask{
          idx.dim(),
          list_size,
          raft::util::FastIntDiv<int32_t>(static_cast<int32_t>(idx.veclen()))};

        // ensure that the correct number of indices are masked out
        count_if_async(handle_, n_elems, mask, list_counts + kMasked);

        // The packed data must match the data of the list built by `extend` where the mask is set.
        // (The padding is not compared.)
        auto& extend_data = extend_index.lists()[label]->data;
        count_if_async(
          handle_,
          n_elems,
          [mask,
           list_data   = list_data.data_handle(),
           extend_data = extend_data.data_handle()] __device__(uint32_t i) {
            return mask(i) && !(list_data[i] == extend_data[i]);
          },
          list_counts + kPackMismatches);

        auto unpacked_flat_codes_view = raft::make_device_matrix_view<DataT, uint32_t>(
          unpacked_flat_codes.data_handle(), list_size, idx.dim());

        helpers::codepacker::unpack(
          handle_, list_data.view(), idx.veclen(), 0, unpacked_flat_codes_view);

        // The unpacked rows must match the flat codes that were packed.
        count_if_async(
          handle_,
          list_size * idx.dim(),
          [flat_codes          = flat_codes_view.data_handle(),
           unpacked_flat_codes = unpacked_flat_codes_view.data_handle()] __device__(uint32_t i) {
            return !(flat_codes[i] == unpacked_flat_codes[i]);
          },
          list_counts + kUnpackMismatches);
      }
    }

    auto counts_host = raft::make_host_matrix<uint32_t, uint32_t>(idx.n_lists(), kNumCounts);
    raft::copy(counts_host.data_handle(), counts.data_handle(), counts.size(), stream_);
    raft::resource::sync_stream(handle_);

    for (uint32_t label = 0; label < idx.n_lists(); label++) {
      uint32_t list_size = list_sizes.data_handle()[label];

      if (list_size > 0) {
        ASSERT_EQ(uint64_t{counts_host(label, kMasked)}, uint64_t{list_size} * uint64_t(ps.dim))
          << "wrong number of masked elements in list " << label;
        ASSERT_EQ(counts_host(label, kPackMismatches), 0u)
          << "packed data of list " << label << " differs from the data added with extend";
        ASSERT_EQ(counts_host(label, kUnpackMismatches), 0u)
          << "unpacked rows of list " << label << " differ from the rows that were packed";
      }
    }
  }

  void testFilter()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivfflat(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<T> distances_ivfflat(queries_size);
    std::vector<T> distances_naive(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      auto* database_filtered_ptr = database.data() + test_ivf_sample_filter::offset * ps.dim;
      cuvs::neighbors::naive_knn<T, DataT, IdxT>(handle_,
                                                 distances_naive_dev.data(),
                                                 indices_naive_dev.data(),
                                                 search_queries.data(),
                                                 database_filtered_ptr,
                                                 ps.num_queries,
                                                 ps.num_db_vecs - test_ivf_sample_filter::offset,
                                                 ps.dim,
                                                 ps.k,
                                                 ps.metric);
      raft::linalg::addScalar(indices_naive_dev.data(),
                              indices_naive_dev.data(),
                              IdxT(test_ivf_sample_filter::offset),
                              queries_size,
                              stream_.get());
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      // unless something is really wrong with clustering, this could serve as a lower bound on
      // recall
      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);

      auto distances_ivfflat_dev = raft::make_device_matrix<T, IdxT>(handle_, ps.num_queries, ps.k);
      auto indices_ivfflat_dev =
        raft::make_device_matrix<IdxT, IdxT>(handle_, ps.num_queries, ps.k);

      {
        ivf_flat::search_params search_params;
        search_params.n_probes = ps.nprobe;

        // Create IVF Flat index. By default, reuse the index trained by testIVFFlat, extended with
        // the whole database: that is what `build` with `add_data_on_build = true` does.
        std::optional<ivf_flat::index<DataT, IdxT>> own_index;
        if (ps.independent_builds) {
          ivf_flat::index_params index_params;
          index_params.n_lists          = ps.nlist;
          index_params.metric           = ps.metric;
          index_params.adaptive_centers = ps.adaptive_centers;

          index_params.add_data_on_build        = true;
          index_params.kmeans_trainset_fraction = 0.5;
          index_params.metric_arg               = 0;

          auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
            (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
          own_index.emplace(ivf_flat::build(handle_, index_params, database_view));
        }
        const auto& index = own_index ? *own_index : full_index();

        // Create Bitset filter
        auto removed_indices =
          raft::make_device_vector<IdxT, int64_t>(handle_, test_ivf_sample_filter::offset);
        raft::linalg::map_offset(handle_, removed_indices.view(), raft::identity_op{});
        raft::resource::sync_stream(handle_);

        cuvs::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
          handle_, removed_indices.view(), ps.num_db_vecs);
        auto bitset_filter_obj =
          cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());

        // Search with the filter
        auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
          search_queries.data(), ps.num_queries, ps.dim);
        ivf_flat::search(handle_,
                         search_params,
                         index,
                         search_queries_view,
                         indices_ivfflat_dev.view(),
                         distances_ivfflat_dev.view(),
                         bitset_filter_obj);

        raft::update_host(
          distances_ivfflat.data(), distances_ivfflat_dev.data_handle(), queries_size, stream_);
        raft::update_host(
          indices_ivfflat.data(), indices_ivfflat_dev.data_handle(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }
      float eps = std::is_same_v<DataT, half> ? 0.005 : 0.001;
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivfflat,
                                  distances_naive,
                                  distances_ivfflat,
                                  ps.num_queries,
                                  ps.k,
                                  eps,
                                  min_recall));
    }
  }

  void SetUp() override
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);

    if (!ps.independent_builds) {
      auto key = make_build_key(ps);
      auto it  = std::find_if(index_cache_.begin(), index_cache_.end(), [&key](const auto& entry) {
        return entry->key == key;
      });
      if (it != index_cache_.end()) {
        cached_ = *it;
        // Mark the entry as the most recently used one.
        std::rotate(index_cache_.begin(), it, std::next(it));
      }
    }
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    // Only indexes that passed all checks are cached.
    if (loaded_.has_value() && full_.has_value() && !::testing::Test::HasFailure()) {
      cacheIndexes();
    }
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

  static void TearDownTestSuite() { index_cache_.clear(); }

 private:
  /**
   * Everything that determines the database and the indexes of a case. SetUp() draws the database
   * first from a fixed seed, so it depends only on (num_db_vecs, dim) and DataT; each fixture
   * instantiation has its own cache. The other parameters (num_queries, k, nprobe) only affect the
   * queries and the searches.
   */
  struct build_key {
    IdxT num_db_vecs;
    IdxT dim;
    IdxT nlist;
    cuvs::distance::DistanceType metric;
    bool adaptive_centers;
    bool host_dataset;
    bool kernel_copy_overlapping;

    bool operator==(const build_key&) const = default;
  };

  static auto make_build_key(const AnnIvfFlatInputs<IdxT>& p) -> build_key
  {
    return build_key{p.num_db_vecs,
                     p.dim,
                     p.nlist,
                     p.metric,
                     p.adaptive_centers,
                     p.host_dataset,
                     p.kernel_copy_overlapping};
  }

  /** The indexes of a case that passed all checks. They are not modified after construction. */
  struct built_indexes {
    build_key key;
    /** See `loaded_`. */
    index<DataT, IdxT> loaded;
    /** See `full_index()`. */
    index<DataT, IdxT> full;
    size_t bytes;
  };

  /**
   * Least-recently-used cache of the indexes of earlier cases, so that cases that differ only in
   * the search parameters do not build and check the same indexes again. It holds at most
   * kIndexCacheBytes of device memory and is cleared in TearDownTestSuite().
   */
  static constexpr size_t kIndexCacheBytes = size_t{256} << 20;
  inline static std::vector<std::shared_ptr<const built_indexes>> index_cache_;

  static auto device_bytes(const index<DataT, IdxT>& idx) -> size_t
  {
    size_t bytes = idx.centers().size() * sizeof(float);
    for (const auto& list : idx.lists()) {
      if (list) { bytes += list->data_byte_size() + list->indices_capacity() * sizeof(IdxT); }
    }
    return bytes;
  }

  /** Whether a cache entry of two indexes of `index_bytes` each would fit into the cache. */
  auto cacheable(size_t index_bytes) const -> bool
  {
    return !ps.independent_builds && 2 * index_bytes <= kIndexCacheBytes;
  }

  void cacheIndexes()
  {
    size_t bytes = device_bytes(*loaded_) + device_bytes(*full_);
    if (bytes > kIndexCacheBytes) { return; }
    index_cache_.insert(index_cache_.begin(),
                        std::make_shared<const built_indexes>(built_indexes{
                          make_build_key(ps), std::move(*loaded_), std::move(*full_), bytes}));
    loaded_.reset();
    full_.reset();
    // Evict the least recently used entries that do not fit anymore.
    size_t total = 0;
    auto it      = index_cache_.begin();
    for (; it != index_cache_.end() && total + (*it)->bytes <= kIndexCacheBytes; ++it) {
      total += (*it)->bytes;
    }
    index_cache_.erase(it, index_cache_.end());
  }

  /**
   * The index trained by testIVFFlat (`trained_`), extended with the whole database at once. This
   * is what `build` with `add_data_on_build = true` produces: it trains the index and then extends
   * it with the dataset and no explicit indices.
   */
  auto full_index() -> const index<DataT, IdxT>&
  {
    if (cached_) { return cached_->full; }
    if (!full_) {
      RAFT_EXPECTS(trained_.has_value(),
                   "testIVFFlat() must run before testPacker() and testFilter()");
      auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
        (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
      const std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
      full_.emplace(ivf_flat::extend(handle_, database_view, no_opt, *trained_));
    }
    return *full_;
  }

  raft::resources handle_;
  cuda::stream_ref stream_;
  AnnIvfFlatInputs<IdxT> ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;

  // The indexes of a case are built once and shared by its sub-tests (testIVFFlat, testPacker and
  // testFilter), unless they are reused from an earlier case with the same build_key (`cached_`).
  /** Trained, but empty (`add_data_on_build = false`); `extend` does not modify it. */
  std::optional<index<DataT, IdxT>> trained_;
  /** `trained_` extended with two halves of the database, serialized and deserialized. */
  std::optional<index<DataT, IdxT>> loaded_;
  /** See `full_index()`. */
  std::optional<index<DataT, IdxT>> full_;
  std::shared_ptr<const built_indexes> cached_;
};

const std::vector<AnnIvfFlatInputs<int64_t>> inputs = {
  // test various dims (aligned and not aligned to vector sizes)
  {1000, 10000, 1, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 3, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 3, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 4, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 4, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded, false},
  // The same as the {5, CosineExpanded, false} entry above, but with independent_builds:
  // covers kmeans_trainset_fraction = 1 (testPacker) and add_data_on_build = true (testFilter).
  {1000,
   10000,
   5,
   16,
   40,
   1024,
   cuvs::distance::DistanceType::CosineExpanded,
   false,
   false,
   false,
   true},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded, true},
  // The same as the {8, CosineExpanded, true} entry above, but with independent_builds.
  {1000,
   10000,
   8,
   16,
   40,
   1024,
   cuvs::distance::DistanceType::CosineExpanded,
   true,
   false,
   false,
   true},

  // test dims that do not fit into kernel shared memory limits
  {1000, 10000, 2048, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2048, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 2049, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2049, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 2050, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 2050, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  // TODO: Re-enable test after adjusting parameters for higher recall. See
  // https://github.com/nvidia/cuvs/issues/1091
  // {1000, 10000, 2051, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 2051, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 2052, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 2052, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 2053, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2053, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 2056, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2056, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},

  // various random combinations
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2Expanded, false},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, false},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false},

  // host input data
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2Expanded, false, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},

  // // host input data with prefetching for kernel copy overlapping
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},

  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::InnerProduct, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {10000, 131072, 8, 10, 50, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {10000, 131072, 8, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, true},

  {1000, 10000, 4096, 20, 50, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 4096, 20, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false},

  // test splitting the big query batches  (> max gridDim.y) into smaller batches
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::InnerProduct, false},
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000000, 1024, 32, 10, 256, 256, cuvs::distance::DistanceType::InnerProduct, false},
  {1000000, 1024, 32, 10, 256, 256, cuvs::distance::DistanceType::CosineExpanded, false},
  {98306, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::InnerProduct, true},
  {98306, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::CosineExpanded, true},

  // test radix_sort for getting the cluster selection
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 2,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   cuvs::distance::DistanceType::L2Expanded,
   false},
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   cuvs::distance::DistanceType::InnerProduct,
   false},
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   cuvs::distance::DistanceType::CosineExpanded,
   false},

  // The following two test cases should show very similar recall.
  // num_queries, num_db_vecs, dim, k, nprobe, nlist, metric, adaptive_centers
  {20000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::L2Expanded, false},
  {100000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::L2Expanded, false}};

}  // namespace cuvs::neighbors::ivf_flat
