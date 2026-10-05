/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"
#include "../cagra_padded_build_helpers.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace cuvs::neighbors::cagra {

class AnnCagraBugMultiCTACrash : public ::testing::TestWithParam<cagra::search_algo> {
 public:
  using data_type = half;

 protected:
  void run()
  {
    // The bug is in the random seed selection: when no sampled node is closer than the initial
    // (inf) best distance, the seed index was left uninitialized and later used to read the graph.
    // This only needs every query-to-node distance to be inf (see SetUp), whatever the graph, so
    // a random graph of the original degree replaces the (slow) graph build.
    build_padded_.emplace(res, raft::make_const_mdspan(dataset->view()));
    cagra::device_padded_index<data_type> cagra_index(
      res, metric, build_padded_->view, raft::make_const_mdspan(graph->view()));
    raft::resource::sync_stream(res);

    cagra::search_params cagra_search_params;
    cagra_search_params.itopk_size        = 32;
    cagra_search_params.thread_block_size = 256;
    cagra_search_params.search_width      = 1;
    cagra_search_params.max_iterations    = 0;
    cagra_search_params.algo = ::testing::TestWithParam<cagra::search_algo>::GetParam();

    // NOTE: when using one resource/stream for everything, the bug is NOT reproducible
    raft::resources res_search;
    cagra::search(res_search,
                  cagra_search_params,
                  cagra_index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors->view(),
                  distances->view());

    std::vector<uint32_t> neighbors_h(n_queries * k);
    raft::update_host(neighbors_h.data(),
                      neighbors->data_handle(),
                      neighbors_h.size(),
                      raft::resource::get_cuda_stream(res_search));
    raft::resource::sync_stream(res_search);

    // An uninitialized seed index must not leak into the results: each one is either a dataset
    // index or the invalid index.
    for (size_t i = 0; i < neighbors_h.size(); i++) {
      if (neighbors_h[i] == std::numeric_limits<uint32_t>::max()) { continue; }
      ASSERT_LT(neighbors_h[i], n_samples) << "query " << i / k;
    }
  }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_samples, n_dim));
    graph.emplace(raft::make_device_matrix<uint32_t, int64_t>(res, n_samples, graph_degree));
    queries.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_queries, n_dim));
    neighbors.emplace(raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k));
    distances.emplace(raft::make_device_matrix<float, int64_t>(res, n_queries, k));
    raft::random::RngState r(1234ULL);
    InitDataset(res, dataset->data_handle(), n_samples, n_dim, metric, r);
    raft::random::uniformInt(res,
                             r,
                             graph->data_handle(),
                             n_samples * graph_degree,
                             uint32_t{0},
                             static_cast<uint32_t>(n_samples));
    // NOTE: when initializing queries with "normal" data, the bug is NOT reproducible.
    // upper_bound<half>() is +inf, so the distance from a query to any node is inf.
    raft::linalg::map(
      res, queries->view(), raft::const_op<data_type>{raft::upper_bound<data_type>()});
    // InitDataset(res, queries->data_handle(), n_queries, n_dim, metric, r);
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    build_padded_.reset();
    dataset.reset();
    graph.reset();
    queries.reset();
    neighbors.reset();
    distances.reset();
    raft::resource::sync_stream(res);
  }

 private:
  raft::resources res;
  std::optional<cuvs::neighbors::test::padded_device_matrix_for_cagra<data_type>> build_padded_{};
  std::optional<raft::device_matrix<data_type, int64_t>> dataset  = std::nullopt;
  std::optional<raft::device_matrix<uint32_t, int64_t>> graph     = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> queries  = std::nullopt;
  std::optional<raft::device_matrix<uint32_t, int64_t>> neighbors = std::nullopt;
  std::optional<raft::device_matrix<float, int64_t>> distances    = std::nullopt;

  constexpr static int64_t n_samples                   = 1183514;
  constexpr static int64_t n_dim                       = 100;
  constexpr static int64_t graph_degree                = 32;
  constexpr static int64_t n_queries                   = 30;
  constexpr static int64_t k                           = 10;
  constexpr static cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

TEST_P(AnnCagraBugMultiCTACrash, AnnCagraBugMultiCTACrash) { this->run(); }

INSTANTIATE_TEST_CASE_P(AnnCagraBugMultiCTACrashReproducer,
                        AnnCagraBugMultiCTACrash,
                        ::testing::Values(cagra::search_algo::MULTI_CTA));

}  // namespace cuvs::neighbors::cagra
