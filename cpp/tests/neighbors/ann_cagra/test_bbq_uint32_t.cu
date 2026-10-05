/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_cagra_bbq.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::cagra {

// The cases of one parameter set share one BBQ graph build, and the cases of one metric share one
// dense reference build (AnnCagraBbqTest::bbq_build, AnnCagraBbqTest::dense_reference_recall).
TEST_P(AnnCagraBbqTest, AnnCagraBbqSearchRecall) { this->testSearchRecall(); }
TEST_P(AnnCagraBbqTest, AnnCagraBbqGraphShape) { this->testGraphShape(); }
TEST_P(AnnCagraBbqTest, AnnCagraBbqGraphOnlyBuild) { this->testGraphOnlyBuild(); }
TEST_P(AnnCagraBbqTest, AnnCagraBbqSerializeRoundTrip) { this->testSerializeRoundTrip(); }
TEST_P(AnnCagraBbqTest, AnnCagraBbqUnsupportedParams) { this->testUnsupportedParams(); }

INSTANTIATE_TEST_CASE_P(AnnCagraBbqTest, AnnCagraBbqTest, ::testing::ValuesIn(bbq_inputs));

}  // namespace cuvs::neighbors::cagra
