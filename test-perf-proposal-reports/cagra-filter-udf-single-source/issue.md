**Title:** [TEST] CAGRA filter-UDF test re-links the search kernels once per UDF source (≈ 72% of its runtime with a cold JIT cache)

`NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST` (`cpp/tests/neighbors/ann_cagra/test_filter_udf.cu`) spends most of its time in
nvJitLink rather than on the GPU. The CAGRA UDF fragment key contains the UDF source text
(`cpp/src/neighbors/detail/cagra/jit_lto_kernels/sample_filter_udf.cuh:78-86`). Each distinct
(source, data type) therefore triggers a full LTO link of the search kernels: SINGLE_CTA, MULTI_CTA and the two
MULTI_KERNEL kernels. With a cold CUDA JIT cache that costs ≈ 2 s per (source, data type).

The test uses 5 UDF sources for float (`accept_all`, `reject_all`, `high_filtering_rate`, `threshold`, `tenant`) and 1
for half (`threshold`). That is 6 link sets, and they take ≈ 12 s of the ≈ 15–17 s run. The GPU is busy for < 1% of
the run. Four of the sources differ only in a constant (`return true`, `return false`, `source_id >= 704`,
`source_id >= 192`).

CI starts with an empty CUDA JIT cache (`~/.nv/ComputeCache`, which nvJitLink uses through the driver), so CI pays
the cold cost on every run. With a warm cache the test takes ≈ 3–4 s.

**Proposal (test-only):** express the four threshold-style predicates as one UDF that reads its lower bound from
`filter_data` (`nullptr` = accept all). Keep the tenant UDF as a second source. That gives 3 link sets instead of 6,
which saves ≈ 6 s with a cold cache. Every test case keeps its semantics and assertions. Two distinct sources
are still linked in one process, so cache keying on the source stays covered.
