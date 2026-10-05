### [TEST] HNSW ACE tests run each `npartitions=2` build three times

**Problem**

`AnnHnswAceBuild` in `NEIGHBORS_ANN_HNSW_ACE_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST` takes its cases from
`generate_hnsw_ace_inputs()` (`cpp/tests/neighbors/ann_hnsw_ace.cuh`). That generator crosses
`npartitions = {0, 1, 2, 4}` with dim `{64, 128}` × use_disk `{false, true}` × metric `{L2, IP}`, giving 32 cases per
executable.

ACE resolves `npartitions` 0 (auto) and 1 to 2 before any other partitioning logic
(`ace_resolve_partition_count`, `cpp/src/neighbors/detail/cagra/cagra_build.cuh:1132-1141`). The memory estimates,
disk-mode decision, memory-limit adjustment, partition labelling and per-partition builds all use the resolved value.
As a result, the 0, 1 and 2 cases run the same build: the logged partition splits are identical, and the dataset is
seeded. **24 of the 32 cases are triplicates**, so 16 per executable (64 across the four) are redundant ACE builds.

These four executables spend most of their time in this group: about 59% of the float profile is ACE partition
sub-builds.

**Proposal**

* Cross only the distinct partition counts `{2, 4}` with the other axes.
* Add one `npartitions=0` case (dim 64, in-memory, L2) and one `npartitions=1` case (dim 64, disk, IP), so the
  resolution path and the "adjusted to 2" warning are still exercised.

This is 32 → 18 cases per executable, with every distinct build kept. The estimated saving is about 26–30% of each
executable's ctest time, ≈ 14 s across the four.

**Note**

`cagra::graph_build_params::ace_params::npartitions` is documented as "automatically derived based on available host
and GPU memory" when 0 (`cpp/include/cuvs/neighbors/cagra.hpp:177-179`). The implementation currently uses a fixed
default of 2. If auto-selection is implemented later, the test axis should be widened again.
