### CAGRA gtests rebuild identical indices in most cases

**Problem**

`NEIGHBORS_ANN_CAGRA_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST` spend most of their time building indices, mostly the
IVF-PQ and NN-descent graph builds. Most of these builds are repeats of an index that an earlier case already built.
These four executables took 313 / 200 / 251 / 265 s under `ctest -j8`.

Nsight Systems profiles of each executable (RTX 6000 Ada, one fixture group per process) show several sources of
repeats.

* **`_U32` / `_I64` twins.**
  * `AnnCagra_U32` / `AnnCagra_I64` and `AnnCagraIndexMerge_U32` / `_I64` build, serialize and merge identical
    indices. They differ only in the output index type of `search`
    (`ann_cagra/test_float_uint32_t.cu:16-17, 26-27`, `test_half_uint32_t.cu:13-14, 20-21`).
  * This costs about 105 s of 366 s profiled in float, and 118 s of 257 s in half.
* **AnnCagraTest instantiates cases it can't tell apart.**
  * The fixture doesn't read `merge_strategy`, `itopk_size` (never set at `ann_cagra.cuh:451-455`) or
    `search_width`.
  * `host_dataset` only adds an unused D2H copy, because no input selects ACE (`:467-476`).
  * 179 of the 459 `inputs` entries are therefore exact duplicates for this fixture, and 94-118 of them execute
    per TEST_P. Profiled cost: 18-29 s per executable.
* **The same index is rebuilt for every search variant.**
  * Builds depend only on (n_rows, dim, metric, degree, build algo, refinement), but each search-algo /
    max_queries / team_size / k case rebuilds.
  * AnnCagraTest float: 596 executed cases for 107 distinct builds.
  * IndexMerge float: 524 cases for 101 distinct half-index pairs. The PHYSICAL / LOGICAL twins alone cost 17-22 s
    per executable.
  * FilterTest: 60 cases for 13 builds.
  * AnnCagraMultiPartitionTest: 48 builds for 13 distinct partition sets (`ann_cagra.cuh:2153-2173`). Search and
    FilteredSearch, and the SINGLE_CTA / MULTI_CTA variants, all rebuild. This costs 11-16 s per executable of
    NN-descent on 10k rows.
* **Likely bug: `AnnCagraIndexFilteredMergeTest` ignores `graph_degree`** (`ann_cagra.cuh:1263-1289`).
  * Every case builds degree-64 graphs, so the `{32, 47, 64}` sweep only produces 20 duplicates, and the printed
    `degree=` is wrong.
  * #819 added the degree axis and wired it, together with a degree-based recall relaxation, into every other
    fixture. FilteredMerge (#1496) already existed when #819 was merged, but #819 did not touch it.

The analyses estimate the savings without losing coverage at about 185 s (float), 131 s (half), 54-68 s (int8) and
60 s (uint8) of profiled time for an index cache plus the folds. Reusing the MultiPartition sets adds 11-16 s per
executable.

**Proposed fix (tests only)**

1. Fold the `_U32` / `_I64` TEST_Ps into one body. It builds, serializes and merges once, and then searches and
   checks into both uint32 and int64 outputs.
2. Instantiate AnnCagraTest and IndexMerge with input lists that drop the entries the fixture ignores (first
   occurrence kept).
3. Add a small per-test-suite cache for built indices:
   * It is keyed by the build parameters, and it also checks that the generated dataset is bitwise identical.
   * The fixtures only read cached indices: search, serialize and merge.
   * It is cleared in `TearDownTestSuite`.
   * It is used by AnnCagraTest, FilterTest, IndexMerge, FilteredMerge (for the halves) and MultiPartition (for the
     partition sets).
4. Make FilteredMerge honour `graph_degree` and apply the same degree-based recall relaxation as IndexMerge.

Every search, serialize round trip, merge, filter and check still runs for every distinct case, with the same
thresholds. Only FilteredMerge changes what it tests (item 4).

**Expected impact**

* The case count drops from 2520 / 1986 / 1128 / 1128 to 1417 / 883 / 943 / 943 (float / half / int8 / uint8).
* Executed cases drop from 1456 / 1114 / 746 / 807 to 782 / 510 / 632 / 683.
* AnnCagraTest builds drop from 596 / 540 / 298 / 321 to 107 / 93 / 107 / 117.
* IndexMerge half-pair builds drop from 524 / 468 / 262 / 285 to 101 / 87 / 101 / 111.
* MultiPartition partition-set builds drop from 48 to 13 (46 to 12 for half).
* Estimated runtime saving: roughly 40-55 % for float and half and 35-40 % for int8 and uint8. Not yet measured.
