# vamana-test-index-reuse

Test-only change to `NEIGHBORS_ANN_VAMANA_TEST`. Only `generate_inputs()` in `cpp/tests/neighbors/ann_vamana.cuh` changes
(see `change.patch`). The fixture, the checks, the thresholds and the four `ann_vamana/test_*_uint32_t.cu` files
(TEST_P bodies, instantiations) are untouched.

**Short version.** Contrary to the proposal's name, there is no index to reuse. Every one of the 315 parameter sets
per data type has distinct build parameters, and no axis that varies is search-only. The only redundancy is
`reverse_batchsize`. It does not change the graph (proof and evidence below), yet the degree-32 block crosses it
with everything else. The change keeps `reverse_batchsize = 100` for the full degree-32 sweep, because 100 runs the
multi-chunk reverse path. It keeps `reverse_batchsize = 1e6` (single chunk) for each degree-32 configuration at one
dim (137), and the single-chunk path also stays in every degree 64/128/256 case.
**1260 → 812 cases (and builds), ≈ 37 s (≈ 13%) less, estimated from measured per-case times.**

## 1. Parameter grid (before)

`AnnVamanaInputs` fields and how `testVamana` uses them (`ann_vamana.cuh`, line numbers before the patch):

| field | values | used by | effect |
|---|---|---|---|
| `n_rows` | 1000 | `SetUp` data, build | build |
| `dim` | 15 values: 1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 384, 512, 619, 1024 | `SetUp` data, build, codebooks (`:150-155`: float 64, int8 384) and therefore the sector-aligned serialize (`:176-183`) | build |
| `graph_degree` | 32 / 64 / 128 / 256 (one per block) | build, CheckGraph, recall guard `:186` | build |
| `visited_size` | 64, 256 / 128, 512 / 256 / 512, 1024 | build | build |
| `max_fraction` | 0.06, 0.1 (deg 32), else 0.06 | build (batch schedule) | build |
| `metric`, `host_dataset` | L2Expanded, false (constant) | build | none varies |
| `reverse_batchsize` | 100, 1e6 (deg 32), else 1e6 | build (reverse-edge chunking only) | **no effect on the graph**, see §2 |
| `insert_iters` | 1.0, 1.5 (deg 32), else 1.0 | build (number of insert passes) | build |
| `n_queries`, `k`, `algo`, `max_queries`, `min_recall` | 100, 10, AUTO, 10, 0.2 (constant) | recall check (deg < 256) | constant, so never the only difference between two cases |
| `itopk_size`, `search_width` | 64 or 32, 1 | **unused**: `testVamana` doesn't copy them into `search_params` (`:223-226`) | none |

Blocks (per data type; there are 4 data types: float, half, int8, uint8):

| block (lines) | cases | builds | checks | time, s (all 4 dtypes)¹ |
|---|---|---|---|---|
| deg 32: 15 dims × vs {64, 256} × mf {0.06, 0.1} × rb {100, 1e6} × iters {1.0, 1.5} (`:312-328`) | 240 | 240 | graph, serialize, recall | 86.4 (rb 100: 46.1, rb 1e6: 40.3) |
| deg 64: 15 dims × vs {128, 512} (`:330-346`) | 30 | 30 | graph, serialize, recall | 30.6 |
| deg 128: 15 dims × vs 256 (`:349-365`) | 15 | 15 | graph, serialize, recall | 21.7 |
| deg 256: 15 dims × vs {512, 1024} (`:368-385`) | 30 | 30 | graph, serialize (no recall, `:186`) | 144.0 |
| **total** | **315** | **315** | | **282.6** |

¹ Sum of gtest per-case times of one single-process run of the unmodified test on an RTX 6000 Ada, with the kvikio
zero-fill fix applied and before the reverse-batch clamp, which did not change the time (277.5 → 277.2 s). Source:
`/tmp/claude-1000/-home-coder/ae252f72-9cc0-46ee-a543-8482b2c783fe/scratchpad/ab_p2b/NEIGHBORS_ANN_VAMANA_TEST.s1.r1.json`
(`ab_p2b/…` below). The second run (`s1.r2`) gives the same numbers to ±0.1 s.

**What the profile already said holds:** every case builds a distinct index. No two cases differ only in
search or serialization parameters: the search parameters are constant, and the serialization path is chosen by
(dtype, dim). So a static index cache (as in the CAGRA/IVF reuse changes) would never hit. The build is ≥ 90%
of the run, and the per-case recall reference (`naive_knn`) takes ≈ 0.06 s for the whole executable, so caching
references would not help either.

## 2. Why `reverse_batchsize` does not change the graph

`batched_insert_vamana` (`cpp/src/neighbors/detail/vamana/vamana_build.cuh`) inserts the nodes in batches. After each
insert batch, it adds reverse edges to every node that received an edge in that batch:

1. The edge lists (`create_reverse_edge_list`, `:442`), their sort by distance and then by destination (`:452-474`,
   `cub::DeviceMergeSort`, stable), `unique_dests` (`:477-478`) and `unique_indices` (`:481-488`) are computed once
   per insert batch. They don't use `reverse_batchsize`.
2. The only use of `reverse_batchsize` is the chunking of the sorted, **distinct** destinations:
   `max_reverse_batch = min(reverse_batchsize, N)` (`:149`). The loop at `:499-555` processes destinations
   `[rev_start, rev_start + reverse_batch)`, with a shorter last chunk (`:501-503`). The chunks are disjoint and
   cover all destinations.
3. Each chunk computes each destination's new list from data that no other destination in the same pass writes:
   * `populate_reverse_list_struct` (`vamana_structs.cuh:1185-1218`) fills entry `i` from
     `unique_indices[rev_start + i]` and the sorted `edge_src`. The candidates and their order are the same
     whichever chunk the destination falls in.
   * `recompute_reverse_dists` (`:1223-1243`) and `RobustPruneKernel` (`robust_prune.cuh:119`, grid-stride over
     entries) read only the destination's own graph row (`graph(queryId, j)`, `:133`, `:154`), its own candidate
     list, and the dataset. They write only to the entry (`:326-330`). The per-block scratch `s_coords_mem`
     (`:108`) is private to the block and rewritten for every entry.
   * `SortPairsKernel` (`greedy_search.cuh:59-72`) sorts each entry independently. `write_graph_edges_kernel` (`vamana_structs.cuh:1144-1157`)
     then writes only the rows of the chunk's destinations.

   Destinations are distinct, so no chunk reads a row that an earlier chunk of the same pass wrote. The result
   per destination, and so the graph after the pass, is independent of the chunking. Grid sizes
   (`num_blocks = min(maxBlocks, reverse_batch)`, `:513`) only change which block handles which entry.
4. The only randomness is `rand()`: N calls for the insert permutation (`:59-72`) and one for the medoid (`:316`).
   Neither depends on `reverse_batchsize`.

With N = 1000, `reverse_batchsize = 100` splits most passes into several chunks: an insert batch of 60 or 100 nodes ×
degree 32 gives up to 1,920 or 3,200 reverse edges, to up to 1,000 distinct nodes. 1e6 (clamped to N since `e8be51ec`) always runs one
chunk. Before the clamp, 1e6 also ran one chunk (`unique_dests ≤ N < 1e6`), so the argument holds for upstream too.

**Empirical check with existing logs (no new GPU run).** The phase-1 nsys profile ran each data type in 4 gtest
shards (`nsys_groups/NEIGHBORS_ANN_VAMANA_TEST/g0xx_…_s{0..3}of4.log` + `_gtest.json`). gtest sharding sends case
`i` to shard `i % 4`. In the degree-32 block, `i % 4 = 2·rb_index + iters_index`. So shard 0 holds the
`(rb 100, iters 1.0)` cases, shard 2 the `(rb 1e6, iters 1.0)` cases with otherwise identical parameters, in the
same order, and likewise shards 1 and 3. Every build makes exactly N + 1 `rand()` calls, so the k-th case of shard 0
and the k-th case of shard 2 start from the same `rand()` state. Result over the 4 data types:

* **480 of 480** `(rb 100, rb 1e6)` pairs log the same `Total edges` count. 152 of them are full graphs
  (32,000 edges), which says little. The other **328 of 328** match too, while the count is sensitive to the
  insert order. The same deg-32 case started from a different `rand()` state (full single-process run
  `ab_p2b/…s1.r1` vs the shard runs) gives a different count in 649 of 652 non-full cases.
* 462 of 480 log the same recall. The other 18 differ by 0.001 (one neighbour in 1,000). The CAGRA multi-CTA oracle
  shows the same noise between two runs of the same binary: 36 of 1,140 recall values differ between `s1.r1` and
  `s1.r2`, while all 1,260 edge counts are identical.

So the rb 1e6 deg-32 cases re-check the graph of their rb 100 sibling, with the same assertions. They differ
only in the `rand()` state they happen to start from.

## 3. The change

`generate_inputs()`:
* The degree-32 product uses `{100}` instead of `{100, 1000000}` for `reverse_batchsize`. It still crosses all 15 dims
  × vs {64, 256} × mf {0.06, 0.1} × iters {1.0, 1.5}, i.e. 120 cases per data type, all with the recall check.
* A loop appends, for the 8 degree-32 configurations at dim 137, a copy with `reverse_batchsize = 1000000`. 137 is
  odd and ≥ `kRobustPruneCandCacheMinDim` (128), so RobustPrune takes its smem candidate cache and distance-reuse
  branches (`robust_prune.cuh:99-102,152`).
* The degree 64/128/256 blocks are unchanged (all `reverse_batchsize = 1e6`, i.e. one chunk, all 15 dims).

### Coverage argument

* **Every data type × degree × visited_size × max_fraction × insert_iters × dim combination is still built,
  graph-checked, serialized and (for deg < 256) recall-checked with `reverse_batchsize = 100`.** That is a full 2³
  factorial over (vs, mf, iters) at every dim, with the same thresholds (fill > 0.75, `max_degree ≥ min(deg, dim)`,
  recall ≥ 0.2).
* **Every (degree 32, vs, mf, rb, iters) build configuration** still runs for every data type, including all 8 with
  rb = 1e6.
* **The multi-chunk reverse path** (several chunks plus a short last chunk) runs in all 120 deg-32 sweep cases.
  **The single-chunk path** runs in the 8 deg-32 sentinels and all 75 deg 64/128/256 cases per data type, at
  every dim.
* **What is no longer run:** the rb 1e6 copies of the deg-32 cases at the 14 other dims (112 per data type). By §2
  each of them builds and checks the same graph as its rb 100 sibling, up to the `rand()` start state. The kernels and
  template instantiations are the same, and so are the sort/prune shared-memory sizes: they depend on degree,
  visited_size and dim, not on rb (`vamana_build.cuh:213-231`).
* The codebook / sector-aligned serialization cases (float dim 64, int8 dim 384) keep all deg-32 configurations
  (rb 100) and all deg 64/128/256 cases.
* Nothing else changes: same fixture, `SetUp` seeds, thresholds and serialization calls.

## 4. Test counts and names

| | per data type | total (`--gtest_list_tests`) |
|---|---|---|
| before | 315 (`AnnVamana/0`–`/314`) | **1260** |
| after | 203 (`AnnVamana/0`–`/202`) | **812** |

Builds: 1260 → 812 (one per case, before and after). Suites: `AnnVamanaTest/AnnVamanaTest{F,F16,I8,U8}_U32`.

Index mapping (same for every data type), new → old:
* `/0`–`/119` (deg 32, rb 100): old `4·(n div 2) + (n mod 2)`, i.e. `/0,1,4,5,8,9,…,236,237`.
* `/120`–`/127` (deg 32, rb 1e6, dim 137): old `/130,131,134,135,138,139,142,143`.
* `/128`–`/202` (deg 64/128/256): old `/240`–`/314` (shift by −112).

Removed: old `/2,3,6,7,…` (`i mod 4 ∈ {2,3}`, `i < 240`) except `/130,131,134,135,138,139,142,143`. That is 112 per data
type, 448 in total. No in-repo filter or script references these indices: `grep -rln AnnVamana` outside build
directories finds only the test sources.

Quick check after building:
```
./gtests/NEIGHBORS_ANN_VAMANA_TEST --gtest_list_tests | grep -c 'AnnVamana/'   # 812 (was 1260)
```

## 5. Expected savings

Sum of the measured per-case times of the removed cases (unmodified test, single process, RTX 6000 Ada,
`ab_p2b/NEIGHBORS_ANN_VAMANA_TEST.s1.r{1,2}.json`, both runs give the same sums):

| | float | half | int8 | uint8 | total |
|---|---|---|---|---|---|
| removed cases, s | 9.6 | 9.3 | 9.1 | 9.0 | **37.0 of 282.6 (13.1%)** |

The rb 1e6 cases are slightly cheaper than their rb 100 siblings (40.3 vs 46.1 s in total), because one chunk means fewer
launches. So the rb 100 copies are kept for the multi-chunk path, not for speed. Against the current ≈ 277 s, expect
**≈ 240 s**. Under `ctest -j8` the saving scales roughly the same (the test is build-bound; GreedySearch uses ≤ 15
blocks per launch).

## 6. Considered and not done

* **Static index cache / reuse across search-only variants**: no two cases share build parameters (§1), so it
  would never hit.
* **`max_queries` 10 → 0 for the CAGRA recall oracle** (hotspots idea 6, ≈ 5 s). Not in the patch because it changes a
  search parameter of the checked combination. If wanted, it should also pin `algo = MULTI_CTA`. With
  `max_queries = 10`, AUTO resolves to MULTI_CTA on every GPU with > 5 SMs (`search_plan.cuh:124-131`:
  SINGLE_CTA only if `max_queries ≥ 2·num_sm`). With 100 queries per batch, GPUs with ≤ 50 SMs (e.g. T4) would
  switch to SINGLE_CTA. CAGRA's own tests cover `max_queries < n_queries` batching (`ann_cagra.cuh:2834-2850`).
* **Reusing the iters 1.0 graph as the first pass of iters 1.5**: the first pass is identical given the same
  `rand()` state, but there is no API to continue a build, so this would need a library change.
* **Sharing data or builds across data types.** int8 and uint8 get the same values from `uniformInt(…, 1, 20)`
  (`ann_vamana.cuh:286-289`). In the shard runs, all 315 int8 cases log the same edge count as the matching uint8 case.
  But the kernels are different instantiations, so both must build. (Side finding: neither type ever sees values
  above 19, so sign- and range-specific distance code, i.e. negative int8 or uint8 > 127, isn't exercised here.)
* **Coverage-reducing options** from `NEIGHBORS_ANN_VAMANA_TEST.md` (trim the deg 64/128/256 dims, half-fraction of
  the deg-32 factorial, `max_fraction` 0.1 for deg ≥ 64) are out of scope. The deg-256 block (144 s, 51%) is
  untouched and still has no recall check (`ann_vamana.cuh:186`).

## 7. Risks

* **The `rand()` sequence of later cases shifts.** `rand()` is never seeded, so each case's insert order depends on
  how many builds ran before it in the process. This is already true under sharding or `--gtest_filter`. After
  the change, the remaining cases (except the first) start from different states than before. The margins are
  large: in the current run the minimum fill ratio is 0.829 (deg 64; threshold 0.75), and the minimum recall is 0.928
  (deg 32; threshold 0.2). The 4-shard profile runs (different start states for every case) passed as well.
* **If a future library change makes `reverse_batchsize` affect the graph** (e.g. chunks that read each other's
  rows), the sweep would cover rb 1e6 at deg 32 at one dim only. The comment in `generate_inputs()` states the
  assumption, so such a change should restore `{100, 1000000}`. A stronger option is a test that builds twice
  from the same `srand()` state with rb 100 and rb 1e6 and compares the graphs. It is not added here: it would cost
  an extra build per case and needs a GPU run to confirm bit-exact determinism first. The logs above suggest the
  build is deterministic: identical edge counts in all 1,260 cases across runs and binaries.
* **Test indices shift** (see §4). Nothing in the repo filters on them.

## 8. Checks done

* `clang-format` 20.1.8 with `cpp/.clang-format`: no changes to the edited header.
* Compile check: each `ann_vamana/test_*_uint32_t.cu` compiled from an overlay copy of `cpp/tests` (`/tmp/vtir/overlay`,
  with `cpp/src` symlinked for the `../../src/...` includes), using its exact `compile_commands.json` command with
  the source and `-I…/cpp/tests` pointed at the overlay, `-o` in `/tmp/vtir/obj`, a single `-arch=sm_89`, and `nice -n 19`.
  `nvcc -M` confirmed the overlay `ann_vamana.cuh`, `test_utils.cuh`, `naive_knn.cuh` and
  `cagra_padded_build_helpers.cuh` are used, with no dependency on the original `cpp/tests`.
  * float: OK (74 s), half: OK (82 s), int8: OK (55 s), uint8: OK (54 s; compiled after the timing session ended).
    Exit 0 under `-Werror` / `-Werror=all-warnings`. The only message is nvcc's `compiler-bindir` redefinition
    warning from the conda env's `NVCC_PREPEND_FLAGS`, which is unrelated.
* `git -C /home/coder/cuvs apply --check change.patch`: OK.
* Counts and index mapping computed by replicating `generate_inputs()` (`$TMPDIR/vtir/count.py`). The product
  order (`iters` fastest, then rb, mf, vs, dim) was confirmed by decoding `value_param` in the gtest JSON.
* Not run: GPU tests (timing benchmarks were running).
