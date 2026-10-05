# cagra-bbq-test-graph-reuse

Test-only change to `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST`. Each BBQ graph is built once per parameter set and each
dense reference once per metric, instead of once per TEST_P. Every check still runs on every parameter set with the
same thresholds. Test names and the case count (5 TEST_Ps × 27 = 135) do not change.

Background: `../NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST.md` ideas 1, 4 and 5(a), and T1 in
`../PROFILING_SUMMARY.md`.

## What changed

All of it is in `cpp/tests/neighbors/ann_cagra_bbq.cuh` (fixture `AnnCagraBbqTest`). `test_bbq_uint32_t.cu` only gets a
2-line comment (`:12-13`). Both files are formatted with clang-format 20.1.8 (`cpp/.clang-format`).

* **Two suite-level caches** (`:449-454`): `std::map<bbq_key_type, bbq_build_result> bbq_builds_` and
  `std::map<dense_key_type, double> dense_reference_recalls_`. Both are cleared in a new public
  `static void TearDownTestSuite()` (`:70-75`), the same pattern as `dynamic_batching.cuh:113-115`.
* **Cache keys** (`:80-101`).
  * `dense_key()` = (data seed, n_queries, n_rows, dim, k, graph_degree, metric). This is everything the database, the
    queries, the ground truth and `default_index_params()` depend on. The seed is now a named constant `kDataSeed`
    (`:81`), also used by `SetUp` (`:429`), so the key cannot drift from the data.
  * `bbq_key()` = `dense_key()` + `layout` + `second_layout`. This adds what the codes and the BBQ graph depend on.
  * `min_recall_ratio` is left out. It is only a threshold.
* **`bbq_build()`** (`:149-178`). On a miss, it quantizes the database and calls `cagra::build` with
  `attach_dataset_on_build = false`. That is the call `testGraphOnlyBuild` used to make. The result is stored as a
  `bbq_build_result` (`:103-118`): a host copy of the graph, plus the build's `metric()`, `graph_size()`,
  `graph_degree()`, `dataset().n_rows()` and `dataset().quantizers.empty()`.
* **`bbq_index(built, owning_codes)`** (`:180-191`) gives each test its own `device_bbq_index<float>`. It calls
  `index(res, metric, codes_view, host_graph_view)`, the same constructor `build_from_bbq_dataset` uses when
  `attach_dataset_on_build` is true (`cpp/src/neighbors/detail/cagra/cagra_build.cuh:2986-2989`). The graph does not
  depend on the flag, because it is computed before that branch (`:2972-2982`). The constructor copies the host graph
  to a new device matrix that the index owns.
* **`dense_reference_recall(dataset, ground_truth)`** (`:193-211`). On a miss, it runs the dense build →
  `update_dataset` → search that `testSearchRecall` used to run inline, and caches the recall.
* **Test bodies.**
  * `testSearchRecall` (`:274-319`) uses `dense_reference_recall` and `bbq_index(bbq_build(), codes)`. It still calls
    `update_dataset(std::move(bbq_index), padded)`, so the BBQ → dense rebinding path is still tested.
  * `testGraphShape` (`:322-344`) and `testSerializeRoundTrip` (`:351-386`) take their index from
    `bbq_index(bbq_build(), codes)` instead of `cagra::build`. The asserts are unchanged.
  * `testGraphOnlyBuild` (`:388-398`) asserts the same four properties on the values `bbq_build()` recorded from the
    graph-only build.
  * `testUnsupportedParams` is unchanged.

## What is reused, and what is not

| artefact | before | after |
|---|---|---|
| BBQ NN-descent build + optimize | 4 per param (SearchRecall, GraphShape, SerializeRoundTrip, GraphOnlyBuild) = 108 | 1 per param = 27 |
| dense NN-descent build + optimize | 1 per param = 27 | 1 per dense key (= per metric here) = 3 |
| **total graph builds** | **135** | **30** |
| quantization (D2H + CPU/file cache + upload) | 5 per param | 5 per param (unchanged) |
| naive ground truth, searches, serialize/deserialize | per case | per case (unchanged) |

What is shared is a **host copy of the graph** plus a few scalars, and one `double` per metric. Index objects, codes and
device buffers are never shared. Each test quantizes its own codes and builds its own index from the const host graph.
It then moves, serializes or searches that index, so a test cannot hand a mutated object to another test.

Memory: 27 × 4000 × 32 × 4 B ≈ 13.8 MB of host memory at peak, plus 3 doubles. The caches hold **no GPU memory**.
gtest runs every parameter set of one TEST_P before moving to the next TEST_P, so a cache that only kept the most
recent parameter set would never hit. That is why all 27 graphs are kept until `TearDownTestSuite`.

## Expected effect

From the profile (`NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST.md`, ideas 1+4+5, combined): 135 → 30 NN-descent builds,
**≈ 11.5 s of the 16.4 s gtest time under nsys (≈ 70%)**. In the plain single-process run that is 14.0 s → about 5 s.
Nothing was run here.

Per TEST_P, in one process with the default order:

* `AnnCagraBbqSearchRecall` keeps its 27 BBQ builds (it runs first and fills the cache). It drops 24 of its 27 dense
  builds: ≈ 6 s → ≈ 3.7 s.
* `AnnCagraBbqGraphShape`, `AnnCagraBbqGraphOnlyBuild` and `AnnCagraBbqSerializeRoundTrip` make no builds. Each case
  is quantize + host-graph upload + checks or searches, a few ms. ≈ 7.6 s (plain run) → well under 1 s.
* `AnnCagraBbqUnsupportedParams` is unchanged.

If a TEST_P runs alone (`--gtest_filter`), it fills the cache itself. That is one build per param, the same as today.

## Case-count changes

None. The executable still has 5 TEST_Ps × 27 params = 135 cases, with the same names and the same parameter indices.

## Risks

* **The default attach branch of `build_from_bbq_dataset` is no longer executed by this executable.** That branch is
  `cagra_build.cuh:2986-2989`, a single `return index(res, params.metric, dataset, graph)`. The graph-only branch
  (`:2990-2992`) is still executed by `bbq_build()`. `bbq_index()` calls the same constructor with the same
  arguments, so the GraphShape asserts (`size()`/`dim()` reporting the quantized shape) still exercise that
  constructor. If the attach branch ever does more than call the constructor, `bbq_index()` must follow it, or
  GraphShape should go back to its own default-params build (+27 builds).
* **One sample of NN-descent per param instead of four.** The builds are not bit-reproducible (static `i_primes`).
  Before, each TEST_P drew its own graph; now all of them check the same graph. Code coverage is the same. A rare bad
  graph now fails every check of that param instead of one.
* **Correlated reference jitter.** The 9 layouts of a metric now share one dense reference recall instead of drawing
  9. The `> 0.8` baseline assert and the `min_recall_ratio` check still run for every param. But if the one reference
  is unusually high or low, all 9 ratio checks for that metric move together. This was rated low risk in the analysis.
* **Failure attribution.** If `cagra::build` throws, the exception surfaces in the first test that asks for that
  param (normally `AnnCagraBbqSearchRecall/<i>`). Nothing is cached, so later TEST_Ps retry the build and fail on their
  own. `AnnCagraBbqGraphOnlyBuild` now reports values recorded from the build rather than asserting on the live index.
  The values and messages are the same.
* **Cache keys.** The keys include the seed and every input of the data, the codes and the graph. The internal
  `/tmp/bbq-n*-d*-b*-l*-m*.bin` code cache in `cpp/internal/cuvs_internal/preprocessing/bbq_cpu_quantize.hpp:459-469`
  still has no data seed or data hash. That is a pre-existing hazard and out of scope here; today's only other user,
  `ann_nn_descent_bbq.cuh`, uses 2000 × 256 and does not collide.
* **gtest options.** `--gtest_shuffle`, `--gtest_filter` and `--gtest_repeat` stay correct, because lookups are by key
  and `TearDownTestSuite` clears the caches between repeats. Only the hit rate changes.

## How to verify

1. Compile-only check (done, no build dir touched). The command for `test_bbq_uint32_t.cu` was taken from
   `cpp/build/latest/compile_commands.json`, with `-o` pointed at `$TMPDIR` and all `--generate-code` flags replaced
   by `-arch=sm_89` (`-Werror` kept). It compiles with no errors or warnings (about 58 s). The only output is
   `nvcc warning : incompatible redefinition for option 'compiler-bindir'`. That comes from the environment
   (`NVCC_PREPEND_FLAGS=-ccbin=...` plus the command's own `-ccbin`), not from the source.
2. `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST --gtest_list_tests`: expect 135 cases, the same names as before.
3. Run the executable. All 135 should pass. The `CAGRA BBQ build (...) recall=..., dense reference=...` INFO lines
   should show one dense reference value per metric, repeated across the 9 layouts.
4. Build count: under `nsys profile -t cuda,nvtx`, count the `cagra::detail::build_from_bbq_dataset` NVTX ranges. Expect
   27 (was 108), and 3 dense `cagra::build` ranges (was 27). Or count the `local_join` launches: 30 × 20 = 600, was 2700.
5. Isolation: run each TEST_P alone, e.g. `--gtest_filter='*AnnCagraBbqSerializeRoundTrip*'` and
   `--gtest_filter='*GraphOnlyBuild*'`. Each should pass and build once per param.
6. Timing: `ctest -R NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` (or the executable alone) before and after.
