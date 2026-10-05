### CAGRA search falls back to one launch per query when the query row pitch isn't 16 B aligned

**Problem**

Since #1846, `cagra::detail::search_main_core` (`cpp/src/neighbors/detail/cagra/cagra_search.cuh:94-165`) only
searches queries in batches when `dim * sizeof(T)` is a multiple of 16 bytes, which means float `dim % 4 == 0`, half
`dim % 8 == 0`, and int8/uint8 `dim % 16 == 0`. Any other dense `[n, dim]` input is copied into a padded
`[n, stride]` buffer. Because the kernels step through queries by `dim` (`setup_workspace`:
`queries_ptr += dim * query_id`), that padded buffer can't be batched, so the search then runs **one plan call per
query**. Queries passed with CAGRA row padding, which `search_main` accepts and the iterative graph build uses, take
the same per-query loop.

Per-query plan calls are very expensive:

* SINGLE_CTA launches 1-block grids, one per query.
* MULTI_KERNEL runs `n_queries × ~itopk iterations × (4 kernels + a blocking D2H of terminate_flag + sync)`
  (`search_multi_kernel.cuh:586-592`).
* The iterative CAGRA graph build searches every row of the dataset (8192-row chunks) through this path, and so does
  `add_nodes` / `extend`.

Nsight Systems profiles of the CAGRA gtests attribute the following time to the fallback: about 54 s in
`NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST`, about 50 s in `_UINT8_`, about 51 s in `_INT8_` (dim 8 is unaligned for
1-byte types) and about 30 s in `_HALF_`. The examples:

* In half, an unaligned iterative-build case spends 137 ms in search, against 1.3 ms when aligned.
* FilterTest's MULTI_KERNEL cases with 100 queries run 826k iterations in int8.

Users with dims such as 100, 102, 200 (uint8) or 769 (float) pay the same cost on every search.

**Why the restriction isn't needed**

The comment says dense rows "can be misaligned between rows and trigger misaligned access in CAGRA search". But the
only device code that reads the query buffer is `setup_workspace_standard_impl` / `setup_workspace_vpq_impl`
(`jit_lto_kernels/setup_workspace_impl.cuh:53-60,143-163`), and both use scalar `DATA_T` loads. The 16 B vector loads
are on the dataset rows, which do need the padding. There are two further signs that dense queries are safe:

* Before #1846 every dim was searched in batches straight from the caller's buffer.
* `search_multi_partition` still passes dense `[n, dim]` queries to the same kernels in batches, including the dim-8
  int8/uint8 multi-partition tests.

**Proposal**

Search dense queries in place, in batches, for any dim. For CAGRA-padded queries, pack each `max_queries` batch into
a dense workspace buffer with `raft::copy_matrix` and search the batch. Kernels and JIT fragments don't change.
Alternatively, the query leading dimension could be passed through the `setup_workspace` fragment and every search
kernel. That avoids the pack copy for padded input, but it changes the JIT fragment signatures across about 20 files
in exchange for skipping one small D2D copy.
