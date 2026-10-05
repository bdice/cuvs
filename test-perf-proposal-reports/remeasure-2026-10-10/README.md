# Remeasurement on the rebased branch (2026-10-10)

Raw data behind the tables in `../SUMMARY.md`. Base: `staging-test-optimizations-local` 6fbb981c (`upstream/main` f751cadd + #2699).
RTX 6000 Ada, single process unless noted.

* `standalone.tsv`: each remaining commit cherry-picked alone onto the base, timed against the base on every executable it touches
  (2 interleaved repetitions, order reversed in the second; `KVIKIO_COMPAT_MODE=ON`). Columns: rep, exe, variant, wall_s, rc, load1,
  tests, failures, gtest_s. The base's first run in each group had a cold JIT cache; `summarize_standalone.py` drops it.
* `base_recheck.tsv`: 2 more warm-cache runs of the base for every executable above.
* `syncfree_chain.tsv`: `c4ef930b` on top of `9c688a3c` + `afe7630e` (`sf_chain`) against that pair (`sf_pre`), 3 repetitions.
* `suite_runs.txt`: FILTER_UDF cold/warm JIT cache; whole suite with upstream vs per-executable `PERCENT` on the base; base +
  `PERCENT` vs the branch tip; CI's four shards with an empty and then their own JIT cache.
* `ci_shards_f751cadd.txt`: the test list of each CI shard in nightly run 37894118256.
* `summarize_standalone.py`, `shard_impact.py`: the tables in the summary. They read the original layout (`ab_standalone/results.tsv`,
  `ab_recheck/`, `ab_syncfree/`, `ci_logs/shard*.txt`, `suite_std_base_r2.log`); adjust the paths to rerun them on these files.
