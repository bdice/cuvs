#!/bin/bash
# Standalone A/B: each executable against the base and every single-commit variant that touches it.
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
S=/tmp/claude-1000/-home-coder/ae252f72-9cc0-46ee-a543-8482b2c783fe/scratchpad
OUT=$J/ab_standalone
export KVIKIO_COMPAT_MODE=ON
clean() { rm -rf /tmp/cagra_ace_test_disk /tmp/hnsw_ace_disk_test; }
group() {  # exe variant...
  deps_check
  local exe=$1; shift
  local spec="base=$V/base"
  for v in "$@"; do [ -d $V/$v ] && spec="$spec $v=$V/$v"; done
  clean; $S/measure_ab2.sh $OUT 2 "$spec" $exe > /dev/null
  echo "done $exe $(date +%T)"
}
. $J/deps_guard.sh; deps_check
echo "start $(date +%T)"
group NEIGHBORS_ANN_CAGRA_BUGS_TEST bug_repro cagra_batch
group NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST bbq_reuse graph_core
group NEIGHBORS_ANN_CAGRA_MERGE_TEST graph_core cagra_batch
group UTIL_TEST kvikio
group NEIGHBORS_ANN_BRUTE_FORCE_TEST kvikio
group NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST ace_dedupe graph_core kvikio
group NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST ace_dedupe graph_core kvikio hnswhalf
group NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST ace_dedupe graph_core kvikio
group NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST ace_dedupe graph_core kvikio
group NEIGHBORS_HNSW_TEST hnswhalf
group NEIGHBORS_ANN_NN_DESCENT_TEST helpers
group NEIGHBORS_ANN_IVF_RABITQ_TEST helpers kmeans kvikio
group NEIGHBORS_ANN_IVF_SQ_TEST ivfsq_reuse recompute batchio kmeans kvikio
group NEIGHBORS_ANN_SCANN_TEST kmeans
group NEIGHBORS_ALL_NEIGHBORS_TEST kmeans
group PREPROCESSING_TEST kmeans
group CLUSTER_TEST kmeans
group NEIGHBORS_TIERED_INDEX_TEST kmeans cagra_batch
group NEIGHBORS_DYNAMIC_BATCHING_TEST kmeans cagra_batch
group NEIGHBORS_MG_TEST kmeans batchio
group NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST cagra_reuse graph_core cagra_batch kmeans kvikio
group NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST cagra_reuse graph_core cagra_batch kmeans
group NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST cagra_reuse graph_core cagra_batch kmeans
group NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST cagra_reuse graph_core cagra_batch kmeans
group NEIGHBORS_ANN_IVF_FLAT_TEST ivfflat_train helpers recompute syncfree pack batchio kmeans
group NEIGHBORS_ANN_IVF_PQ_TEST ivfpq_reuse helpers kmeans recompute syncfree
group NEIGHBORS_ANN_VAMANA_TEST vamana_test
echo "end $(date +%T)"
