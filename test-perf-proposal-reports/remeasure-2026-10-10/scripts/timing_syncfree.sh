#!/bin/bash
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
S=/tmp/claude-1000/-home-coder/ae252f72-9cc0-46ee-a543-8482b2c783fe/scratchpad
export KVIKIO_COMPAT_MODE=ON
. $J/deps_guard.sh; deps_check
echo "start $(date +%T)"
for exe in NEIGHBORS_ANN_IVF_PQ_TEST NEIGHBORS_ANN_IVF_FLAT_TEST; do
  $S/measure_ab2.sh $J/ab_syncfree 3 "sf_pre=$V/sf_pre sf_chain=$V/sf_chain" $exe > /dev/null
  echo "done $exe $(date +%T)"
done
echo "end $(date +%T)"
