#!/bin/bash
# Base rep 1 of each standalone group was the group's first run and paid cold JIT linking for the rebased libcuvs
# (e.g. CAGRA float 310 vs 166 s). Re-time base alone with a warm cache, 2 reps, for every executable timed.
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
S=/tmp/claude-1000/-home-coder/ae252f72-9cc0-46ee-a543-8482b2c783fe/scratchpad
export KVIKIO_COMPAT_MODE=ON
. $J/deps_guard.sh; deps_check
echo "start $(date +%T)"
EXES=$(grep -oE '^group [A-Z0-9_]+' $J/timing_variants.sh | cut -d' ' -f2 | tr '\n' ' ')
rm -rf /tmp/cagra_ace_test_disk /tmp/hnsw_ace_disk_test
$S/measure_ab2.sh $J/ab_recheck 2 "base=$V/base" $EXES > /dev/null
echo "end $(date +%T)"
