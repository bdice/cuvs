#!/bin/bash
# Snapshot the full branch tip (all 21 commits) as variant "tip".
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
B=/home/coder/cuvs/cpp/build/latest
cd /home/coder/cuvs
. $J/deps_guard.sh
eval "$(sed -n '/^snap() {/,/^}/p; /^build() {/,/^}/p' $J/build_variants.sh)"
export SCCACHE_SERVER_PORT=4227 SCCACHE_NO_DIST_COMPILE=1 SCCACHE_ERROR_LOG=$J/sccache_private.log
echo "start $(date +%T)"
git checkout -q test-perf-proposals || exit 1
[ -f $V/tip/libcuvs.so ] || { build tip && snap tip && echo "snap tip $(git rev-parse --short HEAD)"; }
deps_check
echo "end $(date +%T)"
