#!/bin/bash
# c4ef930b (sync-free IVF checks) only applies on top of the IVF-PQ / IVF-Flat test-reuse commits.
# Build prereq = base + 9c688a3c + afe7630e, then prereq + c4ef930b; snapshot both.
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
B=/home/coder/cuvs/cpp/build/latest
BASE=$(git -C /home/coder/cuvs rev-parse staging-test-optimizations-local)
cd /home/coder/cuvs
. $J/deps_guard.sh
eval "$(sed -n '/^snap() {/,/^}/p; /^build() {/,/^}/p' $J/build_variants.sh)"
export SCCACHE_SERVER_PORT=4227 SCCACHE_NO_DIST_COMPILE=1 SCCACHE_ERROR_LOG=$J/sccache_private.log
echo "start $(date +%T)"
git checkout -q --detach $BASE && git cherry-pick --no-edit 9c688a3c afe7630e > $J/cp_sf_pre.log 2>&1 || { echo "prereq pick failed"; git cherry-pick --abort; exit 1; }
[ -f $V/sf_pre/libcuvs.so ] || { build sf_pre && snap sf_pre && echo "snap sf_pre"; }
git cherry-pick --no-edit c4ef930b > $J/cp_sf_chain.log 2>&1 || { echo "syncfree pick failed"; git cherry-pick --abort; exit 1; }
[ -f $V/sf_chain/libcuvs.so ] || { build sf_chain && snap sf_chain && echo "snap sf_chain"; }
deps_check
git checkout -q test-perf-proposals
echo "end $(date +%T)"
