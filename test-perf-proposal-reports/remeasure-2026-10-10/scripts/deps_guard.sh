# Sourced by the pipeline scripts: fingerprint the local raft/rmm checkouts and libraries.
deps_fp() {
  for r in raft rmm; do
    git -C /home/coder/$r rev-parse HEAD
    git -C /home/coder/$r status --porcelain --untracked-files=no | md5sum
  done
  stat -c '%Y %s' /home/coder/rmm/cpp/build/conda/cuda-13.3/release/librmm.so /home/coder/raft/cpp/build/conda/cuda-13.3/release/libraft.so 2>/dev/null
}
deps_check() {  # exit if the dependencies changed since $DEPS_REF was recorded
  if [ "$(deps_fp)" != "$(cat $J/deps_ref.txt)" ]; then echo "DEPS CHANGED at $(date +%T); stopping"; exit 3; fi
}
