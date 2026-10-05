#!/bin/bash
# Build the CPU-only probe against a source tree that contains util/kvikio_io.hpp.
# Usage: build.sh <cpp/src dir of the tree to test> <output name>
#   e.g. build.sh /path/to/unpatched/cpp/src probe_before
#        build.sh /path/to/patched/cpp/src   probe
# Paths below match the devcontainer this was run in; adjust as needed.
set -e
cd "$(dirname "$0")"
HDR_ROOT=${1:?cpp/src directory}
OUT=${2:-probe}
INC="-I${HDR_ROOT} -I/home/coder/cuvs/cpp/include -isystem /home/coder/raft/cpp/include -isystem /home/coder/raft/cpp/build/conda/cuda-13.3/release/include -isystem /home/coder/rmm/cpp/include -isystem /home/coder/rmm/cpp/build/conda/cuda-13.3/release/include -isystem /home/coder/kvikio/cpp/include -isystem /home/coder/kvikio/cpp/build/conda/cuda-13.3/release/include -isystem /home/coder/kvikio/cpp/build/conda/cuda-13.3/release/_deps/bs_thread_pool-src/include -isystem /home/coder/.conda/envs/rapids/include -I/home/coder/.conda/envs/rapids/targets/x86_64-linux/include -I/home/coder/.conda/envs/rapids/targets/x86_64-linux/include/cccl"
DEFS="-DBS_THREAD_POOL_ENABLE_PAUSE=1 -DKVIKIO_CUFILE_FOUND -DKVIKIO_CUFILE_VERSION_API_FOUND -DKVIKIO_LIBCURL_FOUND -DRAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO"
KLIB=/home/coder/kvikio/cpp/build/conda/cuda-13.3/release
CLIB=/home/coder/.conda/envs/rapids/lib
nice -n 19 /home/coder/.conda/envs/rapids/bin/x86_64-conda-linux-gnu-c++ -std=gnu++20 -O1 -Wall -Werror $DEFS $INC probe.cpp -o "$OUT" \
  -L$KLIB -lkvikio -L$CLIB -lrapids_logger -Wl,-rpath,$KLIB -Wl,-rpath,$CLIB -pthread

# Stand-ins used with LD_PRELOAD (never touch the GPU):
#   libcufile.so.0 : records cuFile calls and fails registration
#   no_cufile.so   : makes dlopen("libcufile.so.0") fail
nice -n 19 gcc -shared -fPIC -O1 -w -Wl,-soname,libcufile.so.0 stub_cufile.c -o libcufile.so.0
nice -n 19 gcc -shared -fPIC -O1 -w no_cufile.c -o no_cufile.so -ldl

# Run, for example:
#   CUDA_VISIBLE_DEVICES= KVIKIO_COMPAT_MODE=ON LD_PRELOAD=$PWD/libcufile.so.0 ./probe $TMPDIR/f.bin
