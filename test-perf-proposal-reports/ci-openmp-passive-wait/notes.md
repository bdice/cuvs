# ci-openmp-passive-wait: run the C++ tests with `OMP_WAIT_POLICY=passive`

Change: `ci/test_cpp.sh` exports `OMP_WAIT_POLICY=passive` before running `ctest -j8`. No library or test code changes.

## Why

* The conda environment's OpenMP runtime is LLVM libomp. `_openmp_mutex` is the `*_kmp_llvm` build, and `libgomp.so.1`
  is a symlink to `libomp.so`. The local dev environment has this; the CI test environment probably does too, because
  `mkl` / `llvm-openmp` are in the solve. See "GNU libgomp" below for the other case.
* libomp keeps idle workers spinning for `KMP_BLOCKTIME` (default 200 ms) after every parallel region. The test
  executables alternate short OpenMP regions (NN-descent host loops, HNSW, test references) with GPU work, so the
  workers of every process spin on all cores most of the time.
* Under `ctest -j8`, up to 8 such processes each have a 36-thread team on a 36-thread host. Spinning workers of one
  process take cores from the busy threads of the others.
* `OMP_WAIT_POLICY=passive` makes libomp default to `KMP_BLOCKTIME=0`, so workers sleep right after a region. This was
  checked with `KMP_SETTINGS=1`: the default prints `KMP_BLOCKTIME=200ms`, while `OMP_WAIT_POLICY=passive` prints
  `KMP_BLOCKTIME=0ms`. Under GNU libgomp, the same variable disables spinning (`GOMP_SPINCOUNT` 0).

## Earlier evidence (unpatched NN-descent, single process)

Filter `AnnNNDescentTest/AnnNNDescentTestI8_U32.AnnNNDescent/2??:AnnNNDescentBbqTest/*`, 2 repetitions:

| setting | wall | CPU (user+sys) |
|---|---|---|
| default (`KMP_BLOCKTIME=200ms`) | 16.6 / 16.8 s | 579 / 589 s |
| `KMP_BLOCKTIME=0` | 14.6 / 14.9 s | 141 / 141 s |

That run predates removing the per-iteration NN-descent thread (`../nn-descent-host-loop/`), which removed most of the
two-team contention, so the remaining single-process gain is expected to be smaller. The CPU saving matters most
under `ctest -j8`.

## Outcome

Rejected. See `rejected.md`: the suite got about 1% slower (no test concurrency today, so there is no contention to
remove), and the NN-descent-style host loops got 15–37% slower from waking sleeping workers.
