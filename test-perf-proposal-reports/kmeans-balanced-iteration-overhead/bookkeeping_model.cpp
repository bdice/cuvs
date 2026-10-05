// Host-only model of balancing_em_iters' Random-donor bookkeeping: original (sync per call) vs.
// deferred (records on "device", accounted when the loop is about to end).
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
using IdxT = int64_t;
template <typename I>
I next_donor_seed(I& i_primes, I n_rows) {
  constexpr static std::array kPrimes{29,   71,   113,  173,  229,  281,  349,  409,  463,  541,
                                      601,  659,  733,  809,  863,  941,  1013, 1069, 1151, 1223,
                                      1291, 1373, 1451, 1511, 1583, 1657, 1733, 1811, 1889, 1987,
                                      2053, 2129, 2213, 2287, 2357, 2423, 2531, 2617, 2687, 2741};
  I ofst;
  do { i_primes = (i_primes + 1) % kPrimes.size(); ofst = kPrimes[i_primes]; } while (n_rows % ofst == 0);
  return ofst;
}
static uint64_t H(uint64_t a, uint64_t b) { a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2); a *= 0xff51afd7ed558ccdULL; return a ^ (a >> 33); }
struct Model { uint64_t p_unb, p_upd; };  // outcome probabilities (per 100)
struct Out { bool unb; bool upd; };
// outcome of a balancing call given the current state and the seed it would use
static Out outcome(const Model& m, uint64_t state, IdxT seed) {
  bool unb = H(state, 1) % 100 < m.p_unb;
  bool under = unb && (H(state, 2) % 100 < m.p_upd);  // some underfull cluster
  bool upd = under && (H(state, seed) % 7 != 0);        // donor found depends on the seed
  return {unb, upd};
}
struct Result { std::vector<IdxT> seeds_used; uint32_t n_iters; IdxT i_primes; uint64_t state; };
Result original(const Model& m, uint32_t n_iters, uint32_t pullback, IdxT n_rows, IdxT i_primes, uint64_t state) {
  Result r{};
  uint32_t counter = pullback;
  for (uint32_t iter = 0; iter < n_iters; iter++) {
    if (iter > 0) {
      bool updated = false;
      bool unb = H(state, 1) % 100 < m.p_unb;
      if (unb) {  // n_pairs > 0
        IdxT seed = next_donor_seed(i_primes, n_rows);
        Out o = outcome(m, state, seed);
        updated = o.upd;
        r.seeds_used.push_back(seed);
        if (updated) state = H(state, seed);
      }
      if (updated) { if (counter++ >= pullback) { counter -= pullback; n_iters++; } }
    }
    state = H(state, 12345 + iter);  // E/M
  }
  r.n_iters = n_iters; r.i_primes = i_primes; r.state = state; return r;
}
Result deferred(const Model& m, uint32_t n_iters, uint32_t pullback, IdxT n_rows, IdxT& i_primes_ref, uint64_t state) {
  Result r{};
  constexpr uint32_t K = 3;
  IdxT seeds_i_primes = i_primes_ref;
  std::vector<IdxT> host_seeds, seeds, records;
  auto reserve = [&](uint32_t n) {
    if (host_seeds.size() < n) { while (host_seeds.size() < n) host_seeds.push_back(next_donor_seed(seeds_i_primes, n_rows)); seeds = host_seeds; }
    if (records.size() < K * size_t(n)) records.resize(K * size_t(n), 0);
  };
  reserve(n_iters);
  uint32_t counter = pullback, n_acc = 1;
  for (uint32_t iter = 0; iter < n_iters; iter++) {
    if (iter > 0) {  // "kernel"
      IdxT* rec = records.data() + K * iter;
      bool unb = H(state, 1) % 100 < m.p_unb;
      if (unb) {
        rec[2] = 1;
        bool under = H(state, 2) % 100 < m.p_upd;
        if (under) {
          IdxT n_used = 0;
          for (uint32_t p = 0; p < iter; p++) if (records[K * p + 2] != 0) n_used++;
          IdxT seed = seeds.at(n_used);
          Out o = outcome(m, state, seed);
          r.seeds_used.push_back(seed);
          if (o.upd) { rec[1] = 1; state = H(state, seed); }
        } else {
          // the original consumed a seed here too (n_pairs > 0); record it for comparison
          IdxT n_used = 0;
          for (uint32_t p = 0; p < iter; p++) if (records[K * p + 2] != 0) n_used++;
          r.seeds_used.push_back(seeds.at(n_used));
        }
      }
    }
    state = H(state, 12345 + iter);
    if (iter + 1 == n_iters && n_acc < n_iters) {
      uint32_t n_pending = n_iters - n_acc;
      std::vector<IdxT> hr(records.begin() + K * n_acc, records.begin() + K * (n_acc + n_pending));
      n_acc = n_iters;
      for (uint32_t k = 0; k < n_pending; k++) {
        const IdxT* rec = hr.data() + K * k;
        if (rec[2] != 0) next_donor_seed(i_primes_ref, n_rows);
        if (rec[1] > 0 && counter++ >= pullback) { counter -= pullback; n_iters++; }
      }
      if (n_iters > n_acc) reserve(n_iters);
    }
  }
  r.n_iters = n_iters; r.i_primes = i_primes_ref; r.state = state; return r;
}
int main() {
  long cases = 0, bad = 0, extended = 0;
  for (uint64_t p_unb : {0, 5, 20, 50, 80, 100})
    for (uint64_t p_upd : {0, 30, 70, 100})
      for (uint32_t n_iters : {0u, 1u, 2u, 3u, 10u, 20u, 25u})
        for (uint32_t pullback : {2u, 5u})
          for (IdxT n_rows : {IdxT(1), IdxT(29), IdxT(29 * 71), IdxT(65536), IdxT(2 * 3 * 29 * 71 * 113)})
            for (uint64_t s0 = 0; s0 < 20; s0++)
              for (IdxT ip0 : {IdxT(0), IdxT(17), IdxT(39)}) {
                Model m{p_unb, p_upd};
                Result a = original(m, n_iters, pullback, n_rows, ip0, s0 * 7919);
                IdxT ip = ip0;
                Result b = deferred(m, n_iters, pullback, n_rows, ip, s0 * 7919);
                cases++;
                if (b.n_iters > n_iters) extended++;
                if (a.seeds_used != b.seeds_used || a.n_iters != b.n_iters || a.i_primes != b.i_primes || a.state != b.state) {
                  if (bad++ < 5) printf("MISMATCH p_unb=%lu p_upd=%lu n_iters=%u pb=%u n_rows=%ld s0=%lu\n", p_unb, p_upd, n_iters, pullback, (long)n_rows, s0);
                }
              }
  printf("cases=%ld mismatches=%ld (cases with added iterations: %ld)\n", cases, bad, extended);
  return bad != 0;
}
