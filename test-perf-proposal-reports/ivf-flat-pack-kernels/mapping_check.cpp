// Host-only check that the new IVF-Flat pack/unpack kernels (one thread per element) copy exactly
// the same (src -> dst) element pairs as the old ones (one thread per row).
//
// The kernel bodies below are transcribed from cpp/src/neighbors/ivf_flat/ivf_flat_helpers.cuh
// (old: before the change; new: after the change), with blockIdx/threadIdx turned into loops over
// the launch configuration. Every "thread" is executed sequentially.
//
// For each case the source buffer holds unique values (its own index) and the destination buffer
// is pre-filled with a sentinel. The check asserts that
//   * the old and the new kernels leave bit-identical destination buffers, and
//   * neither kernel reads or writes out of bounds, and neither writes any element twice.
// Unique source values + identical destinations + no double writes means that the new kernel
// writes exactly the same (src -> dst) pairs, so the GPU result does not depend on thread order.
//
// Build and run:  g++ -O2 -std=c++17 -o mapping_check mapping_check.cpp && ./mapping_check

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kIndexGroupSize = 32;
constexpr uint32_t kPackBlockSize  = 256;
constexpr uint32_t kPackMaxGridY   = 65535;
constexpr uint32_t kSentinel       = 0xFFFFFFFFu;

uint32_t round_down(uint32_t x) { return x & ~(kIndexGroupSize - 1); }
uint32_t mod(uint32_t x) { return x & (kIndexGroupSize - 1); }
uint32_t div_up(uint32_t a, uint32_t b) { return a / b + (a % b != 0); }

struct dim3_t {
  uint32_t x, y;
};

// Position source: either `offset + row` or `indices[row]`.
struct positions {
  bool use_indices;
  uint32_t offset;
  const std::vector<uint32_t>* indices;
  uint32_t operator()(uint32_t row) const { return use_indices ? (*indices)[row] : offset + row; }
};

// A memory buffer that records out-of-bounds accesses and double writes.
struct buffer {
  std::vector<uint32_t> data;
  std::vector<uint8_t> writes;
  bool oob = false;
  bool double_write = false;

  uint32_t read(size_t i)
  {
    if (i >= data.size()) {
      oob = true;
      return 0;
    }
    return data[i];
  }
  void write(size_t i, uint32_t v)
  {
    if (i >= data.size()) {
      oob = true;
      return;
    }
    if (writes[i]++) { double_write = true; }
    data[i] = v;
  }
};

buffer make_src(size_t n)
{
  buffer b;
  b.data.resize(n);
  std::iota(b.data.begin(), b.data.end(), 0u);
  b.writes.assign(n, 0);
  return b;
}

buffer make_dst(size_t n)
{
  buffer b;
  b.data.assign(n, kSentinel);
  b.writes.assign(n, 0);
  return b;
}

// ---------------------------------------------------------------- old kernels (one thread per row)
// Launch: blocks = ceil(n_rows / 256), threads = 256. Arithmetic in uint32_t as in the original.

void old_pack(buffer& codes,
              buffer& list_data,
              uint32_t n_rows,
              uint32_t dim,
              uint32_t veclen,
              const positions& pos)
{
  uint32_t blocks = div_up(n_rows, kPackBlockSize);
  for (uint32_t bx = 0; bx < blocks; bx++) {
    for (uint32_t tx = 0; tx < kPackBlockSize; tx++) {
      uint32_t tid = bx * kPackBlockSize + tx;
      if (tid >= n_rows) continue;  // the original reads indices[tid] before this check
      const uint32_t offset = pos(tid);
      // pack_1(codes + tid * dim, list_data, dim, veclen, offset)
      const uint32_t flat_code    = tid * dim;
      const uint32_t group_offset = round_down(offset);
      const uint32_t ingroup_id   = mod(offset) * veclen;
      for (uint32_t l = 0; l < dim; l += veclen) {
        for (uint32_t j = 0; j < veclen; j++) {
          list_data.write(group_offset * dim + l * kIndexGroupSize + ingroup_id + j,
                          codes.read(flat_code + l + j));
        }
      }
    }
  }
}

void old_unpack(buffer& list_data,
                buffer& codes,
                uint32_t n_rows,
                uint32_t dim,
                uint32_t veclen,
                const positions& pos)
{
  uint32_t blocks = div_up(n_rows, kPackBlockSize);
  for (uint32_t bx = 0; bx < blocks; bx++) {
    for (uint32_t tx = 0; tx < kPackBlockSize; tx++) {
      uint32_t tid = bx * kPackBlockSize + tx;
      if (tid >= n_rows) continue;
      const uint32_t offset = pos(tid);
      // unpack_1(list_data, codes + tid * dim, dim, veclen, offset)
      const uint32_t flat_code    = tid * dim;
      const uint32_t group_offset = round_down(offset);
      const uint32_t ingroup_id   = mod(offset) * veclen;
      for (uint32_t l = 0; l < dim; l += veclen) {
        for (uint32_t j = 0; j < veclen; j++) {
          codes.write(flat_code + l + j,
                      list_data.read(group_offset * dim + l * kIndexGroupSize + ingroup_id + j));
        }
      }
    }
  }
}

// ---------------------------------------------------------- new kernels (one thread per element)

// pack_launch_config(); `max_grid_y` defaults to kPackMaxGridY, smaller values exercise the
// grid-stride loop over row groups.
void new_launch_config(
  uint32_t n_rows, uint32_t dim, uint32_t max_grid_y, dim3_t& blocks, dim3_t& threads)
{
  const uint32_t group_elems = kIndexGroupSize * dim;
  const uint32_t n_groups    = div_up(n_rows, kIndexGroupSize);
  threads                    = {std::min(kPackBlockSize, group_elems), 1};
  blocks                     = {div_up(group_elems, threads.x), std::min(n_groups, max_grid_y)};
}

size_t interleaved_offset(uint32_t pos, uint32_t dim, uint32_t veclen, uint32_t l, uint32_t j)
{
  return size_t(round_down(pos)) * dim + l * kIndexGroupSize + mod(pos) * veclen + j;
}

void new_pack(buffer& codes,
              buffer& list_data,
              uint32_t n_rows,
              uint32_t dim,
              uint32_t veclen,
              const positions& pos,
              uint32_t max_grid_y)
{
  dim3_t blocks, threads;
  new_launch_config(n_rows, dim, max_grid_y, blocks, threads);
  for (uint32_t by = 0; by < blocks.y; by++) {
    for (uint32_t bx = 0; bx < blocks.x; bx++) {
      for (uint32_t tx = 0; tx < threads.x; tx++) {
        // ---- kernel body
        const uint32_t ix = bx * threads.x + tx;
        if (ix >= kIndexGroupSize * dim) { continue; }
        const uint32_t chunk_size   = kIndexGroupSize * veclen;
        const uint32_t chunk_ix     = ix / chunk_size;
        const uint32_t in_chunk     = ix - chunk_ix * chunk_size;
        const uint32_t row_in_group = in_chunk / veclen;
        const uint32_t j            = in_chunk - row_in_group * veclen;
        const uint32_t l            = chunk_ix * veclen;
        const uint32_t n_groups     = div_up(n_rows, kIndexGroupSize);
        for (uint32_t group = by; group < n_groups; group += blocks.y) {
          const uint32_t row = group * kIndexGroupSize + row_in_group;
          if (row < n_rows) {
            const uint32_t dst_ix = pos(row);
            list_data.write(interleaved_offset(dst_ix, dim, veclen, l, j),
                            codes.read(size_t(row) * dim + l + j));
          }
        }
        // ---- end of kernel body
      }
    }
  }
}

void new_unpack(buffer& list_data,
                buffer& codes,
                uint32_t n_rows,
                uint32_t dim,
                uint32_t veclen,
                const positions& pos,
                uint32_t max_grid_y)
{
  dim3_t blocks, threads;
  new_launch_config(n_rows, dim, max_grid_y, blocks, threads);
  for (uint32_t by = 0; by < blocks.y; by++) {
    for (uint32_t bx = 0; bx < blocks.x; bx++) {
      for (uint32_t tx = 0; tx < threads.x; tx++) {
        // ---- kernel body
        const uint32_t ix = bx * threads.x + tx;
        if (ix >= kIndexGroupSize * dim) { continue; }
        const uint32_t row_in_group = ix / dim;
        const uint32_t c            = ix - row_in_group * dim;
        const uint32_t j            = c % veclen;
        const uint32_t l            = c - j;
        const uint32_t n_groups     = div_up(n_rows, kIndexGroupSize);
        for (uint32_t group = by; group < n_groups; group += blocks.y) {
          const uint32_t row = group * kIndexGroupSize + row_in_group;
          if (row < n_rows) {
            const uint32_t src_ix = pos(row);
            codes.write(size_t(row) * dim + c,
                        list_data.read(interleaved_offset(src_ix, dim, veclen, l, j)));
          }
        }
        // ---- end of kernel body
      }
    }
  }
}

// ------------------------------------------------------------------------------------ driver

struct stats {
  size_t cases    = 0;
  size_t elements = 0;
  size_t failures = 0;
};

bool same(const buffer& a, const buffer& b)
{
  return a.data == b.data && a.writes == b.writes && !a.oob && !b.oob && !a.double_write &&
         !b.double_write;
}

void check_case(stats& st,
                uint32_t n_rows,
                uint32_t dim,
                uint32_t veclen,
                const positions& pos,
                uint32_t list_rows,
                uint32_t max_grid_y,
                const std::string& what)
{
  const size_t flat_size = size_t(n_rows) * dim;
  const size_t list_size = size_t(list_rows) * dim;

  // pack: codes -> list
  {
    buffer src_old = make_src(flat_size), src_new = make_src(flat_size);
    buffer dst_old = make_dst(list_size), dst_new = make_dst(list_size);
    old_pack(src_old, dst_old, n_rows, dim, veclen, pos);
    new_pack(src_new, dst_new, n_rows, dim, veclen, pos, max_grid_y);
    size_t written = std::count(dst_new.writes.begin(), dst_new.writes.end(), 1);
    if (!same(dst_old, dst_new) || src_old.oob || src_new.oob || written != flat_size) {
      st.failures++;
      std::printf("FAIL pack   %s\n", what.c_str());
    }
  }
  // unpack: list -> codes
  {
    buffer src_old = make_src(list_size), src_new = make_src(list_size);
    buffer dst_old = make_dst(flat_size), dst_new = make_dst(flat_size);
    old_unpack(src_old, dst_old, n_rows, dim, veclen, pos);
    new_unpack(src_new, dst_new, n_rows, dim, veclen, pos, max_grid_y);
    size_t written = std::count(dst_new.writes.begin(), dst_new.writes.end(), 1);
    if (!same(dst_old, dst_new) || src_old.oob || src_new.oob || written != flat_size) {
      st.failures++;
      std::printf("FAIL unpack %s\n", what.c_str());
    }
  }
  st.cases += 2;
  st.elements += 2 * flat_size;
}

}  // namespace

int main()
{
  const std::vector<uint32_t> dims = {1,  2,  3,   4,   5,   6,   8,    12,   16,   17,   24,  32, 33,
                                      48, 64, 100, 128, 136, 256, 1000, 1024, 2048, 2049, 2056, 4096};
  const std::vector<uint32_t> n_rows_list = {1, 2, 7, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 3000};
  const std::vector<uint32_t> offsets     = {0, 1, 5, 31, 32, 33, 42, 64, 100};
  const std::vector<uint32_t> veclens     = {1, 2, 4, 8, 16};
  const std::vector<uint32_t> grid_ys     = {kPackMaxGridY, 1, 3};
  constexpr size_t kMaxElements           = size_t(1) << 19;

  std::mt19937 rng(42);
  stats st;
  for (uint32_t dim : dims) {
    for (uint32_t veclen : veclens) {
      if (dim % veclen != 0) continue;
      for (uint32_t n_rows : n_rows_list) {
        if (size_t(n_rows) * dim > kMaxElements) continue;
        for (uint32_t max_grid_y : grid_ys) {
          // contiguous positions: offset + row
          for (uint32_t offset : offsets) {
            const uint32_t list_rows = div_up(offset + n_rows, kIndexGroupSize) * kIndexGroupSize;
            positions pos{false, offset, nullptr};
            check_case(st,
                       n_rows,
                       dim,
                       veclen,
                       pos,
                       list_rows,
                       max_grid_y,
                       "dim=" + std::to_string(dim) + " veclen=" + std::to_string(veclen) +
                         " n_rows=" + std::to_string(n_rows) + " offset=" + std::to_string(offset) +
                         " grid_y<=" + std::to_string(max_grid_y));
          }
          // explicit positions: a random injective map rows -> [0, list_rows)
          {
            const uint32_t list_rows =
              div_up(n_rows + n_rows / 2 + 5, kIndexGroupSize) * kIndexGroupSize;
            std::vector<uint32_t> all(list_rows);
            std::iota(all.begin(), all.end(), 0u);
            std::shuffle(all.begin(), all.end(), rng);
            std::vector<uint32_t> indices(all.begin(), all.begin() + n_rows);
            positions pos{true, 0, &indices};
            check_case(st,
                       n_rows,
                       dim,
                       veclen,
                       pos,
                       list_rows,
                       max_grid_y,
                       "dim=" + std::to_string(dim) + " veclen=" + std::to_string(veclen) +
                         " n_rows=" + std::to_string(n_rows) +
                         " indices grid_y<=" + std::to_string(max_grid_y));
          }
        }
      }
    }
  }
  std::printf("%zu cases (%zu pack + %zu unpack), %zu elements copied per variant, %zu failures\n",
              st.cases,
              st.cases / 2,
              st.cases / 2,
              st.elements,
              st.failures);
  return st.failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
