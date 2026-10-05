/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include "../util/kvikio_serialize.hpp"
#include "ivf_common.cuh"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cuvs::neighbors::ivf {

template <template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
list<SpecT, SizeT, SpecExtraArgs...>::list(raft::resources const& res,
                                           const spec_type& spec,
                                           size_type n_rows)
  : size{n_rows}, data{res}, indices{res}
{
  auto capacity = raft::round_up_safe<SizeT>(n_rows, spec.align_max);
  if (n_rows < spec.align_max) {
    capacity = raft::bound_by_power_of_two<SizeT>(std::max<SizeT>(n_rows, spec.align_min));
    capacity = std::min<SizeT>(capacity, spec.align_max);
  }
  try {
    data    = raft::make_device_mdarray<value_type>(res, spec.make_list_extents(capacity));
    indices = raft::make_device_vector<index_type, SizeT>(res, capacity);
  } catch (std::bad_alloc& e) {
    RAFT_FAIL(
      "ivf::list: failed to allocate a big enough list to hold all data "
      "(requested size: %zu records, selected capacity: %zu records). "
      "Allocator exception: %s",
      size_t(size),
      size_t(capacity),
      e.what());
  }
  // Fill the index buffer with a pre-defined marker for easier debugging
  raft::matrix::fill(res, indices.view(), ivf::kInvalidRecord<index_type>);
}

template <typename ListT>
CUVS_EXPORT void resize_list(raft::resources const& res,
                             std::shared_ptr<ListT>& orig_list,  // NOLINT
                             const typename ListT::spec_type& spec,
                             typename ListT::size_type new_used_size,
                             typename ListT::size_type old_used_size)
{
  bool skip_resize = false;
  if (orig_list) {
    if (new_used_size <= orig_list->indices.extent(0)) {
      auto shared_list_size = old_used_size;
      if (new_used_size <= old_used_size ||
          orig_list->size.compare_exchange_strong(shared_list_size, new_used_size)) {
        // We don't need to resize the list if:
        //  1. The list exists
        //  2. The new size fits in the list
        //  3. The list doesn't grow or no-one else has grown it yet
        skip_resize = true;
      }
    }
  } else {
    old_used_size = 0;
  }
  if (skip_resize) { return; }
  auto new_list = std::make_shared<ListT>(res, spec, new_used_size);
  if (old_used_size > 0) {
    auto copied_data_extents = spec.make_list_extents(old_used_size);
    auto copied_view         = raft::make_mdspan<typename ListT::value_type,
                                                 typename ListT::size_type,
                                                 raft::row_major,
                                                 false,
                                                 true>(new_list->data.data_handle(), copied_data_extents);
    raft::copy(res,
               raft::make_device_vector_view(copied_view.data_handle(), copied_view.size()),
               raft::make_device_vector_view(orig_list->data.data_handle(), copied_view.size()));
    raft::copy(res,
               raft::make_device_vector_view(new_list->indices.data_handle(), old_used_size),
               raft::make_device_vector_view(orig_list->indices.data_handle(), old_used_size));
  }
  // swap the shared pointer content with the new list
  new_list.swap(orig_list);
}

template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(const raft::resources& handle,
                                             std::ostream& os,
                                             const ListT& ld,
                                             const typename ListT::spec_type& store_spec,
                                             std::optional<typename ListT::size_type> size_override)
{
  if (auto* kvikio_stream = dynamic_cast<cuvs::util::kvikio_ofstream*>(&os);
      kvikio_stream != nullptr) {
    using size_type = typename ListT::size_type;
    const auto size = size_override.value_or(ld.size.load());
    raft::serialize_scalar(handle, *kvikio_stream, size);
    if (size == 0) { return; }

    const auto data_extents = store_spec.make_list_extents(size);
    const auto data_view =
      raft::make_mdspan<const typename ListT::value_type, size_type, raft::row_major, false, true>(
        ld.data.data_handle(), data_extents);
    const auto indices_view = raft::make_device_vector_view(ld.indices.data_handle(), size);
    cuvs::util::detail::serialize_device_mdspan(handle, *kvikio_stream, data_view);
    cuvs::util::detail::serialize_device_mdspan(handle, *kvikio_stream, indices_view);
    return;
  }

  using size_type = typename ListT::size_type;
  auto size       = size_override.value_or(ld.size.load());
  raft::serialize_scalar(handle, os, size);
  if (size == 0) { return; }

  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    raft::make_host_mdarray<typename ListT::value_type, size_type, raft::row_major>(data_extents);
  auto inds_array = raft::make_host_mdarray<typename ListT::index_type, size_type, raft::row_major>(
    raft::make_extents<size_type>(size));
  raft::copy(handle,
             raft::make_host_vector_view(data_array.data_handle(), data_array.size()),
             raft::make_device_vector_view(ld.data.data_handle(), data_array.size()));
  raft::copy(handle,
             raft::make_host_vector_view(inds_array.data_handle(), inds_array.size()),
             raft::make_device_vector_view(ld.indices.data_handle(), inds_array.size()));
  raft::resource::sync_stream(handle);
  raft::serialize_mdspan(handle, os, data_array.view());
  raft::serialize_mdspan(handle, os, inds_array.view());
}

template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(const raft::resources& handle,
                                             std::ostream& os,
                                             const std::shared_ptr<ListT>& ld,
                                             const typename ListT::spec_type& store_spec,
                                             std::optional<typename ListT::size_type> size_override)
{
  if (ld) {
    return serialize_list<ListT>(handle, os, *ld, store_spec, size_override);
  } else {
    return raft::serialize_scalar(handle, os, typename ListT::size_type{0});
  }
}

template <typename ListT>
enable_if_valid_list_t<ListT> deserialize_list(const raft::resources& handle,
                                               std::istream& is,
                                               std::shared_ptr<ListT>& ld,
                                               const typename ListT::spec_type& store_spec,
                                               const typename ListT::spec_type& device_spec)
{
  // Public stream deserializers accept arbitrary streams, which need not be backed by a file that
  // KvikIO can reopen. Keep this host-staged path for those callers; filename overloads use the
  // KvikIO reader overload below.
  using size_type = typename ListT::size_type;
  auto size       = raft::deserialize_scalar<size_type>(handle, is);
  if (size == 0) { return ld.reset(); }
  std::make_shared<ListT>(handle, device_spec, size).swap(ld);
  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    raft::make_host_mdarray<typename ListT::value_type, size_type, raft::row_major>(data_extents);
  auto inds_array = raft::make_host_mdarray<typename ListT::index_type, size_type, raft::row_major>(
    raft::make_extents<size_type>(size));
  raft::deserialize_mdspan(handle, is, data_array.view());
  raft::deserialize_mdspan(handle, is, inds_array.view());
  raft::copy(handle,
             raft::make_device_vector_view(ld->data.data_handle(), data_array.size()),
             raft::make_host_vector_view(data_array.data_handle(), data_array.size()));
  // NB: copying exactly 'size' indices to leave the rest 'kInvalidRecord' intact.
  raft::copy(handle,
             raft::make_device_vector_view(ld->indices.data_handle(), size),
             raft::make_host_vector_view(inds_array.data_handle(), size));
  // Make sure the data is copied from host to device before the host arrays get out of the scope.
  raft::resource::sync_stream(handle);
}

template <typename ListT>
enable_if_valid_list_t<ListT> deserialize_list(const raft::resources& handle,
                                               cuvs::util::kvikio_file_reader& reader,
                                               std::shared_ptr<ListT>& ld,
                                               const typename ListT::spec_type& store_spec,
                                               const typename ListT::spec_type& device_spec)
{
  // Path-backed deserialization reaches this overload and reads list payloads directly into device
  // memory through GDS when available (with KvikIO's compatible I/O fallback otherwise).
  using size_type = typename ListT::size_type;
  auto& is        = reader.stream();
  const auto size = raft::deserialize_scalar<size_type>(handle, is);
  if (size == 0) { return ld.reset(); }

  std::make_shared<ListT>(handle, device_spec, size).swap(ld);

  const auto data_extents = store_spec.make_list_extents(size);
  auto data_view =
    raft::make_mdspan<typename ListT::value_type, size_type, raft::row_major, false, true>(
      ld->data.data_handle(), data_extents);
  auto indices_view = raft::make_device_vector_view(ld->indices.data_handle(), size);
  cuvs::util::detail::deserialize_device_mdspan(handle, reader, data_view);
  cuvs::util::detail::deserialize_device_mdspan(handle, reader, indices_view);
}

namespace detail {

/**
 * Lists whose payload (data + indices) is larger than this are transferred one at a time by
 * serialize_list / deserialize_list, which hand the device pointers to KvikIO (GPUDirect Storage
 * when available). The payloads of smaller lists are batched through a pinned host buffer.
 */
inline constexpr size_t kListDirectIoBytes = size_t{4} << 20;

/** Upper bound of the pinned host buffer used to batch the transfers of the smaller lists. */
inline constexpr size_t kListStagingBytes = cuvs::util::detail::kDeviceSerializationBatchBytes;

constexpr auto mul_saturated(size_t a, size_t b) noexcept -> size_t
{
  return (b != 0 && a > std::numeric_limits<size_t>::max() / b) ? std::numeric_limits<size_t>::max()
                                                                : a * b;
}

/**
 * Bytes of one list record, in the order serialize_list writes them:
 * [head][data payload][mid][indices payload], where `head` is the numpy scalar holding the list
 * size followed (for a non-empty list) by the numpy header of the data, and `mid` is the numpy
 * header of the indices (empty for an empty list).
 */
struct list_record_layout {
  std::string head;
  std::string mid;
  size_t data_bytes    = 0;
  size_t indices_bytes = 0;

  [[nodiscard]] auto payload_bytes() const noexcept -> size_t
  {
    return data_bytes > std::numeric_limits<size_t>::max() - indices_bytes
             ? std::numeric_limits<size_t>::max()
             : data_bytes + indices_bytes;
  }
  /** Whether the list is transferred through the staging buffer (otherwise: one at a time). */
  [[nodiscard]] auto staged() const noexcept -> bool
  {
    return payload_bytes() <= kListDirectIoBytes;
  }
  /** Size of the whole record; only meaningful for staged lists. */
  [[nodiscard]] auto record_bytes() const noexcept -> size_t
  {
    return head.size() + mid.size() + payload_bytes();
  }
};

/**
 * Formats list records exactly as serialize_list does (same numpy writers, same shapes), caching
 * the result by list size. The number of distinct sizes is small in practice: d distinct sizes
 * need at least d * (d - 1) / 2 records in the index.
 */
template <typename ListT>
class list_record_layouts {
 public:
  using size_type  = typename ListT::size_type;
  using value_type = typename ListT::value_type;
  using index_type = typename ListT::index_type;

  list_record_layouts(const raft::resources& res, const typename ListT::spec_type& store_spec)
    : res_{res}, store_spec_{store_spec}
  {
  }

  /** The returned reference stays valid for the lifetime of this object. */
  auto operator()(size_type size) -> const list_record_layout&
  {
    auto it = cache_.find(size);
    if (it == cache_.end()) { it = cache_.emplace(size, make_layout(size)).first; }
    return it->second;
  }

 private:
  [[nodiscard]] auto make_layout(size_type size) const -> list_record_layout
  {
    list_record_layout layout;
    std::ostringstream head;
    raft::serialize_scalar(res_, head, size);
    if (size > 0) {
      const auto extents = store_spec_.make_list_extents(size);
      std::vector<size_t> shape;
      shape.reserve(extents.rank());
      size_t n_elements = 1;
      for (size_t i = 0; i < extents.rank(); ++i) {
        shape.push_back(static_cast<size_t>(extents.extent(i)));
        n_elements = mul_saturated(n_elements, shape.back());
      }
      cuvs::util::detail::write_numpy_header<value_type>(head, shape);
      std::ostringstream mid;
      cuvs::util::detail::write_numpy_header<index_type>(mid, {static_cast<size_t>(size)});
      layout.mid           = mid.str();
      layout.data_bytes    = mul_saturated(n_elements, sizeof(value_type));
      layout.indices_bytes = static_cast<size_t>(size) * sizeof(index_type);
    }
    layout.head = head.str();
    return layout;
  }

  const raft::resources& res_;
  typename ListT::spec_type store_spec_;
  std::unordered_map<size_type, list_record_layout> cache_;
};

/** Waits for the stream when leaving the scope, so that no async copy outlives its host buffer. */
class sync_stream_on_exit {
 public:
  explicit sync_stream_on_exit(cudaStream_t stream) : stream_{stream} {}
  ~sync_stream_on_exit() { (void)cudaStreamSynchronize(stream_); }
  sync_stream_on_exit(const sync_stream_on_exit&)            = delete;
  sync_stream_on_exit& operator=(const sync_stream_on_exit&) = delete;
  sync_stream_on_exit(sync_stream_on_exit&&)                 = delete;
  sync_stream_on_exit& operator=(sync_stream_on_exit&&)      = delete;

 private:
  cudaStream_t stream_;
};

/**
 * Serialize all lists of an index, in order. The output is byte-for-byte the same as calling
 * `serialize_list(handle, os, list, store_spec, sizes(label))` for every non-null list and writing
 * an empty list for every null one.
 *
 * The payloads of consecutive lists are copied into a pinned host buffer (at most
 * kListStagingBytes) with one stream synchronization per batch, and then passed to the stream as
 * plain writes, which a kvikio_ofstream coalesces into a few large writes. Lists with a payload
 * larger than kListDirectIoBytes still go through serialize_list.
 *
 * @param list_at callable `(uint32_t label) -> const ListT*`; nullptr denotes a missing list.
 */
template <typename ListT, typename ListAccessor>
void serialize_lists(const raft::resources& handle,
                     std::ostream& os,
                     const typename ListT::spec_type& store_spec,
                     raft::host_vector_view<const uint32_t, uint32_t> sizes,
                     ListAccessor&& list_at)
{
  using size_type     = typename ListT::size_type;
  const auto n_lists  = sizes.extent(0);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  auto list_size      = [&](uint32_t label) -> size_type {
    return list_at(label) != nullptr ? static_cast<size_type>(sizes(label)) : size_type{0};
  };

  list_record_layouts<ListT> layouts(handle, store_spec);
  size_t staged_bytes = 0;
  for (uint32_t label = 0; label < n_lists; label++) {
    const auto& layout = layouts(list_size(label));
    if (layout.staged()) { staged_bytes += layout.payload_bytes(); }
  }
  auto staging =
    raft::make_pinned_vector<char, size_t>(handle, std::min(staged_bytes, kListStagingBytes));
  sync_stream_on_exit sync_guard{stream};

  uint32_t first_unwritten = 0;  // lists [first_unwritten, label) are staged, but not written yet
  size_t staged            = 0;  // bytes of their payloads in `staging`
  auto write_staged        = [&](uint32_t end) {
    if (staged > 0) { raft::resource::sync_stream(handle); }
    const char* payload = staging.data_handle();
    for (; first_unwritten < end; first_unwritten++) {
      const auto& layout = layouts(list_size(first_unwritten));
      os.write(layout.head.data(), static_cast<std::streamsize>(layout.head.size()));
      if (layout.payload_bytes() == 0) { continue; }
      os.write(payload, static_cast<std::streamsize>(layout.data_bytes));
      payload += layout.data_bytes;
      os.write(layout.mid.data(), static_cast<std::streamsize>(layout.mid.size()));
      os.write(payload, static_cast<std::streamsize>(layout.indices_bytes));
      payload += layout.indices_bytes;
    }
    RAFT_EXPECTS(os.good(), "ivf::serialize_lists: error writing the lists");
    staged = 0;
  };

  for (uint32_t label = 0; label < n_lists; label++) {
    const auto size = list_size(label);
    if (size == 0) { continue; }  // only the size is written, together with the staged lists
    const auto& layout = layouts(size);
    const ListT* list  = list_at(label);
    if (!layout.staged()) {
      write_staged(label);
      ivf::serialize_list<ListT>(handle, os, *list, store_spec, size);
      first_unwritten = label + 1;
      continue;
    }
    if (staged + layout.payload_bytes() > staging.size()) { write_staged(label); }
    char* dst = staging.data_handle() + staged;
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dst, list->data.data_handle(), layout.data_bytes, cudaMemcpyDefault, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(dst + layout.data_bytes,
                                  list->indices.data_handle(),
                                  layout.indices_bytes,
                                  cudaMemcpyDefault,
                                  stream));
    staged += layout.payload_bytes();
  }
  write_staged(n_lists);
}

/**
 * Deserialize all lists of an index from an arbitrary input stream, one list at a time.
 *
 * @param assign callable `(uint32_t label, std::shared_ptr<ListT> list)`; null for empty lists.
 */
template <typename ListT, typename ListAssign>
void deserialize_lists(const raft::resources& handle,
                       std::istream& is,
                       const typename ListT::spec_type& store_spec,
                       const typename ListT::spec_type& device_spec,
                       raft::device_vector_view<const uint32_t, uint32_t> list_sizes,
                       ListAssign&& assign)
{
  for (uint32_t label = 0; label < list_sizes.extent(0); label++) {
    std::shared_ptr<ListT> list;
    ivf::deserialize_list(handle, is, list, store_spec, device_spec);
    assign(label, std::move(list));
  }
}

/**
 * Deserialize all lists of an index from a file. The resulting lists are the same as those of
 * calling `deserialize_list(handle, reader, ...)` for every label.
 *
 * `list_sizes` (already loaded from the file) is used to plan batches of consecutive lists. Each
 * batch is read into a pinned host buffer (at most kListStagingBytes) with one read, every record
 * is checked to be byte-for-byte what serialize_list writes for the expected size, and the payloads
 * are copied to the device without synchronizing per list. Lists with a payload larger than
 * kListDirectIoBytes are read directly into device memory by deserialize_list. If a record differs
 * from the expected bytes (e.g. an inconsistent or truncated file), the remaining lists are parsed
 * by deserialize_list, which accepts or rejects them exactly as before.
 *
 * @param assign callable `(uint32_t label, std::shared_ptr<ListT> list)`; null for empty lists.
 */
template <typename ListT, typename ListAssign>
void deserialize_lists(const raft::resources& handle,
                       cuvs::util::kvikio_file_reader& reader,
                       const typename ListT::spec_type& store_spec,
                       const typename ListT::spec_type& device_spec,
                       raft::device_vector_view<const uint32_t, uint32_t> list_sizes,
                       ListAssign&& assign)
{
  using size_type    = typename ListT::size_type;
  const auto n_lists = list_sizes.extent(0);
  if (n_lists == 0) { return; }
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();

  auto sizes = raft::make_host_vector<uint32_t, uint32_t>(n_lists);
  raft::copy(handle, sizes.view(), list_sizes);
  raft::resource::sync_stream(handle);

  list_record_layouts<ListT> layouts(handle, store_spec);
  auto layout_of = [&](uint32_t label) -> const list_record_layout& {
    return layouts(static_cast<size_type>(sizes(label)));
  };
  size_t staged_bytes = 0;
  for (uint32_t label = 0; label < n_lists; label++) {
    const auto& layout = layout_of(label);
    if (layout.staged()) { staged_bytes += layout.record_bytes(); }
  }

  auto& is      = reader.stream();
  auto position = [&is]() -> size_t {
    const auto pos = is.tellg();
    RAFT_EXPECTS(pos != std::istream::pos_type(-1),
                 "ivf::deserialize_lists: failed to determine the file position");
    return static_cast<size_t>(static_cast<std::streamoff>(pos));
  };
  const auto lists_begin = is.tellg();
  is.seekg(0, std::ios_base::end);
  const size_t file_end = position();
  is.seekg(lists_begin);
  RAFT_EXPECTS(is.good(), "ivf::deserialize_lists: failed to seek in the file");

  auto staging =
    raft::make_pinned_vector<char, size_t>(handle, std::min(staged_bytes, kListStagingBytes));
  sync_stream_on_exit sync_guard{stream};
  bool copies_pending = false;

  auto deserialize_one = [&](uint32_t label) {
    std::shared_ptr<ListT> list;
    ivf::deserialize_list(handle, reader, list, store_spec, device_spec);
    assign(label, std::move(list));
  };

  uint32_t label = 0;
  while (label < n_lists) {
    // Plan a batch of consecutive staged lists [label, batch_end).
    uint32_t batch_end = label;
    size_t batch_bytes = 0;
    for (; batch_end < n_lists; batch_end++) {
      const auto& layout = layout_of(batch_end);
      if (!layout.staged() || batch_bytes + layout.record_bytes() > staging.size()) { break; }
      batch_bytes += layout.record_bytes();
    }
    if (batch_end == label) {
      deserialize_one(label++);
      continue;
    }

    const size_t batch_begin = position();
    const size_t read_bytes  = std::min(batch_bytes, file_end - std::min(file_end, batch_begin));
    if (copies_pending) {
      raft::resource::sync_stream(handle);  // the previous batch is still being copied from
      copies_pending = false;
    }
    // KvikIO reads into host memory with its (multi-threaded) POSIX backend.
    if (read_bytes > 0) { reader.read_device(staging.data_handle(), read_bytes); }

    size_t offset = 0;
    for (; label < batch_end; label++) {
      const auto size    = static_cast<size_type>(sizes(label));
      const auto& layout = layout_of(label);
      if (offset + layout.record_bytes() > read_bytes) { break; }
      const char* head    = staging.data_handle() + offset;
      const char* data    = head + layout.head.size();
      const char* mid     = data + layout.data_bytes;
      const char* indices = mid + layout.mid.size();
      if (std::memcmp(head, layout.head.data(), layout.head.size()) != 0 ||
          std::memcmp(mid, layout.mid.data(), layout.mid.size()) != 0) {
        break;
      }
      std::shared_ptr<ListT> list;
      if (size > 0) {
        list = std::make_shared<ListT>(handle, device_spec, size);
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          list->data.data_handle(), data, layout.data_bytes, cudaMemcpyDefault, stream));
        // NB: copying exactly 'size' indices to leave the rest 'kInvalidRecord' intact.
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          list->indices.data_handle(), indices, layout.indices_bytes, cudaMemcpyDefault, stream));
        copies_pending = true;
      }
      assign(label, std::move(list));
      offset += layout.record_bytes();
    }
    if (offset != read_bytes) { is.seekg(static_cast<std::streamoff>(batch_begin + offset)); }
    if (label < batch_end) {
      // The file does not match the plan; let the per-list parser handle (or reject) the rest.
      while (label < n_lists) {
        deserialize_one(label++);
      }
    }
  }
  raft::resource::sync_stream(handle);
}

}  // namespace detail
}  // namespace cuvs::neighbors::ivf
