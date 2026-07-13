// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


/*
#include "spyre_address_computation.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "spyre_tensor_impl.h"

namespace spyre {

// Spyre stick size in bytes (Sen1.0).
constexpr int64_t STICK_BYTES = 128;

// ---------------------------------------------------------------------------
// GatherIdxMeta
//
// Mirrors deeptools gather_idx_info for a single torch.gather call.
// All addresses and strides are in units of sticks (not bytes, not elements).
//
//   base_addr           ≡ gii_.base_addr_          (start stick of value tensor)
//   skip_addr[d]        ≡ gii_.skip_addr_[d]        (sticks per unit of dim d)
//   idx_prev_cum_size[d]≡ gii_.idx_prev_cum_size_[d](inner-dim cumulative sizes)
// ---------------------------------------------------------------------------
struct GatherIdxMeta {
  int64_t base_addr;                         // in sticks
  std::vector<int64_t> skip_addr;            // per gather-dim, in sticks
  std::vector<int64_t> idx_prev_cum_size;    // per gather-dim
};

// ---------------------------------------------------------------------------
// build_gather_idx_meta
//
// Computes GatherIdxMeta from the value tensor layout and virtual offset.
// Mirrors deeptools constructDCIGatherIdxDataConvert.
//
// For torch.gather(input, dim, index):
//   - There is exactly ONE gather dimension: `dim`.
//   - The flat index value directly indexes along that dimension.
//   - idx_prev_cum_size[0] = 1  (innermost, no decomposition needed).
//   - skip_addr[0] = input.stride(dim) / elements_per_stick
//
// For multi-dim gather (N gather dimensions encoded as a flat integer):
//   - gather_dims must be provided in innermost-first order.
//   - idx_prev_cum_size[d] = product of input.size(k) for all k inner to d
//                          = input.stride(d)  (element units, row-major)
//   - skip_addr[d] = input.stride(gather_dims[d]) / elements_per_stick
// ---------------------------------------------------------------------------
static GatherIdxMeta build_gather_idx_meta(
    const at::Tensor& value_tensor,
    const std::vector<int64_t>& gather_dims,   // innermost first
    int64_t virtual_offset_bytes) {

  const int64_t ndim = value_tensor.dim();
  const int64_t element_size = value_tensor.element_size();
  const int64_t elements_per_stick = STICK_BYTES / element_size;

  for (int64_t d : gather_dims) {
    TORCH_CHECK(d >= 0 && d < ndim,
                "gather dim ", d, " out of range for ", ndim, "D tensor");
    TORCH_CHECK(value_tensor.stride(d) % elements_per_stick == 0,
                "value_tensor.stride(", d, ") = ", value_tensor.stride(d),
                " is not divisible by elements_per_stick=", elements_per_stick,
                ". Pad the value tensor so stride(", d, ") is a multiple of ",
                elements_per_stick, ".");
  }

  GatherIdxMeta meta;

  // base_addr_ : segment-relative start stick.
  meta.base_addr = virtual_offset_bytes / STICK_BYTES;

  // skip_addr_[d] and idx_prev_cum_size_[d] — innermost first, matching
  // the order deeptools expects when iterating from outer to inner.
  //
  // For a contiguous row-major tensor, input.stride(d) in element units equals
  // the product of all sizes at indices > d (inner dims), which is exactly
  // what deeptools calls the "cumulative size" of those inner positions.
  //
  // idx_prev_cum_size_[d] = input.stride(gather_dims[d])   (element units)
  //   → at innermost d where stride == 1: idx_prev_cum_size = 1
  //
  // skip_addr_[d] = input.stride(gather_dims[d]) / elements_per_stick
  meta.skip_addr.resize(gather_dims.size());
  meta.idx_prev_cum_size.resize(gather_dims.size());

  for (size_t i = 0; i < gather_dims.size(); ++i) {
    int64_t host_stride_elements = value_tensor.stride(gather_dims[i]);
    meta.skip_addr[i] = host_stride_elements / elements_per_stick;
    // idx_prev_cum_size[i] is the number of elements "consumed" by one unit
    // of the *inner* gather dimensions combined. For the innermost gather dim
    // this is 1 by definition (no inner gather dims). For outer dims it equals
    // the element stride of the next-inner gather dim (the cumulative inner
    // product), matching deeptools' idx_prev_cum_size_ semantics.
    if (i == 0) {
      // innermost: no inner gather dims, so cumulative size is 1.
      meta.idx_prev_cum_size[i] = 1;
    } else {
      // outer dim d: idx_prev_cum_size = stride of the gather dim one level
      // inner (gather_dims[i-1]).  This is the number of flat-index units that
      // correspond to one step in the current outer gather dimension.
      meta.idx_prev_cum_size[i] = value_tensor.stride(gather_dims[i - 1]);
    }
  }

  return meta;
}

// ---------------------------------------------------------------------------
// indices_to_addresses_nd
//
// Converts a flat integer index tensor into uint32 HBM stick addresses.
// Implements the deeptools ConvertData_gather_idx algorithm exactly.
//
// Algorithm (mirrors deeptools, iterating from outermost to innermost gather
// dimension):
//
//   addr[j] = base_addr
//
//   For d = outermost ... second-innermost:
//     coord   = indexVal[j] / idx_prev_cum_size[d]
//     addr[j] += coord * skip_addr[d]
//     indexVal[j] -= coord * idx_prev_cum_size[d]
//
//   Innermost (idx_prev_cum_size == 1, no division needed):
//     addr[j] += indexVal[j] * skip_addr[0]
//
// @param indices              Integer tensor (int32 or int64), any shape
// @param value_tensor         The value tensor being indexed; on spyre device
// @param dim                  The single gather dimension (for torch.gather)
// @param virtual_offset_bytes Byte offset of value_tensor start in HBM
// @return                     int32 tensor of stick addresses, same shape as
//                             indices, same device as indices
// ---------------------------------------------------------------------------
at::Tensor indices_to_addresses_nd(
    const at::Tensor& indices,
    const at::Tensor& value_tensor,
    int64_t dim,
    int64_t virtual_offset_bytes) {

  TORCH_CHECK(value_tensor.is_privateuseone(),
              "value_tensor must reside on the spyre device");
  TORCH_CHECK(dim >= 0 && dim < value_tensor.dim(),
              "dim=", dim, " out of range for ", value_tensor.dim(), "D tensor");

  // For a standard torch.gather there is exactly one gather dimension.
  // gather_dims is innermost-first, so a single-dim gather is just {dim}.
  const std::vector<int64_t> gather_dims = {dim};

  // Build GII-equivalent metadata (no data access, compile-time metadata only).
  const GatherIdxMeta meta = build_gather_idx_meta(
      value_tensor, gather_dims, virtual_offset_bytes);

  // Mirror deeptools: bring index tensor to CPU for the host compute step.
  const auto original_device = indices.device();
  const auto indices_cpu = indices.cpu().to(at::kLong);
  const auto indices_shape = indices_cpu.sizes();
  const int64_t num_elems = indices_cpu.numel();
  const auto indices_flat = indices_cpu.reshape({num_elems});
  const auto idx_acc = indices_flat.accessor<int64_t, 1>();

  // Validate bounds.
  const int64_t dim_size = value_tensor.size(dim);
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(idx_acc[j] >= 0 && idx_acc[j] < dim_size,
                "Index value ", idx_acc[j], " at flat position ", j,
                " is out of bounds for gather dim ", dim,
                " with size ", dim_size);
  }

  const size_t num_dims = meta.skip_addr.size();

  // --- Core address computation — exact port of ConvertData_gather_idx ---
  //
  // Step 1: initialise addr[] = base_addr_  and  indexVal[] = input[j]
  std::vector<int64_t> addr(num_elems, meta.base_addr);
  std::vector<int64_t> index_val(num_elems);
  for (int64_t j = 0; j < num_elems; ++j) {
    index_val[j] = idx_acc[j];
  }

  if (num_dims > 0) {
    if (meta.idx_prev_cum_size[0] == 1) {
      // Fast path (standard case): innermost idx_prev_cum_size is 1.
      // Iterate outer → second-innermost, then handle innermost separately.

      // Outer dimensions (ri=0 is outermost, ends at second-innermost).
      for (size_t ri = 0; ri < num_dims - 1; ++ri) {
        const size_t i = num_dims - 1 - ri;   // outermost first
        const int64_t skip          = meta.skip_addr[i];
        const int64_t cum_size      = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }

      // Innermost dimension: idx_prev_cum_size[0] == 1 so coord == index_val.
      const int64_t skip_inner = meta.skip_addr[0];
      for (int64_t j = 0; j < num_elems; ++j) {
        addr[j] += index_val[j] * skip_inner;
      }

    } else {
      // General path: iterate all dimensions outer → inner including innermost.
      for (size_t ri = 0; ri < num_dims; ++ri) {
        const size_t i = num_dims - 1 - ri;
        const int64_t skip     = meta.skip_addr[i];
        const int64_t cum_size = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }
    }
  }

  // --- Write result as int32 (SENUINT32 equivalent) ---
  auto addr_tensor = at::empty(
      {num_elems}, at::TensorOptions().dtype(at::kInt));
  auto addr_acc = addr_tensor.accessor<int32_t, 1>();
  std::cout<<"TANU \n";
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(addr[j] >= 0 &&
                addr[j] <= static_cast<int64_t>(
                    std::numeric_limits<uint32_t>::max()),
                "Computed stick address ", addr[j],
                " does not fit in uint32 for index element ", j);
    std::cout<<"Stick Address : "<<addr[j]<<"\n";
    addr_acc[j] = static_cast<int32_t>(addr[j]);
  }

  // Reshape back to the original indices shape and move to original device.
  // Mirrors deeptools host-to-device transfer step.
  return addr_tensor.reshape(indices_shape).to(original_device);
}

}  // namespace spyre

*/

/*
#include "spyre_address_computation.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <flex/flex.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {

// Spyre stick size in bytes (Sen1.0).
constexpr int64_t STICK_BYTES = 128;

// ---------------------------------------------------------------------------
// get_virtual_offset_bytes
//
// Extracts the absolute HBM byte address of a Spyre tensor's storage from its
// CompositeAddress. This mirrors the deeptools base_addr_ calculation:
//
//   baseVirtAddress = segment_id * SEGMENT_SIZE + offset_within_segment
//   base_addr_      = baseVirtAddress / bytesPerStick
//
// The formula uses flex::SEGMENT_SIZE (the byte size of one HBM segment) and
// LogicalAddress.{region_id, offset} to reconstruct the flat virtual address.
// ---------------------------------------------------------------------------
static int64_t get_virtual_offset_bytes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "get_virtual_offset_bytes: tensor must be on spyre device");
  auto* impl =
      dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(impl != nullptr,
              "get_virtual_offset_bytes: tensor is not a SpyreTensorImpl");

  auto* ctx = static_cast<SharedOwnerCtx*>(
      impl->storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "get_virtual_offset_bytes: null SharedOwnerCtx");

  const auto& chunks = ctx->composite_addr.chunks();
  TORCH_CHECK(!chunks.empty(),
              "get_virtual_offset_bytes: CompositeAddress has no chunks");

  // Use the first chunk — interleaved (multi-chunk) layouts are not supported
  // for gather/scatter index computation (same restriction as deeptools).
  const auto& addr = chunks[0].addr;

  // Reconstruct absolute byte address:
  //   region_id maps to a 128 MB HBM segment; each segment = SEGMENT_SIZE bytes.
  int64_t virtual_addr =
      static_cast<int64_t>(addr.region_id) *
          static_cast<int64_t>(flex::SEGMENT_SIZE) +
      static_cast<int64_t>(addr.offset);

  return virtual_addr;
}

// ---------------------------------------------------------------------------
// GatherIdxMeta
//
// Mirrors deeptools gather_idx_info for a single torch.gather call.
// All addresses and strides are in units of sticks (not bytes, not elements).
//
//   base_addr           ≡ gii_.base_addr_          (start stick of value tensor)
//   skip_addr[d]        ≡ gii_.skip_addr_[d]        (sticks per unit of dim d)
//   idx_prev_cum_size[d]≡ gii_.idx_prev_cum_size_[d](inner-dim cumulative sizes)
// ---------------------------------------------------------------------------
struct GatherIdxMeta {
  int64_t base_addr;                         // in sticks
  std::vector<int64_t> skip_addr;            // per gather-dim, in sticks
  std::vector<int64_t> idx_prev_cum_size;    // per gather-dim
};

// ---------------------------------------------------------------------------
// build_gather_idx_meta
//
// Computes GatherIdxMeta from the value tensor layout and virtual offset.
// Mirrors deeptools constructDCIGatherIdxDataConvert.
//
// For torch.gather(input, dim, index):
//   - There is exactly ONE gather dimension: `dim`.
//   - The flat index value directly indexes along that dimension.
//   - idx_prev_cum_size[0] = 1  (innermost, no decomposition needed).
//   - skip_addr[0] = input.stride(dim) / elements_per_stick
//
// For multi-dim gather (N gather dimensions encoded as a flat integer):
//   - gather_dims must be provided in innermost-first order.
//   - idx_prev_cum_size[d] = product of input.size(k) for all k inner to d
//                          = input.stride(d)  (element units, row-major)
//   - skip_addr[d] = input.stride(gather_dims[d]) / elements_per_stick
// ---------------------------------------------------------------------------
static GatherIdxMeta build_gather_idx_meta(
    const at::Tensor& value_tensor,
    const std::vector<int64_t>& gather_dims,   // innermost first
    int64_t virtual_offset_bytes) {

  const int64_t ndim = value_tensor.dim();
  const int64_t element_size = value_tensor.element_size();
  const int64_t elements_per_stick = STICK_BYTES / element_size;

  for (int64_t d : gather_dims) {
    TORCH_CHECK(d >= 0 && d < ndim,
                "gather dim ", d, " out of range for ", ndim, "D tensor");
    TORCH_CHECK(value_tensor.stride(d) % elements_per_stick == 0,
                "value_tensor.stride(", d, ") = ", value_tensor.stride(d),
                " is not divisible by elements_per_stick=", elements_per_stick,
                ". Pad the value tensor so stride(", d, ") is a multiple of ",
                elements_per_stick, ".");
  }

  GatherIdxMeta meta;

  // base_addr_ : segment-relative start stick.
  meta.base_addr = virtual_offset_bytes / STICK_BYTES;

  // skip_addr_[d] and idx_prev_cum_size_[d] — innermost first, matching
  // the order deeptools expects when iterating from outer to inner.
  //
  // For a contiguous row-major tensor, input.stride(d) in element units equals
  // the product of all sizes at indices > d (inner dims), which is exactly
  // what deeptools calls the "cumulative size" of those inner positions.
  //
  // idx_prev_cum_size_[d] = input.stride(gather_dims[d])   (element units)
  //   → at innermost d where stride == 1: idx_prev_cum_size = 1
  //
  // skip_addr_[d] = input.stride(gather_dims[d]) / elements_per_stick
  meta.skip_addr.resize(gather_dims.size());
  meta.idx_prev_cum_size.resize(gather_dims.size());

  for (size_t i = 0; i < gather_dims.size(); ++i) {
    int64_t host_stride_elements = value_tensor.stride(gather_dims[i]);
    meta.skip_addr[i] = host_stride_elements / elements_per_stick;
    // idx_prev_cum_size[i] is the number of elements "consumed" by one unit
    // of the *inner* gather dimensions combined. For the innermost gather dim
    // this is 1 by definition (no inner gather dims). For outer dims it equals
    // the element stride of the next-inner gather dim (the cumulative inner
    // product), matching deeptools' idx_prev_cum_size_ semantics.
    if (i == 0) {
      // innermost: no inner gather dims, so cumulative size is 1.
      meta.idx_prev_cum_size[i] = 1;
    } else {
      // outer dim d: idx_prev_cum_size = stride of the gather dim one level
      // inner (gather_dims[i-1]).  This is the number of flat-index units that
      // correspond to one step in the current outer gather dimension.
      meta.idx_prev_cum_size[i] = value_tensor.stride(gather_dims[i - 1]);
    }
  }

  return meta;
}

// ---------------------------------------------------------------------------
// indices_to_addresses_nd
//
// Converts a flat integer index tensor into uint32 HBM stick addresses.
// Implements the deeptools ConvertData_gather_idx algorithm exactly.
//
// Algorithm (mirrors deeptools, iterating from outermost to innermost gather
// dimension):
//
//   addr[j] = base_addr
//
//   For d = outermost ... second-innermost:
//     coord   = indexVal[j] / idx_prev_cum_size[d]
//     addr[j] += coord * skip_addr[d]
//     indexVal[j] -= coord * idx_prev_cum_size[d]
//
//   Innermost (idx_prev_cum_size == 1, no division needed):
//     addr[j] += indexVal[j] * skip_addr[0]
//
// @param indices              Integer tensor (int32 or int64), any shape
// @param value_tensor         The value tensor being indexed; on spyre device
// @param dim                  The single gather dimension (for torch.gather)
// @param virtual_offset_bytes Byte offset of value_tensor start in HBM
// @return                     int32 tensor of stick addresses, same shape as
//                             indices, same device as indices
// ---------------------------------------------------------------------------
at::Tensor indices_to_addresses_nd(
    const at::Tensor& indices,
    const at::Tensor& value_tensor,
    int64_t dim,
    int64_t virtual_offset_bytes) {

  TORCH_CHECK(value_tensor.is_privateuseone(),
              "value_tensor must reside on the spyre device");
  TORCH_CHECK(dim >= 0 && dim < value_tensor.dim(),
              "dim=", dim, " out of range for ", value_tensor.dim(), "D tensor");

  // If the caller passes virtual_offset_bytes == 0 (the default from the pass),
  // derive it from the tensor's actual HBM allocation. This mirrors what
  // deeptools constructDCIGatherIdxDataConvert does: it reads the minimum
  // virtual address of the value tensor across cores and uses that as base_addr_.
  if (virtual_offset_bytes == 0) {
    virtual_offset_bytes = get_virtual_offset_bytes(value_tensor);
  }

  // For a standard torch.gather there is exactly one gather dimension.
  // gather_dims is innermost-first, so a single-dim gather is just {dim}.
  const std::vector<int64_t> gather_dims = {dim};

  // Build GII-equivalent metadata (no data access, compile-time metadata only).
  const GatherIdxMeta meta = build_gather_idx_meta(
      value_tensor, gather_dims, virtual_offset_bytes);

  // Mirror deeptools: bring index tensor to CPU for the host compute step.
  const auto original_device = indices.device();
  const auto indices_cpu = indices.cpu().to(at::kLong);
  const auto indices_shape = indices_cpu.sizes();
  const int64_t num_elems = indices_cpu.numel();
  const auto indices_flat = indices_cpu.reshape({num_elems});
  const auto idx_acc = indices_flat.accessor<int64_t, 1>();

  // Validate bounds.
  const int64_t dim_size = value_tensor.size(dim);
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(idx_acc[j] >= 0 && idx_acc[j] < dim_size,
                "Index value ", idx_acc[j], " at flat position ", j,
                " is out of bounds for gather dim ", dim,
                " with size ", dim_size);
  }

  const size_t num_dims = meta.skip_addr.size();

  // --- Core address computation — exact port of ConvertData_gather_idx ---
  //
  // Step 1: initialise addr[] = base_addr_  and  indexVal[] = input[j]
  std::vector<int64_t> addr(num_elems, meta.base_addr);
  std::vector<int64_t> index_val(num_elems);
  for (int64_t j = 0; j < num_elems; ++j) {
    index_val[j] = idx_acc[j];
  }

  if (num_dims > 0) {
    if (meta.idx_prev_cum_size[0] == 1) {
      // Fast path (standard case): innermost idx_prev_cum_size is 1.
      // Iterate outer → second-innermost, then handle innermost separately.

      // Outer dimensions (ri=0 is outermost, ends at second-innermost).
      for (size_t ri = 0; ri < num_dims - 1; ++ri) {
        const size_t i = num_dims - 1 - ri;   // outermost first
        const int64_t skip          = meta.skip_addr[i];
        const int64_t cum_size      = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }

      // Innermost dimension: idx_prev_cum_size[0] == 1 so coord == index_val.
      const int64_t skip_inner = meta.skip_addr[0];
      for (int64_t j = 0; j < num_elems; ++j) {
        addr[j] += index_val[j] * skip_inner;
      }

    } else {
      // General path: iterate all dimensions outer → inner including innermost.
      for (size_t ri = 0; ri < num_dims; ++ri) {
        const size_t i = num_dims - 1 - ri;
        const int64_t skip     = meta.skip_addr[i];
        const int64_t cum_size = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }
    }
  }

  // --- Write result as int32 (SENUINT32 equivalent) ---
  auto addr_tensor = at::empty(
      {num_elems}, at::TensorOptions().dtype(at::kInt));
  auto addr_acc = addr_tensor.accessor<int32_t, 1>();
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(addr[j] >= 0 &&
                addr[j] <= static_cast<int64_t>(
                    std::numeric_limits<uint32_t>::max()),
                "Computed stick address ", addr[j],
                " does not fit in uint32 for index element ", j);
    addr_acc[j] = static_cast<int32_t>(addr[j]);
  }

  // Reshape back to the original indices shape and move to original device.
  // Mirrors deeptools host-to-device transfer step.
  return addr_tensor.reshape(indices_shape).to(original_device);
}

}  // namespace spyre
*/

// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
#include "spyre_address_computation.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {

// Spyre stick size in bytes (Sen1.0).
constexpr int64_t STICK_BYTES = 128;

// ---------------------------------------------------------------------------
// get_virtual_offset_bytes
//
// Returns the intra-segment byte offset of a Spyre tensor's storage, which is
// used as virtual_offset_bytes in the address calculation.
//
// deeptools base_addr_ calculation:
//   base_addr_ = baseVirtAddress / bytesPerStick
// where baseVirtAddress is segment-relative (i.e. the byte offset within the
// tensor data segment, starting from 0).
//
// In the Flex memory model, LogicalAddress.region_id identifies the memory
// type (Tensor, Program, …) and LogicalAddress.offset is the byte offset
// within that segment.  The hardware uses only the intra-segment offset when
// computing stick addresses — region_id is a routing tag, not a multiplier.
// Therefore base_addr_ = addr.offset / STICK_BYTES.
// ---------------------------------------------------------------------------
static int64_t get_virtual_offset_bytes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "get_virtual_offset_bytes: tensor must be on spyre device");
  auto* impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(impl != nullptr,
              "get_virtual_offset_bytes: tensor is not a SpyreTensorImpl");

  auto* ctx = static_cast<SharedOwnerCtx*>(
      impl->storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "get_virtual_offset_bytes: null SharedOwnerCtx");

  const auto& chunks = ctx->composite_addr.chunks();
  TORCH_CHECK(!chunks.empty(),
              "get_virtual_offset_bytes: CompositeAddress has no chunks");

  // In the Flex memory model, LogicalAddress.region_id holds the absolute
  // encoded virtual byte address of the allocation (observed from allocator
  // log: "region_id=0x1000000080, offset=0x0").
  // LogicalAddress.offset is always 0 for fresh tensor allocations.
  // The complete byte address = region_id + offset.
  // Dividing by STICK_BYTES gives the stick address (base_addr_ equivalent).
  const auto& addr = chunks[0].addr;
  return static_cast<int64_t>(addr.region_id) +
         static_cast<int64_t>(addr.offset);
}

// ---------------------------------------------------------------------------
// GatherIdxMeta
//
// Mirrors deeptools gather_idx_info for a single torch.gather call.
// All addresses and strides are in units of sticks (not bytes, not elements).
//
//   base_addr           ≡ gii_.base_addr_          (start stick of value tensor)
//   skip_addr[d]        ≡ gii_.skip_addr_[d]        (sticks per unit of dim d)
//   idx_prev_cum_size[d]≡ gii_.idx_prev_cum_size_[d](inner-dim cumulative sizes)
// ---------------------------------------------------------------------------
struct GatherIdxMeta {
  int64_t base_addr;                         // in sticks
  std::vector<int64_t> skip_addr;            // per gather-dim, in sticks
  std::vector<int64_t> idx_prev_cum_size;    // per gather-dim
};

// ---------------------------------------------------------------------------
// build_gather_idx_meta
//
// Computes GatherIdxMeta from the value tensor layout and virtual offset.
// Mirrors deeptools constructDCIGatherIdxDataConvert.
//
// For torch.gather(input, dim, index):
//   - There is exactly ONE gather dimension: `dim`.
//   - The flat index value directly indexes along that dimension.
//   - idx_prev_cum_size[0] = 1  (innermost, no decomposition needed).
//   - skip_addr[0] = input.stride(dim) / elements_per_stick
//
// For multi-dim gather (N gather dimensions encoded as a flat integer):
//   - gather_dims must be provided in innermost-first order.
//   - idx_prev_cum_size[d] = product of input.size(k) for all k inner to d
//                          = input.stride(d)  (element units, row-major)
//   - skip_addr[d] = input.stride(gather_dims[d]) / elements_per_stick
// ---------------------------------------------------------------------------
static GatherIdxMeta build_gather_idx_meta(
    const at::Tensor& value_tensor,
    const std::vector<int64_t>& gather_dims,   // innermost first
    int64_t virtual_offset_bytes) {

  const int64_t ndim = value_tensor.dim();
  const int64_t element_size = value_tensor.element_size();
  const int64_t elements_per_stick = STICK_BYTES / element_size;

  for (int64_t d : gather_dims) {
    TORCH_CHECK(d >= 0 && d < ndim,
                "gather dim ", d, " out of range for ", ndim, "D tensor");
    TORCH_CHECK(value_tensor.stride(d) % elements_per_stick == 0,
                "value_tensor.stride(", d, ") = ", value_tensor.stride(d),
                " is not divisible by elements_per_stick=", elements_per_stick,
                ". Pad the value tensor so stride(", d, ") is a multiple of ",
                elements_per_stick, ".");
  }

  GatherIdxMeta meta;

  // base_addr_ : segment-relative start stick.
  meta.base_addr = virtual_offset_bytes / STICK_BYTES;

  // skip_addr_[d] and idx_prev_cum_size_[d] — innermost first, matching
  // the order deeptools expects when iterating from outer to inner.
  //
  // For a contiguous row-major tensor, input.stride(d) in element units equals
  // the product of all sizes at indices > d (inner dims), which is exactly
  // what deeptools calls the "cumulative size" of those inner positions.
  //
  // idx_prev_cum_size_[d] = input.stride(gather_dims[d])   (element units)
  //   → at innermost d where stride == 1: idx_prev_cum_size = 1
  //
  // skip_addr_[d] = input.stride(gather_dims[d]) / elements_per_stick
  meta.skip_addr.resize(gather_dims.size());
  meta.idx_prev_cum_size.resize(gather_dims.size());

  for (size_t i = 0; i < gather_dims.size(); ++i) {
    int64_t host_stride_elements = value_tensor.stride(gather_dims[i]);
    meta.skip_addr[i] = host_stride_elements / elements_per_stick;
    // idx_prev_cum_size[i] is the number of elements "consumed" by one unit
    // of the *inner* gather dimensions combined. For the innermost gather dim
    // this is 1 by definition (no inner gather dims). For outer dims it equals
    // the element stride of the next-inner gather dim (the cumulative inner
    // product), matching deeptools' idx_prev_cum_size_ semantics.
    if (i == 0) {
      // innermost: no inner gather dims, so cumulative size is 1.
      meta.idx_prev_cum_size[i] = 1;
    } else {
      // outer dim d: idx_prev_cum_size = stride of the gather dim one level
      // inner (gather_dims[i-1]).  This is the number of flat-index units that
      // correspond to one step in the current outer gather dimension.
      meta.idx_prev_cum_size[i] = value_tensor.stride(gather_dims[i - 1]);
    }
  }

  return meta;
}

// ---------------------------------------------------------------------------
// indices_to_addresses_nd
//
// Converts a flat integer index tensor into uint32 HBM stick addresses.
// Implements the deeptools ConvertData_gather_idx algorithm exactly.
//
// Algorithm (mirrors deeptools, iterating from outermost to innermost gather
// dimension):
//
//   addr[j] = base_addr
//
//   For d = outermost ... second-innermost:
//     coord   = indexVal[j] / idx_prev_cum_size[d]
//     addr[j] += coord * skip_addr[d]
//     indexVal[j] -= coord * idx_prev_cum_size[d]
//
//   Innermost (idx_prev_cum_size == 1, no division needed):
//     addr[j] += indexVal[j] * skip_addr[0]
//
// @param indices              Integer tensor (int32 or int64), any shape
// @param value_tensor         The value tensor being indexed; on spyre device
// @param dim                  The single gather dimension (for torch.gather)
// @param virtual_offset_bytes Byte offset of value_tensor start in HBM
// @return                     int32 tensor of stick addresses, same shape as
//                             indices, same device as indices
// ---------------------------------------------------------------------------
at::Tensor indices_to_addresses_nd(
    const at::Tensor& indices,
    const at::Tensor& value_tensor,
    int64_t dim,
    int64_t virtual_offset_bytes) {

  TORCH_CHECK(value_tensor.is_privateuseone(),
              "value_tensor must reside on the spyre device");
  TORCH_CHECK(dim >= 0 && dim < value_tensor.dim(),
              "dim=", dim, " out of range for ", value_tensor.dim(), "D tensor");

  // --- DIAGNOSTIC: print raw CompositeAddress fields ---
  {
    auto* dbg_impl = dynamic_cast<SpyreTensorImpl*>(
        value_tensor.unsafeGetTensorImpl());
    std::cerr << "[addr_dbg] is_SpyreTensorImpl=" << (dbg_impl != nullptr) << "\n";
    if (dbg_impl) {
      auto* dbg_raw_ctx = dbg_impl->storage().data_ptr().get_context();
      auto* dbg_raw_data = dbg_impl->storage().data_ptr().get();
      std::cerr << "[addr_dbg] data_ptr.get()=" << dbg_raw_data
                << "  get_context()=" << dbg_raw_ctx
                << "  same=" << (dbg_raw_data == dbg_raw_ctx) << "\n";
      std::cerr << "[addr_dbg] storage nbytes=" << dbg_impl->storage().nbytes()
                << "  allocator=" << dbg_impl->storage().allocator() << "\n";
      auto* dbg_ctx = static_cast<SharedOwnerCtx*>(dbg_raw_ctx);
      if (dbg_ctx) {
        const auto& dbg_chunks = dbg_ctx->composite_addr.chunks();
        std::cerr << "[addr_dbg] composite_addr chunks=" << dbg_chunks.size() << "\n";
        for (size_t ci = 0; ci < dbg_chunks.size(); ++ci) {
          std::cerr << "[addr_dbg]   chunk[" << ci << "]"
                    << " region_id=" << dbg_chunks[ci].addr.region_id
                    << " offset=" << dbg_chunks[ci].addr.offset
                    << " size=" << dbg_chunks[ci].size
                    << " domain_id=" << dbg_chunks[ci].domain_id << "\n";
        }
        // Print raw bytes of first chunk addr for diagnosis
        const auto& first_addr = dbg_chunks[0].addr;
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&first_addr);
        std::cerr << "[addr_dbg] LogicalAddress raw bytes:";
        for (size_t b = 0; b < sizeof(first_addr) && b < 32; ++b)
          std::cerr << " " << std::hex << (int)raw[b] << std::dec;
        std::cerr << "\n";
      }
    }
    std::cerr << "[addr_dbg] virtual_offset_bytes(caller)=" << virtual_offset_bytes
              << "  value_tensor.shape=" << value_tensor.sizes()
              << "  dim=" << dim
              << "  element_size=" << value_tensor.element_size()
              << "\n";
  }

  // If the caller passes virtual_offset_bytes == 0 (the default from the pass),
  // derive it from the tensor's actual HBM allocation. This mirrors what
  // deeptools constructDCIGatherIdxDataConvert does: it reads the minimum
  // virtual address of the value tensor across cores and uses that as base_addr_.
  if (virtual_offset_bytes == 0) {
    virtual_offset_bytes = get_virtual_offset_bytes(value_tensor);
    std::cerr << "[addr_dbg] derived virtual_offset_bytes=" << virtual_offset_bytes
              << "  => base_addr(sticks)=" << (virtual_offset_bytes / STICK_BYTES)
              << "\n";
  }

  // For a standard torch.gather there is exactly one gather dimension.
  // gather_dims is innermost-first, so a single-dim gather is just {dim}.
  const std::vector<int64_t> gather_dims = {dim};

  // Build GII-equivalent metadata (no data access, compile-time metadata only).
  const GatherIdxMeta meta = build_gather_idx_meta(
      value_tensor, gather_dims, virtual_offset_bytes);

  // Mirror deeptools: bring index tensor to CPU for the host compute step.
  const auto original_device = indices.device();
  const auto indices_cpu = indices.cpu().to(at::kLong);
  const auto indices_shape = indices_cpu.sizes();
  const int64_t num_elems = indices_cpu.numel();
  const auto indices_flat = indices_cpu.reshape({num_elems});
  const auto idx_acc = indices_flat.accessor<int64_t, 1>();

  // Validate bounds.
  const int64_t dim_size = value_tensor.size(dim);
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(idx_acc[j] >= 0 && idx_acc[j] < dim_size,
                "Index value ", idx_acc[j], " at flat position ", j,
                " is out of bounds for gather dim ", dim,
                " with size ", dim_size);
  }

  const size_t num_dims = meta.skip_addr.size();

  // --- Core address computation — exact port of ConvertData_gather_idx ---
  //
  // Step 1: initialise addr[] = base_addr_  and  indexVal[] = input[j]
  std::cerr << "[addr_dbg] base_addr=" << meta.base_addr
            << "  skip_addr[0]=" << (meta.skip_addr.empty() ? -1 : meta.skip_addr[0])
            << "  num_elems=" << num_elems << "\n";

  std::vector<int64_t> addr(num_elems, meta.base_addr);
  std::vector<int64_t> index_val(num_elems);
  for (int64_t j = 0; j < num_elems; ++j) {
    index_val[j] = idx_acc[j];
  }

  if (num_dims > 0) {
    if (meta.idx_prev_cum_size[0] == 1) {
      // Fast path (standard case): innermost idx_prev_cum_size is 1.
      // Iterate outer → second-innermost, then handle innermost separately.

      // Outer dimensions (ri=0 is outermost, ends at second-innermost).
      for (size_t ri = 0; ri < num_dims - 1; ++ri) {
        const size_t i = num_dims - 1 - ri;   // outermost first
        const int64_t skip          = meta.skip_addr[i];
        const int64_t cum_size      = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }

      // Innermost dimension: idx_prev_cum_size[0] == 1 so coord == index_val.
      const int64_t skip_inner = meta.skip_addr[0];
      for (int64_t j = 0; j < num_elems; ++j) {
        addr[j] += index_val[j] * skip_inner;
      }

    } else {
      // General path: iterate all dimensions outer → inner including innermost.
      for (size_t ri = 0; ri < num_dims; ++ri) {
        const size_t i = num_dims - 1 - ri;
        const int64_t skip     = meta.skip_addr[i];
        const int64_t cum_size = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }
    }
  }

  // --- Write result as int32 (SENUINT32 equivalent) ---
  auto addr_tensor = at::empty(
      {num_elems}, at::TensorOptions().dtype(at::kInt));
  auto addr_acc = addr_tensor.accessor<int32_t, 1>();
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(addr[j] >= 0 &&
                addr[j] <= static_cast<int64_t>(
                    std::numeric_limits<uint32_t>::max()),
                "Computed stick address ", addr[j],
                " does not fit in uint32 for index element ", j);
    addr_acc[j] = static_cast<int32_t>(addr[j]);
  }

  // --- DIAGNOSTIC: print first few computed addresses ---
  {
    int64_t show = std::min(num_elems, int64_t(8));
    std::cerr << "[addr_dbg] first " << show << " computed addresses (sticks): ";
    for (int64_t j = 0; j < show; ++j)
      std::cerr << addr[j] << " ";
    std::cerr << "\n";
  }

  // Reshape back to the original indices shape and move to original device.
  // Mirrors deeptools host-to-device transfer step.
  return addr_tensor.reshape(indices_shape).to(original_device);
}

}  // namespace spyre
*/

/*
#include "spyre_address_computation.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {

// Spyre stick size in bytes (Sen1.0).
constexpr int64_t STICK_BYTES = 128;

// ---------------------------------------------------------------------------
// get_virtual_offset_bytes
//
// Returns the HBM byte address of the start of the segment containing this
// tensor — used as virtual_offset_bytes so that:
//
//   base_addr_ = virtual_offset_bytes / STICK_BYTES
//
// In the Flex memory model, LogicalAddress.region_id holds the absolute
// encoded virtual byte address of the allocation, which includes a
// 128-byte (1-stick) allocator alignment header:
//
//   region_id = segment_base + 0x80   (e.g. 0x400000080)
//
// The SDSC compiler assigns each tensor's start_address to the segment base
// (e.g. 0x400000000), matching SEGMENT_OFFSETS[] in constants.py.
// The address tensor must therefore use the segment base, not region_id.
//
// SEGMENT_SIZE = 0x400000000 (from constants.py).  Masking region_id down
// to the nearest segment boundary recovers the segment base:
//
//   segment_base = region_id & ~(SEGMENT_SIZE - 1)
//                = region_id - (region_id % SEGMENT_SIZE)
// ---------------------------------------------------------------------------
static constexpr int64_t SPYRE_SEGMENT_SIZE = 0x400000000LL;  // from constants.py

static int64_t get_virtual_offset_bytes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "get_virtual_offset_bytes: tensor must be on spyre device");
  auto* impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(impl != nullptr,
              "get_virtual_offset_bytes: tensor is not a SpyreTensorImpl");

  auto* ctx = static_cast<SharedOwnerCtx*>(
      impl->storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "get_virtual_offset_bytes: null SharedOwnerCtx");

  const auto& chunks = ctx->composite_addr.chunks();
  TORCH_CHECK(!chunks.empty(),
              "get_virtual_offset_bytes: CompositeAddress has no chunks");

  // region_id is the absolute HBM byte address of the allocation.
  // It includes a 128-byte allocator alignment header past the segment base.
  // Mask down to the segment boundary to get the address the SDSC uses.
  const int64_t region_id =
      static_cast<int64_t>(chunks[0].addr.region_id) +
      static_cast<int64_t>(chunks[0].addr.offset);
  return region_id - (region_id % SPYRE_SEGMENT_SIZE);
}

// ---------------------------------------------------------------------------
// GatherIdxMeta
//
// Mirrors deeptools gather_idx_info for a single torch.gather call.
// All addresses and strides are in units of sticks (not bytes, not elements).
//
//   base_addr           ≡ gii_.base_addr_          (start stick of value tensor)
//   skip_addr[d]        ≡ gii_.skip_addr_[d]        (sticks per unit of dim d)
//   idx_prev_cum_size[d]≡ gii_.idx_prev_cum_size_[d](inner-dim cumulative sizes)
// ---------------------------------------------------------------------------
struct GatherIdxMeta {
  int64_t base_addr;                         // in sticks
  std::vector<int64_t> skip_addr;            // per gather-dim, in sticks
  std::vector<int64_t> idx_prev_cum_size;    // per gather-dim
};

// ---------------------------------------------------------------------------
// build_gather_idx_meta
//
// Computes GatherIdxMeta from the value tensor layout and virtual offset.
// Mirrors deeptools constructDCIGatherIdxDataConvert.
//
// For torch.gather(input, dim, index):
//   - There is exactly ONE gather dimension: `dim`.
//   - The flat index value directly indexes along that dimension.
//   - idx_prev_cum_size[0] = 1  (innermost, no decomposition needed).
//   - skip_addr[0] = input.stride(dim) / elements_per_stick
//
// For multi-dim gather (N gather dimensions encoded as a flat integer):
//   - gather_dims must be provided in innermost-first order.
//   - idx_prev_cum_size[d] = product of input.size(k) for all k inner to d
//                          = input.stride(d)  (element units, row-major)
//   - skip_addr[d] = input.stride(gather_dims[d]) / elements_per_stick
// ---------------------------------------------------------------------------
static GatherIdxMeta build_gather_idx_meta(
    const at::Tensor& value_tensor,
    const std::vector<int64_t>& gather_dims,   // innermost first
    int64_t virtual_offset_bytes) {

  const int64_t ndim = value_tensor.dim();
  const int64_t element_size = value_tensor.element_size();
  const int64_t elements_per_stick = STICK_BYTES / element_size;

  for (int64_t d : gather_dims) {
    TORCH_CHECK(d >= 0 && d < ndim,
                "gather dim ", d, " out of range for ", ndim, "D tensor");
    TORCH_CHECK(value_tensor.stride(d) % elements_per_stick == 0,
                "value_tensor.stride(", d, ") = ", value_tensor.stride(d),
                " is not divisible by elements_per_stick=", elements_per_stick,
                ". Pad the value tensor so stride(", d, ") is a multiple of ",
                elements_per_stick, ".");
  }

  GatherIdxMeta meta;

  // base_addr_ : segment-relative start stick.
  meta.base_addr = virtual_offset_bytes / STICK_BYTES;

  // skip_addr_[d] and idx_prev_cum_size_[d] — innermost first, matching
  // the order deeptools expects when iterating from outer to inner.
  //
  // For a contiguous row-major tensor, input.stride(d) in element units equals
  // the product of all sizes at indices > d (inner dims), which is exactly
  // what deeptools calls the "cumulative size" of those inner positions.
  //
  // idx_prev_cum_size_[d] = input.stride(gather_dims[d])   (element units)
  //   → at innermost d where stride == 1: idx_prev_cum_size = 1
  //
  // skip_addr_[d] = input.stride(gather_dims[d]) / elements_per_stick
  meta.skip_addr.resize(gather_dims.size());
  meta.idx_prev_cum_size.resize(gather_dims.size());

  for (size_t i = 0; i < gather_dims.size(); ++i) {
    int64_t host_stride_elements = value_tensor.stride(gather_dims[i]);
    meta.skip_addr[i] = host_stride_elements / elements_per_stick;
    // idx_prev_cum_size[i] is the number of elements "consumed" by one unit
    // of the *inner* gather dimensions combined. For the innermost gather dim
    // this is 1 by definition (no inner gather dims). For outer dims it equals
    // the element stride of the next-inner gather dim (the cumulative inner
    // product), matching deeptools' idx_prev_cum_size_ semantics.
    if (i == 0) {
      // innermost: no inner gather dims, so cumulative size is 1.
      meta.idx_prev_cum_size[i] = 1;
    } else {
      // outer dim d: idx_prev_cum_size = stride of the gather dim one level
      // inner (gather_dims[i-1]).  This is the number of flat-index units that
      // correspond to one step in the current outer gather dimension.
      meta.idx_prev_cum_size[i] = value_tensor.stride(gather_dims[i - 1]);
    }
  }

  return meta;
}

// ---------------------------------------------------------------------------
// indices_to_addresses_nd
//
// Converts a flat integer index tensor into uint32 HBM stick addresses.
// Implements the deeptools ConvertData_gather_idx algorithm exactly.
//
// Algorithm (mirrors deeptools, iterating from outermost to innermost gather
// dimension):
//
//   addr[j] = base_addr
//
//   For d = outermost ... second-innermost:
//     coord   = indexVal[j] / idx_prev_cum_size[d]
//     addr[j] += coord * skip_addr[d]
//     indexVal[j] -= coord * idx_prev_cum_size[d]
//
//   Innermost (idx_prev_cum_size == 1, no division needed):
//     addr[j] += indexVal[j] * skip_addr[0]
//
// @param indices              Integer tensor (int32 or int64), any shape
// @param value_tensor         The value tensor being indexed; on spyre device
// @param dim                  The single gather dimension (for torch.gather)
// @param virtual_offset_bytes Byte offset of value_tensor start in HBM
// @return                     int32 tensor of stick addresses, same shape as
//                             indices, same device as indices
// ---------------------------------------------------------------------------
at::Tensor indices_to_addresses_nd(
    const at::Tensor& indices,
    const at::Tensor& value_tensor,
    int64_t dim,
    int64_t virtual_offset_bytes) {

  TORCH_CHECK(value_tensor.is_privateuseone(),
              "value_tensor must reside on the spyre device");
  TORCH_CHECK(dim >= 0 && dim < value_tensor.dim(),
              "dim=", dim, " out of range for ", value_tensor.dim(), "D tensor");

  // If the caller passes virtual_offset_bytes == 0 (the default from the pass),
  // derive it from the tensor's actual HBM allocation.
  // base_addr_ = region_id / STICK_BYTES, where region_id is the absolute
  // HBM byte address of the value tensor's allocation start.
  if (virtual_offset_bytes == 0) {
    virtual_offset_bytes = get_virtual_offset_bytes(value_tensor);
  }

  // For a standard torch.gather there is exactly one gather dimension.
  // gather_dims is innermost-first, so a single-dim gather is just {dim}.
  const std::vector<int64_t> gather_dims = {dim};

  // Build GII-equivalent metadata (no data access, compile-time metadata only).
  const GatherIdxMeta meta = build_gather_idx_meta(
      value_tensor, gather_dims, virtual_offset_bytes);

  // Mirror deeptools: bring index tensor to CPU for the host compute step.
  const auto original_device = indices.device();
  const auto indices_cpu = indices.cpu().to(at::kLong);
  const auto indices_shape = indices_cpu.sizes();
  const int64_t num_elems = indices_cpu.numel();
  const auto indices_flat = indices_cpu.reshape({num_elems});
  const auto idx_acc = indices_flat.accessor<int64_t, 1>();

  // Validate bounds.
  const int64_t dim_size = value_tensor.size(dim);
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(idx_acc[j] >= 0 && idx_acc[j] < dim_size,
                "Index value ", idx_acc[j], " at flat position ", j,
                " is out of bounds for gather dim ", dim,
                " with size ", dim_size);
  }

  const size_t num_dims = meta.skip_addr.size();

  // --- Core address computation — exact port of ConvertData_gather_idx ---
  //
  // Step 1: initialise addr[] = base_addr_  and  indexVal[] = input[j]
  std::vector<int64_t> addr(num_elems, meta.base_addr);
  std::vector<int64_t> index_val(num_elems);
  for (int64_t j = 0; j < num_elems; ++j) {
    index_val[j] = idx_acc[j];
  }

  if (num_dims > 0) {
    if (meta.idx_prev_cum_size[0] == 1) {
      // Fast path (standard case): innermost idx_prev_cum_size is 1.
      // Iterate outer → second-innermost, then handle innermost separately.

      // Outer dimensions (ri=0 is outermost, ends at second-innermost).
      for (size_t ri = 0; ri < num_dims - 1; ++ri) {
        const size_t i = num_dims - 1 - ri;   // outermost first
        const int64_t skip          = meta.skip_addr[i];
        const int64_t cum_size      = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }

      // Innermost dimension: idx_prev_cum_size[0] == 1 so coord == index_val.
      const int64_t skip_inner = meta.skip_addr[0];
      for (int64_t j = 0; j < num_elems; ++j) {
        addr[j] += index_val[j] * skip_inner;
      }

    } else {
      // General path: iterate all dimensions outer → inner including innermost.
      for (size_t ri = 0; ri < num_dims; ++ri) {
        const size_t i = num_dims - 1 - ri;
        const int64_t skip     = meta.skip_addr[i];
        const int64_t cum_size = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }
    }
  }

  // --- Write result as int32 (SENUINT32 equivalent) ---
  auto addr_tensor = at::empty(
      {num_elems}, at::TensorOptions().dtype(at::kInt));
  auto addr_acc = addr_tensor.accessor<int32_t, 1>();
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(addr[j] >= 0 &&
                addr[j] <= static_cast<int64_t>(
                    std::numeric_limits<uint32_t>::max()),
                "Computed stick address ", addr[j],
                " does not fit in uint32 for index element ", j);
    addr_acc[j] = static_cast<int32_t>(addr[j]);
  }

  // Reshape back to the original indices shape and move to original device.
  // Mirrors deeptools host-to-device transfer step.
  return addr_tensor.reshape(indices_shape).to(original_device);
}

}  // namespace spyre
*/


#include "spyre_address_computation.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {

// Spyre stick size in bytes (Sen1.0).
constexpr int64_t STICK_BYTES = 128;

// ---------------------------------------------------------------------------
// get_virtual_offset_bytes
//
// Returns the HBM byte address of the start of the segment containing this
// tensor — used as virtual_offset_bytes so that:
//
//   base_addr_ = virtual_offset_bytes / STICK_BYTES
//
// In the Flex memory model, LogicalAddress.region_id holds the absolute
// encoded virtual byte address of the allocation, which includes a
// 128-byte (1-stick) allocator alignment header:
//
//   region_id = segment_base + 0x80   (e.g. 0x400000080)
//
// The SDSC compiler assigns each tensor's start_address to the segment base
// (e.g. 0x400000000), matching SEGMENT_OFFSETS[] in constants.py.
// The address tensor must therefore use the segment base, not region_id.
//
// SEGMENT_SIZE = 0x400000000 (from constants.py).  Masking region_id down
// to the nearest segment boundary recovers the segment base:
//
//   segment_base = region_id & ~(SEGMENT_SIZE - 1)
//                = region_id - (region_id % SEGMENT_SIZE)
// ---------------------------------------------------------------------------
static constexpr int64_t SPYRE_SEGMENT_SIZE = 0x400000000LL;  // from constants.py

static int64_t get_virtual_offset_bytes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "get_virtual_offset_bytes: tensor must be on spyre device");
  auto* impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(impl != nullptr,
              "get_virtual_offset_bytes: tensor is not a SpyreTensorImpl");

  auto* ctx = static_cast<SharedOwnerCtx*>(
      impl->storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "get_virtual_offset_bytes: null SharedOwnerCtx");

  const auto& chunks = ctx->composite_addr.chunks();
  TORCH_CHECK(!chunks.empty(),
              "get_virtual_offset_bytes: CompositeAddress has no chunks");

  // region_id is the absolute HBM byte address of the allocation.
  // It includes a 128-byte allocator alignment header past the segment base.
  // Mask down to the segment boundary to get the address the SDSC uses.
  const int64_t region_id =
      static_cast<int64_t>(chunks[0].addr.region_id) +
      static_cast<int64_t>(chunks[0].addr.offset);
  return region_id - (region_id % SPYRE_SEGMENT_SIZE);
}

// ---------------------------------------------------------------------------
// GatherIdxMeta
//
// Mirrors deeptools gather_idx_info for a single torch.gather call.
// All addresses and strides are in units of sticks (not bytes, not elements).
//
//   base_addr           ≡ gii_.base_addr_          (start stick of value tensor)
//   skip_addr[d]        ≡ gii_.skip_addr_[d]        (sticks per unit of dim d)
//   idx_prev_cum_size[d]≡ gii_.idx_prev_cum_size_[d](inner-dim cumulative sizes)
// ---------------------------------------------------------------------------
struct GatherIdxMeta {
  int64_t base_addr;                         // in sticks
  std::vector<int64_t> skip_addr;            // per gather-dim, in sticks
  std::vector<int64_t> idx_prev_cum_size;    // per gather-dim
};

// ---------------------------------------------------------------------------
// build_gather_idx_meta
//
// Computes GatherIdxMeta from the value tensor layout and virtual offset.
// Mirrors deeptools constructDCIGatherIdxDataConvert.
//
// For torch.gather(input, dim, index):
//   - There is exactly ONE gather dimension: `dim`.
//   - The flat index value directly indexes along that dimension.
//   - idx_prev_cum_size[0] = 1  (innermost, no decomposition needed).
//   - skip_addr[0] = input.stride(dim) / elements_per_stick
//
// For multi-dim gather (N gather dimensions encoded as a flat integer):
//   - gather_dims must be provided in innermost-first order.
//   - idx_prev_cum_size[d] = product of input.size(k) for all k inner to d
//                          = input.stride(d)  (element units, row-major)
//   - skip_addr[d] = input.stride(gather_dims[d]) / elements_per_stick
// ---------------------------------------------------------------------------
static GatherIdxMeta build_gather_idx_meta(
    const at::Tensor& value_tensor,
    const std::vector<int64_t>& gather_dims,   // innermost first
    int64_t virtual_offset_bytes) {

  const int64_t ndim = value_tensor.dim();
  const int64_t element_size = value_tensor.element_size();
  const int64_t elements_per_stick = STICK_BYTES / element_size;

  for (int64_t d : gather_dims) {
    TORCH_CHECK(d >= 0 && d < ndim,
                "gather dim ", d, " out of range for ", ndim, "D tensor");
    TORCH_CHECK(value_tensor.stride(d) % elements_per_stick == 0,
                "value_tensor.stride(", d, ") = ", value_tensor.stride(d),
                " is not divisible by elements_per_stick=", elements_per_stick,
                ". Pad the value tensor so stride(", d, ") is a multiple of ",
                elements_per_stick, ".");
  }

  GatherIdxMeta meta;

  // base_addr_ : segment-relative start stick.
  meta.base_addr = virtual_offset_bytes / STICK_BYTES;

  // skip_addr_[d] and idx_prev_cum_size_[d] — innermost first, matching
  // the order deeptools expects when iterating from outer to inner.
  //
  // For a contiguous row-major tensor, input.stride(d) in element units equals
  // the product of all sizes at indices > d (inner dims), which is exactly
  // what deeptools calls the "cumulative size" of those inner positions.
  //
  // idx_prev_cum_size_[d] = input.stride(gather_dims[d])   (element units)
  //   → at innermost d where stride == 1: idx_prev_cum_size = 1
  //
  // skip_addr_[d] = input.stride(gather_dims[d]) / elements_per_stick
  meta.skip_addr.resize(gather_dims.size());
  meta.idx_prev_cum_size.resize(gather_dims.size());

  for (size_t i = 0; i < gather_dims.size(); ++i) {
    int64_t host_stride_elements = value_tensor.stride(gather_dims[i]);
    meta.skip_addr[i] = host_stride_elements / elements_per_stick;
    // idx_prev_cum_size[i] is the number of elements "consumed" by one unit
    // of the *inner* gather dimensions combined. For the innermost gather dim
    // this is 1 by definition (no inner gather dims). For outer dims it equals
    // the element stride of the next-inner gather dim (the cumulative inner
    // product), matching deeptools' idx_prev_cum_size_ semantics.
    if (i == 0) {
      // innermost: no inner gather dims, so cumulative size is 1.
      meta.idx_prev_cum_size[i] = 1;
    } else {
      // outer dim d: idx_prev_cum_size = stride of the gather dim one level
      // inner (gather_dims[i-1]).  This is the number of flat-index units that
      // correspond to one step in the current outer gather dimension.
      meta.idx_prev_cum_size[i] = value_tensor.stride(gather_dims[i - 1]);
    }
  }

  return meta;
}

// ---------------------------------------------------------------------------
// indices_to_addresses_nd
//
// Converts a flat integer index tensor into uint32 HBM stick addresses.
// Implements the deeptools ConvertData_gather_idx algorithm exactly.
//
// Algorithm (mirrors deeptools, iterating from outermost to innermost gather
// dimension):
//
//   addr[j] = base_addr
//
//   For d = outermost ... second-innermost:
//     coord   = indexVal[j] / idx_prev_cum_size[d]
//     addr[j] += coord * skip_addr[d]
//     indexVal[j] -= coord * idx_prev_cum_size[d]
//
//   Innermost (idx_prev_cum_size == 1, no division needed):
//     addr[j] += indexVal[j] * skip_addr[0]
//
// @param indices              Integer tensor (int32 or int64), any shape
// @param value_tensor         The value tensor being indexed; on spyre device
// @param dim                  The single gather dimension (for torch.gather)
// @param virtual_offset_bytes Byte offset of value_tensor start in HBM
// @return                     int32 tensor of stick addresses, same shape as
//                             indices, same device as indices
// ---------------------------------------------------------------------------
at::Tensor indices_to_addresses_nd(
    const at::Tensor& indices,
    const at::Tensor& value_tensor,
    int64_t dim,
    int64_t virtual_offset_bytes) {

  TORCH_CHECK(value_tensor.is_privateuseone(),
              "value_tensor must reside on the spyre device");
  TORCH_CHECK(dim >= 0 && dim < value_tensor.dim(),
              "dim=", dim, " out of range for ", value_tensor.dim(), "D tensor");

  // If the caller passes virtual_offset_bytes == 0 (the default from the pass),
  // derive it from the tensor's actual HBM allocation.
  // base_addr_ = region_id / STICK_BYTES, where region_id is the absolute
  // HBM byte address of the value tensor's allocation start.
  if (virtual_offset_bytes == 0) {
    virtual_offset_bytes = get_virtual_offset_bytes(value_tensor);
  }

  // For a standard torch.gather there is exactly one gather dimension.
  // gather_dims is innermost-first, so a single-dim gather is just {dim}.
  const std::vector<int64_t> gather_dims = {dim};

  // Build GII-equivalent metadata (no data access, compile-time metadata only).
  const GatherIdxMeta meta = build_gather_idx_meta(
      value_tensor, gather_dims, virtual_offset_bytes);

  // Mirror deeptools: bring index tensor to CPU for the host compute step.
  const auto original_device = indices.device();
  const auto indices_cpu = indices.cpu().to(at::kLong);
  const auto indices_shape = indices_cpu.sizes();
  const int64_t num_elems = indices_cpu.numel();
  const auto indices_flat = indices_cpu.reshape({num_elems});
  const auto idx_acc = indices_flat.accessor<int64_t, 1>();

  // Validate bounds.
  const int64_t dim_size = value_tensor.size(dim);
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(idx_acc[j] >= 0 && idx_acc[j] < dim_size,
                "Index value ", idx_acc[j], " at flat position ", j,
                " is out of bounds for gather dim ", dim,
                " with size ", dim_size);
  }

  const size_t num_dims = meta.skip_addr.size();

  // ---------------------------------------------------------------------------
  // Outer-dimension (non-gathered) batch offset.
  //
  // torch.gather on a multi-dimensional input applies each index value in the
  // context of the *same position in all non-gathered dimensions*.  For example,
  // for a (B, V, D) tensor gathered on dim=1 with index shape (B, N, D):
  //
  //   output[b, n, d] = input[b, index[b,n,d], d]
  //
  // The stick address must therefore include the contribution of every
  // non-gathered dimension.  The indices tensor has the same shape as the
  // output, so we can reconstruct those coordinates directly from the flat
  // output position j.
  //
  // For each dimension k != dim:
  //   coord_k = (j / output_stride[k]) % output_size[k]
  //   addr[j] += coord_k * (value_tensor.stride(k) / elements_per_stick)
  //
  // For dim itself, the index value idx_acc[j] is used (handled below).
  // The innermost dimension (last dim of the output, which is the last dim of
  // the value tensor) always has stride(last) == 1, so its contribution is
  // zero sticks and is correctly skipped by the divisibility check in
  // build_gather_idx_meta.
  // ---------------------------------------------------------------------------
  const int64_t ndim = value_tensor.dim();
  const int64_t element_size = value_tensor.element_size();
  const int64_t elements_per_stick = STICK_BYTES / element_size;

  // Compute the flat-output stride for each dimension of the indices tensor
  // (same shape as output).  indices_shape == indices_cpu.sizes().
  std::vector<int64_t> out_stride(ndim, 1);
  for (int64_t k = ndim - 2; k >= 0; --k) {
    out_stride[k] = out_stride[k + 1] * indices_shape[k + 1];
  }

  // --- Core address computation — exact port of ConvertData_gather_idx ---
  //
  // Step 1: initialise addr[] = base_addr_  and  indexVal[] = input[j]
  std::vector<int64_t> addr(num_elems, meta.base_addr);
  std::vector<int64_t> index_val(num_elems);
  for (int64_t j = 0; j < num_elems; ++j) {
    index_val[j] = idx_acc[j];
  }

  // Step 1b: add the contribution of every non-gathered dimension.
  for (int64_t k = 0; k < ndim; ++k) {
    if (k == dim) continue;                           // handled via index_val below
    const int64_t val_stride_k = value_tensor.stride(k);
    if (val_stride_k % elements_per_stick != 0) continue; // sub-stick dim, no stick offset
    const int64_t skip_k = val_stride_k / elements_per_stick;
    if (skip_k == 0) continue;
    const int64_t sz_k = indices_shape[k];
    for (int64_t j = 0; j < num_elems; ++j) {
      const int64_t coord_k = (j / out_stride[k]) % sz_k;
      addr[j] += coord_k * skip_k;
    }
  }

  if (num_dims > 0) {
    if (meta.idx_prev_cum_size[0] == 1) {
      // Fast path (standard case): innermost idx_prev_cum_size is 1.
      // Iterate outer → second-innermost, then handle innermost separately.

      // Outer dimensions (ri=0 is outermost, ends at second-innermost).
      for (size_t ri = 0; ri < num_dims - 1; ++ri) {
        const size_t i = num_dims - 1 - ri;   // outermost first
        const int64_t skip          = meta.skip_addr[i];
        const int64_t cum_size      = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }

      // Innermost dimension: idx_prev_cum_size[0] == 1 so coord == index_val.
      const int64_t skip_inner = meta.skip_addr[0];
      for (int64_t j = 0; j < num_elems; ++j) {
        addr[j] += index_val[j] * skip_inner;
      }

    } else {
      // General path: iterate all dimensions outer → inner including innermost.
      for (size_t ri = 0; ri < num_dims; ++ri) {
        const size_t i = num_dims - 1 - ri;
        const int64_t skip     = meta.skip_addr[i];
        const int64_t cum_size = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j]       += coord * skip;
          index_val[j]  -= coord * cum_size;
        }
      }
    }
  }

  // --- Write result as int32 (SENUINT32 equivalent) ---
  auto addr_tensor = at::empty(
      {num_elems}, at::TensorOptions().dtype(at::kInt));
  auto addr_acc = addr_tensor.accessor<int32_t, 1>();
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(addr[j] >= 0 &&
                addr[j] <= static_cast<int64_t>(
                    std::numeric_limits<uint32_t>::max()),
                "Computed stick address ", addr[j],
                " does not fit in uint32 for index element ", j);
    addr_acc[j] = static_cast<int32_t>(addr[j]);
  }
  

for (int i = 0; i < std::min<int64_t>(20, num_elems); ++i) {
    std::cout
        << "i=" << i
        << " idx=" << idx_acc[i]
        << " addr=" << addr[i]
        << std::endl;
}

std::cout << "shape = " << value_tensor.sizes() << std::endl;

for (int i = 0; i < value_tensor.dim(); ++i) {
    std::cout << "stride[" << i << "] = "
              << value_tensor.stride(i) << std::endl;
}

std::cout << "base_addr = " << meta.base_addr << std::endl;

for (size_t i = 0; i < meta.skip_addr.size(); ++i) {
    std::cout << "skip[" << i << "] = "
              << meta.skip_addr[i] << std::endl;
}

std::cout<<"value_tensor.shape = "<<value_tensor.sizes()<<"\n";
//std::cout<<"value_tensor.stride() = "<<value_tensor.stride()<<"\n";
for (int i = 0; i < value_tensor.dim(); ++i) {
    std::cout << "stride[" << i << "] = "
              << value_tensor.stride(i) << std::endl;
}
std::cout<<"indices.shape = "<<indices.sizes()<<"\n";
//std::cout<<"indices = "<<indices<<"\n";
std::cout<<"dim = "<<dim<<"\n";

  // Reshape back to the original indices shape and move to original device.
  // Mirrors deeptools host-to-device transfer step.
  return addr_tensor.reshape(indices_shape).to(original_device);
}

}  // namespace spyre

