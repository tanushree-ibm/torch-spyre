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
static constexpr int64_t SPYRE_SEGMENT_SIZE =
    0x400000000LL;  // from constants.py

static int64_t get_virtual_offset_bytes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "get_virtual_offset_bytes: tensor must be on spyre device");
  std::cout<<" get_virtual_offset_bytes start \n";
  auto* impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(impl != nullptr,
              "get_virtual_offset_bytes: tensor is not a SpyreTensorImpl");

  auto* ctx =
      static_cast<SharedOwnerCtx*>(impl->storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "get_virtual_offset_bytes: null SharedOwnerCtx");

  const auto& chunks = ctx->composite_addr.chunks();
  TORCH_CHECK(!chunks.empty(),
              "get_virtual_offset_bytes: CompositeAddress has no chunks");

  // region_id is the absolute HBM byte address of the allocation.
  // It includes a 128-byte allocator alignment header past the segment base.
  // Mask down to the segment boundary to get the address the SDSC uses.
  const int64_t region_id = static_cast<int64_t>(chunks[0].addr.region_id) +
                            static_cast<int64_t>(chunks[0].addr.offset);

  std::cout<<" chunks[0].addr.region_id = "<<chunks[0].addr.region_id<<"\n";
  std::cout<<" chunks[0].addr.offset = "<<chunks[0].addr.offset<<"\n";
  std::cout<<" region_id - (region_id % SPYRE_SEGMENT_SIZE) = "<<(region_id - (region_id % SPYRE_SEGMENT_SIZE))<<"\n";
  return region_id - (region_id % SPYRE_SEGMENT_SIZE);

}

// ---------------------------------------------------------------------------
// GatherIdxMeta
//
// Mirrors deeptools gather_idx_info for a single torch.gather call.
// All addresses and strides are in units of sticks (not bytes, not elements).
//
//   base_addr           ≡ gii_.base_addr_          (start stick of value
//   tensor) skip_addr[d]        ≡ gii_.skip_addr_[d]        (sticks per unit of
//   dim d) idx_prev_cum_size[d]≡ gii_.idx_prev_cum_size_[d](inner-dim
//   cumulative sizes)
// ---------------------------------------------------------------------------
struct GatherIdxMeta {
  int64_t base_addr;                       // in sticks
  std::vector<int64_t> skip_addr;          // per gather-dim, in sticks
  std::vector<int64_t> idx_prev_cum_size;  // per gather-dim
};

// ---------------------------------------------------------------------------
// stick_skip_for_host_stride
//
// A Spyre device layout stores dimensions in tile-major order (see
// docs/source/user_guide/tensors_and_layouts.md): dimensions that don't fit
// in a single stick get an extra tiling dimension inserted *ahead* of them in
// device_size/stride_map, so the number of sticks skipped when advancing one
// element along a PyTorch dimension is generally NOT
// `host_stride / elements_per_stick`. For example a (3, 128) fp16 tensor has
// device_size=[2, 3, 64] (col-tile, row, within-stick): advancing one row
// only skips 1 stick (rows are adjacent within a tile), not
// stride(0)/64 == 2, because the col-tile dimension — not the row
// dimension — is the one multiplied by 2.
//
// The correct skip is the product of device_size over every dimension
// strictly between the matching device dimension and the stick dimension
// (the last device dimension). Returns false if no device dimension's
// stride_map matches `host_stride` (e.g. it is folded into the within-stick
// part of a sticked dimension).
// ---------------------------------------------------------------------------
static bool stick_skip_for_host_stride(const SpyreTensorLayout& layout,
                                       int64_t host_stride, int64_t* skip_out) {
  std::cout<<" stick_skip_for_host_stride start \n";
  const auto& device_size = layout.device_size;
  const auto& stride_map = layout.stride_map;
  const int64_t stick_dim = static_cast<int64_t>(device_size.size()) - 1;
  std::cout<<"device_size = "<<device_size<<"\n";
  std::cout<<"stride_map = "<<stride_map<<"\n";

  for (int64_t k = 0; k < stick_dim; ++k) {
    std::cout<<"stride_map["<<k<<"] = "<<stride_map[k]<<"\n";
    std::cout<<"host_stride = "<<host_stride<<"\n";
    if (stride_map[k] == host_stride) {
      int64_t skip = 1;
      for (int64_t m = k + 1; m < stick_dim; ++m) {
        std::cout<<"m = "<<m<<"\n";
        std::cout<<"device_size["<<m<<"] = "<<device_size[m]<<"\n";
        skip *= device_size[m];
        std::cout<<"skip = "<<skip<<"\n";
      }
      *skip_out = skip;
      std::cout<<"skip_out = "<<skip<<"\n";
      std::cout<<"reurn true \n";
      return true;
    }
  }
  std::cout<<"return false \n";
  return false;
}

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
//   - skip_addr[0] = stick skip of `dim` in the tensor's SpyreTensorLayout.
//
// For multi-dim gather (N gather dimensions encoded as a flat integer):
//   - gather_dims must be provided in innermost-first order.
//   - idx_prev_cum_size[d] = product of input.size(k) for all k inner to d
//                          = input.stride(d)  (element units, row-major)
//   - skip_addr[d] = stick skip of gather_dims[d] in the SpyreTensorLayout.
// ---------------------------------------------------------------------------
static GatherIdxMeta build_gather_idx_meta(
    const at::Tensor& value_tensor,
    const std::vector<int64_t>& gather_dims,  // innermost first
    int64_t virtual_offset_bytes) {
  std::cout<<" build_gather_idx_meta start\n";
  const int64_t ndim = value_tensor.dim();
  const int64_t element_size = value_tensor.element_size();
  const int64_t elements_per_stick = STICK_BYTES / element_size;
  const SpyreTensorLayout layout = get_spyre_tensor_layout(value_tensor);

  std::cout<<" ndim = "<<ndim<<"\n";
  std::cout<<" element_size = "<<element_size<<"\n";
  std::cout<<" elements_per_stick = "<<elements_per_stick<<"\n";

  for (int64_t d : gather_dims) {
    TORCH_CHECK(d >= 0 && d < ndim, "gather dim ", d, " out of range for ",
                ndim, "D tensor");
    TORCH_CHECK(value_tensor.stride(d) % elements_per_stick == 0,
                "value_tensor.stride(", d, ") = ", value_tensor.stride(d),
                " is not divisible by elements_per_stick=", elements_per_stick,
                ". Pad the value tensor so stride(", d, ") is a multiple of ",
                elements_per_stick, ".");
  }

  GatherIdxMeta meta;

  // base_addr_ : segment-relative start stick.
  meta.base_addr = virtual_offset_bytes / STICK_BYTES;
  std::cout<<" meta.base_addr = "<<meta.base_addr<<"\n";

  // skip_addr_[d] and idx_prev_cum_size_[d] — innermost first, matching
  // the order deeptools expects when iterating from outer to inner.
  //
  // For a contiguous row-major tensor, input.stride(d) in element units equals
  // the product of all sizes at indices > d (inner dims), which is exactly
  // what deeptools calls the "cumulative size" of those inner positions.
  //
  // idx_prev_cum_size_[d] = input.stride(gather_dims[d])   (element units)
  //   → at innermost d where stride == 1: idx_prev_cum_size = 1
  meta.skip_addr.resize(gather_dims.size());
  meta.idx_prev_cum_size.resize(gather_dims.size());

  for (size_t i = 0; i < gather_dims.size(); ++i) {
    int64_t host_stride_elements = value_tensor.stride(gather_dims[i]);
    std::cout<<" gather_dims["<<i<<"] = "<<gather_dims[i]<<"\n";
    std::cout<<"value_tensor.stride(gather_dims[i]) = "<<value_tensor.stride(gather_dims[i])<<"\n";
    std::cout<<" host_stride_elements = "<<host_stride_elements<<"\n";
    int64_t skip = 0;
    TORCH_CHECK(
        stick_skip_for_host_stride(layout, host_stride_elements, &skip),
        "build_gather_idx_meta: no device dimension in the SpyreTensorLayout "
        "matches host stride ",
        host_stride_elements, " for gather dim ", gather_dims[i],
        "; this dimension's on-device layout does not "
        "support stick-level gather addressing.");
    std::cout<<" skip = "<<skip<<"\n";
    meta.skip_addr[i] = skip;
    std::cout<<" meta.skip_addr["<<i<<"]="<<meta.skip_addr[i]<<"\n";
    // idx_prev_cum_size[i] is the number of elements "consumed" by one unit
    // of the *inner* gather dimensions combined. For the innermost gather dim
    // this is 1 by definition (no inner gather dims). For outer dims it equals
    // the element stride of the next-inner gather dim (the cumulative inner
    // product), matching deeptools' idx_prev_cum_size_ semantics.
    if (i == 0) {
      // innermost: no inner gather dims, so cumulative size is 1.
      std::cout<<" i = "<<i<<"\n";
      meta.idx_prev_cum_size[i] = 1;
      std::cout<<" meta.idx_prev_cum_size["<<i<<"]="<<meta.idx_prev_cum_size[i]<<"\n";
    } else {
      // outer dim d: idx_prev_cum_size = stride of the gather dim one level
      // inner (gather_dims[i-1]).  This is the number of flat-index units that
      // correspond to one step in the current outer gather dimension.
      std::cout<<" i = "<<i<<"\n";
      std::cout<<" value_tensor.stride(gather_dims["<<i-1<<"]) = "<<value_tensor.stride(gather_dims[i - 1])<<"\n";
      meta.idx_prev_cum_size[i] = value_tensor.stride(gather_dims[i - 1]);
      std::cout<<"meta.idx_prev_cum_size["<<i<<"] = "<<meta.idx_prev_cum_size[i]<<"\n";
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
at::Tensor indices_to_addresses_nd(const at::Tensor& indices,
                                   const at::Tensor& value_tensor, int64_t dim,
                                   int64_t virtual_offset_bytes) {
  TORCH_CHECK(value_tensor.is_privateuseone(),
              "value_tensor must reside on the spyre device");
  TORCH_CHECK(dim >= 0 && dim < value_tensor.dim(), "dim=", dim,
              " out of range for ", value_tensor.dim(), "D tensor");
  std::cout<<" indices_to_addresses_nd start \n";

  // If the caller passes virtual_offset_bytes == 0 (the default from the pass),
  // derive it from the tensor's actual HBM allocation.
  // base_addr_ = region_id / STICK_BYTES, where region_id is the absolute
  // HBM byte address of the value tensor's allocation start.
  std::cout<<" virtual_offset_bytes = "<<virtual_offset_bytes<<"\n";
  if (virtual_offset_bytes == 0) {
    virtual_offset_bytes = get_virtual_offset_bytes(value_tensor);
    std::cout<<" virtual_offset_bytes = "<<virtual_offset_bytes<<"\n";
  }

  // For a standard torch.gather there is exactly one gather dimension.
  // gather_dims is innermost-first, so a single-dim gather is just {dim}.
  const std::vector<int64_t> gather_dims = {dim};

  // Build GII-equivalent metadata (no data access, compile-time metadata only).
  const GatherIdxMeta meta =
      build_gather_idx_meta(value_tensor, gather_dims, virtual_offset_bytes);

  // Mirror deeptools: bring index tensor to CPU for the host compute step.
  const auto original_device = indices.device();
  const auto indices_cpu = indices.cpu().to(at::kLong);
  const auto indices_shape = indices_cpu.sizes();
  const int64_t num_elems = indices_cpu.numel();
  const auto indices_flat = indices_cpu.reshape({num_elems});
  const auto idx_acc = indices_flat.accessor<int64_t, 1>();
  std::cout<<" indices_shape = "<<indices_shape<<"\n";
  std::cout<<" num_elems = "<<num_elems<<"\n";
  std::cout<<" indices_flat = "<<indices_flat<<"\n";

  // Validate bounds.
  const int64_t dim_size = value_tensor.size(dim);
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(idx_acc[j] >= 0 && idx_acc[j] < dim_size, "Index value ",
                idx_acc[j], " at flat position ", j,
                " is out of bounds for gather dim ", dim, " with size ",
                dim_size);
  }

  const size_t num_dims = meta.skip_addr.size();
  std::cout<<" meta.skip_addr.size() = "<<meta.skip_addr.size()<<"\n";
  std::cout<<" num_dims = "<<num_dims<<"\n";

  // ---------------------------------------------------------------------------
  // Outer-dimension (non-gathered) batch offset.
  //
  // torch.gather on a multi-dimensional input applies each index value in the
  // context of the *same position in all non-gathered dimensions*.  For
  // example, for a (B, V, D) tensor gathered on dim=1 with index shape (B, N,
  // D):
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
  //   addr[j] += coord_k * stick_skip_for_host_stride(value_tensor.stride(k))
  //
  // For dim itself, the index value idx_acc[j] is used (handled below).
  // The innermost dimension (last dim of the output, which is the last dim of
  // the value tensor) always has stride(last) == 1, so its contribution is
  // zero sticks and is correctly skipped (no device dimension's stride_map
  // matches a sub-stick host stride).
  // ---------------------------------------------------------------------------
  const int64_t ndim = value_tensor.dim();
  const SpyreTensorLayout value_layout = get_spyre_tensor_layout(value_tensor);
  std::cout<<" ndim = "<<ndim<<"\n";
  std::cout<<" value_layout = "<<value_layout.toString()<<"\n";

  // Compute the flat-output stride for each dimension of the indices tensor
  // (same shape as output).  indices_shape == indices_cpu.sizes().
  std::vector<int64_t> out_stride(ndim, 1);
  for (int64_t k = ndim - 2; k >= 0; --k) {
    std::cout<<" out_stride[k + 1]="<<out_stride[k + 1] <<"\n";
    std::cout<<" indices_shape[k + 1]="<<indices_shape[k + 1]<<"\n";
    out_stride[k] = out_stride[k + 1] * indices_shape[k + 1];
    std::cout<<" k="<<k<<"out_stride[k]="<<out_stride[k]<<"\n";
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
  std::cout<<"value_tensor.stride(0) = "<<value_tensor.stride(0)<<"\n";
  std::cout<<"value_tensor.stride(1) = "<<value_tensor.stride(1)<<"\n";
  for (int64_t k = 0; k < ndim; ++k) {
    std::cout<<" k="<<k<<"\n";
    if (k == dim) continue;  // handled via index_val below
    int64_t skip_k = 0;
    std::cout<<"value_tensor.stride("<<k<<")="<<value_tensor.stride(k)<<"\n";
    if (!stick_skip_for_host_stride(value_layout, value_tensor.stride(k),
                                    &skip_k) ||
        skip_k == 0) {
      continue;  // sub-stick dim, no stick offset
    }
    std::cout<<" skip_k : "<<skip_k<<"\n";
    const int64_t sz_k = indices_shape[k];
    for (int64_t j = 0; j < num_elems; ++j) {
      const int64_t coord_k = (j / out_stride[k]) % sz_k;
      addr[j] += coord_k * skip_k;
    }
  }

  std::cout<<"num_dims = "<<num_dims<<"\n";
  if (num_dims > 0) {
    std::cout<<"meta.idx_prev_cum_size[0] = "<<meta.idx_prev_cum_size[0]<<"\n";
    if (meta.idx_prev_cum_size[0] == 1) {
      // Fast path (standard case): innermost idx_prev_cum_size is 1.
      // Iterate outer → second-innermost, then handle innermost separately.

      // Outer dimensions (ri=0 is outermost, ends at second-innermost).
      for (size_t ri = 0; ri < num_dims - 1; ++ri) {
        const size_t i = num_dims - 1 - ri;  // outermost first
        const int64_t skip = meta.skip_addr[i];
        const int64_t cum_size = meta.idx_prev_cum_size[i];

        std::cout<<"i = "<<i<<"\n";
        std::cout<<"skip = "<<skip<<"\n";
        std::cout<<"cum_size = "<<cum_size<<"\n";

        for (int64_t j = 0; j < num_elems; ++j) {
          std::cout<<"index_val["<<j<<"] = "<<index_val[j]<<"\n";
          const int64_t coord = index_val[j] / cum_size;
          std::cout<<" coord = "<<"index_val[j] / cum_size"<<"= "<<coord<<"\n";
          std::cout<<"coord * skip = "<<coord * skip<<"\n";
          std::cout<<"coord * cum_size = "<<coord * cum_size<<"\n";
          addr[j] += coord * skip;
          index_val[j] -= coord * cum_size;
          std::cout<<"addr[j] = "<<addr[j]<<"\n";
          std::cout<<"index_val[j] = "<<index_val[j]<<"\n";
        }
      }

      // Innermost dimension: idx_prev_cum_size[0] == 1 so coord == index_val.
      const int64_t skip_inner = meta.skip_addr[0];
      for (int64_t j = 0; j < num_elems; ++j) {
        addr[j] += index_val[j] * skip_inner;
      }

    } else {
      // General path: iterate all dimensions outer → inner including innermost.
      std::cout<<"General path: iterate all dimensions outer → inner including innermost. \n";
      for (size_t ri = 0; ri < num_dims; ++ri) {
        const size_t i = num_dims - 1 - ri;
        const int64_t skip = meta.skip_addr[i];
        const int64_t cum_size = meta.idx_prev_cum_size[i];
        for (int64_t j = 0; j < num_elems; ++j) {
          const int64_t coord = index_val[j] / cum_size;
          addr[j] += coord * skip;
          index_val[j] -= coord * cum_size;
        }
      }
    }
  }

  // --- Write result as int32 (SENUINT32 equivalent) ---
  auto addr_tensor =
      at::empty({num_elems}, at::TensorOptions().dtype(at::kInt));
  auto addr_acc = addr_tensor.accessor<int32_t, 1>();
  for (int64_t j = 0; j < num_elems; ++j) {
    TORCH_CHECK(
        addr[j] >= 0 && addr[j] <= static_cast<int64_t>(
                                       std::numeric_limits<uint32_t>::max()),
        "Computed stick address ", addr[j],
        " does not fit in uint32 for index element ", j);
    std::cout<<" Address : "<<static_cast<int32_t>(addr[j])<<"\n";
    addr_acc[j] = static_cast<int32_t>(addr[j]);
  }

  // Reshape back to the original indices shape and move to original device.
  // Mirrors deeptools host-to-device transfer step.
  return addr_tensor.reshape(indices_shape).to(original_device);
}

}  // namespace spyre



