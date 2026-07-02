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

#include <vector>

#include "spyre_tensor_impl.h"

namespace spyre {

/*at::Tensor indices_to_addresses_nd(const at::Tensor& logical_indices,
                                   const at::Tensor& value_tensor, int64_t dim,
                                   int64_t virtual_offset) {
  // Store the original device for later
  std::cout<<"TANU indices_to_addresses_nd \n";
  auto original_device = logical_indices.device();

  // Only move indices to CPU for computation (value tensor stays on device)
  auto indices_cpu = logical_indices.cpu();

  // Get shapes (no need to move value_tensor to CPU for metadata)
  auto indices_shape = indices_cpu.sizes();
  auto value_shape = value_tensor.sizes();
  int64_t value_ndim = value_shape.size();

  // DeepTools stick size (128 bytes for Sen1.0)
  constexpr int64_t STICK_SIZE_BYTES = 128;

  // Use virtual_offset as the base address for address computation
  // This allows for paged attention and other use cases requiring non-zero base
  // addresses
  int64_t base_address = virtual_offset;
  int64_t element_size = value_tensor.element_size();

  // Use the tensor's actual strides from SpyreTensorLayout.
  // The stride_map contains host memory offsets for each device dimension.
  // Since device dimensions include the stick dimension, we need to use
  // the tensor's logical strides which correspond to host memory layout.
  auto value_strides = value_tensor.strides();

  // Determine indexing mode based on indices shape
  bool is_multidim_index =
      (indices_shape.size() > 0 &&
       indices_shape[indices_shape.size() - 1] == value_ndim);

  at::Tensor addresses;

  if (is_multidim_index) {
    // Multi-dimensional indexing: indices shape is [..., ndim]
    // Each index tuple specifies coordinates in all dimensions

    auto flat_indices = indices_cpu.reshape({-1, value_ndim});
    int64_t num_indices = flat_indices.size(0);

    addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kFloat));
    auto flat_indices_accessor = flat_indices.accessor<int64_t, 2>();
    auto addresses_accessor = addresses.accessor<float, 1>();

    for (int64_t i = 0; i < num_indices; ++i) {
      int64_t element_offset = 0;

      // Compute offset using all dimensions with actual tensor strides
      for (int64_t d = 0; d < value_ndim; ++d) {
        int64_t coord = flat_indices_accessor[i][d];

        // Validate coordinate
        TORCH_CHECK(coord >= 0 && coord < value_shape[d], "Index ", coord,
                    " out of bounds for dimension ", d,
                    " (size: ", value_shape[d], ")");

        element_offset += coord * value_strides[d];
      }

      // Convert to byte address
      int64_t byte_address = base_address + (element_offset * element_size);

      // Verify stick alignment
      TORCH_CHECK(byte_address % STICK_SIZE_BYTES == 0, "Address ",
                  byte_address, " is not stick-aligned. ",
                  "Element offset: ", element_offset);

      // Convert to stick address
      int64_t stick_address = byte_address / STICK_SIZE_BYTES;
      addresses_accessor[i] = static_cast<float>(stick_address);
    }

    // Reshape back to original shape (without last dimension)
    std::vector<int64_t> output_shape(indices_shape.begin(),
                                      indices_shape.end() - 1);
    addresses = addresses.reshape(output_shape);

  } else {
    // Single-dimension indexing: indices shape is [...]
    // Index along specified dimension (default: dim=0, row-major)

    TORCH_CHECK(dim >= 0 && dim < value_ndim, "Dimension ", dim,
                " out of bounds for ", value_ndim, "D tensor");

    auto flat_indices = indices_cpu.reshape({-1});
    int64_t num_indices = flat_indices.size(0);

    addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kLong));
    auto flat_indices_accessor = flat_indices.accessor<int64_t, 1>();
    auto addresses_accessor = addresses.accessor<int64_t, 1>();

    // Compute stride for the indexed dimension using actual tensor strides
    int64_t dim_stride_elements = value_strides[dim];
    int64_t dim_stride_bytes = dim_stride_elements * element_size;

    for (int64_t i = 0; i < num_indices; ++i) {
      int64_t index = flat_indices_accessor[i];

      // Validate index
      TORCH_CHECK(index >= 0 && index < value_shape[dim], "Index ", index,
                  " out of bounds for dimension ", dim,
                  " (size: ", value_shape[dim], ")");

      // Compute address: base + (index * stride)
      int64_t byte_address = base_address + (index * dim_stride_bytes);

      // Verify stick alignment
      TORCH_CHECK(byte_address % STICK_SIZE_BYTES == 0, "Address ",
                  byte_address, " for index ", index,
                  " is not stick-aligned. Stride: ", dim_stride_bytes,
                  " bytes");

      // Convert to stick address
      int64_t stick_address = byte_address / STICK_SIZE_BYTES;
      addresses_accessor[i] = static_cast<int64_t>(stick_address);
    }

    // Reshape back to original indices shape
    addresses = addresses.reshape(indices_shape);
  }

  // Move the result back to the original device (spyre)
  return addresses.to(original_device);
}*/

// Single consolidated function for host to device coordinate conversion
std::vector<std::vector<int64_t>> convertHostCoOrdinatesToDeviceCoOrdinates(
    const std::vector<std::vector<int64_t>>& host_cords, int64_t rank,
    int64_t elements_per_stick = 64)
{
    std::cout<<"convertHostCoOrdinatesToDeviceCoOrdinates \n";
    std::vector<std::vector<int64_t>> device_cords;
    //int rank = 3;//host_cords.size();
    std::cout<<"rank = "<<rank<<"\n";
    for (const auto& coord : host_cords) {
        std::vector<int64_t> device_cord;
        
        if(rank==1)
        {
            device_cord.push_back(coord[0]/ elements_per_stick);
            device_cord.push_back(coord[0]% elements_per_stick);
        }
        else if(rank==2){ ///2D
        device_cord.push_back(coord[1]/ elements_per_stick);
        device_cord.push_back(coord[0]);
        device_cord.push_back(coord[1]% elements_per_stick);

        }else if(rank==3){

        //3D
        device_cord.push_back(coord[1]);
        device_cord.push_back(coord[2]/elements_per_stick);
        device_cord.push_back(coord[0]);
        device_cord.push_back(coord[2]%elements_per_stick);

        }
        device_cords.push_back(device_cord);
    }
    return device_cords;
}

// Single consolidated compute_addresses function for all N-D tensors
at::Tensor indices_to_addresses_nd(
    const at::Tensor& indices,
    const at::Tensor& input,
    int64_t dim,
    int64_t virtual_offset) 
{
    std::cout<<"indices_to_addresses_nd start \n";
    // Get tensor properties
    SpyreTensorLayout input_layout = get_spyre_tensor_layout(input);
    auto device_stride = input_layout.stride_map;
    int64_t element_size = input.element_size();
    /*auto dim_map = input_layout.dim_map;
    std::cout<<"TANU dim_map = "<<dim_map.size();
    std::cout<<"dim_map[0] = "<<dim_map[0];
    std::cout<<"dim_map[1] = "<<dim_map[1];
    std::cout<<"dim_map[2] = "<<dim_map[2];*/

    int64_t ndim = input.dim();

    std::cout<<"input.dim() = "<<input.dim()<<"\n";
    std::cout<<"dim = "<<dim<<"\n";
    std::cout<<"input_layout = "<<input_layout.toString()<<"\n";

    constexpr int64_t STICK_SIZE = 128;
    
    int64_t elements_per_stick = STICK_SIZE / element_size;
    std::cout<<"elements_per_stick = "<<elements_per_stick<<"\n";

    // Move indices to CPU
    auto original_device = indices.device();

    //auto indices_cpu = indices.cpu().contiguous();
    auto indices_cpu = indices.cpu();
    std::cout<<"indices : "<<indices_cpu<<"\n";
    std::cout << "dtype = "
          << indices_cpu.scalar_type()
          << "\n";

    TORCH_CHECK(
    indices_cpu.scalar_type() == at::kLong,
    "Expected int64 indices");

    std::cout << "sizes  = "
          << indices_cpu.sizes()
          << "\n";

    std::cout << "strides = "
          << indices_cpu.strides()
          << "\n";
    
    auto indices_shape = indices_cpu.sizes();

    // Accessor
    auto acc = indices_cpu.accessor<int64_t, 1>();
    // Preprocessing (O(m))
    std::unordered_map<int64_t, int64_t> reverse_index;

    for (int j = 0; j < indices_cpu.size(0); j++) {
        reverse_index[acc[j]] = j;
    }


    // Extract host coordinates (works for any N-D tensor)
    std::vector<std::vector<int64_t>> host_cords;
    int64_t total = indices.numel();

    std::cout<<"Total no. of indices : "<<total<<"\n";
    

    // Find coordinates:
    // out[i, index[i, j]] for dim = 1
    //std::vector<std::vector<int64_t>> host_cords;

    for (int64_t i = 0; i < input.size(0); ++i) {
        for (int64_t j = 0; j < input.size(1); ++j) {

            std::vector<int64_t> input_cord{i,j};
            std::vector<int64_t> output_cord(input_cord);

            std::cout<<"i,j : "<<i<<j<<"\n";
            std::cout<<"input_cord[dim] : "<<input_cord[dim]<<"\n";
            auto it = reverse_index.find(input_cord[dim]);
            if (it != reverse_index.end()) {
                std::cout<<"TANU \n";

            //output_cord[dim] = it->second;
            host_cords.push_back(output_cord);
            }
            
        }
    }

    for (auto it : host_cords) {
        std::cout
            << "host_cords : "
            << it[0] << ","
            << it[1] << "\n";
    }
 
    
    // Convert to device coordinates (works for all dimensions)
    auto device_cords = convertHostCoOrdinatesToDeviceCoOrdinates(
        host_cords, ndim, elements_per_stick);
    

    // Calculate addresses
    int64_t numElements = device_cords.size();
    auto ind_addresses = at::zeros(
    {numElements}, 
    at::TensorOptions().dtype(at::kLong));  // int64_t
    auto ind_addresses_accessor = ind_addresses.accessor<int64_t, 1>();

    std::cout<<"ndim : "<<ndim<<"\n";

    for (size_t i = 0; i < device_cords.size(); ++i) {
        const auto& coord = device_cords[i];

        // Calculate element offset
        int64_t element_offset = 0;
        for (size_t k = 0; k < coord.size(); ++k) {
            element_offset += coord[k] * device_stride[k];
        }

        // Convert to stick address
        int64_t byte_address = virtual_offset + element_offset * element_size;
        int64_t stick_address = byte_address / STICK_SIZE;
        
        ind_addresses_accessor[i] = stick_address;
        std::cout<<"byte_address : "<<byte_address<<" , stick_address : "<<stick_address<<"\n";
    }
    auto final_address = ind_addresses.reshape({2,64});

    //auto final_address = ind_addresses.reshape(indices_shape);
    
    std::cout<<"final_address = \n";
    std::cout<<final_address<<"\n";

    std::cout<<"Stick Address = \n";
    if(ndim==3){
        std::vector<int> values;
        int sticks = final_address.size(2);
        for (int c = 0; c < sticks; c += 64) {
            values.push_back(c);
        }

        int rows = final_address.size(0);
        int cols = final_address.size(1);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                for (auto v : values) {
                    std::cout << final_address[r][c][v].item<int64_t>()  << ", ";
                }
                std::cout << "\n";
            }
        }
    } else if(ndim==2){
        std::vector<int> values;
        int sticks = final_address.size(1);
        for (int c = 0; c < sticks; c += 64) {
            values.push_back(c);
        }

        int rows = final_address.size(0);
        //int cols = final_address.size(1);
        for (int r = 0; r < rows; r++) {
        // for (int c = 0; c < cols; c++) {
                for (auto v : values) {
                    std::cout << final_address[r][v].item<int64_t>()  << ", ";
                }
                std::cout << "\n";
            //}
        }
    }
    return ind_addresses.reshape({2,64}).to(original_device);
    //return input;
}

}  // namespace spyre
