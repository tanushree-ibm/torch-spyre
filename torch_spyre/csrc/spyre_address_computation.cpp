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
#include "spyre_tensor_impl.h"  // Add this header

namespace spyre {

// Single consolidated function for host to device coordinate conversion
std::vector<std::vector<int64_t>> convertHostCoOrdinatesToDeviceCoOrdinates(
    const std::vector<std::vector<int64_t>>& host_cords,
    int64_t elements_per_stick = 64)
{
    std::vector<std::vector<int64_t>> device_cords;
    
    for (const auto& coord : host_cords) {
        std::vector<int64_t> device_coord;
        
        // Copy all dimensions except the last
        for (size_t i = 0; i < coord.size() - 1; ++i) {
            device_coord.push_back(coord[i]);
        }
        
        // Split last dimension: [stick_index, element_in_stick]
        int64_t last_dim = coord.back();
        device_coord.push_back(last_dim / elements_per_stick);
        device_coord.push_back(last_dim % elements_per_stick);
        
        device_cords.push_back(device_coord);
    }
    
    return device_cords;
}

// Single consolidated compute_addresses function for all N-D tensors
at::Tensor compute_addresses_from_input_indices(
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& indices,
    int64_t virtual_offset) 
{
    // Get tensor properties
    SpyreTensorLayout input_layout = get_spyre_tensor_layout(input);
    auto device_stride = input_layout.stride_map;
    int64_t element_size = input.element_size();
    int64_t ndim = input.dim();
    
    constexpr int64_t STICK_SIZE = 128;
    int64_t elements_per_stick = STICK_SIZE / element_size;

    // Move indices to CPU
    auto original_device = indices.device();
    auto indices_cpu = indices.cpu();
    auto indices_shape = indices_cpu.sizes();
    auto acc = indices_cpu.accessor<int64_t, 3>();

    // Extract host coordinates (works for any N-D tensor)
    std::vector<std::vector<int64_t>> host_cords;
    std::cout<<"Host CoOrdinates :";
    for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
        for (int64_t j = 0; j < indices_cpu.size(1); ++j) {
            for (int64_t k = 0; k < indices_cpu.size(2); ++k) {
            int64_t val = acc[i][j][k];
            
            // Build N-D coordinate by inserting val at position 'dim'
            std::vector<int64_t> coord = {i, j, k};  // Initialize with indices positions
            coord[dim] = val;  // Replace dimension 'dim' with the indexed value
            host_cords.push_back(coord);
            std::cout<<"\n";
            }
        }
    }

    // Convert to device coordinates (works for all dimensions)
    auto device_cords = convertHostCoOrdinatesToDeviceCoOrdinates(
        host_cords, elements_per_stick);

    // Calculate addresses
    int64_t numElements = device_cords.size();
    auto ind_addresses = at::zeros({numElements}, at::TensorOptions().dtype(at::kFloat));
    auto ind_addresses_accessor = ind_addresses.accessor<float, 1>();

    std::cout<<"Device CoOrdinate \n";
    for (size_t i = 0; i < device_cords.size(); ++i) {
        const auto& coord = device_cords[i];
        
        // Calculate element offset
        int64_t element_offset = 0;
        for (size_t k = 0; k < coord.size(); ++k) {
            element_offset += coord[k] * device_stride[k];
            std::cout<<coord[k]<<" , ";
        }
        std::cout<<"\n";
        // Convert to stick address
        int64_t byte_address = virtual_offset + element_offset * element_size;
        int64_t stick_address = byte_address / STICK_SIZE;
        
        ind_addresses_accessor[i] = static_cast<float>(stick_address);
    }

    return ind_addresses.reshape(indices_shape).to(original_device);
}

} // namespace spyre
