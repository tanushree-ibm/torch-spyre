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

std::vector<std::vector<int64_t>> convertHostCoOrdinatesToDeviceCoOrdinates(std::vector<std::vector<int64_t>>& host_cords)
{
    std::vector<std::vector<int64_t>> device_cords;
    for (const auto& coord : host_cords) {
        int64_t row = coord[0];
        int64_t col = coord[1];
        device_cords.push_back({col/64, row, col%64});
    }
    return device_cords;
}
at::Tensor compute_addresses_from_input_indices(
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& indices,
    int64_t virtual_offset) {

    std::cout << "input.size() = " << input.sizes() << "\n";
    std::cout << "dim = " << dim << "\n";
    std::cout << "indices.size() = " << indices.sizes() << "\n";

    SpyreTensorLayout input_layout = get_spyre_tensor_layout(input);
    // Print layout information
    std::cout << input_layout.toString() << std::endl;
    
    // Get device size and stride from layout
    auto device_size = input_layout.device_size;
    auto device_stride = input_layout.stride_map;
    int64_t element_size = input.element_size();
  

    // Store the original device for later
    auto original_device = indices.device();

    // Ensure indices tensor is on CPU
    auto indices_cpu = indices.cpu();

    std::cout << "indices_cpu : " << indices_cpu << "\n";

    // Get the shape of indices
    auto indices_shape = indices_cpu.sizes();

    // Accessor
    auto acc = indices_cpu.accessor<int64_t, 2>();

    // Find coordinates:
    // out[i, index[i, j]] for dim = 1
    std::vector<std::vector<int64_t>> host_cords;

    for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
        for (int64_t j = 0; j < indices_cpu.size(1); ++j) {

            int64_t val = acc[i][j]; //index[i][j]

            std::cout
                << "indices[" << i << "][" << j << "] = "
                << val << "\n";

            if (dim == 1) {
                host_cords.push_back({i, val}); //[i, index[i, j]] 
            } else {
                host_cords.push_back({val, j}); //[index[i, j], j] 
            }
        }
    }

    for (auto it : host_cords) {
        std::cout
            << "host_cords : "
            << it[0] << ","
            << it[1] << "\n";
    }

    int64_t numElements = host_cords.size();
    auto ind_addresses = at::zeros(
        {numElements},
        at::TensorOptions().dtype(at::kFloat));

    auto ind_addresses_accessor =
        ind_addresses.accessor<float, 1>();

    constexpr int64_t STICK_SIZE = 128;

    std::vector<int64_t> stick_addresses;

    int i = 0;

    std::vector<std::vector<int64_t>> device_cords = convertHostCoOrdinatesToDeviceCoOrdinates(host_cords);
    for (auto it : device_cords) {
        std::cout
            << "device_cords : "
            << it[0] << ","
            << it[1] << ","
            << it[2] << "\n";
    }

    for (const auto& coord : device_cords) {

        // Compute linear element_offset=row×stride0​+col×stride1​
        int64_t element_offset =
            coord[0] * device_stride[0] +
            coord[1] * device_stride[1] +
            coord[2] * device_stride[2];

        std::cout
            << "element_offset = "
            << element_offset << "\n";

        // Convert to byte_address=virtual_offset+element_offset×element_size
        int64_t byte_address =
            virtual_offset +
            element_offset * element_size;

        std::cout
            << "byte_address = "
            << byte_address << "\n";
        //TBD
        // Ensure stick alignment
        /*TORCH_CHECK(
            byte_address % STICK_SIZE == 0,
            "Address is not stick aligned!");*/

        // Convert to stick_address=byte_address​/STICK_SIZE
        int64_t stick_address =
            byte_address / STICK_SIZE;

        std::cout
            << "stick_address = "
            << stick_address << "\n\n";

        stick_addresses.push_back(stick_address);

        ind_addresses_accessor[i++] =
            static_cast<int64_t>(stick_address);
    }


    auto reshaped_addresses_ind =
        ind_addresses.reshape(indices_shape);

    std::cout
        << "reshaped_addresses = "
        << reshaped_addresses_ind << "\n";

    // Move the result back to the original device (spyre)
    return reshaped_addresses_ind.to(original_device);
}
} // namespace spyre
