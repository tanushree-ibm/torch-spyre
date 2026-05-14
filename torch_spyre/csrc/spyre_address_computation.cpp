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

namespace spyre {

at::Tensor compute_addresses_from_input_indices(
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& indices,
    int64_t virtual_offset,
    const std::vector<int64_t>& device_size,
    const std::vector<int64_t>& device_stride,
    int64_t element_size) {

    std::cout << "input.size() = " << input.sizes() << "\n";
    std::cout << "dim = " << dim << "\n";
    std::cout << "indices.size() = " << indices.sizes() << "\n";

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
    std::vector<std::pair<int64_t, int64_t>> dim_cords;

    for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
        for (int64_t j = 0; j < indices_cpu.size(1); ++j) {

            int64_t val = acc[i][j]; //index[i][j]

            std::cout
                << "indices[" << i << "][" << j << "] = "
                << val << "\n";

            if (dim == 1) {
                dim_cords.push_back({i, val}); //[i, index[i, j]] 
            } else {
                dim_cords.push_back({val, j}); //[index[i, j], j] 
            }
        }
    }

    for (auto it : dim_cords) {
        std::cout
            << "dim_cords : "
            << it.first << ","
            << it.second << "\n";
    }

    int64_t numElements = dim_cords.size();
    auto ind_addresses = at::zeros(
        {numElements},
        at::TensorOptions().dtype(at::kFloat));

    auto ind_addresses_accessor =
        ind_addresses.accessor<float, 1>();

    constexpr int64_t STICK_SIZE = 128;

    std::vector<int64_t> stick_addresses;

    int i = 0;

    for (const auto& coord : dim_cords) {

        int64_t row = coord.first;
        int64_t col = coord.second;

        // Compute linear element_offset=row×stride0​+col×stride1​
        int64_t element_offset =
            row * device_stride[0] +
            col * device_stride[1];

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

        // Ensure stick alignment
        TORCH_CHECK(
            byte_address % STICK_SIZE == 0,
            "Address is not stick aligned!");

        // Convert to stick_address=byte_address​/STICK_SIZE
        int64_t stick_address =
            byte_address / STICK_SIZE;

        std::cout
            << "stick_address = "
            << stick_address << "\n\n";

        stick_addresses.push_back(stick_address);

        ind_addresses_accessor[i++] =
            static_cast<float>(stick_address);
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
