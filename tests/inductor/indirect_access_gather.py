# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

import torch._dynamo
torch._dynamo.config.dynamic_shapes = False

# TODO : Address computation for multicore : WIP
# Until then set `SENCORES=1` testing indirect access



def test_gather_3d_128():
    print("\n" + "=" * 70)
    print("3D Gather Test on Spyre (Input Shape: 2 x 8 x 128)")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    # -----------------------------
    # 1. CPU Input (ground truth)
    # -----------------------------
    input_cpu = torch.randn(2, 4, 128, dtype=torch.float16)
    #print("CPU Input Shape:", input_cpu.shape)

    # No padding needed here
    input_spyre = input_cpu.to("spyre")
    #print("Spyre Input Shape:", input_spyre.shape)

    # -----------------------------
    # 2. Build index tensor
    # Gather 4 rows from dim=1 for each batch
    # -----------------------------
    row_indices = torch.tensor(
        [
            #[2, 0, 1, 2],  # batch 0
            #[3, 1, 0, 3],  # batch 1
            [2, 0],  # batch 0
            [3, 1],  # batch 1
        ],
        dtype=torch.int64,
    )

    # Expand across last dim so gather(dim=1) copies full rows
    #index_cpu = row_indices.unsqueeze(-1).expand(-1, -1, 128)  # (2, 4, 128)
    index_cpu = (row_indices.unsqueeze(-1).expand(-1, -1, 128).contiguous()) # (2, 4, 128)
    print(index_cpu.dtype)
    print(index_cpu.is_contiguous())
    print(index_cpu)


    index_spyre = index_cpu.to("spyre")
    #print("Index Shape:", index_spyre.shape)

    # -----------------------------
    # 3. Expected result (CPU)
    # -----------------------------
    expected = torch.stack([
        torch.stack([
            input_cpu[0, 2],
            input_cpu[0, 0],
            #input_cpu[0, 1],
            #input_cpu[0, 2],
        ]),
        torch.stack([
            input_cpu[1, 3],
            input_cpu[1, 1],
            #input_cpu[1, 0],
            #input_cpu[1, 3],
        ]),
    ])
    #print("Expected Shape:", expected.shape)

    # -----------------------------
    # 4. Compile + Run on Spyre
    # -----------------------------
    compiled_fn = torch.compile(gather_fn)

    result_spyre = compiled_fn(input_spyre, 1, index_spyre)
    result_cpu = result_spyre.cpu()

    print("Result Shape (Spyre):", result_spyre.shape)

    # -----------------------------
    # 5. Validate
    # -----------------------------
    print("Result CPU:")
    print(result_cpu)
    #print("Expected:")
    #print(expected)

    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Mismatch!\nMax diff: {torch.max(torch.abs(result_cpu - expected))}"
    )

    #print("✓ Test passed!")

    return result_spyre

if __name__ == "__main__":
    import os

    try:
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "1"
        test_gather_3d_128() # Values are wrongly fetched

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
