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



def test_index_select():
    def index_select_fn(input, dim, index):
        return torch.index_select(input, dim, index)  

    input_cpu = torch.randn((3, 128), dtype=torch.float16)
    input_spyre = input_cpu.to("spyre")

    row_indices = torch.tensor([0,1,2],dtype=torch.int64,)
    index_cpu = row_indices
    #print(index_cpu.dtype)
    #print(index_cpu)

    index_spyre = index_cpu.to("spyre")
    compiled_fn = torch.compile(index_select_fn)
    expected = compiled_fn(input_cpu, 0, index_cpu)
    print(expected)

    result_spyre = compiled_fn(input_spyre, 0, index_spyre)
    result_cpu = result_spyre.cpu()
    print("Result Shape (Spyre):", result_spyre.shape)

    print(result_cpu)
    assert torch.allclose(result_cpu[0], expected[0], rtol=1e-3, atol=1e-3), \
        f"Result mismatch!\nExpected: {expected}\nGot: {result_cpu}"
    print("✓ Assertion passed: Result matches expected values")

    

if __name__ == "__main__":
    import os

    try:
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "1"
        test_index_select() # Values are wrongly fetched

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

