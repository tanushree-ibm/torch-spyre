import torch
def test_indirect_indices_input_address():
    """Test indirect_add using stick addresses as expected by DeepTools.

    DeepTools indirect access operates at stick granularity (128 bytes).
    For FP16 data, each stick holds 64 elements (128 bytes / 2 bytes per element).

    Memory layout for 128 FP16 elements in [2, 64] device layout:
    - Stick 0: elements 0-63   (row 0)
    - Stick 1: elements 64-127 (row 1)
    """
    
    # Data tensors (128 FP16 elements = 2 sticks)
    #input = torch.tensor(([[10, 20, 30], [40, 50, 60]]), dtype=torch.float16, device="spyre")
    input = torch.randn((2,128), dtype=torch.float16, device="spyre")
    print(input.cpu())


    # IMPORTANT: DeepTools indirect access only works at stick boundaries!
    # Each stick holds 64 FP16 elements (128 bytes / 2 bytes per element)
    # We can only access elements at stick boundaries: 0, 64, 128, etc.
    # Stick 0: elements 0-63, Stick 1: elements 64-127
    #index1 = torch.tensor(([[0, 64, 0], [64, 0, 64]]), dtype=torch.int64, device="spyre")   # Only stick-aligned indices
    index1 = torch.tensor(([[0, 64], [0, 64]]), dtype=torch.int64, device="spyre")   # Only stick-aligned indices


    print(
    f"Index 1: {index1.tolist()} "
    f"(sticks: {[[x // 64 for x in row] for row in index1.tolist()]})")

    dim = 1
    # Convert to stick addresses
    # DeepTools expects stick addresses (in units of 128 bytes), not byte addresses
    # The indices_to_address function now correctly converts to stick addresses
    address_a = torch.ops.spyre.indices_to_input_address(
        input,
        dim,
        index1,
        virtual_offset=0,  # Base address for input_a (in bytes, will be converted to sticks)
    )
    print(address_a)

if __name__ == "__main__":
    test_indirect_add_address()
