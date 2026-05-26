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


def test_indices_to_input_address_2d():
    """Comprehensive test suite for torch.ops.spyre.indices_to_input_address"""
    
    test_cases = [
        # (input_shape, indices, dim, virtual_offset)
        ((2, 64), [[0], [0]], 1, 0),
        ((4, 128), [[0, 64], [0, 64], [0, 64], [0, 64]], 1, 0),
        ((2, 512), [[0, 64, 128], [64, 128, 192]], 1, 0),
        ((8, 128), [[0], [64], [0], [64], [0], [64], [0], [64]], 1, 0),
    ]
    
    for input_shape, indices_list, dim, offset in test_cases:
        input_tensor = torch.randn(input_shape, dtype=torch.float16, device="spyre")
        indices = torch.tensor(indices_list, dtype=torch.int64, device="spyre")
        
        result = torch.ops.spyre.indices_to_input_address(
            input_tensor, dim, indices, offset
        )
        print(result)


if __name__ == "__main__":
    test_indirect_indices_input_address()
    test_indices_to_input_address_2d()


    """((128, 64), [[0, 64], [0, 64]], 0, 0),
        ((3, 256), [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 1, 0),
        ((2, 128), [[0, 64], [0, 64]], 1, 128),
        ((1, 256), [[0, 64, 128, 192]], 1, 0),
        ((2, 512), [[448, 384, 320, 256], [192, 128, 64, 0]], 1, 0),
        # Additional edge cases
        ((16, 4096), [[0, 64, 128, 192] for _ in range(16)], 1, 0),
        ((2, 4096), [[0, 4032], [64, 4032]], 1, 0),
        ((4, 512), [[0, 128, 256], [64, 192, 320], [128, 256, 384], [192, 320, 448]], 1, 0),
        ((3, 192), [[0, 64], [64, 128], [0, 64]], -1, 0),
        ((2, 256), [[0, 64, 128, 192]], 1, 1024),
        ((32, 8192), [[i * 64 for i in range(10)] for _ in range(32)], 1, 0),"""