import torch

def test_3d_minimal():
    """
    Minimal 3D tensor address computation test
    
    Input shape: [2, 2, 128] (FP16)
    - 2 batches
    - 2 rows per batch
    - 128 columns (2 sticks of 64 elements each)
    """
    
    print("MINIMAL 3D ADDRESS COMPUTATION TEST")
    print("="*80)
    
    # Create minimal 3D input
    input_shape = (2, 2, 128)
    input_tensor = torch.randn(input_shape, dtype=torch.float16, device="spyre")
    
    print(f"Input shape: {input_shape}")
    print(f"Elements per stick: 64 (128 bytes / 2 bytes)")
    print(f"Total sticks: {2 * 2 * 2} (batch × rows × tiles)")
    
    # Test: Index along dim=2 (columns)
    print("\nIndexing along dim=2 (columns)")
    print("-"*80)
    
    # Select first element (col 0) and second stick (col 64) for each batch/row
    indices = torch.tensor([
        [[0, 64], [0, 64]],  # Batch 0: both rows
        [[0, 64], [0, 64]]   # Batch 1: both rows
    ], dtype=torch.int64, device="spyre")
    
    print(f"Indices shape: {indices.shape}")
    print(f"Indices:\n{indices.cpu()}")
    
    result = torch.ops.spyre.indices_to_input_address(
        input_tensor,
        dim=2,
        indices=indices,
        virtual_offset=0
    )
    
    print(f"\nResult shape: {result.shape}")
    print(f"Result addresses:\n{result.cpu()}")
    
    # Manual calculation (column-major layout)
    print("\n" + "="*80)
    print("MANUAL VERIFICATION (Column-Major)")
    print("="*80)
    
    # For [2, 2, 128] with column-major:
    # Device strides: [batch_stride, row_stride, 1]
    # Each tile (64 elements) = 2 batches × 2 rows = 4 elements = 0.125 sticks
    
    coords = [
        (0, 0, 0),   # Batch 0, Row 0, Col 0
        (0, 0, 64),  # Batch 0, Row 0, Col 64
        (0, 1, 0),   # Batch 0, Row 1, Col 0
        (0, 1, 64),  # Batch 0, Row 1, Col 64
        (1, 0, 0),   # Batch 1, Row 0, Col 0
        (1, 0, 64),  # Batch 1, Row 0, Col 64
        (1, 1, 0),   # Batch 1, Row 1, Col 0
        (1, 1, 64),  # Batch 1, Row 1, Col 64
    ]
    
    print(f"Shape: {input_tensor.shape}")
    print(f"Strides: {input_tensor.stride()}")  # Output: (256, 128, 1)
    print("\nCoordinate → Stick Address:")
    host_stride = input_tensor.stride()

    addresses = []
    for batch, row, col in coords:
        tile = col // 64
        col_in_tile = col % 64
        #device stride_map =[128, 64, 256, 1],
        element_offset = batch * 128 + row * 64 + tile * 256 + col_in_tile * 1
        byte_address = element_offset * 2  # FP16 = 2 bytes
        stick_address = byte_address // 128
        addresses.append(stick_address)
        
        print(f"  [{batch}, {row}, {col:3d}] → offset={element_offset:4d}, "
              f"bytes={byte_address:5d}, stick={stick_address}")
    print(addresses)
    expected = torch.tensor(addresses, dtype=torch.int64).reshape(indices.shape)
    print(expected)
    # Expected result
    #expected = torch.tensor([
    #    [[0, 4], [1, 5]],  # Batch 0
    #    [[2, 6], [3, 7]]   # Batch 1
    #], dtype=torch.int64)
    
    print(f"\nExpected addresses:\n{expected}")
    
    # Verify
    match = torch.allclose(result.cpu(), expected, rtol=0, atol=0)
    #match = torch.allclose(result.cpu(), res, rtol=0, atol=0)
    print(f"\n{'✓ PASSED' if match else '✗ FAILED'}: Addresses match = {match}")
    
    return match

# Run test
if __name__ == "__main__":
    test_3d_minimal()
