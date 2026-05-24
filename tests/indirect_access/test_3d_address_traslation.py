import torch

def test_3d_address_translation():
    """
    Test 3D tensor address translation for torch.ops.spyre.indices_to_input_address
    
    Tensor shape: [2, 4, 256] (FP16)
    - 2 batches
    - 4 rows per batch
    - 256 columns (4 sticks of 64 elements each)
    """
    
    #print("=" * 80)
    print("3D TENSOR ADDRESS TRANSLATION TEST")
    #print("=" * 80)
    
    # Create 3D input tensor
    batch_size = 2
    num_rows = 4
    num_cols = 256
    input_shape = (batch_size, num_rows, num_cols)
    
    input_tensor = torch.randn(input_shape, dtype=torch.float16, device="spyre")
    
    print(f"\nInput tensor shape: {input_tensor.shape}")
    print(f"Data type: {input_tensor.dtype} (2 bytes per element)")
    print(f"Elements per stick: 64 (128 bytes / 2 bytes)")
    print(f"Total sticks per row: {num_cols // 64}")
    
    # Test Case 1: Index along dim=2 (last dimension - columns)
    #print("\n" + "-" * 80)
    print("TEST CASE 1: Indexing along dim=2 (columns)")
    #print("-" * 80)
    
    # Indices: Select specific columns (stick-aligned)
    indices_dim2 = torch.tensor([
        [[0, 64, 128], [0, 64, 128], [0, 64, 128], [0, 64, 128]],  # Batch 0
        [[0, 64, 128], [0, 64, 128], [0, 64, 128], [0, 64, 128]]   # Batch 1
    ], dtype=torch.int64, device="spyre")
    
    print(f"Indices shape: {indices_dim2.shape}")
    print(f"Indices:\n{indices_dim2.cpu()}")
    
    result_dim2 = torch.ops.spyre.indices_to_input_address(
        input_tensor,
        dim=2,
        indices=indices_dim2,
        virtual_offset=0
    )
    
    print(f"\nResult shape: {result_dim2.shape}")
    print(f"Result addresses:\n{result_dim2.cpu()}")
    
    """
    # Manual verification for dim=2
    print("\nManual Verification (dim=2):")
    print("Host → Device → Address")
    
    test_coords = [
        ([0, 0, 0], "Batch 0, Row 0, Col 0"),
        ([0, 0, 64], "Batch 0, Row 0, Col 64"),
        ([0, 0, 128], "Batch 0, Row 0, Col 128"),
        ([1, 2, 64], "Batch 1, Row 2, Col 64"),
    ]
    
    for host_coord, desc in test_coords:
        b, r, c = host_coord
        
        # Convert to device coordinate: [b, r, c/64, c%64]
        device_coord = [b, r, c // 64, c % 64]
        
        # Calculate address (assuming device_stride = [1024, 256, 64, 1])
        # These strides depend on your actual device layout
        device_stride = [1024, 256, 64, 1]  # Example strides
        element_offset = sum(device_coord[i] * device_stride[i] for i in range(4))
        byte_address = element_offset * 2  # 2 bytes for FP16
        stick_address = byte_address // 128
        
        print(f"\n{desc}:")
        print(f"  Host coord:   {host_coord}")
        print(f"  Device coord: {device_coord}")
        print(f"  Element offset: {element_offset}")
        print(f"  Byte address: {byte_address}")
        print(f"  Stick address: {stick_address}")
    
    # Test Case 2: Index along dim=1 (middle dimension - rows)
    print("\n" + "-" * 80)
    print("TEST CASE 2: Indexing along dim=1 (rows)")
    print("-" * 80)
    
    indices_dim1 = torch.tensor([
        [[0, 2], [0, 2]],  # Batch 0: select rows 0 and 2
        [[1, 3], [1, 3]]   # Batch 1: select rows 1 and 3
    ], dtype=torch.int64, device="spyre")
    
    print(f"Indices shape: {indices_dim1.shape}")
    print(f"Indices:\n{indices_dim1.cpu()}")
    
    result_dim1 = torch.ops.spyre.indices_to_input_address(
        input_tensor,
        dim=1,
        indices=indices_dim1,
        virtual_offset=0
    )
    
    print(f"\nResult shape: {result_dim1.shape}")
    print(f"Result addresses:\n{result_dim1.cpu()}")
    
    # Test Case 3: Index along dim=0 (first dimension - batches)
    print("\n" + "-" * 80)
    print("TEST CASE 3: Indexing along dim=0 (batches)")
    print("-" * 80)
    
    indices_dim0 = torch.tensor([
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]]
    ], dtype=torch.int64, device="spyre")
    
    print(f"Indices shape: {indices_dim0.shape}")
    print(f"Indices:\n{indices_dim0.cpu()}")
    
    result_dim0 = torch.ops.spyre.indices_to_input_address(
        input_tensor,
        dim=0,
        indices=indices_dim0,
        virtual_offset=0
    )
    
    print(f"\nResult shape: {result_dim0.shape}")
    print(f"Result addresses:\n{result_dim0.cpu()}")
    
    # Test Case 4: Edge cases
    print("\n" + "-" * 80)
    print("TEST CASE 4: Edge Cases")
    print("-" * 80)
    
    # All zeros
    indices_zeros = torch.zeros((2, 4, 3), dtype=torch.int64, device="spyre")
    result_zeros = torch.ops.spyre.indices_to_input_address(
        input_tensor, dim=2, indices=indices_zeros, virtual_offset=0
    )
    print(f"All zeros indices - Result shape: {result_zeros.shape}")
    print(f"All zeros addresses:\n{result_zeros.cpu()}")
    
    # Maximum indices
    indices_max = torch.tensor([
        [[192, 192, 192], [192, 192, 192], [192, 192, 192], [192, 192, 192]],
        [[192, 192, 192], [192, 192, 192], [192, 192, 192], [192, 192, 192]]
    ], dtype=torch.int64, device="spyre")
    result_max = torch.ops.spyre.indices_to_input_address(
        input_tensor, dim=2, indices=indices_max, virtual_offset=0
    )
    print(f"\nMax indices (192) - Result shape: {result_max.shape}")
    print(f"Max addresses:\n{result_max.cpu()}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)"""


def test_3d_detailed_verification():
    """Detailed step-by-step verification of 3D address translation"""
    
    print("\n" + "=" * 80)
    print("DETAILED 3D ADDRESS TRANSLATION VERIFICATION")
    print("=" * 80)
    
    # Simple 3D tensor for easy verification
    input_tensor = torch.randn((2, 2, 128), dtype=torch.float16, device="spyre")
    
    print(f"\nInput shape: {input_tensor.shape}")
    print("Layout: [batch=2, rows=2, cols=128]")
    print("Sticks per row: 2 (128 / 64)")
    
    # Test specific coordinates
    test_cases = [
        {
            'name': 'First element',
            'indices': torch.tensor([[[0]], [[0]]], dtype=torch.int64, device="spyre"),
            'dim': 2,
            'expected_host': [[0, 0, 0], [1, 0, 0]],
            'expected_device': [[0, 0, 0, 0], [1, 0, 0, 0]],
        },
        {
            'name': 'Second stick boundary',
            'indices': torch.tensor([[[64]], [[64]]], dtype=torch.int64, device="spyre"),
            'dim': 2,
            'expected_host': [[0, 0, 64], [1, 0, 64]],
            'expected_device': [[0, 0, 1, 0], [1, 0, 1, 0]],
        },
        {
            'name': 'Multiple positions',
            'indices': torch.tensor([[[0, 64]], [[0, 64]]], dtype=torch.int64, device="spyre"),
            'dim': 2,
            'expected_host': [[0, 0, 0], [0, 0, 64], [1, 0, 0], [1, 0, 64]],
            'expected_device': [[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0]],
        }
    ]
    
    for test in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Test: {test['name']}")
        print(f"{'-' * 80}")
        
        result = torch.ops.spyre.indices_to_input_address(
            input_tensor,
            dim=test['dim'],
            indices=test['indices'],
            virtual_offset=0
        )
        
        print(f"Indices: {test['indices'].cpu()}")
        print(f"Result shape: {result.shape}")
        print(f"Result addresses: {result.cpu()}")
        
        print(f"\nExpected transformations:")
        for i, (host, device) in enumerate(zip(test['expected_host'], test['expected_device'])):
            print(f"  {i+1}. Host {host} → Device {device}")


if __name__ == "__main__":
    # Run basic test
    test_3d_address_translation()
    
    # Run detailed verification
    #test_3d_detailed_verification()

