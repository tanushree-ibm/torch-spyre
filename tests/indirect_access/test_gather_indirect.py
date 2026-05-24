import torch
import pytest

class TestIndicesToInputAddress:
    """Test suite for torch.ops.spyre.indices_to_input_address"""
    
    @pytest.mark.parametrize("input_shape,indices_pattern,dim,virtual_offset,expected_shape,test_name", [
        # Test 8.1: Very large input tensor
        (
            (64, 16384),
            lambda: [[i * 64 for i in range(20)] for _ in range(64)],
            1,
            0,
            (64, 20),
            "very_large_input_tensor"
        ),
        # Test 8.2: Many indices per row
        (
            (4, 8192),
            lambda: [[i * 64 for i in range(128)] for _ in range(4)],
            1,
            0,
            (4, 128),
            "many_indices_per_row"
        ),
        # Test 8.3: Maximum offset with large input
        (
            (8, 4096),
            lambda: [[0, 64, 128] for _ in range(8)],
            1,
            65536,
            (8, 3),
            "maximum_offset_large_input"
        ),
    ])
    def test_stress_cases(self, input_shape, indices_pattern, dim, virtual_offset, expected_shape, test_name):
        """Stress test cases for indices_to_input_address operation"""
        
        # Create input tensor
        input_tensor = torch.randn(input_shape, dtype=torch.float16, device="spyre")
        
        # Create indices tensor using the pattern function
        indices = torch.tensor(indices_pattern(), dtype=torch.int64, device="spyre")
        
        # Call the operation
        result = torch.ops.spyre.indices_to_input_address(
            input_tensor,
            dim,
            indices,
            virtual_offset
        )
        
        # Assertions
        assert result.shape == expected_shape, \
            f"Test '{test_name}': Expected shape {expected_shape}, got {result.shape}"
        
        assert result.dtype == torch.float32, \
            f"Test '{test_name}': Expected dtype float32, got {result.dtype}"
        
        assert result.device.type == "spyre", \
            f"Test '{test_name}': Expected device 'spyre', got {result.device.type}"
        
        print(f"✓ Test '{test_name}' passed: shape={result.shape}, dtype={result.dtype}")


# Alternative: Single test with all cases
def test_indices_to_input_address_stress_suite():
    """Single test function with all stress test cases"""
    
    test_cases = [
        {
            'name': 'Test 8.1: Very large input tensor',
            'input_shape': (64, 16384),
            'indices': [[i * 64 for i in range(20)] for _ in range(64)],
            'dim': 1,
            'virtual_offset': 0,
            'expected_shape': (64, 20)
        },
        {
            'name': 'Test 8.2: Many indices per row',
            'input_shape': (4, 8192),
            'indices': [[i * 64 for i in range(128)] for _ in range(4)],
            'dim': 1,
            'virtual_offset': 0,
            'expected_shape': (4, 128)
        },
        {
            'name': 'Test 8.3: Maximum offset with large input',
            'input_shape': (8, 4096),
            'indices': [[0, 64, 128] for _ in range(8)],
            'dim': 1,
            'virtual_offset': 65536,
            'expected_shape': (8, 3)
        }
    ]
    
    print("\n" + "=" * 80)
    print("Running Stress Test Suite for indices_to_input_address")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        try:
            print(f"\n{test_case['name']}")
            print("-" * 80)
            
            # Create input tensor
            input_tensor = torch.randn(
                test_case['input_shape'],
                dtype=torch.float16,
                device="spyre"
            )
            print(f"Input shape: {input_tensor.shape}")
            
            # Create indices tensor
            indices = torch.tensor(
                test_case['indices'],
                dtype=torch.int64,
                device="spyre"
            )
            print(f"Indices shape: {indices.shape}")
            print(f"Dimension: {test_case['dim']}")
            print(f"Virtual offset: {test_case['virtual_offset']}")
            
            # Call the operation
            result = torch.ops.spyre.indices_to_input_address(
                input_tensor,
                test_case['dim'],
                indices,
                test_case['virtual_offset']
            )
            
            # Verify results
            assert result.shape == test_case['expected_shape'], \
                f"Shape mismatch: expected {test_case['expected_shape']}, got {result.shape}"
            
            assert result.dtype == torch.float32, \
                f"Dtype mismatch: expected float32, got {result.dtype}"
            
            assert result.device.type == "spyre", \
                f"Device mismatch: expected 'spyre', got {result.device.type}"
            
            print(f"✓ PASSED")
            print(f"  Output shape: {result.shape}")
            print(f"  Output dtype: {result.dtype}")
            print(f"  Output device: {result.device}")
            passed += 1
            
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(test_cases)} total")
    print("=" * 80)
    
    assert failed == 0, f"{failed} test(s) failed"


# Compact version with subtest
def test_indices_to_input_address_compact():
    """Compact test with all cases using pytest subtests"""
    
    test_data = [
        ((64, 16384), 20, 64, 1, 0, (64, 20), "8.1_very_large"),
        ((4, 8192), 128, 4, 1, 0, (4, 128), "8.2_many_indices"),
        ((8, 4096), 3, 8, 1, 65536, (8, 3), "8.3_max_offset"),
    ]
    
    for input_shape, num_indices, num_rows, dim, offset, expected_shape, test_id in test_data:
        # Generate indices
        indices_list = [[i * 64 for i in range(num_indices)] for _ in range(num_rows)]
        
        # Create tensors
        input_tensor = torch.randn(input_shape, dtype=torch.float16, device="spyre")
        indices = torch.tensor(indices_list, dtype=torch.int64, device="spyre")
        
        # Run operation
        result = torch.ops.spyre.indices_to_input_address(
            input_tensor, dim, indices, offset
        )
        print(result)
        """
        # Verify
        assert result.shape == expected_shape, \
            f"Test {test_id}: shape mismatch"
        assert result.dtype == torch.float32, \
            f"Test {test_id}: dtype mismatch"
        
        print(f"✓ Test {test_id} passed")"""


if __name__ == "__main__":
    # Run the single comprehensive test
    #test_indices_to_input_address_stress_suite()
    
    # Or run the compact version
    test_indices_to_input_address_compact()

