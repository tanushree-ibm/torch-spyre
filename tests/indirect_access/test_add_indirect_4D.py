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
    input = torch.randn((1,1,2,128), dtype=torch.float16, device="spyre")
    print(input.cpu())

    # IMPORTANT: DeepTools indirect access only works at stick boundaries!
    # Each stick holds 64 FP16 elements (128 bytes / 2 bytes per element)
    # We can only access elements at stick boundaries: 0, 64, 128, etc.
    # Stick 0: elements 0-63, Stick 1: elements 64-127
    #index1 = torch.tensor(([[0, 64, 0], [64, 0, 64]]), dtype=torch.int64, device="spyre")   # Only stick-aligned indices
    index1 = torch.tensor(([[[[0, 64], [0, 64]]]]), dtype=torch.int64, device="spyre")   # Only stick-aligned indices


    print(
    f"Index 1: {index1.tolist()} ")
    #f"(sticks: {[[x // 64 for x in row] for row in index1.tolist()]})")

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

def test_indirect_indices_input_address_4D():
    """Test indirect_add using stick addresses as expected by DeepTools.

    DeepTools indirect access operates at stick granularity (128 bytes).
    For FP16 data, each stick holds 64 elements (128 bytes / 2 bytes per element).

    Memory layout for 128 FP16 elements in [2, 64] device layout:
    - Stick 0: elements 0-63   (row 0)
    - Stick 1: elements 64-127 (row 1)
    """
    
    # Data tensors (128 FP16 elements = 2 sticks)
    #input = torch.tensor(([[10, 20, 30], [40, 50, 60]]), dtype=torch.float16, device="spyre")
    #input = torch.randn((1,1,2,128), dtype=torch.float16, device="spyre")
    #print(input.cpu())

    # Paged attention dimensions - REDUCED for testing
    NUM_PAGES = 8       # Total pages in cache
    PAGE_SIZE = 4       # Tokens per page
    NUM_HEADS = 2       # Attention heads
    HEAD_SIZE = 64      # Dimension per head

    # 4D input: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    input = torch.randn(
        NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE,
        dtype=torch.float16
    )
    print("TANU \n")
    print(input.cpu())

    # Pad HEAD_SIZE to stick size (64 is already aligned)
    input_tensor = input.to("spyre")
    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"Total elements: {input_tensor.numel():,}")
    # print(f"Input Tensor : {input_tensor}")

    # Index tensor: select 2 pages (e.g., pages 1 and 5)
    # Shape: [BATCH_SIZE] where BATCH_SIZE=2
    page_indices = torch.tensor([1, 5], dtype=torch.int64)

    # CRITICAL: Index tensor should be 1D for page selection
    # We're gathering along dim=0 (NUM_PAGES dimension)
    # The index shape should match the output shape for non-gathered dims
    # Output will be: [2, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    
    # Expand index to match output dimensions (except gathered dim)
    index_tensor = torch.zeros(2, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.int64)
    index_tensor[0, 0, 0, :len(page_indices)] = page_indices
    index_tensor = index_tensor.to("spyre")

    print(f"Page indices: {page_indices.tolist()}")
    print(f"Index Tensor Shape: {index_tensor.shape}")
    print(f"Index Tensor : ")
    print(index_tensor.cpu())


    # Indices: which pages to gather (e.g., pages 2 and 5)
    page_indices = torch.tensor([2, 5], dtype=torch.int64, device="spyre")

    # Expand indices to match full shape
    indices_expanded = page_indices.view(-1, 1, 1, 1).expand(
        -1, PAGE_SIZE, NUM_HEADS, HEAD_SIZE
    )  # Shape: [2, 4, 2, 64]
    print(indices_expanded)

    dim = 0
    address_a = torch.ops.spyre.indices_to_input_address(
        input_tensor,
        dim,
        indices_expanded.to("spyre"),
        virtual_offset=0,  # Base address for input_a (in bytes, will be converted to sticks)
    )
    print(address_a)
    """

    # IMPORTANT: DeepTools indirect access only works at stick boundaries!
    # Each stick holds 64 FP16 elements (128 bytes / 2 bytes per element)
    # We can only access elements at stick boundaries: 0, 64, 128, etc.
    # Stick 0: elements 0-63, Stick 1: elements 64-127
    #index1 = torch.tensor(([[0, 64, 0], [64, 0, 64]]), dtype=torch.int64, device="spyre")   # Only stick-aligned indices
    index1 = torch.tensor(([[[[0, 64], [0, 64]]]]), dtype=torch.int64, device="spyre")   # Only stick-aligned indices


    print(
    f"Index 1: {index1.tolist()} ")
    #f"(sticks: {[[x // 64 for x in row] for row in index1.tolist()]})")

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
    print(address_a)"""

if __name__ == "__main__":
    test_indirect_indices_input_address()
    test_indirect_indices_input_address_4D()
