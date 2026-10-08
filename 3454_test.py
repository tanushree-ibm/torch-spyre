import torch
DEVICE = "spyre"

# Destination view has element stride 2 in the last dimension.
rows, cols = 4096, 4096
src   = torch.rand(rows, cols, device=DEVICE, dtype=torch.bfloat16)
alloc = torch.empty(rows * cols * 2, device=DEVICE, dtype=torch.bfloat16)
alloc.view(rows, cols * 2)[:, ::2].copy_(src)

# InductorError: Unexpected stick expression 2*(Mod(d1, 32))
# expected Mod(var, 64), a bare variable, 0, or any of those with a constant offset
