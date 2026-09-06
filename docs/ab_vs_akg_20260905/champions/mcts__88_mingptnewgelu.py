import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    y = x * tl.sigmoid(1.5957691216057308 * (x + 0.044715 * x * x * x))
    tl.store(y_ptr + offs, y)

def run(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return y