import torch
import triton
import triton.language as tl

@triton.jit
def scalar_mul_kernel(a_ptr, out_ptr, s_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    s = tl.load(s_ptr)
    out = a * s
    tl.store(out_ptr + offs, out, mask=mask)

def run(A, s):
    out = torch.empty_like(A)
    n = A.numel()
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, device=A.device, dtype=A.dtype)
    elif s.device != A.device or s.dtype != A.dtype:
        s = s.to(A.device, A.dtype)
    grid = (triton.cdiv(n, 4096),)
    scalar_mul_kernel[grid](A, out, s, n, BLOCK_SIZE=4096, num_warps=8)
    return out
