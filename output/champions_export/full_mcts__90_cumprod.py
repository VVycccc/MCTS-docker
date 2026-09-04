import torch
import triton
import triton.language as tl

dim = 1

@triton.jit
def _mul(a, b):
    return a * b

@triton.jit
def cumprod_kernel(ptr_in, ptr_out, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid

    arange = tl.arange(0, BLOCK_SIZE)
    running = 1.0
    for col_start in range(0, N, BLOCK_SIZE):
        offs = col_start + arange
        mask = offs < N
        x = tl.load(ptr_in + row * N + offs, mask=mask, other=1.0)

        cumprod_x = tl.associative_scan(x, axis=0, combine_fn=_mul)
        result = cumprod_x * running

        block_prod = tl.sum(tl.where(arange == BLOCK_SIZE - 1, cumprod_x, 0.0))
        running = running * block_prod

        tl.store(ptr_out + row * N + offs, result, mask=mask)


def get_inputs():
    return [torch.rand(32768, 32768)]


def get_init_inputs():
    return [dim]


def run(x):
    d = get_init_inputs()[0]
    if d != 1 or x.dim() != 2:
        return torch.cumprod(x, dim=d)
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 2048
    grid = (M,)
    cumprod_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return out
