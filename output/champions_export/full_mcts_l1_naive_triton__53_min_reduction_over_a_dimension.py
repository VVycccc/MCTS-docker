import torch
import triton
import triton.language as tl

@triton.jit
def min_reduce_kernel(x_ptr, out_ptr, OUTER, REDUCE, INNER,
                      BLOCK_INNER: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_i = pid_n * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    mask_i = offs_i < INNER

    acc = tl.full((BLOCK_INNER,), float('inf'), tl.float32)

    o = pid_m
    for d_start in range(0, REDUCE, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < REDUCE
        ptrs = x_ptr + o * REDUCE * INNER + offs_d[:, None] * INNER + offs_i[None, :]
        mask_2d = mask_d[:, None] & mask_i[None, :]
        x = tl.load(ptrs, mask=mask_2d, other=float('inf'))
        m = tl.min(x, axis=0)
        acc = tl.minimum(acc, m)

    tl.store(out_ptr + o * INNER + offs_i, acc, mask=mask_i)


def run(x):
    dim = 1
    OUTER = x.shape[0]
    REDUCE = x.shape[dim]
    INNER = x.shape[2]
    out = torch.empty((OUTER, INNER), device=x.device, dtype=x.dtype)
    BLOCK_INNER = 128
    BLOCK_D = 128
    grid = (OUTER, triton.cdiv(INNER, BLOCK_INNER))
    min_reduce_kernel[grid](x, out, OUTER, REDUCE, INNER,
                            BLOCK_INNER=BLOCK_INNER, BLOCK_D=BLOCK_D, num_warps=8)
    return out
