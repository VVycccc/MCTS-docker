import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1024}, num_warps=4),
        triton.Config({'BLOCK_M': 2048}, num_warps=8),
        triton.Config({'BLOCK_M': 4096}, num_warps=8),
    ],
    key=['M'],
)
@triton.jit
def diag_matmul_kernel(ptr_a, ptr_b, ptr_c, N, M,
                       BLOCK_M: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    a = tl.load(ptr_a + pid_n)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    b = tl.load(ptr_b + pid_n * M + offs_m, mask=mask_m, other=0.0)
    c = a * b
    tl.store(ptr_c + pid_n * M + offs_m, c, mask=mask_m)


def run(A, B):
    N, M = B.shape
    C = torch.empty_like(B)
    grid = lambda meta: (N, triton.cdiv(M, meta['BLOCK_M']))
    diag_matmul_kernel[grid](A, B, C, N, M)
    return C
