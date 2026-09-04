import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/9_Matmul_Subtract_Multiply_ReLU_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_sub_mul_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    subtract_value,
    multiply_value,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_m < M

    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
        else:
            k_mask = (k_start + offs_k) < K
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc=acc, allow_tf32=True)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    b_ptrs = b_ptr + offs_n
    if EVEN_N:
        bias = tl.load(b_ptrs)
    else:
        bias = tl.load(b_ptrs, mask=mask_n, other=0.0)
    acc += bias[None, :]

    acc = acc - subtract_value
    acc = acc * multiply_value
    acc = tl.maximum(acc, 0.0)

    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    if EVEN_M and EVEN_N:
        tl.store(y_ptrs, acc)
    elif EVEN_M:
        tl.store(y_ptrs, acc, mask=mask_n[None, :])
    elif EVEN_N:
        tl.store(y_ptrs, acc, mask=mask_m[:, None])
    else:
        tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    M, K = x.shape
    w = _weights['linear.weight']
    b = _weights['linear.bias']
    N = w.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

    matmul_sub_mul_relu_kernel[grid](
        x, w, b, y,
        M, N, K,
        2.0,
        1.5,
        EVEN_M=M % 128 == 0, EVEN_N=N % 128 == 0, EVEN_K=K % 32 == 0,
    )

    return y
