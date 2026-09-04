import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/9_Matmul_Subtract_Multiply_ReLU_weights.pt"
_W = None
_device = None


@triton.jit
def _linear_sub_mul_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    subtract_value, multiply_value,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        # weight is [N, K] row-major; load [BLOCK_N, BLOCK_K] along contiguous K
        b_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

        if EVEN_M and EVEN_N and EVEN_K:
            a = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
        else:
            if EVEN_K:
                a_mask = offs_m[:, None] < M
                b_mask = offs_n[:, None] < N
            else:
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, tl.trans(b_tile), acc, allow_tf32=True)

    if EVEN_N:
        bias = tl.load(b_ptr + offs_n)
    else:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = acc - subtract_value
    acc = acc * multiply_value
    acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))
    else:
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def run(x):
    global _W, _device
    if _W is None or _device != str(x.device):
        state = torch.load(_weights_path, map_location="cpu", weights_only=True)
        _W = {k: v.to(x.device) for k, v in state.items()}
        _device = str(x.device)

    weight = _W["linear.weight"]
    bias = _W["linear.bias"]

    x = x.contiguous()
    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    EVEN_M = (M % BLOCK_M == 0)
    EVEN_N = (N % BLOCK_N == 0)
    EVEN_K = (K % BLOCK_K == 0)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _linear_sub_mul_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        2.0, 1.5,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_K=EVEN_K,
        num_warps=8, num_stages=3,
    )
    return out
