import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_matmul_sub_mul_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    subtract_value,
    multiply_value,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            if EVEN_M:
                x = tl.load(x_ptrs)
            else:
                x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
            if EVEN_N:
                w = tl.load(w_ptrs)
            else:
                w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_rem = K - k * BLOCK_K
            k_mask = offs_k < k_rem
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x, w, acc=acc, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if EVEN_N:
        b = tl.load(b_ptr + offs_n)
    else:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    acc = acc - subtract_value
    acc = acc * multiply_value
    acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


_W = None
_W_device = None

def run(x):
    global _W, _W_device
    if _W is None or _W_device != str(x.device):
        _weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/9_Matmul_Subtract_Multiply_ReLU_weights.pt"
        weights = torch.load(_weights_path, map_location='cpu', weights_only=True)
        _W = {k: v.to(x.device) for k, v in weights.items()}
        _W['weight_t_fp16'] = _W['linear.weight'].t().contiguous().to(torch.float16)
        _W_device = str(x.device)

    weight_t = _W['weight_t_fp16']
    bias = _W['linear.bias']

    M, K = x.shape
    N = weight_t.shape[1]
    x_fp16 = x.to(torch.float16)

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8
    EVEN_K = K % BLOCK_K == 0
    EVEN_M = M % BLOCK_M == 0
    EVEN_N = N % BLOCK_N == 0

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fused_matmul_sub_mul_relu_kernel[grid](
        x_fp16, weight_t, bias, out,
        M, N, K,
        2.0,
        1.5,
        x_fp16.stride(0), x_fp16.stride(1),
        weight_t.stride(1), weight_t.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        EVEN_K=EVEN_K,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        num_stages=3,
        num_warps=8,
    )

    return out
