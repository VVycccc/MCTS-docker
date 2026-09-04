import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp_weights.pt"
_W = None

def _init_weights(device):
    global _W
    w = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _W = {k: v.to(device) for k, v in w.items()}
    _W['w_sum'] = _W['linear.weight'].sum(dim=0)   # [in_features]
    _W['bias_sum'] = _W['linear.bias'].sum()       # scalar

@triton.jit
def matmul_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                     mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(b))

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def sum_reduce_kernel(in_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    acc = 0.0
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        vals = tl.load(in_ptr + pid * N + offs_n, mask=mask_n, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, acc)

@triton.jit
def matvec_kernel(x_ptr, w_ptr, bias_ptr, out_ptr, M, K, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    acc = 0.0
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        if EVEN_K:
            x_row = tl.load(x_ptr + pid * K + offs_k)
            w_vals = tl.load(w_ptr + offs_k)
        else:
            mask_k = offs_k < K
            x_row = tl.load(x_ptr + pid * K + offs_k, mask=mask_k, other=0.0)
            w_vals = tl.load(w_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(x_row * w_vals, axis=0)
    bias_val = tl.load(bias_ptr)
    acc += bias_val
    tl.store(out_ptr + pid, acc)

def run(x):
    global _W
    if _W is None or str(next(iter(_W.values())).device) != str(x.device):
        _init_weights(x.device)

    w_sum = _W['w_sum']       # [in_features]
    bias_sum = _W['bias_sum']  # scalar

    x = x.to(torch.float32)
    M, K = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK_K = 8192
    matvec_kernel[(M,)](x, w_sum, bias_sum, out, M, K, BLOCK_K=BLOCK_K, EVEN_K=(K % BLOCK_K == 0), num_warps=8, num_stages=2)
    return out.unsqueeze(1)
