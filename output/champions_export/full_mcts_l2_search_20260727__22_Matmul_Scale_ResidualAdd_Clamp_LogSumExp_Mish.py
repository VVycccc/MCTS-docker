import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def fused_matmul_expsum_kernel(x_ptr, w_ptr, b_ptr, sum_ptr, M, N, K,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                GROUP_M: tl.constexpr):
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

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x, w, acc=acc, allow_tf32=True)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    acc = acc * 4.0
    acc = tl.minimum(tl.maximum(acc, -10.0), 10.0)
    exp_val = tl.exp(acc)
    row_sum = tl.sum(exp_val, axis=1)
    tl.atomic_add(sum_ptr + offs_m, row_sum, mask=offs_m < M)


@triton.jit
def elementwise_kernel(x_ptr, out_ptr, total, scale_factor, clamp_min, clamp_max,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x * scale_factor
    x = x + x
    x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)

    max_val = float('-inf')
    for off in range(0, N, BLOCK_N):
        mask = (off + offs_n) < N
        x = tl.load(x_ptr + pid * N + off + offs_n, mask=mask, other=float('-inf'))
        current_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, current_max)

    sum_val = 0.0
    for off in range(0, N, BLOCK_N):
        mask = (off + offs_n) < N
        x = tl.load(x_ptr + pid * N + off + offs_n, mask=mask, other=float('-inf'))
        sum_val += tl.sum(tl.exp(x - max_val), axis=0)

    result = tl.log(sum_val) + max_val
    tl.store(out_ptr + pid, result)


@triton.jit
def final_lse_mish_kernel(sum_ptr, out_ptr, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    s = tl.load(sum_ptr + offs, mask=mask, other=1.0)
    lse = tl.log(s)

    sp = tl.log(1.0 + tl.exp(lse))
    tanh_sp = 1.0 - 2.0 / (tl.exp(2.0 * sp) + 1.0)
    mish_val = lse * tanh_sp
    result = lse * mish_val

    tl.store(out_ptr + offs, result, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    matmul_weight = _weights['matmul.weight']
    matmul_bias = _weights['matmul.bias']

    x = x.contiguous()
    M, K = x.shape
    N = matmul_weight.shape[0]

    lse_sum = torch.zeros(M, device=x.device, dtype=torch.float32)
    final_out = torch.empty(M, 1, device=x.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    grid_mm = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    fused_matmul_expsum_kernel[grid_mm](x, matmul_weight, matmul_bias, lse_sum, M, N, K,
                                         BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages=3, num_warps=8)

    BLOCK_FINAL = 256
    grid_final = (triton.cdiv(M, BLOCK_FINAL),)
    final_lse_mish_kernel[grid_final](lse_sum, final_out.view(-1), M, BLOCK_SIZE=BLOCK_FINAL)

    return final_out
