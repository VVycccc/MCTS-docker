import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/62_Matmul_GroupNorm_LeakyReLU_Sum_weights.pt"

_W = None
_W_device = None


# 1) Linear: out[M, N] = x[M, K] @ W_t[K, N] + b[N]
@triton.jit
def _matmul_bias_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(x_ptr + offs_m[:, None] * K + (k + offs_k)[None, :],
                    mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(w_ptr + (k + offs_k)[:, None] * N + offs_n[None, :],
                    mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 2) Fused GroupNorm + LeakyReLU + (x+x): process GROUPS_PER_BLOCK groups per program
@triton.jit
def _groupnorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, G, C_PER_G, eps, negative_slope,
                      BLOCK_C: tl.constexpr, GROUPS_PER_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_g_blocks = tl.cdiv(G, GROUPS_PER_BLOCK)
    row = pid // num_g_blocks
    g0 = (pid % num_g_blocks) * GROUPS_PER_BLOCK

    offs_g = tl.arange(0, GROUPS_PER_BLOCK)
    offs_c = tl.arange(0, BLOCK_C)
    g = g0 + offs_g
    cols = g[:, None] * C_PER_G + offs_c[None, :]
    mask = (g[:, None] < G) & (offs_c[None, :] < C_PER_G) & (cols < N)

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=1) / C_PER_G
    diff = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / C_PER_G
    inv_std = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    y = (x - mean[:, None]) * inv_std[:, None] * w + b
    y = tl.where(y > 0, y, negative_slope * y)
    y = y + y
    tl.store(out_ptr + row * N + cols, y, mask=mask)


# 3) LeakyReLU
@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, numel, negative_slope, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x > 0, x, negative_slope * x)
    tl.store(out_ptr + offs, y, mask=mask)


# 4) x + x
@triton.jit
def _add_self_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + x, mask=mask)


def run(x):
    global _W, _W_device
    if _W is None or _W_device != str(x.device):
        _W = {k: v.to(x.device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
        _W_device = str(x.device)

    fc_weight = _W['fc.weight']   # [8192, 8192]
    if 'fc.weight_t' not in _W:
        _W['fc.weight_t'] = fc_weight.t().contiguous()
    fc_weight_t = _W['fc.weight_t']  # [8192, 8192] as [K, N]
    fc_bias = _W['fc.bias']       # [8192]
    gn_weight = _W['gn.weight']   # [8192]
    gn_bias = _W['gn.bias']       # [8192]

    num_groups = 512
    eps = 1e-5
    negative_slope = 0.01

    x = x.contiguous()
    M, K = x.shape
    N = fc_weight.shape[0]
    # 1) matmul + bias
    h = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 64, 32, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_bias_kernel[grid](x, fc_weight_t, fc_bias, h, M, N, K,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              GROUP_M=GROUP_M, num_warps=8, num_stages=3)

    # 2) fused group norm + leaky relu + (x+x)
    c_per_g = N // num_groups
    BLOCK_C = triton.next_power_of_2(c_per_g)
    GROUPS_PER_BLOCK = 32
    out = torch.empty_like(h)
    num_g_blocks = triton.cdiv(num_groups, GROUPS_PER_BLOCK)
    _groupnorm_kernel[(M * num_g_blocks,)](h, gn_weight, gn_bias, out,
                                           N, num_groups, c_per_g, eps, negative_slope,
                                           BLOCK_C=BLOCK_C, GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
                                           num_warps=4)

    return out
