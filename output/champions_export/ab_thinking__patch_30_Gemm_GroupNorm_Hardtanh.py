import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/30_Gemm_GroupNorm_Hardtanh_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=True)

    b_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b_vals[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def group_norm_hardtanh_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, num_groups,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = N // num_groups
    start = pid_g * group_size

    # Pass 1: compute mean
    sum_val = 0.0
    for off in range(0, group_size, BLOCK_SIZE):
        offs = start + off + tl.arange(0, BLOCK_SIZE)
        mask = offs < (start + group_size)
        vals = tl.load(x_ptr + pid_m * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    mean = sum_val / group_size

    # Pass 2: compute variance
    sum_sq = 0.0
    for off in range(0, group_size, BLOCK_SIZE):
        offs = start + off + tl.arange(0, BLOCK_SIZE)
        mask = offs < (start + group_size)
        vals = tl.load(x_ptr + pid_m * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        diff = vals - mean
        sum_sq += tl.sum(diff * diff)
    var = sum_sq / group_size

    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize, apply affine, and apply HardTanh (fused)
    for off in range(0, group_size, BLOCK_SIZE):
        offs = start + off + tl.arange(0, BLOCK_SIZE)
        mask = offs < (start + group_size)
        vals = tl.load(x_ptr + pid_m * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        normalized = (vals - mean) * inv_std
        w_vals = tl.load(w_ptr + offs, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + offs, mask=mask, other=0.0)
        out = w_vals * normalized + b_vals
        out = tl.maximum(tl.minimum(out, max_val), min_val)
        tl.store(y_ptr + pid_m * stride_ym + offs * stride_yn, out, mask=mask)


def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    gemm_weight = _weights['gemm.weight']
    gemm_bias = _weights['gemm.bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    M, K = x.shape
    N = gemm_weight.shape[0]
    num_groups = 16
    eps = 1e-5
    hardtanh_min = -2.0
    hardtanh_max = 2.0

    # --- GEMM (autotuned) ---
    y_gemm = torch.empty(M, N, device=x.device, dtype=torch.float32)
    grid_gemm = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    gemm_kernel[grid_gemm](
        x, gemm_weight, gemm_bias, y_gemm,
        M, N, K,
        x.stride(0), x.stride(1),
        gemm_weight.stride(0), gemm_weight.stride(1),
        y_gemm.stride(0), y_gemm.stride(1),
    )

    # --- Fused GroupNorm + HardTanh ---
    y_out = torch.empty_like(y_gemm)
    BLOCK_SIZE_GN = 256
    grid_gn = (M, num_groups)
    group_norm_hardtanh_kernel[grid_gn](
        y_gemm, gn_weight, gn_bias, y_out,
        M, N, num_groups,
        y_gemm.stride(0), y_gemm.stride(1),
        y_out.stride(0), y_out.stride(1),
        eps,
        hardtanh_min, hardtanh_max,
        BLOCK_SIZE=BLOCK_SIZE_GN,
    )

    return y_out
