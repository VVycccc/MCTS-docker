import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = self.hardtanh(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]

import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/30_Gemm_GroupNorm_Hardtanh_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
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
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n)
    acc += b[None, :]

    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, acc, mask=mask_mn)


@triton.jit
def groupnorm_hardtanh_kernel(
    x_ptr, y_ptr,
    gn_weight_ptr, gn_bias_ptr,
    M, N, num_groups,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    hardtanh_min: tl.constexpr, hardtanh_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // num_groups
    pid_g = pid % num_groups

    channels_per_group = N // num_groups
    group_start = pid_g * channels_per_group

    sum_val = 0.0
    sum_sq = 0.0

    for off in range(0, channels_per_group, BLOCK_SIZE):
        offs = group_start + off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + pid_m * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    mean = sum_val / channels_per_group
    var = sum_sq / channels_per_group - mean * mean
    eps = 1e-5
    inv_std = 1.0 / tl.sqrt(var + eps)

    for off in range(0, channels_per_group, BLOCK_SIZE):
        offs = group_start + off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + pid_m * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        gn_w = tl.load(gn_weight_ptr + offs, mask=mask, other=0.0)
        gn_b = tl.load(gn_bias_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std * gn_w + gn_b
        y = tl.maximum(tl.minimum(y, hardtanh_max), hardtanh_min)
        tl.store(y_ptr + pid_m * stride_ym + offs * stride_yn, y, mask=mask)


def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    M, K = x.shape
    N = _weights['gemm.weight'].shape[0]

    gemm_weight = _weights['gemm.weight']
    gemm_bias = _weights['gemm.bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    # --- GEMM ---
    y_gemm = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid_gemm = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    gemm_kernel[grid_gemm](
        x, gemm_weight, gemm_bias, y_gemm,
        M, N, K,
        x.stride(0), x.stride(1),
        gemm_weight.stride(0), gemm_weight.stride(1),
        y_gemm.stride(0), y_gemm.stride(1),
    )

    # --- GroupNorm + HardTanh (fused) ---
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GN = 512
    grid_gn = (M * num_groups,)
    groupnorm_hardtanh_kernel[grid_gn](
        y_gemm, out,
        gn_weight, gn_bias,
        M, N, num_groups,
        y_gemm.stride(0), y_gemm.stride(1),
        out.stride(0), out.stride(1),
        hardtanh_min=hardtanh_min, hardtanh_max=hardtanh_max,
        BLOCK_SIZE=BLOCK_SIZE_GN,
        num_warps=4,
    )

    return out
