import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def conv2d_gemm_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    C_out = 64
    C_in = 8
    H = 128
    W = 128
    H_out = 126
    W_out = 126
    K = 72  # C_in * 3 * 3

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < H_out * W_out
    mask_n = offs_n < C_out

    oh = offs_m // W_out
    ow = offs_m % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = (K + BLOCK_K - 1) // BLOCK_K
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        ic = offs_k // 9
        kh = (offs_k % 9) // 3
        kw = offs_k % 3

        # Load input: (BLOCK_M, BLOCK_K)
        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        x_ptrs = x_ptr + pid_b * C_in * H * W + ic[None, :] * H * W + ih * W + iw
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load weights: (BLOCK_K, BLOCK_N)
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_vals = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_vals, w_vals, acc=acc, allow_tf32=True)

    # Add bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Store in NCHW format
    out_ptrs = out_ptr + pid_b * C_out * H_out * W_out + offs_n[None, :] * H_out * W_out + offs_m[:, None]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gn_stats_kernel(x_ptr, mean_ptr, rstd_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_groups = 16
    B = 128
    C = 64
    H = 126
    W = 126
    channels_per_group = 4
    HW = H * W
    b = pid // num_groups
    g = pid % num_groups
    total = 0.0
    total_sq = 0.0
    count = channels_per_group * HW
    num_iters = (HW + BLOCK - 1) // BLOCK
    for c_offset in range(4):
        c = g * channels_per_group + c_offset
        base = b * C * HW + c * HW
        for it in range(num_iters):
            s = it * BLOCK
            offs = s + tl.arange(0, BLOCK)
            mask = offs < HW
            x_val = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
            total += tl.sum(x_val, axis=0)
            total_sq += tl.sum(x_val * x_val, axis=0)
    mean = total / count
    var = total_sq / count - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def fused_norm_scale_pool_clamp_kernel(
    x_ptr, mean_ptr, rstd_ptr, gn_weight_ptr, gn_bias_ptr, scale_ptr, out_ptr,
    BLOCK: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    C_out = 64
    H = 126
    W = 126
    H_out = 31
    W_out = 31
    K = 4
    HW = H * W
    HW_out = H_out * W_out
    channels_per_group = 4
    num_groups = 16
    b = pid_bc // C_out
    c = pid_bc % C_out
    g = c // channels_per_group
    mean = tl.load(mean_ptr + b * num_groups + g)
    rstd = tl.load(rstd_ptr + b * num_groups + g)
    w = tl.load(gn_weight_ptr + c)
    bias = tl.load(gn_bias_ptr + c)
    s = tl.load(scale_ptr + c)
    alpha = rstd * w * s
    beta = bias * s
    offs = pid_spatial * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW_out
    oh = offs // W_out
    ow = offs % W_out
    result = tl.full((BLOCK,), -1e30, dtype=tl.float32)
    for i in range(4):
        for j in range(4):
            ih = oh * K + i
            iw = ow * K + j
            x_val = tl.load(x_ptr + b * C_out * HW + c * HW + ih * W + iw, mask=mask, other=-1e30)
            y = (x_val - mean) * alpha + beta
            result = tl.maximum(result, y)
    result = tl.maximum(result, 0.0)
    result = tl.minimum(result, 1.0)
    tl.store(out_ptr + b * C_out * HW_out + c * HW_out + offs, result, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)
    device = x.device
    B = 128
    C_out = 64
    H_out = 126
    W_out = 126
    num_groups = 16
    H_pool = 31
    W_pool = 31
    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']
    scale = _weights['scale']

    # Conv2d using GEMM approach with tensor cores
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    conv_out = torch.empty(B, C_out, H_out, W_out, device=device, dtype=torch.float32)
    grid_conv = (triton.cdiv(H_out * W_out, BLOCK_M), triton.cdiv(C_out, BLOCK_N), B)
    conv2d_gemm_kernel[grid_conv](x, conv_weight, conv_bias, conv_out,
                                  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                  num_stages=3, num_warps=8)

    # GroupNorm stats
    BLOCK = 1024
    mean = torch.empty(B, num_groups, device=device, dtype=torch.float32)
    rstd = torch.empty(B, num_groups, device=device, dtype=torch.float32)
    grid_gn_stats = (B * num_groups,)
    gn_stats_kernel[grid_gn_stats](conv_out, mean, rstd, BLOCK=BLOCK)

    # Fused norm + scale + pool + clamp
    out = torch.empty(B, C_out, H_pool, W_pool, device=device, dtype=torch.float32)
    grid_fused = (B * C_out, triton.cdiv(H_pool * W_pool, BLOCK))
    fused_norm_scale_pool_clamp_kernel[grid_fused](
        conv_out, mean, rstd, gn_weight, gn_bias, scale, out, BLOCK=BLOCK
    )
    return out
