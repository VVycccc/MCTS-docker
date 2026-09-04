import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/19_ConvTranspose2d_GELU_GroupNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def conv_transpose_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    M_total: tl.constexpr, K_total: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_total
    mask_n = offs_n < C_out

    # Decode output positions: m -> (n, oh, ow)
    ow = offs_m % W_out
    tmp_m = offs_m // W_out
    oh = tmp_m % H_out
    n_idx = tmp_m // H_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        # Decode k: c_in = k // 9, kh = (k % 9) // 3, kw = k % 3
        c_in_k = offs_k // 9
        kh_k = (offs_k % 9) // 3
        kw_k = offs_k % 3

        # Input indices: ih = oh - kh, iw = ow - kw
        ih = oh[:, None] - kh_k[None, :]
        iw = ow[:, None] - kw_k[None, :]

        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_m[:, None] & mask_k[None, :]

        x_idx = n_idx[:, None] * (C_in * H * W) + c_in_k[None, :] * (H * W) + ih * W + iw
        x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

        # Weight: precomputed [K_total, C_out], contiguous
        w_idx = offs_k[:, None] * C_out + offs_n[None, :]
        w_val = tl.load(w_ptr + w_idx, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_val, w_val, acc=acc, allow_tf32=True)

    # Add bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Fused GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))

    # Store output in [N, C_out, H_out, W_out] layout
    out_idx = n_idx[:, None] * (C_out * H_out * W_out) + offs_n[None, :] * (H_out * W_out) + oh[:, None] * W_out + ow[:, None]
    tl.store(out_ptr + out_idx, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offs, result, mask=mask)


@triton.jit
def group_norm_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    N, C, H, W, num_groups,
    BLOCK: tl.constexpr,
    CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    c_start = g * CPG
    count = CPG * H * W

    sum_val = 0.0
    sum_sq = 0.0

    total_elems = CPG * H * W
    for off in range(0, total_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total_elems
        w_pos = idx % W
        tmp = idx // W
        h_pos = tmp % H
        c_offset = tmp // H
        c = c_start + c_offset
        elem_idx = n * (C * H * W) + c * (H * W) + h_pos * W + w_pos
        x_val = tl.load(x_ptr + elem_idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_val, axis=0)
        sum_sq += tl.sum(x_val * x_val, axis=0)

    mean = sum_val / count
    var = sum_sq / count - mean * mean

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


@triton.jit
def group_norm_apply_kernel(
    x_ptr, mean_ptr, var_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, num_groups,
    BLOCK: tl.constexpr,
    CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * H * W
    mask = offs < total

    w_pos = offs % W
    tmp = offs // W
    h_pos = tmp % H
    tmp = tmp // H
    c = tmp % C
    n = tmp // C

    g = c // CPG
    group_idx = n * num_groups + g

    mean = tl.load(mean_ptr + group_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + group_idx, mask=mask, other=0.0)
    weight = tl.load(w_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)

    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    eps = 1e-5
    normalized = (x_val - mean) / tl.sqrt(var + eps)
    result = normalized * weight + bias

    tl.store(out_ptr + offs, result, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv_transpose.weight']
    conv_bias = _weights['conv_transpose.bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    N, C_in, H, W = x.shape
    C_out = conv_weight.shape[1]
    H_out = (H - 1) + 3
    W_out = (W - 1) + 3
    num_groups = 8
    CPG = C_out // num_groups

    BLOCK = 1024

    # Step 1: ConvTranspose2d via im2col + GEMM
    reshaped_weight = conv_weight.permute(0, 2, 3, 1).contiguous().reshape(C_in * 9, C_out)

    gelu_out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    M_total = N * H_out * W_out
    K_total = C_in * 9
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(C_out, BLOCK_N))
    conv_transpose_gemm_kernel[grid](
        x, reshaped_weight, conv_bias, gelu_out,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        N=N, C_in=C_in, C_out=C_out, H=H, W=W, H_out=H_out, W_out=W_out,
        M_total=M_total, K_total=K_total,
        num_stages=2, num_warps=8,
    )

    # Step 2: GELU (fused into conv kernel)

    # Step 3: GroupNorm — compute stats
    mean = torch.empty(N * num_groups, device=x.device, dtype=torch.float32)
    var = torch.empty(N * num_groups, device=x.device, dtype=torch.float32)
    grid_stats = (N * num_groups,)
    group_norm_stats_kernel[grid_stats](
        gelu_out, mean, var,
        N, C_out, H_out, W_out, num_groups,
        BLOCK=BLOCK, CPG=CPG,
    )

    # Step 3: GroupNorm — apply normalization
    out = torch.empty_like(gelu_out)
    total = gelu_out.numel()
    grid = (triton.cdiv(total, BLOCK),)
    group_norm_apply_kernel[grid](
        gelu_out, mean, var, gn_weight, gn_bias, out,
        N, C_out, H_out, W_out, num_groups,
        BLOCK=BLOCK, CPG=CPG,
    )

    return out
