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
def conv_transpose2d_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, C_out, H, W, H_out, W_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C_out * H_out * W_out
    mask = offs < total

    ow = offs % W_out
    tmp = offs // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    c_out = tmp % C_out
    n = tmp // C_out

    acc = tl.load(b_ptr + c_out, mask=mask, other=0.0).to(tl.float32)

    for c_in in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = oh - kh
                iw = ow - kw
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
                x_idx = n * (C_in * H * W) + c_in * (H * W) + ih * W + iw
                w_idx = c_in * (C_out * 9) + c_out * 9 + kh * 3 + kw
                x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
                w_val = tl.load(w_ptr + w_idx, mask=valid, other=0.0)
                acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    # Fused GELU
    result = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
    tl.store(out_ptr + offs, result, mask=mask)


@triton.jit
def group_norm_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, num_groups,
    BLOCK: tl.constexpr,
    CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    c_start = g * CPG
    count = CPG * H * W

    # Pass 1: compute mean and rstd
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
    eps = 1e-5
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine
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
        weight = tl.load(w_ptr + c, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
        normalized = (x_val - mean) * rstd
        result = normalized * weight + bias
        tl.store(out_ptr + elem_idx, result, mask=mask)


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

    BLOCK = 256

    # Step 1: ConvTranspose2d + GELU fused
    gelu_out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    total = N * C_out * H_out * W_out
    grid = (triton.cdiv(total, BLOCK),)
    conv_transpose2d_gelu_kernel[grid](
        x, conv_weight, conv_bias, gelu_out,
        N, C_in, C_out, H, W, H_out, W_out,
        BLOCK=BLOCK,
    )

    # Step 2: GroupNorm (stats + apply fused)
    out = torch.empty_like(gelu_out)
    grid_gn = (N * num_groups,)
    group_norm_fused_kernel[grid_gn](
        gelu_out, gn_weight, gn_bias, out,
        N, C_out, H_out, W_out, num_groups,
        BLOCK=BLOCK, CPG=CPG,
    )

    return out
