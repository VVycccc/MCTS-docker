import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    C_in, C_out, H, W,
    sx_b, sx_c, sx_h, sx_w,
    sw_ic, sw_oc,
    sy_b, sy_c, sy_h, sy_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C_out
    oc = pid_bc % C_out

    num_h_blocks = tl.cdiv(H, BLOCK_H)
    oh_start = (pid_hw // num_h_blocks) * BLOCK_H
    ow_start = (pid_hw % num_h_blocks) * BLOCK_W

    offs_h = oh_start + tl.arange(0, BLOCK_H)
    offs_w = ow_start + tl.arange(0, BLOCK_W)
    mask_h = offs_h < H
    mask_w = offs_w < W

    bias_val = tl.load(b_ptr + oc)
    acc = tl.full([BLOCK_H, BLOCK_W], bias_val, tl.float32)

    for ic in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = offs_h + 1 - kh
                iw = offs_w + 1 - kw
                in_mask_h = (ih >= 0) & (ih < H)
                in_mask_w = (iw >= 0) & (iw < W)
                x_ptrs = x_ptr + b * sx_b + ic * sx_c + ih[:, None] * sx_h + iw[None, :] * sx_w
                x_val = tl.load(x_ptrs, mask=in_mask_h[:, None] & in_mask_w[None, :], other=0.0)
                w_val = tl.load(w_ptr + ic * sw_ic + oc * sw_oc + kh * 3 + kw)
                acc += w_val * x_val

    y_ptrs = y_ptr + b * sy_b + oc * sy_c + offs_h[:, None] * sy_h + offs_w[None, :] * sy_w
    tl.store(y_ptrs, acc, mask=mask_h[:, None] & mask_w[None, :])


# Fused kernel: MaxPool2d + Hardtanh + Mean + Tanh
# Eliminates intermediate temp2 tensor and its memory traffic
@triton.jit
def maxpool_hardtanh_mean_tanh_kernel(
    x_ptr, y_ptr,
    C, H_in, W_in, H_out, W_out,
    sx_b, sx_c, sx_h, sx_w,
    sy_b, sy_c,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    base = x_ptr + b * sx_b + c * sx_c
    n_out = H_out * W_out

    acc = tl.zeros([BLOCK], tl.float32)
    NEG = -1.0e30

    for off in range(0, n_out, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < n_out
        oh = idx // W_out
        ow = idx % W_out

        # Clamp indices to valid range for safe pointer computation
        oh_c = tl.minimum(oh, H_out - 1)
        ow_c = tl.minimum(ow, W_out - 1)

        ih0 = 2 * oh_c
        ih1 = 2 * oh_c + 1
        iw0 = 2 * ow_c
        iw1 = 2 * ow_c + 1

        v00 = tl.load(base + ih0 * sx_h + iw0 * sx_w, mask=mask, other=NEG)
        v01 = tl.load(base + ih0 * sx_h + iw1 * sx_w, mask=mask, other=NEG)
        v10 = tl.load(base + ih1 * sx_h + iw0 * sx_w, mask=mask, other=NEG)
        v11 = tl.load(base + ih1 * sx_h + iw1 * sx_w, mask=mask, other=NEG)

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.maximum(tl.minimum(m, 1.0), -1.0)

        acc += tl.where(mask, m, 0.0)

    total = tl.sum(acc, axis=0)
    mean_val = total / n_out
    result = 2.0 * tl.sigmoid(2.0 * mean_val) - 1.0

    tl.store(y_ptr + b * sy_b + c * sy_c, result)


def run(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()
    B, C_in, H, W = x.shape
    C_out = 64
    H_conv = H
    W_conv = W
    H_pool = H_conv // 2
    W_pool = W_conv // 2

    weight = _weights['conv_transpose.weight']
    bias = _weights['conv_transpose.bias']

    temp1 = torch.empty(B, C_out, H_conv, W_conv, device=x.device, dtype=torch.float32)

    BLOCK_H = 32
    BLOCK_W = 32

    grid_conv = (B * C_out, triton.cdiv(H_conv, BLOCK_H) * triton.cdiv(W_conv, BLOCK_W))
    conv_transpose2d_kernel[grid_conv](
        x, weight, bias, temp1,
        C_in, C_out, H_conv, W_conv,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1),
        temp1.stride(0), temp1.stride(1), temp1.stride(2), temp1.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    # Fused maxpool + hardtanh + mean + tanh — eliminates temp2
    output = torch.empty(B, C_out, 1, 1, device=x.device, dtype=torch.float32)

    BLOCK = 256
    grid_fused = (B * C_out,)
    maxpool_hardtanh_mean_tanh_kernel[grid_fused](
        temp1, output,
        C_out, H_conv, W_conv, H_pool, W_pool,
        temp1.stride(0), temp1.stride(1), temp1.stride(2), temp1.stride(3),
        output.stride(0), output.stride(1),
        BLOCK=BLOCK,
    )

    return output
