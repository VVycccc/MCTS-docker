import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/25_Conv2d_Min_Tanh_Tanh_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    w = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _weights = {k: v.to(device) for k, v in w.items()}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ],
    key=['H_out', 'W_out'],
)
@triton.jit
def fused_conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    H, W, H_out, W_out,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    C_OUT: tl.constexpr, C_IN: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_tiles = tl.cdiv(W_out, BLOCK_W)
    num_h_tiles = tl.cdiv(H_out, BLOCK_H)
    pid_w = pid % num_w_tiles
    pid = pid // num_w_tiles
    pid_h = pid % num_h_tiles
    b = pid // num_h_tiles

    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    offs_hw = tl.arange(0, BLOCK_HW)
    local_h = offs_hw // BLOCK_W
    local_w = offs_hw % BLOCK_W
    h_idx = h_start + local_h
    w_idx = w_start + local_w
    mask_hw = (h_idx < H_out) & (w_idx < W_out)

    c_out_offsets = tl.arange(0, C_OUT)
    c_in_offsets = tl.arange(0, C_IN)

    acc = tl.zeros([C_OUT, BLOCK_HW], dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            x_h = h_idx + kh
            x_w = w_idx + kw
            x_mask = (x_h < H)[None, :] & (x_w < W)[None, :] & mask_hw[None, :]
            x_vals = tl.load(
                x_ptr + b * C_IN * H * W + c_in_offsets[:, None] * H * W + x_h[None, :] * W + x_w[None, :],
                mask=x_mask, other=0.0,
            )

            w_vals = tl.load(
                w_ptr + c_out_offsets[:, None] * C_IN * 9 + c_in_offsets[None, :] * 9 + kh * 3 + kw
            )

            acc = tl.dot(w_vals, x_vals, acc=acc, allow_tf32=True)

    bias_vals = tl.load(b_ptr + c_out_offsets)
    acc += bias_vals[:, None]

    min_val = tl.min(acc, axis=0)

    t1 = 2.0 * tl.sigmoid(2.0 * min_val) - 1.0
    t2 = 2.0 * tl.sigmoid(2.0 * t1) - 1.0

    tl.store(
        out_ptr + b * H_out * W_out + h_idx * W_out + w_idx,
        t2, mask=mask_hw,
    )


def run(x):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']

    B, C_in, H, W = x.shape
    C_out = conv_weight.shape[0]
    H_out = H - 2
    W_out = W - 2

    out = torch.empty(B, 1, H_out, W_out, device=x.device, dtype=torch.float32)

    grid = lambda META: (B * triton.cdiv(H_out, META['BLOCK_H']) * triton.cdiv(W_out, META['BLOCK_W']),)
    fused_conv_min_tanh_kernel[grid](
        x, conv_weight, conv_bias, out,
        H, W, H_out, W_out,
        C_OUT=C_out, C_IN=C_in,
    )

    return out
