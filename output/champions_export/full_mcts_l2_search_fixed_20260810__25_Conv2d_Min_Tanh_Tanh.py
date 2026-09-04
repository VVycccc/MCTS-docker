import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/25_Conv2d_Min_Tanh_Tanh_weights.pt"
_weights = None

def _init_weights(device):
    global _weights
    w = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _weights = {k: v.to(device) for k, v in w.items()}


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, C_out, H, W, H_out, W_out,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_tiles = tl.cdiv(W_out, BLOCK_W)
    num_h_tiles = tl.cdiv(H_out, BLOCK_H)
    pid_w = pid % num_w_tiles
    pid = pid // num_w_tiles
    pid_h = pid % num_h_tiles
    pid = pid // num_h_tiles
    c_out = pid % C_out
    b = pid // C_out

    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_h = offs_h < H_out
    mask_w = offs_w < W_out
    mask_2d = mask_h[:, None] & mask_w[None, :]

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    for c_in in range(C_in):
        for kh in range(3):
            for kw in range(3):
                x_h = offs_h + kh
                x_w = offs_w + kw
                x_mask = (x_h < H)[:, None] & (x_w < W)[None, :] & mask_2d
                x_vals = tl.load(
                    x_ptr + b * C_in * H * W + c_in * H * W + x_h[:, None] * W + x_w[None, :],
                    mask=x_mask, other=0.0,
                )
                w_val = tl.load(w_ptr + c_out * C_in * 9 + c_in * 9 + kh * 3 + kw)
                acc += x_vals * w_val

    bias_val = tl.load(b_ptr + c_out)
    acc += bias_val

    tl.store(
        out_ptr + b * C_out * H_out * W_out + c_out * H_out * W_out + offs_h[:, None] * W_out + offs_w[None, :],
        acc, mask=mask_2d,
    )


@triton.jit
def min_tanh_kernel(
    in_ptr, out_ptr,
    B, C_out, H_out, W_out,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
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
    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_h = offs_h < H_out
    mask_w = offs_w < W_out
    mask_2d = mask_h[:, None] & mask_w[None, :]

    min_val = tl.full([BLOCK_H, BLOCK_W], float('inf'), dtype=tl.float32)

    for c in range(C_out):
        vals = tl.load(
            in_ptr + b * C_out * H_out * W_out + c * H_out * W_out + offs_h[:, None] * W_out + offs_w[None, :],
            mask=mask_2d, other=float('inf'),
        )
        min_val = tl.minimum(min_val, vals)

    t1 = 2.0 * tl.sigmoid(2.0 * min_val) - 1.0
    t2 = 2.0 * tl.sigmoid(2.0 * t1) - 1.0

    tl.store(
        out_ptr + b * H_out * W_out + offs_h[:, None] * W_out + offs_w[None, :],
        t2, mask=mask_2d,
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

    conv_out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=torch.float32)

    BLOCK_H = 32
    BLOCK_W = 32
    grid1 = (B * C_out * triton.cdiv(H_out, BLOCK_H) * triton.cdiv(W_out, BLOCK_W),)
    conv2d_kernel[grid1](
        x, conv_weight, conv_bias, conv_out,
        B, C_in, C_out, H, W, H_out, W_out,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    out = torch.empty(B, 1, H_out, W_out, device=x.device, dtype=torch.float32)
    grid2 = (B * triton.cdiv(H_out, BLOCK_H) * triton.cdiv(W_out, BLOCK_W),)
    min_tanh_kernel[grid2](
        conv_out, out,
        B, C_out, H_out, W_out,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    return out
