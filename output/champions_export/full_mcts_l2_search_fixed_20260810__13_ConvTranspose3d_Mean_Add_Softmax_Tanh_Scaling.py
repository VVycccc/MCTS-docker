import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    C_IN, C_OUT, D, H, W,
    BLOCK_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    w_block = pid % num_w_blocks
    rest = pid // num_w_blocks
    h_idx = rest % H
    rest = rest // H
    d_idx = rest % D
    rest = rest // D
    c_out = rest % C_OUT
    n = rest // C_OUT

    w_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    b = tl.load(bias_ptr + c_out)
    acc = tl.zeros((BLOCK_W,), tl.float32) + b

    for c_in in range(0, C_IN):
        for kd in range(0, KD):
            in_d = d_idx * STRIDE + PAD - kd
            if (in_d >= 0) & (in_d < D):
                for kh in range(0, KH):
                    in_h = h_idx * STRIDE + PAD - kh
                    if (in_h >= 0) & (in_h < H):
                        for kw in range(0, KW):
                            in_w = w_offs * STRIDE + PAD - kw
                            valid = (in_w >= 0) & (in_w < W) & w_mask
                            x_vals = tl.load(
                                x_ptr + n * (C_IN * D * H * W) + c_in * (D * H * W) + in_d * (H * W) + in_h * W + in_w,
                                mask=valid, other=0.0,
                            )
                            w_val = tl.load(weight_ptr + c_in * (C_OUT * KD * KH * KW) + c_out * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw)
                            acc += w_val * x_vals

    tl.store(
        out_ptr + n * (C_OUT * D * H * W) + c_out * (D * H * W) + d_idx * (H * W) + h_idx * W + w_offs,
        acc, mask=w_mask,
    )


@triton.jit
def mean_pool_kernel(
    in_ptr, out_ptr,
    C_OUT, D, H, W,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    w_block = pid % num_w_blocks
    rest = pid // num_w_blocks
    h_idx = rest % H
    rest = rest // H
    c_out = rest % C_OUT
    n = rest // C_OUT

    w_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    acc = tl.zeros((BLOCK_W,), tl.float32)
    for d in range(0, D):
        val = tl.load(
            in_ptr + n * (C_OUT * D * H * W) + c_out * (D * H * W) + d * (H * W) + h_idx * W + w_offs,
            mask=w_mask, other=0.0,
        )
        acc += val
    acc = acc / D
    tl.store(
        out_ptr + n * (C_OUT * H * W) + c_out * (H * W) + h_idx * W + w_offs,
        acc, mask=w_mask,
    )


@triton.jit
def add_bias_kernel(
    in_ptr, bias_ptr, out_ptr,
    C_OUT, H, W,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    w_block = pid % num_w_blocks
    rest = pid // num_w_blocks
    h_idx = rest % H
    rest = rest // H
    c_out = rest % C_OUT
    n = rest // C_OUT

    w_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    b = tl.load(bias_ptr + c_out)
    val = tl.load(
        in_ptr + n * (C_OUT * H * W) + c_out * (H * W) + h_idx * W + w_offs,
        mask=w_mask, other=0.0,
    )
    tl.store(
        out_ptr + n * (C_OUT * H * W) + c_out * (H * W) + h_idx * W + w_offs,
        val + b, mask=w_mask,
    )


@triton.jit
def softmax_kernel(
    in_ptr, out_ptr,
    C_OUT: tl.constexpr, H, W,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    w_block = pid % num_w_blocks
    rest = pid // num_w_blocks
    h_idx = rest % H
    n = rest // H

    w_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W
    c_offs = tl.arange(0, C_OUT)

    ptrs = in_ptr + n * (C_OUT * H * W) + c_offs[:, None] * (H * W) + h_idx * W + w_offs[None, :]
    mask2d = w_mask[None, :]
    x = tl.load(ptrs, mask=mask2d, other=-float('inf'))

    m = tl.max(x, axis=0)
    x = x - m[None, :]
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    y = e / s[None, :]

    tl.store(
        out_ptr + n * (C_OUT * H * W) + c_offs[:, None] * (H * W) + h_idx * W + w_offs[None, :],
        y, mask=mask2d,
    )


@triton.jit
def tanh_scale_kernel(
    in_ptr, out_ptr,
    C_OUT, H, W,
    SCALING: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    w_block = pid % num_w_blocks
    rest = pid // num_w_blocks
    h_idx = rest % H
    rest = rest // H
    c_out = rest % C_OUT
    n = rest // C_OUT

    w_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    val = tl.load(
        in_ptr + n * (C_OUT * H * W) + c_out * (H * W) + h_idx * W + w_offs,
        mask=w_mask, other=0.0,
    )
    th = 2.0 * tl.sigmoid(2.0 * val) - 1.0
    out = th * SCALING
    tl.store(
        out_ptr + n * (C_OUT * H * W) + c_out * (H * W) + h_idx * W + w_offs,
        out, mask=w_mask,
    )


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    bias = _weights['bias']
    conv_bias = _weights['conv_transpose.bias']
    conv_weight = _weights['conv_transpose.weight']

    x = x.contiguous()
    B, C_IN, D, H, W = x.shape
    C_OUT = conv_weight.shape[1]
    BLOCK_W = 64

    conv_out = torch.empty(B, C_OUT, D, H, W, device=x.device, dtype=torch.float32)
    grid_conv = (B * C_OUT * D * H * triton.cdiv(W, BLOCK_W),)
    conv_transpose3d_kernel[grid_conv](
        x, conv_weight, conv_bias, conv_out,
        C_IN, C_OUT, D, H, W, BLOCK_W,
        KD=3, KH=3, KW=3, STRIDE=1, PAD=1,
    )

    mean_out = torch.empty(B, C_OUT, H, W, device=x.device, dtype=torch.float32)
    grid_mh = (B * C_OUT * H * triton.cdiv(W, BLOCK_W),)
    mean_pool_kernel[grid_mh](
        conv_out, mean_out,
        C_OUT, D, H, W, BLOCK_W,
    )

    add_out = torch.empty(B, C_OUT, H, W, device=x.device, dtype=torch.float32)
    add_bias_kernel[grid_mh](
        mean_out, bias, add_out,
        C_OUT, H, W, BLOCK_W,
    )

    sm_out = torch.empty(B, C_OUT, H, W, device=x.device, dtype=torch.float32)
    grid_sm = (B * H * triton.cdiv(W, BLOCK_W),)
    softmax_kernel[grid_sm](
        add_out, sm_out,
        C_OUT, H, W, BLOCK_W,
    )

    out_flat = torch.empty(B, C_OUT, H, W, device=x.device, dtype=torch.float32)
    tanh_scale_kernel[grid_mh](
        sm_out, out_flat,
        C_OUT, H, W, 2.0, BLOCK_W,
    )

    return out_flat.unsqueeze(2)
