import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/36_ConvTranspose2d_Min_Sum_GELU_Add_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, C_out, H, W, H_out, W_out,
    stride, padding,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B * C_out * H_out * W_out
    mask = offs < total

    n = offs // (C_out * H_out * W_out)
    rem = offs % (C_out * H_out * W_out)
    c_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    oh = rem2 // W_out
    ow = rem2 % W_out

    acc = tl.load(b_ptr + c_out, mask=mask, other=0.0).to(tl.float32)

    for c_in in range(C_in):
        for kh in range(3):
            for kw in range(3):
                oh_p = oh + padding - kh
                ow_p = ow + padding - kw
                valid = (oh_p % stride == 0) & (ow_p % stride == 0)
                ih = oh_p // stride
                iw = ow_p // stride
                in_bounds = valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
                w_off = ((c_in * C_out + c_out) * 3 + kh) * 3 + kw
                w_val = tl.load(w_ptr + w_off, mask=in_bounds, other=0.0)
                in_off = ((n * C_in + c_in) * H + ih) * W + iw
                x_val = tl.load(x_ptr + in_off, mask=in_bounds, other=0.0)
                acc += w_val * x_val

    out_off = ((n * C_out + c_out) * H_out + oh) * W_out + ow
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def min_channel_kernel(
    x_ptr, out_ptr,
    B, C_out, H_out, W_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B * H_out * W_out
    mask = offs < total

    n = offs // (H_out * W_out)
    rem = offs % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    acc = tl.full([BLOCK], float('inf'), dtype=tl.float32)
    for c in range(C_out):
        off = ((n * C_out + c) * H_out + oh) * W_out + ow
        x = tl.load(x_ptr + off, mask=mask, other=float('inf'))
        acc = tl.minimum(acc, x)

    out_off = (n * H_out + oh) * W_out + ow
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def sum_height_kernel(
    x_ptr, out_ptr,
    B, H_out, W_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B * W_out
    mask = offs < total

    n = offs // W_out
    ow = offs % W_out

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for oh in range(H_out):
        off = (n * H_out + oh) * W_out + ow
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        acc += x

    out_off = n * W_out + ow
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def gelu_add_kernel(
    x_ptr, bias_ptr, out_ptr,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    gelu = 0.5 * x * (1.0 + tl.erf(x * 0.70710678118))
    b = tl.load(bias_ptr)
    y = gelu + b
    tl.store(out_ptr + offs, y, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    bias = _weights['bias']
    conv_bias = _weights['conv_transpose.bias']
    conv_weight = _weights['conv_transpose.weight']

    B, C_in, H, W = x.shape
    C_out = conv_weight.shape[1]
    KH = conv_weight.shape[2]
    KW = conv_weight.shape[3]
    stride = 2
    padding = 1
    output_padding = 1
    H_out = (H - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W - 1) * stride - 2 * padding + KW + output_padding

    BLOCK = 64

    # Step 1: ConvTranspose2d
    conv_out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    total_conv = B * C_out * H_out * W_out
    grid1 = (triton.cdiv(total_conv, BLOCK),)
    conv_transpose2d_kernel[grid1](
        x, conv_weight, conv_bias, conv_out,
        B, C_in, C_out, H, W, H_out, W_out,
        stride, padding,
        BLOCK=BLOCK,
    )

    # Step 2: Min over channels
    min_out = torch.empty(B, 1, H_out, W_out, device=x.device, dtype=torch.float32)
    total_min = B * H_out * W_out
    grid2 = (triton.cdiv(total_min, BLOCK),)
    min_channel_kernel[grid2](
        conv_out, min_out,
        B, C_out, H_out, W_out,
        BLOCK=BLOCK,
    )

    # Step 3: Sum over height
    sum_out = torch.empty(B, 1, 1, W_out, device=x.device, dtype=torch.float32)
    total_sum = B * W_out
    grid3 = (triton.cdiv(total_sum, BLOCK),)
    sum_height_kernel[grid3](
        min_out, sum_out,
        B, H_out, W_out,
        BLOCK=BLOCK,
    )

    # Step 4: GELU + Add bias
    out = torch.empty(B, 1, 1, W_out, device=x.device, dtype=torch.float32)
    grid4 = (triton.cdiv(total_sum, BLOCK),)
    gelu_add_kernel[grid4](
        sum_out, bias, out,
        total_sum,
        BLOCK=BLOCK,
    )

    return out
