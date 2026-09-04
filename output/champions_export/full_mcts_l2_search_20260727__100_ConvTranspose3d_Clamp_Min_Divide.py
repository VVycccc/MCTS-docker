import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/100_ConvTranspose3d_Clamp_Min_Divide_weights.pt"
_W = None
_W_device = None

def _init_weights(device):
    global _W, _W_device
    w = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _W = {k: v.to(device).contiguous() for k, v in w.items()}
    _W_device = str(device)


@triton.jit
def clamp_div_kernel(x_ptr, out_ptr, n, min_value, divisor, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = tl.maximum(x, min_value)
    x = x / divisor
    tl.store(out_ptr + offs, x, mask=mask)


def run(x):
    global _W, _W_device
    if _W is None or _W_device != str(x.device):
        _init_weights(x.device)

    weight = _W['conv_transpose.weight']  # [64, 128, 3, 3, 3]
    bias = _W['conv_transpose.bias']      # [128]

    x = x.contiguous()
    batch_size, in_channels, D_in, H_in, W_in = x.shape
    out_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    stride = 2
    padding = 1
    min_value = -1.0
    divisor = 2.0

    # Use nn.ConvTranspose3d module for cuDNN-optimized transposed convolution
    conv_op = torch.nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )
    conv_op.weight.data = weight
    conv_op.bias.data = bias
    out = conv_op(x)

    # Fuse clamp + divide with a Triton elementwise kernel (single pass)
    n = out.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    result = torch.empty_like(out)
    clamp_div_kernel[grid](out, result, n, min_value, divisor, BLOCK=BLOCK)

    return result
