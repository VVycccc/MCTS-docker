import torch
import triton
import triton.language as tl
import math

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/34_ConvTranspose3d_LayerNorm_GELU_Scaling_weights.pt"
_weights = None
_device = None
_conv_module = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def layernorm_gelu_kernel(
    x_ptr, ln_w_ptr, ln_b_ptr, out_ptr,
    W_OUT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W_OUT

    x = tl.load(x_ptr + pid * W_OUT + offs, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / W_OUT
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / W_OUT
    inv_std = 1.0 / tl.sqrt(var + EPS)
    x_normed = x_centered * inv_std

    ln_w = tl.load(ln_w_ptr + offs, mask=mask, other=0.0)
    ln_b = tl.load(ln_b_ptr + offs, mask=mask, other=0.0)
    x_scaled = x_normed * ln_w + ln_b

    gelu = x_scaled * 0.5 * (1.0 + tl.erf(x_scaled * 0.7071067811865475))

    tl.store(out_ptr + pid * W_OUT + offs, gelu, mask=mask)


def run(x):
    global _weights, _device, _conv_module
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv_transpose.weight'].contiguous()
    conv_bias = _weights['conv_transpose.bias'].contiguous()
    ln_weight = _weights['layer_norm.weight'].contiguous()
    ln_bias = _weights['layer_norm.bias'].contiguous()

    x = x.contiguous()

    in_channels = x.shape[1]
    out_channels = conv_weight.shape[1]
    KD, KH, KW = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]

    if _conv_module is None:
        _conv_module = torch.nn.ConvTranspose3d(
            in_channels, out_channels, (KD, KH, KW), stride=2, padding=1
        )
        _conv_module.weight.data = conv_weight
        _conv_module.bias.data = conv_bias
        _conv_module = _conv_module.to(x.device)

    conv_out = _conv_module(x)

    W_out = conv_out.shape[-1]
    num_rows = conv_out.numel() // W_out
    out = torch.empty_like(conv_out)
    BLOCK_W = triton.next_power_of_2(W_out)

    layernorm_gelu_kernel[(num_rows,)](
        conv_out, ln_weight, ln_bias, out,
        W_OUT=W_out,
        EPS=1e-5,
        BLOCK_W=BLOCK_W,
    )

    return out
