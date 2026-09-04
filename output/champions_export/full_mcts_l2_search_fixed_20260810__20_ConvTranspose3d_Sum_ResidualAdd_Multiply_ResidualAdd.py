import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd_weights.pt"
_weights = None
_device = None
_weight_t = None

def _init_weights(device):
    global _weights, _device, _weight_t
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)
    # Pre-transpose weight from [C_in, C_out, KD, KH, KW] to [KD, KH, KW, C_in, C_out]
    _weight_t = _weights['conv_transpose.weight'].permute(2, 3, 4, 0, 1).contiguous()

@triton.jit
def kernel(
    x_ptr, weight_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
    stride, padding,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    C_IN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_total = N * D_out * H_out * W_out
    mask_m = offs_m < M_total
    mask_n = offs_n < C_out

    # Decode m -> (n, d_out, h_out, w_out)
    w_out = offs_m % W_out
    hw = offs_m // W_out
    h_out = hw % H_out
    dh = hw // H_out
    d_out = dh % D_out
    n = dh // D_out

    c_out = offs_n
    offs_k = tl.arange(0, C_IN)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                d_in_num = d_out + padding - kd
                h_in_num = h_out + padding - kh
                w_in_num = w_out + padding - kw

                d_in = d_in_num // stride
                h_in = h_in_num // stride
                w_in = w_in_num // stride

                valid = mask_m \
                    & ((d_in_num % stride) == 0) \
                    & ((h_in_num % stride) == 0) \
                    & ((w_in_num % stride) == 0) \
                    & (d_in >= 0) & (d_in < D_in) \
                    & (h_in >= 0) & (h_in < H_in) \
                    & (w_in >= 0) & (w_in < W_in)

                in_idx = n[:, None] * (C_in * D_in * H_in * W_in) \
                       + offs_k[None, :] * (D_in * H_in * W_in) \
                       + d_in[:, None] * (H_in * W_in) \
                       + h_in[:, None] * W_in + w_in[:, None]
                x_val = tl.load(x_ptr + in_idx, mask=valid[:, None], other=0.0)

                w_idx = (kd * (KH * KW) + kh * KW + kw) * (C_IN * C_out) \
                      + offs_k[:, None] * C_out \
                      + c_out[None, :]
                w_val = tl.load(weight_ptr + w_idx, mask=mask_n[None, :], other=0.0)

                acc = tl.dot(x_val, w_val, acc=acc, allow_tf32=True)

    # Add conv transpose bias
    conv_bias_val = tl.load(conv_bias_ptr + c_out, mask=mask_n, other=0.0).to(tl.float32)
    acc += conv_bias_val[None, :]

    # Post-conv elementwise: original_x = conv; x = conv+bias; x+=orig; x*=orig; x+=orig
    original_x = acc
    bias_val = tl.load(bias_ptr + c_out, mask=mask_n, other=0.0).to(tl.float32)
    result = acc + bias_val[None, :]
    result = result + original_x
    result = result * original_x
    result = result + original_x

    out_idx = n[:, None] * (C_out * D_out * H_out * W_out) \
            + c_out[None, :] * (D_out * H_out * W_out) \
            + d_out[:, None] * (H_out * W_out) \
            + h_out[:, None] * W_out + w_out[:, None]
    tl.store(out_ptr + out_idx, result, mask=mask_m[:, None] & mask_n[None, :])

def run(x):
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    bias = _weights['bias']
    conv_transpose_bias = _weights['conv_transpose.bias']

    x = x.contiguous()
    N, C_in, D_in, H_in, W_in = x.shape
    C_out = _weights['conv_transpose.weight'].shape[1]
    kD, kH, kW = _weights['conv_transpose.weight'].shape[2], _weights['conv_transpose.weight'].shape[3], _weights['conv_transpose.weight'].shape[4]
    stride = 2
    padding = 1
    output_padding = 1

    D_out = (D_in - 1) * stride - 2 * padding + kD + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding

    out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.float32)

    M_total = N * D_out * H_out * W_out
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    kernel[grid](
        x, _weight_t, conv_transpose_bias, bias, out,
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        stride, padding,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        KD=kD, KH=kH, KW=kW,
        C_IN=C_in,
        num_warps=8, num_stages=3,
    )

    return out
