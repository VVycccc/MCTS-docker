import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def kernel(
    x_ptr, weight_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    D_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    stride: tl.constexpr, padding: tl.constexpr, M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode offs_m -> (n, d_out, h_out, w_out)
    w_out = offs_m % W_out
    hw = offs_m // W_out
    h_out = hw % H_out
    dh = hw // H_out
    d_out = dh % D_out
    n = dh // D_out

    # Conv transpose bias
    conv_b = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        # Decode offs_k -> (c_in, kd, kh, kw)
        kw_i = offs_k % KW
        khr = offs_k // KW
        kh_i = khr % KH
        kdr = khr // KH
        kd_i = kdr % KD
        c_in_i = kdr // KD

        # Input indices [BLOCK_M, BLOCK_K]
        d_in_num = d_out[:, None] + padding - kd_i[None, :]
        h_in_num = h_out[:, None] + padding - kh_i[None, :]
        w_in_num = w_out[:, None] + padding - kw_i[None, :]

        d_in = d_in_num // stride
        h_in = h_in_num // stride
        w_in = w_in_num // stride

        valid = mask_m[:, None] & k_mask[None, :] \
              & ((d_in_num % stride) == 0) \
              & ((h_in_num % stride) == 0) \
              & ((w_in_num % stride) == 0) \
              & (d_in >= 0) & (d_in < D_in) \
              & (h_in >= 0) & (h_in < H_in) \
              & (w_in >= 0) & (w_in < W_in)

        in_idx = n[:, None] * (C_in * D_in * H_in * W_in) \
               + c_in_i[None, :] * (D_in * H_in * W_in) \
               + d_in * (H_in * W_in) + h_in * W_in + w_in

        x_val = tl.load(x_ptr + in_idx, mask=valid, other=0.0).to(tl.float16)

        # Weight index: [c_in, c_out, kd, kh, kw]
        w_idx = c_in_i[:, None] * (C_out * KD * KH * KW) \
              + offs_n[None, :] * (KD * KH * KW) \
              + kd_i[:, None] * (KH * KW) + kh_i[:, None] * KW + kw_i[:, None]

        w_val = tl.load(weight_ptr + w_idx, mask=mask_n[None, :] & k_mask[:, None], other=0.0).to(tl.float16)

        acc = tl.dot(x_val, w_val, acc=acc, allow_tf32=True)

    # Add conv bias
    acc += conv_b[None, :]

    # Post-conv epilogue simplified: result = acc * (2*acc + bias + 1)
    bias_val = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    result = acc * (2.0 * acc + bias_val[None, :] + 1.0)

    # Store
    out_idx = n[:, None] * (C_out * D_out * H_out * W_out) \
            + offs_n[None, :] * (D_out * H_out * W_out) \
            + d_out[:, None] * (H_out * W_out) \
            + h_out[:, None] * W_out + w_out[:, None]

    tl.store(out_ptr + out_idx, result, mask=mask_m[:, None] & mask_n[None, :])

def run(x):
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    bias = _weights['bias']
    conv_transpose_bias = _weights['conv_transpose.bias']
    conv_transpose_weight = _weights['conv_transpose.weight']

    x = x.contiguous()
    N, C_in, D_in, H_in, W_in = x.shape
    C_out = conv_transpose_weight.shape[1]
    kD, kH, kW = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    stride = 2
    padding = 1
    output_padding = 1

    D_out = (D_in - 1) * stride - 2 * padding + kD + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding

    out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.float32)

    M = N * D_out * H_out * W_out
    K_TOTAL = C_in * kD * kH * kW

    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    kernel[grid](
        x, conv_transpose_weight, conv_transpose_bias, bias, out,
        N=N, C_in=C_in, C_out=C_out, D_in=D_in, H_in=H_in, W_in=W_in,
        D_out=D_out, H_out=H_out, W_out=W_out,
        stride=stride, padding=padding, M=M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        KD=kD, KH=kH, KW=kW,
        K_TOTAL=K_TOTAL,
        num_stages=3,
        num_warps=8,
    )

    return out
