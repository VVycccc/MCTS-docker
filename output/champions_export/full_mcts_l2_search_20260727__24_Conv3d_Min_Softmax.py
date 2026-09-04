import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/24_Conv3d_Min_Softmax_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_out, D, H, W, D_out, H_out, W_out,
    C_in: tl.constexpr, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n_ch = tl.program_id(2)

    n = pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (D_out * H_out * W_out)

    d = offs_m // (H_out * W_out)
    hw = offs_m % (H_out * W_out)
    h = hw // W_out
    w = hw % W_out

    offs_n = pid_n_ch * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out

    K = C_in * KD * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        c_in = offs_k // (KD * KH * KW)
        rem = offs_k % (KD * KH * KW)
        kd = rem // (KH * KW)
        rem2 = rem % (KH * KW)
        kh = rem2 // KW
        kw = rem2 % KW

        x_idx = (n * (C_in * D * H * W)
                 + c_in[None, :] * (D * H * W)
                 + (d[:, None] + kd[None, :]) * (H * W)
                 + (h[:, None] + kh[None, :]) * W
                 + (w[:, None] + kw[None, :]))
        x_val = tl.load(x_ptr + x_idx, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w_val = tl.load(w_ptr + w_idx, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_val, w_val, acc=acc, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    y_idx = n * (C_out * D_out * H_out * W_out) + offs_n[None, :] * (D_out * H_out * W_out) + offs_m[:, None]
    tl.store(y_ptr + y_idx, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def min_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (H * W)

    result = tl.full((BLOCK_SIZE,), float('inf'), tl.float32)

    for d_idx in range(D):
        x_idx = n * (C * D * H * W) + c * (D * H * W) + d_idx * (H * W) + offs
        x_val = tl.load(x_ptr + x_idx, mask=mask, other=float('inf'))
        result = tl.minimum(result, x_val)

    y_idx = n * (C * H * W) + c * (H * W) + offs
    tl.store(y_ptr + y_idx, result, mask=mask)


@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    # Pass 1: find max over all channels
    x_max = tl.full((BLOCK_HW,), float('-inf'), tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        x_idx = pid_n * (C * H * W) + offs_c[:, None] * (H * W) + offs[None, :]
        x_val = tl.load(x_ptr + x_idx, mask=mask_c[:, None] & mask[None, :], other=float('-inf'))
        x_max = tl.maximum(x_max, tl.max(x_val, axis=0))

    # Pass 2: compute sum of exp
    x_sum = tl.zeros((BLOCK_HW,), tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        x_idx = pid_n * (C * H * W) + offs_c[:, None] * (H * W) + offs[None, :]
        x_val = tl.load(x_ptr + x_idx, mask=mask_c[:, None] & mask[None, :], other=float('-inf'))
        x_exp = tl.exp(x_val - x_max[None, :])
        x_sum += tl.sum(x_exp, axis=0)

    # Pass 3: normalize and store
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        x_idx = pid_n * (C * H * W) + offs_c[:, None] * (H * W) + offs[None, :]
        x_val = tl.load(x_ptr + x_idx, mask=mask_c[:, None] & mask[None, :], other=float('-inf'))
        y_val = tl.exp(x_val - x_max[None, :]) / x_sum[None, :]
        tl.store(y_ptr + x_idx, y_val, mask=mask_c[:, None] & mask[None, :])


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']

    N, C_in, D, H, W = x.shape
    C_out, _, KD, KH, KW = conv_weight.shape
    D_out = D - KD + 1
    H_out = H - KH + 1
    W_out = W - KW + 1

    # Step 1: Conv3d using tl.dot (tensor cores)
    conv_out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32

    grid_n = N
    grid_m = triton.cdiv(D_out * H_out * W_out, BLOCK_M)
    grid_n_ch = triton.cdiv(C_out, BLOCK_N)

    conv3d_kernel[grid_n, grid_m, grid_n_ch](
        x, conv_weight, conv_bias, conv_out,
        N, C_out, D, H, W, D_out, H_out, W_out,
        C_in=C_in, KD=KD, KH=KH, KW=KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Step 2: Min along dim=2
    min_out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 128
    grid_nc = N * C_out
    grid_hw = triton.cdiv(H_out * W_out, BLOCK_SIZE)
    min_kernel[grid_nc, grid_hw](
        conv_out, min_out,
        N, C_out, H_out, W_out,
        D=D_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 3: Softmax along dim=1
    softmax_out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    BLOCK_HW = 128
    BLOCK_C = 32
    grid_n_s = N
    grid_hw2 = triton.cdiv(H_out * W_out, BLOCK_HW)
    softmax_kernel[grid_n_s, grid_hw2](
        min_out, softmax_out,
        N, C_out, H_out, W_out,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
    )

    return softmax_out
