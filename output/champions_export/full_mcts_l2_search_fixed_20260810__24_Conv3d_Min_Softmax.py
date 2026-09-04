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
def conv3d_dot_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_out, D, H, W, D_out, H_out, W_out,
    C_in: tl.constexpr, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_TOTAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)

    # Spatial output offsets
    offs_n = pid_spatial * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < (D_out * H_out * W_out)

    d_out = offs_n // (H_out * W_out)
    hw = offs_n % (H_out * W_out)
    h_out = hw // W_out
    w_out = hw % W_out

    # Output channel offsets
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < C_out

    # Initialize accumulator with bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
    acc += bias[:, None]

    # K loop: iterate over C_in * KD * KH * KW
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOTAL

        # Decode k -> (c_in, kd, kh, kw)
        c_in_k = offs_k // (KD * KH * KW)
        rem_k = offs_k % (KD * KH * KW)
        kd_k = rem_k // (KH * KW)
        rem2_k = rem_k % (KH * KW)
        kh_k = rem2_k // KW
        kw_k = rem2_k % KW

        # Load weights: a[BLOCK_M, BLOCK_K]
        w_idx = offs_m[:, None] * K_TOTAL + offs_k[None, :]
        a = tl.load(w_ptr + w_idx, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load input (im2col): b[BLOCK_K, BLOCK_N]
        x_idx = (
            pid_n * (C_in * D * H * W)
            + c_in_k[:, None] * (D * H * W)
            + (d_out[None, :] + kd_k[:, None]) * (H * W)
            + (h_out[None, :] + kh_k[:, None]) * W
            + (w_out[None, :] + kw_k[:, None])
        )
        b = tl.load(x_ptr + x_idx, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    # Store output
    y_idx = pid_n * (C_out * D_out * H_out * W_out) + offs_m[:, None] * (D_out * H_out * W_out) + offs_n[None, :]
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
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (H * W)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    x_idx = pid_n * (C * H * W) + offs_c[:, None] * (H * W) + offs[None, :]
    x_val = tl.load(x_ptr + x_idx, mask=mask_c[:, None] & mask[None, :], other=float('-inf'))

    x_max = tl.max(x_val, axis=0)
    x_exp = tl.exp(x_val - x_max[None, :])
    x_sum = tl.sum(x_exp, axis=0)
    y_val = x_exp / x_sum[None, :]

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

    BLOCK_SIZE = 32

    # Step 1: Conv3d (tl.dot + TF32 Tensor Core)
    conv_out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.float32)
    grid_nc = N * C_out
    grid_spatial = triton.cdiv(D_out * H_out * W_out, 128)
    conv3d_dot_kernel[N, grid_spatial](
        x, conv_weight, conv_bias, conv_out,
        N, C_out, D, H, W, D_out, H_out, W_out,
        C_in=C_in, KD=KD, KH=KH, KW=KW,
        BLOCK_M=32, BLOCK_N=128, BLOCK_K=64,
        K_TOTAL=C_in * KD * KH * KW,
        num_stages=2, num_warps=8,
    )

    # Step 2: Min along dim=2
    min_out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    grid_hw = triton.cdiv(H_out * W_out, BLOCK_SIZE)
    min_kernel[grid_nc, grid_hw](
        conv_out, min_out,
        N, C_out, H_out, W_out,
        D=D_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 3: Softmax along dim=1
    softmax_out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    grid_n = N
    grid_hw2 = triton.cdiv(H_out * W_out, BLOCK_SIZE)
    softmax_kernel[grid_n, grid_hw2](
        min_out, softmax_out,
        N, C_out, H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C=32,
    )

    return softmax_out
