import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/15_ConvTranspose3d_BatchNorm_Subtract_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, scale_ptr, shift_ptr, out_ptr,
    N, C_in, D_in, H_in, W_in, D_out, H_out, W_out,
    stride, padding, spatial_size,
    K: tl.constexpr, K3: tl.constexpr, C_IN: tl.constexpr, C_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    total_m = N * D_out * H_out * W_out
    mask_m = offs_m < total_m
    mask_n = offs_n < C_OUT

    ow = offs_m % W_out
    tmp = offs_m // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    n = tmp // D_out

    base_out = n * C_OUT * spatial_size + od * H_out * W_out + oh * W_out + ow

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_IN * K3, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < C_IN * K3

        kw = offs_k % K
        tmp_k = offs_k // K
        kh = tmp_k % K
        tmp_k = tmp_k // K
        kd = tmp_k % K
        c_in = tmp_k // K

        id_num = od[:, None] + padding - kd[None, :]
        ih_num = oh[:, None] + padding - kh[None, :]
        iw_num = ow[:, None] + padding - kw[None, :]

        valid = (id_num % stride == 0) & (ih_num % stride == 0) & (iw_num % stride == 0)
        id_val = id_num // stride
        ih_val = ih_num // stride
        iw_val = iw_num // stride
        valid = valid & (id_val >= 0) & (id_val < D_in) & (ih_val >= 0) & (ih_val < H_in) & (iw_val >= 0) & (iw_val < W_in)
        valid = valid & mask_m[:, None] & mask_k[None, :]

        id_val = tl.where(valid, id_val, 0)
        ih_val = tl.where(valid, ih_val, 0)
        iw_val = tl.where(valid, iw_val, 0)

        x_idx = ((n[:, None] * C_IN + c_in[None, :]) * D_in + id_val) * H_in * W_in + ih_val * W_in + iw_val
        a_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)

        w_idx = ((c_in[:, None] * C_OUT + offs_n[None, :]) * K + kd[:, None]) * K * K + kh[:, None] * K + kw[:, None]
        b_val = tl.load(w_ptr + w_idx, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc = tl.dot(a_val, b_val, acc=acc, allow_tf32=True)

    scale_val = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    shift_val = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc * scale_val[None, :] + shift_val[None, :]

    out_idx = base_out[:, None] + offs_n[None, :] * spatial_size
    tl.store(out_ptr + out_idx, acc, mask=mask_m[:, None] & mask_n[None, :])

def run(x):
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()

    conv_weight = _weights['conv_transpose.weight']
    conv_bias = _weights['conv_transpose.bias']
    bn_weight = _weights['batch_norm.weight']
    bn_bias = _weights['batch_norm.bias']
    bn_running_mean = _weights['batch_norm.running_mean']
    bn_running_var = _weights['batch_norm.running_var']

    N, C_in, D_in, H_in, W_in = x.shape
    C_out = conv_weight.shape[1]
    K = conv_weight.shape[2]
    stride = 2
    padding = 1

    D_out = (D_in - 1) * stride - 2 * padding + K
    H_out = (H_in - 1) * stride - 2 * padding + K
    W_out = (W_in - 1) * stride - 2 * padding + K

    eps = 1e-5
    bn_std = torch.sqrt(bn_running_var + eps)
    bn_scale = bn_weight / bn_std
    bn_shift = bn_bias - bn_running_mean * bn_scale
    combined_scale = bn_scale.contiguous()
    combined_shift = (conv_bias * bn_scale + bn_shift).contiguous()

    conv_out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.float32)
    spatial_size = D_out * H_out * W_out
    total_m = N * spatial_size
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_K = 64
    K3 = K * K * K
    grid = (triton.cdiv(total_m, BLOCK_M),)
    conv_transpose3d_gemm_kernel[grid](
        x, conv_weight, combined_scale, combined_shift, conv_out,
        N, C_in, D_in, H_in, W_in, D_out, H_out, W_out,
        stride, padding, spatial_size,
        K=K, K3=K3, C_IN=C_in, C_OUT=C_out,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=2, num_warps=4,
    )

    spatial_mean = conv_out.mean(dim=(2, 3, 4), keepdim=True)
    conv_out = conv_out - spatial_mean

    return conv_out
