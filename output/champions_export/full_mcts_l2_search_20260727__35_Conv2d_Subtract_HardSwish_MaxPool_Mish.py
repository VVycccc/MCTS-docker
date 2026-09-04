import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/35_Conv2d_Subtract_HardSwish_MaxPool_Mish_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 128, 'BLOCK_W': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_W': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_W': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['out_channels', 'H_out', 'W_out'],
)
@triton.jit
def conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels, H, W, H_out, W_out,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_w = tl.program_id(2)

    pid_b = pid_bh // H_out
    pid_oh = pid_bh % H_out

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_oc = offs_oc < out_channels
    mask_w = offs_w < W_out

    acc = tl.zeros([BLOCK_OC, BLOCK_W], dtype=tl.float32)

    K = in_channels * 9
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        ic = offs_k // 9
        kh = (offs_k % 9) // 3
        kw = offs_k % 3

        w_ptrs = w_ptr + offs_oc[:, None] * K + offs_k[None, :]
        w_val = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)

        ih = pid_oh + kh
        iw = offs_w[None, :] + kw[:, None]
        x_ptrs = x_ptr + pid_b * in_channels * H * W + ic[:, None] * H * W + ih[:, None] * W + iw
        x_val = tl.load(x_ptrs, mask=mask_k[:, None] & mask_w[None, :], other=0.0)

        acc = tl.dot(w_val, x_val, acc=acc, allow_tf32=True)

    b_val = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += b_val[:, None]

    acc = acc - 0.5
    acc = acc * tl.where(acc >= 3.0, 1.0, tl.where(acc <= -3.0, 0.0, (acc + 3.0) / 6.0))

    out_ptrs = out_ptr + pid_b * out_channels * H_out * W_out + offs_oc[:, None] * H_out * W_out + pid_oh * W_out + offs_w[None, :]
    tl.store(out_ptrs, acc, mask=mask_oc[:, None] & mask_w[None, :])


@triton.jit
def maxpool_mish_kernel(
    x_ptr, out_ptr, N, C, H, W, H_out, W_out,
    BLOCK_NCH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)

    total_nch = N * C * H_out
    offs_nch = pid * BLOCK_NCH + tl.arange(0, BLOCK_NCH)
    mask_nch = offs_nch < total_nch

    pid_n = offs_nch // (C * H_out)
    rem = offs_nch % (C * H_out)
    pid_c = rem // H_out
    pid_oh = rem % H_out

    offs_w = tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    ih0 = pid_oh * 2
    iw0 = offs_w * 2

    base = pid_n * C * H * W + pid_c * H * W + ih0 * W
    v00 = tl.load(x_ptr + base[:, None] + iw0[None, :], mask=mask_nch[:, None] & mask_w[None, :], other=-1.0e30)
    v01 = tl.load(x_ptr + base[:, None] + iw0[None, :] + 1, mask=mask_nch[:, None] & mask_w[None, :], other=-1.0e30)
    v10 = tl.load(x_ptr + base[:, None] + W + iw0[None, :], mask=mask_nch[:, None] & mask_w[None, :], other=-1.0e30)
    v11 = tl.load(x_ptr + base[:, None] + W + iw0[None, :] + 1, mask=mask_nch[:, None] & mask_w[None, :], other=-1.0e30)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    sp = tl.where(result > 20.0, result, tl.log(1.0 + tl.exp(result)))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    result = result * tanh_sp

    out_idx = pid_n * C * H_out * W_out + pid_c * H_out * W_out + pid_oh * W_out
    tl.store(out_ptr + out_idx[:, None] + offs_w[None, :], result, mask=mask_nch[:, None] & mask_w[None, :])


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']

    batch_size, in_channels, H, W = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = 3
    H_out = H - kernel_size + 1
    W_out = W - kernel_size + 1

    x = x.contiguous()

    # Step 1: Conv2d + Subtract + HardSwish (fused)
    hs_out = torch.empty(batch_size, out_channels, H_out, W_out, device=x.device, dtype=torch.float32)
    grid_conv = lambda meta: (batch_size * H_out, triton.cdiv(out_channels, meta['BLOCK_OC']), triton.cdiv(W_out, meta['BLOCK_W']))
    conv2d_fused_kernel[grid_conv](
        x, conv_weight, conv_bias, hs_out,
        batch_size, in_channels, out_channels, H, W, H_out, W_out,
    )

    # Step 2: MaxPool2d + Mish (fused)
    pool_H_out = H_out // 2
    pool_W_out = W_out // 2
    out = torch.empty(batch_size, out_channels, pool_H_out, pool_W_out, device=x.device, dtype=torch.float32)
    BLOCK_NCH = 16
    BLOCK_W = 64
    grid_pool = (triton.cdiv(batch_size * out_channels * pool_H_out, BLOCK_NCH),)
    maxpool_mish_kernel[grid_pool](
        hs_out, out,
        batch_size, out_channels, H_out, W_out, pool_H_out, pool_W_out,
        BLOCK_NCH=BLOCK_NCH, BLOCK_W=BLOCK_W,
        num_warps=8,
    )

    return out
