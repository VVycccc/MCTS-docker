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
def conv_t3d_clamp_div_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride, padding,
    min_value, divisor,
    num_spatial_blocks,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_spatial = pid % num_spatial_blocks
    b_idx = pid // num_spatial_blocks

    offs = pid_spatial * BLOCK + tl.arange(0, BLOCK)
    spatial_size = D_out * H_out * W_out
    spatial_mask = offs < spatial_size

    d_idx = offs // (H_out * W_out)
    hw_rem = offs % (H_out * W_out)
    h_idx = hw_rem // W_out
    w_idx = hw_rem % W_out

    # Accumulator for all output channels at once: [BLOCK, C_OUT]
    acc = tl.zeros([BLOCK, C_OUT], dtype=tl.float32)

    HW_in = H_in * W_in
    DHW_in = D_in * HW_in
    KKK = K * K * K
    KK = K * K

    c_out_offs = tl.arange(0, C_OUT)

    # Tile over input channels, flatten kernel positions for better pipelining
    for k_start in range(0, C_IN, BLOCK_K):
        c_in_offs = k_start + tl.arange(0, BLOCK_K)
        c_in_mask = c_in_offs < C_IN

        for k_idx in range(K * K * K):
            kd = k_idx // (K * K)
            kh = (k_idx // K) % K
            kw = k_idx % K

            d_in_raw = d_idx + padding - kd
            h_in_raw = h_idx + padding - kh
            w_in_raw = w_idx + padding - kw
            d_ok = (d_in_raw % stride) == 0
            h_ok = (h_in_raw % stride) == 0
            w_ok = (w_in_raw % stride) == 0
            d_in = d_in_raw // stride
            h_in = h_in_raw // stride
            w_in = w_in_raw // stride
            valid = (spatial_mask & d_ok & h_ok & w_ok
                     & (d_in >= 0) & (d_in < D_in)
                     & (h_in >= 0) & (h_in < H_in)
                     & (w_in >= 0) & (w_in < W_in))

            # Gather input values: [BLOCK, BLOCK_K]
            in_offsets = (b_idx * C_IN * DHW_in + c_in_offs[None, :] * DHW_in
                          + d_in[:, None] * HW_in + h_in[:, None] * W_in + w_in[:, None])
            input_vals = tl.load(x_ptr + in_offsets, mask=valid[:, None] & c_in_mask[None, :], other=0.0)

            # Load weight tile: [BLOCK_K, C_OUT]
            w_offsets = (c_in_offs[:, None] * (C_OUT * KKK) + c_out_offs[None, :] * KKK
                         + kd * KK + kh * K + kw)
            w_vals = tl.load(w_ptr + w_offsets, mask=c_in_mask[:, None], other=0.0)

            # Tensor core matrix multiply
            acc = tl.dot(input_vals, w_vals, acc=acc, allow_tf32=True)

    # Bias, clamp, divide
    bias_vals = tl.load(b_ptr + c_out_offs)
    acc = acc + bias_vals[None, :]
    acc = tl.maximum(acc, min_value)
    acc = acc / divisor

    # Store output: [BLOCK, C_OUT]
    out_offsets = b_idx * (C_OUT * spatial_size) + c_out_offs[None, :] * spatial_size + offs[:, None]
    tl.store(out_ptr + out_offsets, acc, mask=spatial_mask[:, None])


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

    D_out = (D_in - 1) * stride - 2 * padding + kernel_size
    H_out = (H_in - 1) * stride - 2 * padding + kernel_size
    W_out = (W_in - 1) * stride - 2 * padding + kernel_size

    out = torch.empty(batch_size, out_channels, D_out, H_out, W_out,
                      device=x.device, dtype=x.dtype)

    BLOCK = 64
    BLOCK_K = 32
    spatial_size = D_out * H_out * W_out
    num_spatial_blocks = triton.cdiv(spatial_size, BLOCK)
    grid = (batch_size * num_spatial_blocks,)

    conv_t3d_clamp_div_kernel[grid](
        x, weight, bias, out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding,
        min_value, divisor,
        num_spatial_blocks,
        C_IN=in_channels,
        C_OUT=out_channels,
        K=kernel_size,
        BLOCK=BLOCK,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out
