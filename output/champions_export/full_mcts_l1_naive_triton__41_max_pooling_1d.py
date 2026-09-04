import torch
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr, out_ptr,
    batch_features, seq_len, out_len,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_bf = tl.program_id(0)
    pid_o = tl.program_id(1)

    o_start = pid_o * BLOCK
    offs_o = o_start + tl.arange(0, BLOCK)

    x_base = x_ptr + pid_bf * seq_len
    out_base = out_ptr + pid_bf * out_len

    in_start = o_start * STRIDE - PADDING

    # Tile-level safety check
    min_idx = in_start
    max_idx = in_start + (BLOCK - 1) * STRIDE + (KERNEL_SIZE - 1) * DILATION
    tile_safe = (min_idx >= 0) & (max_idx < seq_len) & (o_start + BLOCK <= out_len)

    if tile_safe:
        # Fast path: manual unroll all 8 loads + tree reduction (3 levels)
        arange = tl.arange(0, BLOCK)
        x0 = tl.load(x_base + in_start + 0 * DILATION + arange)
        x1 = tl.load(x_base + in_start + 1 * DILATION + arange)
        x2 = tl.load(x_base + in_start + 2 * DILATION + arange)
        x3 = tl.load(x_base + in_start + 3 * DILATION + arange)
        x4 = tl.load(x_base + in_start + 4 * DILATION + arange)
        x5 = tl.load(x_base + in_start + 5 * DILATION + arange)
        x6 = tl.load(x_base + in_start + 6 * DILATION + arange)
        x7 = tl.load(x_base + in_start + 7 * DILATION + arange)
        # Tree reduction - level 1
        y0 = tl.maximum(x0, x1)
        y1 = tl.maximum(x2, x3)
        y2 = tl.maximum(x4, x5)
        y3 = tl.maximum(x6, x7)
        # Tree reduction - level 2
        z0 = tl.maximum(y0, y1)
        z1 = tl.maximum(y2, y3)
        # Tree reduction - level 3
        result = tl.maximum(z0, z1)
        tl.store(out_base + offs_o, result)
    else:
        # Slow path: with masking
        mask_o = offs_o < out_len
        arange = tl.arange(0, BLOCK)
        offs_in = in_start + arange
        mask_x = (offs_in >= 0) & (offs_in < seq_len) & mask_o
        result = tl.load(x_base + offs_in, mask=mask_x, other=float('-inf'))
        for k in range(1, KERNEL_SIZE):
            offs_k = in_start + k * DILATION + arange
            mask_k = (offs_k >= 0) & (offs_k < seq_len) & mask_o
            x_k = tl.load(x_base + offs_k, mask=mask_k, other=float('-inf'))
            result = tl.maximum(result, x_k)
        tl.store(out_base + offs_o, result, mask=mask_o)

def run(x):
    x = x.contiguous()
    batch_size, num_features, sequence_length = x.shape
    kernel_size = 8
    stride = 1
    padding = 4
    dilation = 3

    out_len = (sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty(batch_size, num_features, out_len, device=x.device, dtype=x.dtype)

    batch_features = batch_size * num_features
    BLOCK = 4096
    grid = (batch_features, triton.cdiv(out_len, BLOCK))

    maxpool1d_kernel[grid](
        x, out,
        batch_features, sequence_length, out_len,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PADDING=padding,
        DILATION=dilation,
        BLOCK=BLOCK,
        num_warps=16,
        num_stages=3,
    )

    return out
